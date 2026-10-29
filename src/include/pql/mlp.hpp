// mlp.hpp — The dense learner: a two-layer network over the compiled features.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/kernels.hpp"
#include "pql/learner.hpp"

namespace pql {

// A two-layer network over [self features | per-link neighbour means], which is
// one round of GraphSAGE-style aggregation with the neighbourhood restricted to
// rows visible at the anchor. Kept dense and small: the compiled aggregates
// carry the history, so depth buys much less here than width of the feature
// vocabulary does.
struct MlpLearner : Learner {
	int hidden = 64;
	std::vector<float> w1, b1; // width x hidden
	std::vector<float> w2, b2; // hidden x hidden
	std::vector<float> w3, b3; // hidden x 1

	void Init(int width, int hid, uint64_t seed) {
		hidden = hid;
		w1.assign((size_t)width * hid, 0.0f);
		b1.assign((size_t)hid, 0.0f);
		w2.assign((size_t)hid * hid, 0.0f);
		b2.assign((size_t)hid, 0.0f);
		w3.assign((size_t)hid, 0.0f);
		b3.assign(1, 0.0f);
		uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
		auto rnd = [&]() {
			st ^= st << 13;
			st ^= st >> 7;
			st ^= st << 17;
			return double(int64_t(st >> 11)) / double(1ULL << 53);
		};
		const double s1 = std::sqrt(2.0 / std::max(1, width));
		for (auto &v : w1) {
			v = float((rnd() * 2 - 1) * s1);
		}
		const double s2 = std::sqrt(2.0 / std::max(1, hid));
		for (auto &v : w2) {
			v = float((rnd() * 2 - 1) * s2);
		}
		for (auto &v : w3) {
			v = float((rnd() * 2 - 1) * s2);
		}
	}

	const char *Arch() const override {
		return "mlp";
	}
	bool Explainable() const override {
		return true;
	}
	size_t ParamBytes() const override {
		return (w1.size() + b1.size() + w2.size() + b2.size() + w3.size() + b3.size()) * sizeof(float);
	}
	void EntityInputs(const FeatureSpec &features, std::vector<std::string> &numeric,
	                  const std::vector<FeatureSpec::CatCol> *&cats) const override {
		for (const auto &sc : features.self_cols) {
			numeric.push_back(sc.name);
		}
		cats = &features.cat_cols;
	}

	// Score an index set in batches through MatMulNT. The scratch is grown on the
	// first call and reused, so an epoch's validation pass performs no allocation.
	void ScoreInto(const Dataset &ds, const Target &t, const std::vector<size_t> &idx, EvalScratch &scratch,
	               std::vector<double> &out, const CancelCheck &cancelled = CancelCheck()) const {
		const int width = ds.width;
		const size_t cap = EvalScratch::ROWS;
		out.assign(idx.size(), 0.0);
		scratch.Ensure(cap, hidden, width);
		float *const xb = scratch.x.data();
		for (size_t base = 0; base < idx.size(); base += cap) {
			const int B = (int)std::min(cap, idx.size() - base);
			if (cancelled) {
				cancelled();
			}
			for (int i = 0; i < B; i++) {
				const float *row = ds.Row(idx[base + (size_t)i]);
				std::memcpy(xb + (size_t)i * (size_t)width, row, sizeof(float) * (size_t)width);
			}
			MatMulNT(xb, w1.data(), b1.data(), scratch.h1.data(), B, width, hidden, true);
			MatMulNT(scratch.h1.data(), w2.data(), b2.data(), scratch.h2.data(), B, hidden, hidden, true);
			scratch.o.resize((size_t)B);
			OutputLayer(scratch.h2.data(), w3.data(), b3[0], scratch.o.data(), B, hidden);
			for (int i = 0; i < B; i++) {
				const double o = scratch.o[(size_t)i];
				out[base + (size_t)i] = t.Decode(o, ds.examples[idx[base + (size_t)i]].base);
			}
		}
	}
	void Score(const ScoreContext &ctx, const std::vector<size_t> &rows, std::vector<double> &out) const override {
		EvalScratch scratch;
		ScoreInto(ctx.ds, ctx.target, rows, scratch, out, ctx.cancelled);
	}

	void Fit(const FitContext &ctx, TrainReport &rep) override {
		const Dataset &ds = ctx.ds;
		const Statement &stmt = ctx.stmt;
		const bool binary = ctx.target.classification;
		const CancelCheck &cancelled = ctx.cancelled;
		const int epochs = (int)stmt.options.Num("EPOCHS", 60);
		const double lr = stmt.options.Num("LR", 0.01);
		const double wd = stmt.options.Num("L2", 0.0);
		const uint64_t seed = ctx.seed;
		const int width = ctx.features.width;
		Init(width, (int)stmt.options.Num("HIDDEN", 64), seed);
		std::vector<size_t> tr = ctx.train; // shuffled in place each epoch
		const std::vector<size_t> &va = ctx.val, &te = ctx.test;
		const size_t batch = (size_t)std::max(1.0, stmt.options.Num("BATCH", 64));
		// One block for the whole run: activations, gradients and the batch's
		// labels. The working set is a function of (batch, width, hidden), so it is
		// carved once here and never grown; the block is sized by this same carve
		// (Arena::Carve), which is what keeps a buffer from being left out of it.
		const size_t bw = batch * (size_t)width, bh = batch * (size_t)hidden;
		const size_t n_w1 = (size_t)width * (size_t)hidden, n_w2 = (size_t)hidden * (size_t)hidden;
		Arena arena;
		float *X = nullptr, *H1 = nullptr, *H2 = nullptr, *DH2 = nullptr, *DH1 = nullptr;
		float *DO = nullptr, *YB = nullptr, *OB = nullptr;
		float *gw1 = nullptr, *gb1 = nullptr, *gw2 = nullptr, *gb2 = nullptr, *gw3 = nullptr;
		float *gb3 = nullptr;
		arena.Carve([&](Arena &a) {
			X = a.Zeroed<float>(bw);
			H1 = a.Zeroed<float>(bh);
			H2 = a.Zeroed<float>(bh);
			DH2 = a.Zeroed<float>(bh);
			DH1 = a.Zeroed<float>(bh);
			DO = a.Zeroed<float>(batch);
			YB = a.Zeroed<float>(batch);
			OB = a.Zeroed<float>(batch);
			gw1 = a.Zeroed<float>(n_w1);
			gb1 = a.Zeroed<float>((size_t)hidden);
			gw2 = a.Zeroed<float>(n_w2);
			gb2 = a.Zeroed<float>((size_t)hidden);
			gw3 = a.Zeroed<float>((size_t)hidden);
			gb3 = a.Zeroed<float>(1);
		});
		EvalScratch escratch;
		std::vector<double> scores;

		Adam a1, a2, a3, ab1, ab2, ab3;
		const size_t n_gw1 = n_w1, n_gw2 = n_w2;

		// The best epoch's weights, copied in place so an improving epoch allocates nothing.
		std::vector<float> bw1 = w1, bb1 = b1, bw2 = w2, bb2 = b2, bw3 = w3, bb3 = b3;
		// Regression scores as -MAE, which is negative, so the sentinel must be
		// below every attainable value or selection silently never fires.
		double best_val = -std::numeric_limits<double>::infinity();
		bool have_val = false;
		rep.width = width;

		uint64_t rng = seed | 1ULL;
		auto shuffle = [&](std::vector<size_t> &v) {
			for (size_t i = v.size(); i > 1; i--) {
				rng ^= rng << 13;
				rng ^= rng >> 7;
				rng ^= rng << 17;
				std::swap(v[i - 1], v[rng % i]);
			}
		};

		// Batch-major scratch, allocated once and reused every step.

		for (int ep = 0; ep < epochs; ep++) {
			shuffle(tr);
			for (size_t bstart = 0; bstart < tr.size(); bstart += batch) {
				const size_t bend = std::min(bstart + batch, tr.size());
				if (cancelled) {
					cancelled();
				}
				const int B = (int)(bend - bstart);
				for (int i = 0; i < B; i++) {
					const float *xv = ds.Row(tr[bstart + (size_t)i]);
					std::memcpy(X + (size_t)i * (size_t)width, xv, sizeof(float) * (size_t)width);
					YB[(size_t)i] = (float)ds.examples[tr[bstart + (size_t)i]].label;
				}
				MatMulNT(X, w1.data(), b1.data(), H1, B, width, hidden, true);
				MatMulNT(H1, w2.data(), b2.data(), H2, B, hidden, hidden, true);

				OutputLayer(H2, w3.data(), b3[0], OB, B, hidden);
				for (int i = 0; i < B; i++) {
					const double o = OB[(size_t)i];
					double y = YB[(size_t)i];
					if (binary) {
						DO[(size_t)i] = float(1.0 / (1.0 + std::exp(-o)) - y);
					} else {
						y = ctx.target.Encode(y);
						// Huber, not squared: the reported metric is MAE, which is the
						// conditional median, and squared loss fits the mean. On skewed
						// counts those differ a lot.
						const double d = o - y;
						DO[(size_t)i] = float(d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d));
					}
				}

				std::memset(gw1, 0, sizeof(float) * n_gw1);
				std::memset(gb1, 0, sizeof(float) * (size_t)hidden);
				std::memset(gw2, 0, sizeof(float) * n_gw2);
				std::memset(gb2, 0, sizeof(float) * (size_t)hidden);
				std::memset(gw3, 0, sizeof(float) * (size_t)hidden);
				gb3[0] = 0.0f;

				{
					// gw3[j] += DO[b]*H2[b][j] and dh2[b][j] = gated DO[b]*w3[j], for
					// every b and j. Written the obvious way round, with b outermost, the
					// accumulate into gw3 is a read-modify-write of the same 64 floats on
					// every row, and the compiler has to keep it in memory: it measured
					// 1.7 GFLOP/s. Blocking by j instead holds that slice of gw3 (and of
					// w3) in registers for the whole pass over the batch, which is the
					// same 13x this file gets everywhere else from choosing which axis
					// the innermost loop walks.
					const float *__restrict w3p = w3.data();
					float *__restrict g3 = gw3;
					for (int j0 = 0; j0 < hidden; j0 += 16) {
						const int jn = std::min(16, hidden - j0);
						float g[16], w[16];
						for (int y = 0; y < jn; y++) {
							g[y] = g3[j0 + y];
							w[y] = w3p[j0 + y];
						}
						for (int i = 0; i < B; i++) {
							const float d = DO[(size_t)i];
							const float *__restrict h2 = &H2[(size_t)i * (size_t)hidden] + j0;
							float *__restrict dh2r = &DH2[(size_t)i * (size_t)hidden] + j0;
							for (int y = 0; y < jn; y++) {
								g[y] += d * h2[y];
								dh2r[y] = (h2[y] > 0.0f) ? d * w[y] : 0.0f;
							}
						}
						for (int y = 0; y < jn; y++) {
							g3[j0 + y] = g[y];
						}
					}
					float gb = 0.0f;
					for (int i = 0; i < B; i++) {
						gb += DO[(size_t)i];
					}
					gb3[0] += gb;
				}
				AccumOuter(DH2, H1, gw2, B, hidden, hidden);
				for (int i = 0; i < B; i++) {
					const float *dh2 = &DH2[(size_t)i * (size_t)hidden];
					for (int j = 0; j < hidden; j++) {
						gb2[(size_t)j] += dh2[j];
					}
				}
				BackThroughRelu(DH2, w2.data(), H1, DH1, B, hidden, hidden);
				AccumOuter(DH1, X, gw1, B, width, hidden);
				for (int i = 0; i < B; i++) {
					const float *dh1 = &DH1[(size_t)i * (size_t)hidden];
					for (int j = 0; j < hidden; j++) {
						gb1[(size_t)j] += dh1[j];
					}
				}

				const float scale = 1.0f / float(std::max(1, B));
				for (size_t i = 0; i < n_gw1; i++) {
					gw1[i] *= scale;
				}
				for (int i = 0; i < hidden; i++) {
					gb1[i] *= scale;
					gb2[i] *= scale;
					gw3[i] *= scale;
				}
				for (size_t i = 0; i < n_gw2; i++) {
					gw2[i] *= scale;
				}
				gb3[0] *= scale;

				a1.Step(w1, gw1, lr, wd);
				ab1.Step(b1, gb1, lr);
				a2.Step(w2, gw2, lr, wd);
				ab2.Step(b2, gb2, lr);
				a3.Step(w3, gw3, lr, wd);
				ab3.Step(b3, gb3, lr);
			}
			rep.epochs_run = ep + 1;
			if (cancelled) {
				cancelled();
			}

			if (!va.empty()) {
				ScoreInto(ds, ctx.target, va, escratch, scores);
				const double m = FoldMetric(ds, va, scores, ctx.target);
				// >= so a later epoch wins ties; NaN (undefined metric) falls back to the
				// most recent weights rather than freezing at epoch 1, but only while no
				// epoch has produced a defined metric: after one has, a NaN epoch is a
				// diverged model, and it used to replace the best one.
				if (std::isnan(m) ? !std::isfinite(best_val) : !(m < best_val)) {
					best_val = std::isnan(m) ? best_val : m;
					bw1 = w1;
					bb1 = b1;
					bw2 = w2;
					bb2 = b2;
					bw3 = w3;
					bb3 = b3;
					have_val = true;
				}
			}
		}
		if (have_val) {
			w1 = bw1;
			b1 = bb1;
			w2 = bw2;
			b2 = bb2;
			w3 = bw3;
			b3 = bb3;
			// -inf means no epoch ever produced a defined metric (a single-class fold),
			// which must read as "undefined", not as a very bad score.
			rep.val_metric =
			    std::isfinite(best_val) ? (binary ? best_val : -best_val) : std::numeric_limits<double>::quiet_NaN();
		}
		if (!te.empty()) {
			ScoreInto(ds, ctx.target, te, escratch, scores);
			ReportTest(ds, te, scores, ctx.target, rep);
		}
	}
};

} // namespace pql
