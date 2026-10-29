// sage.hpp — Heterogeneous GraphSAGE: spec, features, prefix sums and the training kernels.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/features.hpp"
#include "pql/kernels.hpp"
#include "pql/learner.hpp"

namespace pql {

// ===========================================================================
// 9b. Heterogeneous GraphSAGE
// ===========================================================================
//
// Transposed from RelML's HeteroGraphSAGE: a W_self per node type, a W_neigh
// per edge type, mean aggregation over neighbours, ReLU between layers.
//
//     h_v = ReLU( W_self . x_v  +  SUM_r W_r . mean_{u in N_r(v)} x_u )
//
// Two things are done differently here, both because the CSR arena is sorted by
// time.
//
// 1. Aggregation is EXACT AND TEMPORAL. RelML has no temporal neighbour
//    sampling, which is why its own benchmark has to keep fact tables out of the
//    graph or a 2010 row message-passes into a 2015 prediction. Here the
//    neighbours visible at an anchor are a prefix of a time-sorted slice, so a
//    running sum over that slice gives the mean of any prefix by subtracting two
//    points. Nothing after the anchor can enter, by construction.
//
// 2. That also makes it O(C) per node instead of O(degree * C): the running sums
//    are built once per epoch and every entity reads two rows out of them.
//
// Layout is chosen for the cache: node features and running sums are row-major
// [N x C] in one arena block, so an aggregation is a contiguous subtract and a
// batch of them is a GEMM through the same kernels the dense layers use.

struct SageSpec {
	struct NodeType {
		int table = -1;
		// Names alongside the indices. Table and column positions are per
		// statement, because the loader reads only what a statement can use, so a
		// stored index means nothing in a later one.
		std::string table_name;
		std::vector<std::string> col_names;
		std::vector<int> cols;      // numeric columns encoding this node type
		std::vector<double> mu, sd; // standardisation, fitted on the train fold
		// Text columns, encoded exactly as the dense path encodes them. They come
		// after the numeric block, so `dim` is the two together.
		std::vector<FeatureSpec::CatCol> cats;
		int dim = 0;
	};
	struct EdgeType {
		int link = -1;         // index into Database::links
		int src_node = -1;     // index into nodes: the child
		int dst_node = -1;     // index into nodes: the parent (the entity)
		std::string child_key; // the child's key column, to tell two links apart
	};
	std::vector<NodeType> nodes;
	std::vector<EdgeType> edges;
	int entity_node = -1;

	int NodeDim(int n) const {
		return nodes[(size_t)n].dim;
	}
};

// Raw node features, [N x dim] row-major, one block per node type.
struct SageFeatures {
	std::vector<std::vector<float>> x; // per node type
};

// Running sums of a source node type's features along one link's CSR order.
// psum[e] holds the sum of the first (e - base) children of that parent, so the
// mean over [lo, hi) is (psum[base+hi] - psum[base+lo]) / (hi - lo).
struct SagePrefix {
	std::vector<std::vector<float>> psum; // per edge type, [(E + P) x src_dim]
	std::vector<std::vector<uint32_t>> off;
};

inline SageFeatures BuildSageFeatures(const Database &db, const SageSpec &spec) {
	SageFeatures f;
	f.x.resize(spec.nodes.size());
	for (size_t n = 0; n < spec.nodes.size(); n++) {
		const SageSpec::NodeType &nt = spec.nodes[n];
		const Frame &fr = db.tables[(size_t)nt.table];
		f.x[n].assign(fr.nrows * (size_t)nt.dim, 0.0f);
		const int stride = nt.dim;
		for (size_t c = 0; c < nt.cols.size(); c++) {
			const Column &col = fr.columns[(size_t)nt.cols[c]];
			const double mu = nt.mu[c], sd = nt.sd[c];
			float *dst = f.x[n].data() + c;
			for (size_t r = 0; r < fr.nrows; r++) {
				dst[r * (size_t)stride] = col.valid[r] ? float((col.num[r] - mu) / sd) : 0.0f;
			}
		}
		size_t base = nt.cols.size();
		for (const auto &cc : nt.cats) {
			const Column &col = fr.columns[(size_t)cc.index];
			float *dst = f.x[n].data() + base;
			for (size_t r = 0; r < fr.nrows; r++) {
				const uint32_t code = col.valid[r] ? col.code[r] : 0u;
				const bool known = code < cc.slot_by_code.size();
				if (cc.one_hot) {
					const int slot = (known && col.valid[r]) ? cc.slot_by_code[code] : -1;
					if (slot >= 0) {
						dst[r * (size_t)stride + (size_t)slot] = 1.0f;
					} else if (cc.has_other) {
						dst[r * (size_t)stride + cc.labels.size()] = 1.0f;
					}
				} else {
					const double fq = (known && col.valid[r]) ? (double)cc.freq_by_code[code] : cc.freq_other;
					dst[r * (size_t)stride] = float((fq - cc.freq_mean) / cc.freq_sd);
				}
			}
			base += (size_t)cc.Width();
		}
	}
	return f;
}

inline SagePrefix BuildSagePrefix(const Database &db, const SageSpec &spec, const SageFeatures &feat) {
	SagePrefix pre;
	pre.psum.resize(spec.edges.size());
	pre.off.resize(spec.edges.size());
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const SageSpec::EdgeType &et = spec.edges[e];
		const Link &lk = db.links[(size_t)et.link];
		const int d = spec.NodeDim(et.src_node);
		const size_t nparents = lk.off.empty() ? 0 : lk.off.size() - 1;
		pre.off[e].assign(nparents + 1, 0u);
		PrefixTotal(lk);
		uint32_t total = 0;
		for (size_t p = 0; p < nparents; p++) {
			pre.off[e][p] = total;
			total += lk.Count((uint32_t)p) + 1u;
		}
		pre.off[e][nparents] = total;
		pre.psum[e].assign((size_t)total * (size_t)d, 0.0f);
		const float *src = feat.x[(size_t)et.src_node].data();
		for (size_t p = 0; p < nparents; p++) {
			const uint32_t b = lk.Begin((uint32_t)p), en = lk.End((uint32_t)p);
			float *run = pre.psum[e].data() + (size_t)pre.off[e][p] * (size_t)d;
			// run[0..d) is the zero row; each step adds one child, contiguous in d.
			for (uint32_t i = b; i < en; i++) {
				const float *xu = src + (size_t)lk.flat[i] * (size_t)d;
				float *cur = run + (size_t)(i - b + 1) * (size_t)d;
				const float *prev = run + (size_t)(i - b) * (size_t)d;
				for (int k = 0; k < d; k++) {
					cur[k] = prev[k] + xu[k];
				}
			}
		}
	}
	return pre;
}

// ---------------------------------------------------------------------------
// Second hop.
//
// A child's embedding is computed at the child's OWN timestamp, aggregating its
// own children up to that instant. Since every visible child satisfies
// t_child <= anchor, this cannot see past the anchor, and because it does not
// depend on the anchor it is computed once per epoch rather than per example.
// That is what makes an exact two-hop affordable; RelML sidesteps the question
// by not sampling temporally at all.
// ---------------------------------------------------------------------------

// How many of an edge's children are visible at each example's anchor.
//
// That count is a binary search over the child link, and it depends only on the
// example's entity row and anchor: both fixed for the whole run. Recomputing it
// per epoch cost two searches per example per epoch, each probing a timestamp
// through an index array. Here it is found once.
struct VisibleCounts {
	std::vector<std::vector<uint32_t>> n; // per edge, per example
	uint32_t At(size_t e, size_t ex) const {
		return n[e][ex];
	}
};

struct SageHop2 {
	// Per entity-edge: the transformed embedding of each child row, [N_child x C].
	std::vector<std::vector<float>> h_child;
	// Prefix sums of those embeddings along the entity link's CSR order.
	std::vector<std::vector<float>> psum;
	std::vector<std::vector<uint32_t>> off;
	bool active = false;
};

// Parameters of the child layer: one W_self per child node type, one W_neigh per
// grandchild edge. Mirrors SageParams but is indexed per entity-edge.
struct SageChildParams {
	int channels = 0;
	std::vector<int> self_dim;                         // per entity-edge
	std::vector<std::vector<float>> w_self;            // [C x self_dim]
	std::vector<std::vector<float>> bias;              // [C]
	std::vector<std::vector<int>> gc_link;             // grandchild link ids
	std::vector<std::vector<int>> gc_dim;              // + degree slot
	std::vector<std::vector<std::vector<float>>> w_gc; // [C x gc_dim]

	size_t ParamCount() const {
		size_t n = 0;
		for (const auto &w : w_self) {
			n += w.size();
		}
		for (const auto &b : bias) {
			n += b.size();
		}
		for (const auto &e : w_gc) {
			for (const auto &w : e) {
				n += w.size();
			}
		}
		return n;
	}
};

// Mean of a neighbourhood visible at `anchor`, written into `out` (length d).
inline void SageMeanAt(const SagePrefix &pre, size_t e, uint32_t parent_row, size_t visible, float *out, int d) {
	const uint32_t base = pre.off[e][parent_row];
	const float *hi = pre.psum[e].data() + (size_t)(base + visible) * (size_t)d;
	const float *lo = pre.psum[e].data() + (size_t)base * (size_t)d;
	if (visible == 0) {
		for (int k = 0; k <= d; k++) {
			out[k] = 0.0f;
		}
		return;
	}
	const float inv = 1.0f / float(visible);
	for (int k = 0; k < d; k++) {
		out[k] = (hi[k] - lo[k]) * inv;
	}
	// Degree rides along with the mean. A mean is scale-stable but says nothing
	// about how many neighbours produced it, and on relational data the count is
	// usually the signal. RelML aggregates by mean alone; RelBench's RDL sums,
	// which keeps degree implicitly. Carrying both keeps mean's conditioning and
	// makes the count learnable.
	out[d] = float(std::log1p(double(visible)));
}

// ---------------------------------------------------------------------------
// Step 1: build the node/edge schema the layers will run over.
// ---------------------------------------------------------------------------

// Numeric columns of a table that can encode a node, excluding keys and clocks
// (an id is identity, not signal; a timestamp is the axis, not a feature).
inline void SageNodeColumns(const Database &db, int table, const Link *incoming, std::vector<int> &out) {
	const Frame &f = db.tables[(size_t)table];
	for (size_t c = 0; c < f.columns.size(); c++) {
		if (!f.columns[c].IsNumeric()) {
			continue;
		}
		if (incoming && ((int)c == incoming->child_key_col || (int)c == incoming->child_time_col)) {
			continue;
		}
		bool is_key = false;
		for (const auto &fk : db.fks) {
			const bool child_side =
			    ToUpper(fk.child_table) == ToUpper(f.name) && ToUpper(fk.child_column) == ToUpper(f.columns[c].name);
			const bool parent_side =
			    ToUpper(fk.parent_table) == ToUpper(f.name) && ToUpper(fk.parent_column) == ToUpper(f.columns[c].name);
			if (child_side || parent_side) {
				is_key = true;
				break;
			}
		}
		if (!is_key) {
			out.push_back((int)c);
		}
	}
}

inline SageSpec BuildSageSpec(const Database &db, const Frame &entity, const Statement &stmt, int anchor_col,
                              int target_col, double train_cutoff) {
	SageSpec spec;
	auto add_node = [&](int table, const Link *incoming) {
		for (size_t n = 0; n < spec.nodes.size(); n++) {
			if (spec.nodes[n].table == table) {
				return (int)n;
			}
		}
		SageSpec::NodeType nt;
		nt.table = table;
		nt.table_name = db.tables[(size_t)table].name;
		SageNodeColumns(db, table, incoming, nt.cols);
		{
			// EXCLUDE means the same thing whichever architecture is asked for.
			const Frame &nf = db.tables[(size_t)table];
			std::vector<int> keep;
			for (int c : nt.cols) {
				if (!IsExcluded(stmt, entity, nf.name, nf.columns[(size_t)c].name)) {
					keep.push_back(c);
				}
			}
			nt.cols.swap(keep);
		}
		if (table == db.Find(entity.name)) {
			// the anchor and the target are not inputs
			std::vector<int> keep;
			for (int c : nt.cols) {
				if (c != anchor_col && c != target_col) {
					keep.push_back(c);
				}
			}
			nt.cols.swap(keep);
		}
		const Frame &f = db.tables[(size_t)table];
		const Column *anch =
		    (table == db.Find(entity.name) && anchor_col >= 0) ? &f.columns[(size_t)anchor_col] : nullptr;
		// A child table has no anchor column, but its rows are dated by the link
		// that reaches it, and that clock bounds its statistics the same way.
		Column clock;
		if (!anch && incoming && incoming->dated && std::isfinite(train_cutoff) && incoming->ctime.size() == f.nrows) {
			clock.name = "clock";
			clock.type = ColType::TIMESTAMP;
			clock.num = incoming->ctime;
			clock.valid.resize(f.nrows);
			for (size_t r = 0; r < f.nrows; r++) {
				clock.valid[r] = std::isfinite(clock.num[r]) ? 1 : 0;
			}
			anch = &clock;
		}
		std::vector<int> usable;
		for (int c : nt.cols) {
			const Column &col = f.columns[(size_t)c];
			double sum = 0, sq = 0;
			uint64_t n = 0;
			for (size_t r = 0; r < f.nrows; r++) {
				if (!col.valid[r]) {
					continue;
				}
				// Statistics come from the training era only.
				if (anch && std::isfinite(train_cutoff) && (!anch->valid[r] || anch->num[r] > train_cutoff)) {
					continue;
				}
				sum += col.num[r];
				sq += col.num[r] * col.num[r];
				n++;
			}
			const double mu = n ? sum / double(n) : 0.0;
			const double var = n > 1 ? (sq - double(n) * mu * mu) / double(n - 1) : 0.0;
			const double sdv = var > 1e-18 ? std::sqrt(var) : 1.0;
			if (!std::isfinite(mu) || !std::isfinite(sdv)) {
				continue; // see FitFeatureSpec: a column with no mean is not a feature
			}
			usable.push_back(c);
			nt.mu.push_back(mu);
			nt.sd.push_back(sdv);
		}
		nt.cols.swap(usable);
		// Text columns on this node type, fitted over the same era as the numerics
		// above: the entity's anchor for the entity itself, the link's clock for a
		// child table.
		for (size_t ci = 0; ci < f.columns.size(); ci++) {
			if (f.columns[ci].type != ColType::CATEGORY) {
				continue;
			}
			if (incoming && ((int)ci == incoming->child_key_col || (int)ci == incoming->child_time_col)) {
				continue;
			}
			if (table == db.Find(entity.name) && ((int)ci == anchor_col || (int)ci == target_col)) {
				continue;
			}
			bool is_key = false;
			for (const auto &fk : db.fks) {
				const bool child_side = ToUpper(fk.child_table) == ToUpper(f.name) &&
				                        ToUpper(fk.child_column) == ToUpper(f.columns[ci].name);
				const bool parent_side = ToUpper(fk.parent_table) == ToUpper(f.name) &&
				                         ToUpper(fk.parent_column) == ToUpper(f.columns[ci].name);
				if (child_side || parent_side) {
					is_key = true;
					break;
				}
			}
			if (is_key) {
				continue;
			}
			if (IsExcluded(stmt, entity, f.name, f.columns[ci].name)) {
				continue;
			}
			FeatureSpec::CatCol cc;
			if (FitCatCol(f, ci, anch, train_cutoff, 16, cc)) {
				nt.cats.push_back(cc);
			}
		}
		for (int c : nt.cols) {
			nt.col_names.push_back(f.columns[(size_t)c].name);
		}
		nt.dim = (int)nt.cols.size();
		for (const auto &cc : nt.cats) {
			nt.dim += cc.Width();
		}
		spec.nodes.push_back(nt);
		return (int)spec.nodes.size() - 1;
	};

	const int entity_table = db.Find(entity.name);
	spec.entity_node = add_node(entity_table, nullptr);
	for (size_t l = 0; l < db.links.size(); l++) {
		const Link &lk = db.links[l];
		if (ToUpper(lk.parent_table) != ToUpper(entity.name)) {
			continue;
		}
		const int ct = db.Find(lk.child_table);
		if (ct < 0) {
			continue;
		}
		SageSpec::EdgeType et;
		et.link = (int)l;
		if (lk.child_key_col >= 0) {
			et.child_key = db.tables[(size_t)ct].columns[(size_t)lk.child_key_col].name;
		}
		et.src_node = add_node(ct, &lk);
		et.dst_node = spec.entity_node;
		if (spec.NodeDim(et.src_node) > 0) {
			spec.edges.push_back(et);
		}
	}
	return spec;
}

// Re-resolve a stored spec against the tables of a later statement.
//
// Everything in a SageSpec that points at the database is an index, and indices
// are per statement: the loader reads only the tables and columns a statement
// can use, so column three of `orders` at TRAIN need not be column three at
// PREDICT. Names are what survive, so names are what this resolves from.
inline void RebindSageSpec(const Database &db, const Frame &entity, SageSpec &spec) {
	for (auto &nt : spec.nodes) {
		const int t = db.Find(nt.table_name);
		if (t < 0) {
			throw std::runtime_error("pql: table '" + nt.table_name + "' used by this model is not available");
		}
		nt.table = t;
		const Frame &f = db.tables[(size_t)t];
		for (size_t c = 0; c < nt.col_names.size(); c++) {
			const int i = f.Find(nt.col_names[c]);
			if (i < 0) {
				throw std::runtime_error("pql: column '" + nt.table_name + "." + nt.col_names[c] +
				                         "' used by this model is missing");
			}
			nt.cols[c] = i;
		}
		for (auto &cc : nt.cats) {
			const int i = f.Find(cc.name);
			if (i < 0 || f.columns[(size_t)i].type != ColType::CATEGORY) {
				throw std::runtime_error("pql: text column '" + nt.table_name + "." + cc.name +
				                         "' used by this model is missing or is not text any more");
			}
			cc.index = i;
			cc.Resolve(f.columns[(size_t)i]);
		}
	}
	for (auto &et : spec.edges) {
		const std::string ct = spec.nodes[(size_t)et.src_node].table_name;
		int found = -1;
		for (size_t l = 0; l < db.links.size(); l++) {
			const Link &lk = db.links[l];
			if (ToUpper(lk.child_table) != ToUpper(ct) || ToUpper(lk.parent_table) != ToUpper(entity.name)) {
				continue;
			}
			const Frame &child = db.At(lk.child_table);
			const std::string key =
			    lk.child_key_col >= 0 ? child.columns[(size_t)lk.child_key_col].name : std::string();
			if (et.child_key.empty() || ToUpper(key) == ToUpper(et.child_key)) {
				found = (int)l;
				break;
			}
		}
		if (found < 0) {
			throw std::runtime_error("pql: the link from '" + ct + "' used by this model is not available");
		}
		et.link = found;
	}
}

// ---------------------------------------------------------------------------
// Step 2: parameters, carved from one arena block.
// ---------------------------------------------------------------------------

struct SageParams {
	int channels = 0;
	int self_dim = 0;
	std::vector<int> src_dim;                // per edge type
	std::vector<float> w_self;               // [C x self_dim]
	std::vector<std::vector<float>> w_neigh; // per edge: [C x src_dim]
	std::vector<float> bias;                 // [C]
	std::vector<float> w_out;                // [C]
	float b_out = 0.0f;

	size_t ParamCount() const {
		size_t n = w_self.size() + bias.size() + w_out.size() + 1;
		for (const auto &w : w_neigh) {
			n += w.size();
		}
		return n;
	}

	void Init(const SageSpec &spec, int C, uint64_t seed) {
		channels = C;
		self_dim = spec.NodeDim(spec.entity_node);
		src_dim.clear();
		w_neigh.assign(spec.edges.size(), {});
		for (size_t e = 0; e < spec.edges.size(); e++) {
			src_dim.push_back(spec.NodeDim(spec.edges[e].src_node) + 1); // + degree
		}
		uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
		auto rnd = [&]() {
			st ^= st << 13;
			st ^= st >> 7;
			st ^= st << 17;
			return double(int64_t(st >> 11)) / double(1ULL << 53);
		};
		auto fill = [&](std::vector<float> &w, size_t n, int fan_in) {
			w.assign(n, 0.0f);
			const double s = std::sqrt(2.0 / double(std::max(1, fan_in)));
			for (auto &v : w) {
				v = float((rnd() * 2 - 1) * s);
			}
		};
		fill(w_self, (size_t)C * (size_t)std::max(1, self_dim), std::max(1, self_dim));
		for (size_t e = 0; e < w_neigh.size(); e++) {
			fill(w_neigh[e], (size_t)C * (size_t)std::max(1, src_dim[e]), std::max(1, src_dim[e]));
		}
		bias.assign((size_t)C, 0.0f);
		fill(w_out, (size_t)C, C);
		b_out = 0.0f;
	}
};

// All per-step buffers come from one block. The working set is an exact function
// of (batch, channels, per-relation widths), all known once the model is built,
// so there is no reason to return to the allocator afterwards; the pointers below
// are carved once and stay valid for the run.
struct SageScratch {
	Arena arena;
	float *xself = nullptr;     // [B x self_dim]
	std::vector<float *> means; // per edge: [B x src_dim]
	float *z = nullptr;         // [B x C] pre-activation
	float *h = nullptr;         // [B x C] post-ReLU
	float *dz = nullptr;        // [B x C]
	float *obuf = nullptr;      // [B] output-layer values
	size_t self_stride = 0;
	std::vector<size_t> mean_stride;

	// The block is sized by the carve itself (Arena::Carve), not by a formula
	// kept in step with it by hand.
	void Ensure(size_t B, const SageParams &p) {
		self_stride = (size_t)std::max(1, p.self_dim);
		means.assign(p.w_neigh.size(), nullptr);
		mean_stride.assign(p.w_neigh.size(), 0);
		for (size_t e = 0; e < p.w_neigh.size(); e++) {
			mean_stride[e] = (size_t)std::max(1, p.src_dim[e]);
		}
		arena.Carve([&](Arena &a) {
			xself = a.Zeroed<float>(B * self_stride);
			for (size_t e = 0; e < p.w_neigh.size(); e++) {
				means[e] = a.Zeroed<float>(B * mean_stride[e]);
			}
			z = a.Zeroed<float>(B * (size_t)p.channels);
			h = a.Zeroed<float>(B * (size_t)p.channels);
			dz = a.Zeroed<float>(B * (size_t)p.channels);
			obuf = a.Zeroed<float>(B);
		});
	}
};

// Gather the batch's own features and its neighbourhood means. This is the only
// part that touches the graph; everything after it is dense linear algebra.
inline void SageGather(const Database &db, const SageSpec &spec, const SageFeatures &feat, const SagePrefix &pre,
                       const Dataset &ds, const std::vector<size_t> &idx, size_t from, int B, const SageParams &params,
                       const VisibleCounts &vc, SageScratch &sc) {
	const Frame &entity = db.tables[(size_t)spec.nodes[(size_t)spec.entity_node].table];
	(void)entity;
	const int sd = std::max(1, params.self_dim);
	const float *xsrc = feat.x[(size_t)spec.entity_node].data();
	for (int i = 0; i < B; i++) {
		const size_t exi = idx[from + (size_t)i];
		const Example &ex = ds.examples[exi];
		if (params.self_dim > 0) {
			std::memcpy(sc.xself + (size_t)i * (size_t)sd, xsrc + (size_t)ex.entity_row * (size_t)params.self_dim,
			            sizeof(float) * (size_t)params.self_dim);
		}
		for (size_t e = 0; e < spec.edges.size(); e++) {
			const int d = params.src_dim[e];
			// src_dim already includes the degree slot, so pass the feature width.
			SageMeanAt(pre, e, ex.entity_row, vc.At(e, exi), sc.means[e] + (size_t)i * (size_t)std::max(1, d), d - 1);
		}
	}
}

// z = W_self . x_self + SUM_r W_r . mean_r ; h = ReLU(z + b) ; out = w_out . h
inline void SageForward(const SageParams &p, int B, SageScratch &sc, std::vector<double> &out) {
	const int C = p.channels;
	if (p.self_dim > 0) {
		MatMulNT(sc.xself, p.w_self.data(), nullptr, sc.z, B, p.self_dim, C, false);
	} else {
		std::memset(sc.z, 0, sizeof(float) * (size_t)B * (size_t)C);
	}
	for (size_t e = 0; e < p.w_neigh.size(); e++) {
		const int d = p.src_dim[e];
		if (d <= 0) {
			continue;
		}
		MatMulAccNT(sc.means[e], p.w_neigh[e].data(), nullptr, sc.z, B, d, C, false);
	}
	// Bias and ReLU as a straight map, then the output layer separately. Folding
	// the dot product into this loop made it a dependent chain C long per row,
	// which is latency the map does not have.
	{
		const float *__restrict bs = p.bias.data();
		for (int i = 0; i < B; i++) {
			float *__restrict zr = sc.z + (size_t)i * (size_t)C;
			float *__restrict hr = sc.h + (size_t)i * (size_t)C;
			for (int k = 0; k < C; k++) {
				const float v = zr[k] + bs[k];
				zr[k] = v;
				hr[k] = v > 0.0f ? v : 0.0f;
			}
		}
	}
	OutputLayer(sc.h, p.w_out.data(), p.b_out, sc.obuf, B, C);
	out.assign((size_t)B, 0.0);
	for (int i = 0; i < B; i++) {
		out[(size_t)i] = sc.obuf[(size_t)i];
	}
}

struct SageGrads {
	std::vector<float> gw_self, gbias, gw_out;
	std::vector<std::vector<float>> gw_neigh;
	float gb_out = 0.0f;

	void Ensure(const SageParams &p) {
		gw_self.assign(p.w_self.size(), 0.0f);
		gbias.assign(p.bias.size(), 0.0f);
		gw_out.assign(p.w_out.size(), 0.0f);
		gw_neigh.resize(p.w_neigh.size());
		for (size_t e = 0; e < gw_neigh.size(); e++) {
			gw_neigh[e].assign(p.w_neigh[e].size(), 0.0f);
		}
		gb_out = 0.0f;
	}
	void Zero() {
		std::fill(gw_self.begin(), gw_self.end(), 0.0f);
		std::fill(gbias.begin(), gbias.end(), 0.0f);
		std::fill(gw_out.begin(), gw_out.end(), 0.0f);
		for (auto &g : gw_neigh) {
			std::fill(g.begin(), g.end(), 0.0f);
		}
		gb_out = 0.0f;
	}
};

// `dout` holds dLoss/dout per row. Layer-1 inputs are raw features, so no
// gradient flows past the aggregation; that is exact for a single layer.
inline void SageBackward(const SageParams &p, int B, const std::vector<float> &dout, SageScratch &sc, SageGrads &g) {
	const int C = p.channels;
	// Blocked by channel, not by row: gw_out and gbias are accumulated over the
	// whole batch, so walking rows on the inside would make both a
	// read-modify-write of the same C floats on every row.
	{
		float *__restrict gwo = g.gw_out.data();
		float *__restrict gbi = g.gbias.data();
		const float *__restrict wo = p.w_out.data();
		for (int k0 = 0; k0 < C; k0 += 16) {
			const int kn = std::min(16, C - k0);
			float go[16], gb[16], w[16];
			for (int y = 0; y < kn; y++) {
				go[y] = gwo[k0 + y];
				gb[y] = gbi[k0 + y];
				w[y] = wo[k0 + y];
			}
			for (int i = 0; i < B; i++) {
				const float d = dout[(size_t)i];
				const float *__restrict hr = sc.h + (size_t)i * (size_t)C + k0;
				const float *__restrict zr = sc.z + (size_t)i * (size_t)C + k0;
				float *__restrict dzr = sc.dz + (size_t)i * (size_t)C + k0;
				for (int y = 0; y < kn; y++) {
					go[y] += d * hr[y];
					const float dv = (zr[y] > 0.0f) ? d * w[y] : 0.0f;
					dzr[y] = dv;
					gb[y] += dv;
				}
			}
			for (int y = 0; y < kn; y++) {
				gwo[k0 + y] = go[y];
				gbi[k0 + y] = gb[y];
			}
		}
		float gbo = 0.0f;
		for (int i = 0; i < B; i++) {
			gbo += dout[(size_t)i];
		}
		g.gb_out += gbo;
	}
	if (p.self_dim > 0) {
		AccumOuter(sc.dz, sc.xself, g.gw_self.data(), B, p.self_dim, C);
	}
	for (size_t e = 0; e < p.w_neigh.size(); e++) {
		const int d = p.src_dim[e];
		if (d > 0) {
			AccumOuter(sc.dz, sc.means[e], g.gw_neigh[e].data(), B, d, C);
		}
	}
}

// Build the child layer: for every entity-edge, find the child's own children,
// and size a W_self plus one W_neigh per grandchild relation.
inline SageChildParams BuildChildLayer(const Database &db, const SageSpec &spec, int C, uint64_t seed) {
	SageChildParams cp;
	cp.channels = C;
	uint64_t st = seed ? seed ^ 0xA5A5A5A5u : 0x1234567u;
	auto rnd = [&]() {
		st ^= st << 13;
		st ^= st >> 7;
		st ^= st << 17;
		return double(int64_t(st >> 11)) / double(1ULL << 53);
	};
	auto fill = [&](std::vector<float> &w, size_t n, int fan) {
		w.assign(n, 0.0f);
		const double sc = std::sqrt(2.0 / double(std::max(1, fan)));
		for (auto &v : w) {
			v = float((rnd() * 2 - 1) * sc);
		}
	};
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const int sd = spec.NodeDim(spec.edges[e].src_node);
		cp.self_dim.push_back(sd);
		cp.w_self.push_back({});
		fill(cp.w_self.back(), (size_t)C * (size_t)std::max(1, sd), std::max(1, sd));
		cp.bias.push_back(std::vector<float>((size_t)C, 0.0f));
		const std::string child_tbl = db.tables[(size_t)spec.nodes[(size_t)spec.edges[e].src_node].table].name;
		std::vector<int> links;
		std::vector<int> dims;
		std::vector<std::vector<float>> ws;
		for (size_t l = 0; l < db.links.size(); l++) {
			if (ToUpper(db.links[l].parent_table) != ToUpper(child_tbl)) {
				continue;
			}
			const int gt = db.Find(db.links[l].child_table);
			if (gt < 0) {
				continue;
			}
			std::vector<int> cols;
			SageNodeColumns(db, gt, &db.links[l], cols);
			if (cols.empty()) {
				continue;
			}
			links.push_back((int)l);
			dims.push_back((int)cols.size() + 1); // + degree
			ws.push_back({});
			fill(ws.back(), (size_t)C * (size_t)dims.back(), dims.back());
		}
		cp.gc_link.push_back(links);
		cp.gc_dim.push_back(dims);
		cp.w_gc.push_back(ws);
	}
	return cp;
}

// Everything the child layer needs to run and to be differentiated.
// Data-only part of the child layer: grandchild features, their running sums,
// and the message each child sees at its own timestamp. None of this depends on
// the weights, so recomputing it per epoch was pure waste; it is built once.
// Everything the child layer reads, laid out in link order.
//
// A child row is reached through the parent's CSR slice, so the natural index
// for it is its position in that flat array, not its row number in its own
// table. Indexing by row number made every child in a batch a scattered read of
// its features and its gate mask; ordering these copies by link position makes
// the whole slice for one entity contiguous, and the backward pass walks it.
// The permutation is applied once, here.
struct ChildStatic {
	// per entity-edge, per grandchild relation: [flat x gc_dim] messages
	std::vector<std::vector<std::vector<float>>> gc_mean;
	// per entity-edge: [flat x self_dim] the child's own features
	std::vector<std::vector<float>> x_csr;
	bool built = false;
};

struct ChildEmbed {
	// per entity-edge: [N_child x C] post-ReLU. The pre-activation is not kept:
	// the only thing backward wants from it is its sign, and ReLU already records
	// that, since h[k] > 0 exactly when z[k] > 0.
	// Rows are link positions, not child row numbers: see ChildStatic.
	// One block of activations, reused across blocks and across edges.
	std::vector<float> hbuf;
	// One bit per channel per child: whether that channel's ReLU gate is open.
	// Backward needs nothing else from h, and reading eight bytes per child
	// instead of C floats turns four scattered cache lines into one.
	std::vector<std::vector<uint64_t>> hmask;
	// per entity-edge, per grandchild relation: [N_child x gc_dim] messages
	std::vector<std::vector<float>> psum; // prefix sums of h along the entity link
	std::vector<std::vector<uint32_t>> off;
};

// Forward the child layer for every child row, each at its own timestamp, then
// build the running sums the entity will read.
// Build the data-only part once: for every grandchild relation, the message each
// child row sees at its OWN timestamp. Weight-independent, so it never needs
// recomputing.
inline void BuildChildStatic(const Database &db, const SageSpec &spec, const SageChildParams &cp,
                             const SageFeatures &feat, ChildStatic &cs) {
	cs.gc_mean.resize(spec.edges.size());
	cs.x_csr.resize(spec.edges.size());
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &elk = db.links[(size_t)spec.edges[e].link];
		const size_t flatn = elk.flat.size();
		const int sd = cp.self_dim[e];
		{
			const float *src = feat.x[(size_t)spec.edges[e].src_node].data();
			std::vector<float> &X = cs.x_csr[e];
			X.assign(flatn * (size_t)std::max(1, sd), 0.0f);
			if (sd > 0) {
				for (size_t i = 0; i < flatn; i++) {
					std::memcpy(&X[i * (size_t)sd], src + (size_t)elk.flat[i] * (size_t)sd, sizeof(float) * (size_t)sd);
				}
			}
		}
		cs.gc_mean[e].resize(cp.gc_link[e].size());
		for (size_t g = 0; g < cp.gc_link[e].size(); g++) {
			const Link &glk = db.links[(size_t)cp.gc_link[e][g]];
			const Frame &gf = db.At(glk.child_table);
			const int gd = cp.gc_dim[e][g];
			const int gfeat = gd - 1;
			std::vector<int> gcols;
			SageNodeColumns(db, db.Find(glk.child_table), &glk, gcols);
			std::vector<float> gx(gf.nrows * (size_t)std::max(1, gfeat), 0.0f);
			for (int c = 0; c < gfeat; c++) {
				const Column &col = gf.columns[(size_t)gcols[(size_t)c]];
				double sum = 0;
				uint64_t n = 0;
				for (size_t r = 0; r < gf.nrows; r++) {
					if (col.valid[r]) {
						sum += col.num[r];
						n++;
					}
				}
				const double mu = n ? sum / double(n) : 0.0;
				double sq = 0;
				for (size_t r = 0; r < gf.nrows; r++) {
					if (col.valid[r]) {
						sq += (col.num[r] - mu) * (col.num[r] - mu);
					}
				}
				const double sd2 = n > 1 ? std::sqrt(sq / double(n - 1)) : 1.0;
				const double inv = sd2 > 1e-12 ? 1.0 / sd2 : 1.0;
				for (size_t r = 0; r < gf.nrows; r++) {
					gx[r * (size_t)std::max(1, gfeat) + (size_t)c] =
					    col.valid[r] ? float((col.num[r] - mu) * inv) : 0.0f;
				}
			}
			const size_t gparents = glk.off.empty() ? 0 : glk.off.size() - 1;
			std::vector<uint32_t> goff(gparents + 1, 0u);
			PrefixTotal(glk);
			uint32_t tot = 0;
			for (size_t p = 0; p < gparents; p++) {
				goff[p] = tot;
				tot += glk.Count((uint32_t)p) + 1u;
			}
			goff[gparents] = tot;
			std::vector<float> gps((size_t)tot * (size_t)std::max(1, gfeat), 0.0f);
			for (size_t p = 0; p < gparents; p++) {
				const uint32_t b = glk.Begin((uint32_t)p), en = glk.End((uint32_t)p);
				float *run = gps.data() + (size_t)goff[p] * (size_t)std::max(1, gfeat);
				for (uint32_t i = b; i < en; i++) {
					const float *xw = gx.data() + (size_t)glk.flat[i] * (size_t)std::max(1, gfeat);
					float *cur = run + (size_t)(i - b + 1) * (size_t)std::max(1, gfeat);
					const float *prev = run + (size_t)(i - b) * (size_t)std::max(1, gfeat);
					for (int k = 0; k < gfeat; k++) {
						cur[k] = prev[k] + xw[k];
					}
				}
			}
			std::vector<float> &M = cs.gc_mean[e][g];
			M.assign(flatn * (size_t)gd, 0.0f);
			for (size_t i = 0; i < flatn; i++) {
				const uint32_t u = elk.flat[i];
				if ((size_t)u >= gparents) {
					continue;
				}
				const double tu = elk.dated ? elk.Time(u) : std::numeric_limits<double>::infinity();
				const size_t vis = Database::VisiblePrefix(glk, gf, u, tu);
				if (vis == 0) {
					continue;
				}
				float *out = &M[i * (size_t)gd];
				const float *hi = gps.data() + (size_t)(goff[u] + vis) * (size_t)std::max(1, gfeat);
				const float *lo = gps.data() + (size_t)goff[u] * (size_t)std::max(1, gfeat);
				// goff is indexed by the grandchild link's parent, which is the child's
				// own row number; only the destination moves to link order.
				const float inv = 1.0f / float(vis);
				for (int k = 0; k < gfeat; k++) {
					out[k] = (hi[k] - lo[k]) * inv;
				}
				out[gfeat] = float(std::log1p(double(vis)));
			}
		}
	}
	cs.built = true;
}

// Per epoch, only the weight-dependent part: the GEMMs, the ReLU, and the
// running sums the entity reads. Buffers are sized on the first call and reused.
inline void ComputeChildEmbeddings(const Database &db, const SageSpec &spec, const SageChildParams &cp,
                                   const ChildStatic &cs, ChildEmbed &ce) {
	const int C = cp.channels;
	ce.hmask.resize(spec.edges.size());
	ce.psum.resize(spec.edges.size());
	ce.off.resize(spec.edges.size());

	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &elk = db.links[(size_t)spec.edges[e].link];
		const size_t N = elk.flat.size(); // rows are link positions, not child rows
		const int sd = cp.self_dim[e];
		const size_t words = ((size_t)C + 63) / 64;
		const size_t full_words = (size_t)C / 64;
		const int tail_bits = C & 63;
		const size_t nparents = elk.off.empty() ? 0 : elk.off.size() - 1;
		if (ce.hmask[e].size() != N * words) {
			ce.hmask[e].assign(N * words, 0ull);
		}
		if (ce.off[e].size() != nparents + 1) {
			ce.off[e].assign(nparents + 1, 0u);
			PrefixTotal(elk);
			uint32_t total = 0;
			for (size_t p = 0; p < nparents; p++) {
				ce.off[e][p] = total;
				total += elk.Count((uint32_t)p) + 1u;
			}
			ce.off[e][nparents] = total;
			ce.psum[e].assign((size_t)total * (size_t)C, 0.0f);
		}
		const float *xc = cs.x_csr[e].data();
		const size_t ngc = cp.gc_link[e].size();
		const float *bs = cp.bias[e].data();

		// The activations are produced and consumed in the same order and nothing
		// else ever reads them, so they are never materialised for the whole child
		// table. A block at a time goes through the products, straight into the
		// running sums and the gate masks, and the block stays in cache. Holding
		// the full [child x C] array meant writing it and reading it back through
		// memory once per epoch, which at a quarter of a million children is 128 MB
		// of traffic per epoch that buys nothing.
		const size_t BLK = 4096;
		if (ce.hbuf.size() < BLK * (size_t)C) {
			ce.hbuf.assign(BLK * (size_t)C, 0.0f);
		}
		float *const hbuf = ce.hbuf.data();
		float *const ps = ce.psum[e].data();

		// Every parent's base row is zero; empty parents are never reached by the
		// walk below, so they are cleared here.
		for (size_t p = 0; p < nparents; p++) {
			float *run = ps + (size_t)ce.off[e][p] * (size_t)C;
			for (int k = 0; k < C; k++) {
				run[k] = 0.0f;
			}
		}

		size_t p = 0;
		for (size_t base = 0; base < N; base += BLK) {
			const int rows = (int)std::min(BLK, N - base);
			bool finished = false;
			if (sd > 0) {
				const bool last = ngc == 0;
				MatMulNT(xc + base * (size_t)sd, cp.w_self[e].data(), last ? bs : nullptr, hbuf, rows, sd, C, last);
				finished = last;
			} else {
				std::fill(hbuf, hbuf + (size_t)rows * (size_t)C, 0.0f);
			}
			for (size_t g = 0; g < ngc; g++) {
				const bool last = g + 1 == ngc;
				const int gd = cp.gc_dim[e][g];
				MatMulAccNT(cs.gc_mean[e][g].data() + base * (size_t)gd, cp.w_gc[e][g].data(), last ? bs : nullptr,
				            hbuf, rows, gd, C, last);
				finished = finished || last;
			}
			if (!finished) {
				// No term at all: the activation is just the bias through ReLU.
				for (int r = 0; r < rows; r++) {
					float *__restrict hr = hbuf + (size_t)r * (size_t)C;
					for (int k = 0; k < C; k++) {
						hr[k] = bs[k] > 0.0f ? bs[k] : 0.0f;
					}
				}
			}
			for (int r = 0; r < rows; r++) {
				const size_t i = base + (size_t)r;
				while (p + 1 < nparents && i >= elk.End((uint32_t)p)) {
					p++;
				}
				// off[p] is Begin(p) + p, so the running row for link position i is
				// i + p + 1 and its predecessor is i + p. No search, no per-parent
				// bookkeeping.
				const float *__restrict hu = hbuf + (size_t)r * (size_t)C;
				float *__restrict cur = ps + (i + p + 1) * (size_t)C;
				const float *__restrict prev = ps + (i + p) * (size_t)C;
				for (int k = 0; k < C; k++) {
					cur[k] = prev[k] + hu[k];
				}
				// The row is in cache here, so the gate mask costs nothing extra.
				uint64_t *mu = &ce.hmask[e][i * words];
				for (size_t w = 0; w < full_words; w++) {
					uint64_t bits = 0;
					const int bb = (int)w * 64;
					for (int y = 0; y < 64; y++) {
						bits |= (uint64_t)(hu[bb + y] > 0.0f) << y;
					}
					mu[w] = bits;
				}
				if (tail_bits) {
					uint64_t bits = 0;
					const int bb = (int)full_words * 64;
					for (int y = 0; y < tail_bits; y++) {
						bits |= (uint64_t)(hu[bb + y] > 0.0f) << y;
					}
					mu[full_words] = bits;
				}
			}
		}
	}
}

// Gather for the two-hop case: the entity's message is the mean of its
// children's EMBEDDINGS (plus degree), not of their raw columns.
inline void SageGather2(const SageSpec &spec, const SageFeatures &feat, const ChildEmbed &ce, const Dataset &ds,
                        const std::vector<size_t> &idx, size_t from, int B, int C, const SageParams &params,
                        const VisibleCounts &vc, SageScratch &sc) {
	// The entity's own columns, exactly as the one-hop gather collects them. This
	// used to be left out here, so xself stayed at the zeros the arena handed
	// back: two-hop models saw none of the entity's own attributes, and w_self
	// took a zero gradient every step and never moved off its initialisation.
	if (params.self_dim > 0) {
		const int sd = std::max(1, params.self_dim);
		const float *xsrc = feat.x[(size_t)spec.entity_node].data();
		for (int i = 0; i < B; i++) {
			const Example &ex = ds.examples[idx[from + (size_t)i]];
			std::memcpy(sc.xself + (size_t)i * (size_t)sd, xsrc + (size_t)ex.entity_row * (size_t)params.self_dim,
			            sizeof(float) * (size_t)params.self_dim);
		}
	}
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const int d = C + 1;
		for (int i = 0; i < B; i++) {
			const size_t exi = idx[from + (size_t)i];
			const Example &ex = ds.examples[exi];
			float *out = sc.means[e] + (size_t)i * (size_t)d;
			const size_t vis = vc.At(e, exi);
			if (vis == 0) {
				for (int k = 0; k <= C; k++) {
					out[k] = 0.0f;
				}
				continue;
			}
			const uint32_t base = ce.off[e][ex.entity_row];
			const float *hi = ce.psum[e].data() + (size_t)(base + vis) * (size_t)C;
			const float *lo = ce.psum[e].data() + (size_t)base * (size_t)C;
			const float inv = 1.0f / float(vis);
			for (int k = 0; k < C; k++) {
				out[k] = (hi[k] - lo[k]) * inv;
			}
			out[C] = float(std::log1p(double(vis)));
		}
	}
}

// Gradients of the child layer.
struct ChildGrads {
	std::vector<std::vector<float>> gw_self, gbias;
	std::vector<std::vector<std::vector<float>>> gw_gc;
	void Ensure(const SageChildParams &cp) {
		gw_self.resize(cp.w_self.size());
		gbias.resize(cp.bias.size());
		gw_gc.resize(cp.w_gc.size());
		for (size_t e = 0; e < cp.w_self.size(); e++) {
			gw_self[e].assign(cp.w_self[e].size(), 0.0f);
			gbias[e].assign(cp.bias[e].size(), 0.0f);
			gw_gc[e].resize(cp.w_gc[e].size());
			for (size_t g = 0; g < cp.w_gc[e].size(); g++) {
				gw_gc[e][g].assign(cp.w_gc[e][g].size(), 0.0f);
			}
		}
	}
	void Zero() {
		for (auto &v : gw_self) {
			std::fill(v.begin(), v.end(), 0.0f);
		}
		for (auto &v : gbias) {
			std::fill(v.begin(), v.end(), 0.0f);
		}
		for (auto &e : gw_gc) {
			for (auto &v : e) {
				std::fill(v.begin(), v.end(), 0.0f);
			}
		}
	}
};

// Push the entity's gradient back through the aggregation into each visible
// child, then through the child layer. Only rows actually touched by this batch
// are cleared and updated, so the cost is O(batch * degree * C) rather than
// O(all children * C).
inline void BuildVisibleCounts(const Database &db, const SageSpec &spec, const Dataset &ds, VisibleCounts &vc) {
	vc.n.assign(spec.edges.size(), std::vector<uint32_t>());
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &lk = db.links[(size_t)spec.edges[e].link];
		const Frame &child = db.At(lk.child_table);
		vc.n[e].resize(ds.examples.size());
		for (size_t i = 0; i < ds.examples.size(); i++) {
			vc.n[e][i] = (uint32_t)Database::VisiblePrefix(lk, child, ds.examples[i].entity_row, ds.examples[i].anchor);
		}
	}
}

// Staging for the child-layer backward.
//
// Every visible child contributes one outer product to each of the child
// layer's weight matrices. Applied a child at a time that is a scalar loop per
// child per relation, and it was where nearly two thirds of a two-hop epoch
// went. Collecting the rows here and flushing them as one product per relation
// turns exactly the same arithmetic into the tiled kernel. The block is sized so
// a flush stays inside the last-level cache.
struct ChildBackScratch {
	static const int CH = 256;
	std::vector<float> gmat;               // [CH x C]  gated message gradients
	std::vector<float> xmat;               // [CH x xw] the children's own features
	std::vector<std::vector<float>> gcmat; // per grandchild relation: [CH x gcw]
	std::vector<float> dmsg;               // [B x (C+1)] message gradients, whole batch
	std::vector<float> gtmp;               // [C x width] one relation's padded gradient
	int xw = 0;                            // padded width of a staged feature row
	std::vector<int> gcw;
	int rows = 0;

	// One column past the features, holding 1.0, so the bias gradient falls out of
	// the same product as the weight gradient instead of costing a separate pass
	// over every staged row. Four columns minimum, below which the tile has less
	// to work with than a vector holds.
	//
	// This used to round up to a multiple of four as well, because AccumOuter's
	// steps along K were sixteen and four wide and anything else fell to a scalar
	// tail. It now specialises on the exact width, so the padding is pure extra
	// arithmetic: dropping it took 5% off a two-hop run.
	static int Pad(int w) {
		return w < 4 ? 4 : w;
	}

	void Ensure(int C, int sd, const std::vector<int> &gd, int B) {
		const size_t need_m = (size_t)B * (size_t)(C + 1);
		if (dmsg.size() < need_m) {
			dmsg.assign(need_m, 0.0f);
		}
		const size_t need_g = (size_t)CH * (size_t)C;
		if (gmat.size() < need_g) {
			gmat.assign(need_g, 0.0f);
		}
		const int want = Pad(sd + 1);
		if (xw != want || xmat.size() < (size_t)CH * (size_t)want) {
			xw = want;
			xmat.assign((size_t)CH * (size_t)xw, 0.0f);
			for (int r = 0; r < CH; r++) {
				xmat[(size_t)r * (size_t)xw + sd] = 1.0f;
			}
		}
		gcw.resize(gd.size());
		if (gcmat.size() < gd.size()) {
			gcmat.resize(gd.size());
		}
		size_t widest = (size_t)xw;
		for (size_t i = 0; i < gd.size(); i++) {
			gcw[i] = Pad(std::max(1, gd[i]));
			widest = std::max(widest, (size_t)gcw[i]);
			const size_t n = (size_t)CH * (size_t)gcw[i];
			if (gcmat[i].size() < n) {
				gcmat[i].assign(n, 0.0f);
			}
		}
		const size_t need_t = (size_t)C * widest;
		if (gtmp.size() < need_t) {
			gtmp.assign(need_t, 0.0f);
		}
		rows = 0;
	}
};

inline void SageBackward2(const Database &db, const SageSpec &spec, const SageParams &p, const SageChildParams &cp,
                          const ChildStatic &cs, const ChildEmbed &ce, const Dataset &ds,
                          const std::vector<size_t> &idx, size_t from, int B, SageScratch &sc, ChildGrads &cg,
                          ChildBackScratch &st, const VisibleCounts &vc) {
	const int C = p.channels;
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &lk = db.links[(size_t)spec.edges[e].link];
		const int sd = cp.self_dim[e];
		const size_t ngc = cp.gc_link[e].size();
		const float *xsrc = cs.x_csr[e].data();
		const float *W = p.w_neigh[e].data(); // [C x (C+1)]
		const int d = C + 1;
		float *__restrict gb = cg.gbias[e].data();
		float *__restrict gws = cg.gw_self[e].data();
		st.Ensure(C, sd, cp.gc_dim[e], B);
		const size_t words = ((size_t)C + 63) / 64;
		const size_t full_words = (size_t)C / 64;
		const int tail_bits = C & 63;
		const uint64_t *maskbase = ce.hmask[e].data();

		// One product for the whole batch. Formed per entity this was a 64x65
		// matrix-vector product with a runtime trip count, and it measured
		// 3 GFLOP/s; as a single GEMM it is the same arithmetic through the tile.
		MatMulNN(sc.dz, W, nullptr, st.dmsg.data(), B, C, d, false);

		auto flush = [&]() {
			const int R = st.rows;
			if (R == 0) {
				return;
			}
			// One product carries the weight gradient and the bias gradient together,
			// the latter through the padded column of ones.
			const int xw = st.xw;
			float *__restrict gt = st.gtmp.data();
			std::fill(gt, gt + (size_t)C * (size_t)xw, 0.0f);
			AccumOuter(st.gmat.data(), st.xmat.data(), gt, R, xw, C);
			for (int c = 0; c < C; c++) {
				const float *__restrict row = gt + (size_t)c * (size_t)xw;
				float *__restrict dst = gws + (size_t)c * (size_t)sd;
				for (int k = 0; k < sd; k++) {
					dst[k] += row[k];
				}
				gb[c] += row[sd];
			}
			for (size_t g = 0; g < ngc; g++) {
				const int gd = cp.gc_dim[e][g], w = st.gcw[g];
				std::fill(gt, gt + (size_t)C * (size_t)w, 0.0f);
				AccumOuter(st.gmat.data(), st.gcmat[g].data(), gt, R, w, C);
				float *__restrict gwg = cg.gw_gc[e][g].data();
				for (int c = 0; c < C; c++) {
					const float *__restrict row = gt + (size_t)c * (size_t)w;
					float *__restrict dst = gwg + (size_t)c * (size_t)gd;
					for (int k = 0; k < gd; k++) {
						dst[k] += row[k];
					}
				}
			}
			st.rows = 0;
		};

		for (int i = 0; i < B; i++) {
			const size_t exi = idx[from + (size_t)i];
			const Example &ex = ds.examples[exi];
			const size_t vis = vc.At(e, exi);
			if (vis == 0) {
				continue;
			}
			const uint32_t b = lk.Begin(ex.entity_row);
			// The message gradient is shared by every child of this entity, so the
			// 1/n is applied once here rather than once per child.
			const float inv = 1.0f / float(vis);
			float *__restrict dm = st.dmsg.data() + (size_t)i * (size_t)d;
			for (int c = 0; c < C; c++) {
				dm[c] *= inv;
			}

			// Each child's contribution is dmean through that child's own ReLU gate.
			// The gate is diagonal, so it distributes over the sum and can be applied
			// per contribution rather than after accumulating.
			// The staging buffers are hoisted out of the loop. Reached through `st`
			// they are reloaded on every child, because the flush below can write to
			// the same object, and that reload cost four times the copy itself.
			float *const gmat = st.gmat.data();
			float *const xmat = st.xmat.data();
			const int xw = st.xw;
			int rows = st.rows;
			for (size_t j = 0; j < vis; j++) {
				const size_t pos = (size_t)b + j;
				const uint64_t *__restrict mu = maskbase + pos * words;
				float *__restrict gr = gmat + (size_t)rows * (size_t)C;
				// gr = dm through the gate, as an AND on the bit pattern rather than a
				// conditional. Written as `bit ? dm[c] : 0` the shift amount varies per
				// lane, clang gives up on it, and the loop measured twenty times
				// slower than this. Zeroing the bits of a float yields +0.0, which is
				// what the conditional produced.
				// Whole words are peeled off with a trip count of 64 known at compile
				// time. Written with a runtime bound of min(64, C - base) the loop does
				// not vectorise, and that alone cost seven times the rest of the work.
				for (size_t w = 0; w < full_words; w++) {
					const uint64_t bits = mu[w];
					const int base = (int)w * 64;
					for (int y = 0; y < 64; y++) {
						uint32_t v;
						std::memcpy(&v, &dm[base + y], sizeof(v));
						v &= (uint32_t)0 - (uint32_t)((bits >> y) & 1u);
						std::memcpy(&gr[base + y], &v, sizeof(v));
					}
				}
				for (int y = 0; y < tail_bits; y++) {
					const int c = (int)full_words * 64 + y;
					uint32_t v;
					std::memcpy(&v, &dm[c], sizeof(v));
					v &= (uint32_t)0 - (uint32_t)((mu[full_words] >> y) & 1u);
					std::memcpy(&gr[c], &v, sizeof(v));
				}
				if (sd > 0) {
					// Only the feature columns are written; the ones column and the pad
					// were set when the buffer was sized and never change.
					std::memcpy(xmat + (size_t)rows * (size_t)xw, xsrc + pos * (size_t)sd, sizeof(float) * (size_t)sd);
				}
				for (size_t g = 0; g < ngc; g++) {
					const int gd = cp.gc_dim[e][g];
					std::memcpy(st.gcmat[g].data() + (size_t)rows * (size_t)st.gcw[g],
					            &cs.gc_mean[e][g][pos * (size_t)gd], sizeof(float) * (size_t)gd);
				}
				if (++rows == ChildBackScratch::CH) {
					st.rows = rows;
					flush();
					rows = st.rows;
				}
			}
			st.rows = rows;
		}
		flush();
	}
}

// The GraphSAGE learner: the spec, the entity layer, the optional child layer, and
// the training loop that fits them.
struct SageLearner : Learner {
	int layers = 1;
	SageSpec spec;
	SageParams params;
	SageChildParams child;

	const char *Arch() const override {
		return "sage";
	}
	bool Explainable() const override {
		return false;
	}
	size_t ParamBytes() const override {
		return (params.ParamCount() + child.ParamCount()) * sizeof(float);
	}
	void EntityInputs(const FeatureSpec &, std::vector<std::string> &numeric,
	                  const std::vector<FeatureSpec::CatCol> *&cats) const override {
		if (spec.entity_node >= 0 && (size_t)spec.entity_node < spec.nodes.size()) {
			const SageSpec::NodeType &nt = spec.nodes[(size_t)spec.entity_node];
			numeric = nt.col_names;
			cats = &nt.cats;
		}
	}

	void Fit(const FitContext &ctx, TrainReport &rep) override {
		const Database &db = ctx.db;
		const Frame &entity = ctx.entity;
		const Statement &stmt = ctx.stmt;
		const Dataset &ds = ctx.ds;
		const bool binary = ctx.target.classification;
		const int anchor_col = ctx.anchor_col, target_col = ctx.target_col;
		const double train_cutoff = ctx.train_cutoff;
		const uint64_t seed = ctx.seed;
		const CancelCheck &cancelled = ctx.cancelled;
		const int epochs = (int)stmt.options.Num("EPOCHS", 60);
		const double lr = stmt.options.Num("LR", 0.01);
		const double wd = stmt.options.Num("L2", 0.0);
		std::vector<size_t> tr = ctx.train; // shuffled in place each epoch
		const std::vector<size_t> &va = ctx.val, &te = ctx.test;
		spec = BuildSageSpec(db, entity, stmt, anchor_col, target_col, train_cutoff);
		if (spec.edges.empty()) {
			throw std::runtime_error("pql: arch='sage' needs at least one linked child table with "
			                         "numeric columns");
		}
		const int C = (int)stmt.options.Num("HIDDEN", 64);
		const int n_layers = (int)stmt.options.Num("LAYERS", 1);
		layers = n_layers;
		params.Init(spec, C, seed);
		const SageFeatures feats = BuildSageFeatures(db, spec);
		const SagePrefix prefix = BuildSagePrefix(db, spec, feats);
		VisibleCounts vcounts;
		BuildVisibleCounts(db, spec, ds, vcounts);
		ChildEmbed ce;
		ChildStatic cstat;
		ChildGrads cg;
		ChildBackScratch cbs;
		if (n_layers >= 2) {
			// The entity now aggregates child EMBEDDINGS, so each message is C wide
			// plus the degree slot.
			child = BuildChildLayer(db, spec, C, seed);
			for (size_t e = 0; e < params.src_dim.size(); e++) {
				params.src_dim[e] = C + 1;
				params.w_neigh[e].assign((size_t)C * (size_t)(C + 1), 0.0f);
				uint64_t st2 = seed * 2654435761u + (uint64_t)e + 1;
				const double sc2 = std::sqrt(2.0 / double(C + 1));
				for (auto &v : params.w_neigh[e]) {
					st2 ^= st2 << 13;
					st2 ^= st2 >> 7;
					st2 ^= st2 << 17;
					v = float(((double(int64_t(st2 >> 11)) / double(1ULL << 53)) * 2 - 1) * sc2);
				}
			}
			cg.Ensure(child);
			BuildChildStatic(db, spec, child, feats, cstat);
		}

		uint64_t srng = seed | 1ULL;
		auto sshuffle = [&](std::vector<size_t> &v) {
			for (size_t i = v.size(); i > 1; i--) {
				srng ^= srng << 13;
				srng ^= srng >> 7;
				srng ^= srng << 17;
				std::swap(v[i - 1], v[srng % i]);
			}
		};
		SageScratch sc;
		SageGrads gr;
		gr.Ensure(params);
		// One optimiser per tensor, for this run only. These used to be
		// `static thread_local`, and were re-assigned only when they had to GROW: a
		// second TRAIN in the same session then started from the previous model's
		// moments and step count, so the same statement on the same data gave a
		// different model depending on what had been trained before it.
		std::vector<Adam> opts(3 + params.w_neigh.size());
		std::vector<Adam> copts;
		std::vector<double> scores;
		std::vector<double> outs;
		std::vector<float> douts;
		SageParams best_p = params;
		// The child weights the current epoch's embeddings were computed from. A
		// validation score is a property of (entity weights, these), so this is
		// what gets restored with best_p; restoring the entity layer alone left a
		// two-hop model with child weights from a later epoch that was never
		// validated, and its metric could not be reproduced by PREDICT.
		SageChildParams ce_child = child, best_c = child;
		double best_m = -std::numeric_limits<double>::infinity();
		bool have_m = false;
		const size_t sbatch = (size_t)std::max(1.0, stmt.options.Num("BATCH", 64));
		sc.Ensure(sbatch, params);

		auto score = [&](const std::vector<size_t> &fold, std::vector<double> &into) {
			into.assign(fold.size(), 0.0);
			for (size_t b = 0; b < fold.size(); b += sbatch) {
				const int B = (int)std::min(sbatch, fold.size() - b);
				if (n_layers >= 2) {
					SageGather2(spec, feats, ce, ds, fold, b, B, C, params, vcounts, sc);
				} else {
					SageGather(db, spec, feats, prefix, ds, fold, b, B, params, vcounts, sc);
				}
				SageForward(params, B, sc, outs);
				for (int i = 0; i < B; i++) {
					into[b + (size_t)i] = ctx.target.Decode(outs[(size_t)i], ds.examples[fold[b + (size_t)i]].base);
				}
			}
		};

		for (int ep = 0; ep < epochs; ep++) {
			sshuffle(tr);
			if (n_layers >= 2) {
				// Child embeddings depend on the child weights, so they are refreshed
				// once per epoch rather than per batch.
				ComputeChildEmbeddings(db, spec, child, cstat, ce);
				ce_child = child;
			}
			for (size_t b = 0; b < tr.size(); b += sbatch) {
				const int B = (int)std::min(sbatch, tr.size() - b);
				if (cancelled) {
					cancelled();
				}
				if (n_layers >= 2) {
					SageGather2(spec, feats, ce, ds, tr, b, B, C, params, vcounts, sc);
				} else {
					SageGather(db, spec, feats, prefix, ds, tr, b, B, params, vcounts, sc);
				}
				SageForward(params, B, sc, outs);
				douts.assign((size_t)B, 0.0f);
				for (int i = 0; i < B; i++) {
					double y = ds.examples[tr[b + (size_t)i]].label;
					if (binary) {
						douts[(size_t)i] = float(1.0 / (1.0 + std::exp(-outs[(size_t)i])) - y);
					} else {
						y = ctx.target.Encode(y);
						const double d = outs[(size_t)i] - y;
						douts[(size_t)i] = float(d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d));
					}
				}
				gr.Zero();
				SageBackward(params, B, douts, sc, gr);
				if (n_layers >= 2) {
					cg.Zero();
					SageBackward2(db, spec, params, child, cstat, ce, ds, tr, b, B, sc, cg, cbs, vcounts);
				}
				const float inv = 1.0f / float(std::max(1, B));
				auto step = [&](std::vector<float> &w, std::vector<float> &g, Adam &opt, double decay) {
					for (auto &v : g) {
						v *= inv;
					}
					opt.Step(w, g.data(), lr, decay);
				};
				step(params.w_self, gr.gw_self, opts[0], wd);
				step(params.bias, gr.gbias, opts[1], 0.0);
				step(params.w_out, gr.gw_out, opts[2], wd);
				for (size_t e = 0; e < gr.gw_neigh.size(); e++) {
					step(params.w_neigh[e], gr.gw_neigh[e], opts[3 + e], wd);
				}
				if (n_layers >= 2) {
					size_t need = 0;
					for (size_t e = 0; e < cg.gw_self.size(); e++) {
						need += 2 + cg.gw_gc[e].size();
					}
					if (copts.size() < need) {
						copts.assign(need, Adam());
					}
					size_t oi = 0;
					for (size_t e = 0; e < cg.gw_self.size(); e++) {
						step(child.w_self[e], cg.gw_self[e], copts[oi++], wd);
						step(child.bias[e], cg.gbias[e], copts[oi++], 0.0);
						for (size_t g = 0; g < cg.gw_gc[e].size(); g++) {
							step(child.w_gc[e][g], cg.gw_gc[e][g], copts[oi++], wd);
						}
					}
				}
				params.b_out -= float(lr * double(gr.gb_out) * double(inv));
			}
			rep.epochs_run = ep + 1;
			if (cancelled) {
				cancelled();
			}
			if (!va.empty()) {
				score(va, scores);
				const double m = FoldMetric(ds, va, scores, ctx.target);
				// >= so a later epoch wins ties. An undefined metric is adopted only
				// while no epoch has ever produced a defined one (a single-class fold
				// keeps the latest weights); once one has, a NaN epoch is a diverged
				// model and must not replace the best.
				if (std::isnan(m) ? !std::isfinite(best_m) : !(m < best_m)) {
					best_m = std::isnan(m) ? best_m : m;
					best_p = params;
					best_c = ce_child;
					have_m = true;
				}
			}
		}
		if (have_m) {
			params = best_p;
			child = best_c;
			rep.val_metric =
			    std::isfinite(best_m) ? (binary ? best_m : -best_m) : std::numeric_limits<double>::quiet_NaN();
		}
		rep.width = (int)(params.ParamCount() + (n_layers >= 2 ? child.ParamCount() : 0));
		if (!te.empty()) {
			score(te, scores);
			ReportTest(ds, te, scores, ctx.target, rep);
		}
	}

	// Score every example of `ds` through a SAGE model of either depth, in the
	// units the caller reports (probability, or a de-standardised value plus the
	// persistence base). RunPredict and RunBacktest both used to gather one hop
	// whatever the model was trained with: a two-hop model's edge widths are C+1
	// (child embeddings), so the one-hop prefix sums, whose stride is the child's
	// raw width, were read past their end and the predictions were garbage.
	void Score(const ScoreContext &ctx, const std::vector<size_t> &rows, std::vector<double> &out) const override {
		const Database &db = ctx.db;
		const Frame &entity = ctx.entity;
		const Dataset &ds = ctx.ds;
		const CancelCheck &cancelled = ctx.cancelled;
		SageSpec spec_s = spec;
		RebindSageSpec(db, entity, spec_s);
		const SageFeatures feats = BuildSageFeatures(db, spec_s);
		VisibleCounts vcounts;
		BuildVisibleCounts(db, spec_s, ds, vcounts);
		const bool two_hop = layers >= 2;
		const int C = params.channels;
		SagePrefix prefix;
		SageChildParams cp;
		ChildStatic cstat;
		ChildEmbed ce;
		if (two_hop) {
			// Link ids are per load, so the child layer's shape is rediscovered the
			// way training discovered it and the trained weights are laid over it. A
			// different shape means the graph is not the one the model was fitted on.
			cp = BuildChildLayer(db, spec_s, C, 1);
			const SageChildParams &tr = child;
			if (cp.self_dim != tr.self_dim || cp.gc_dim != tr.gc_dim) {
				throw std::runtime_error("pql: model '" + ctx.name +
				                         "' was trained on a graph whose child tables or their "
				                         "columns differ from what is loaded now; retrain it");
			}
			cp.w_self = tr.w_self;
			cp.bias = tr.bias;
			cp.w_gc = tr.w_gc;
			BuildChildStatic(db, spec_s, cp, feats, cstat);
			ComputeChildEmbeddings(db, spec_s, cp, cstat, ce);
		} else {
			prefix = BuildSagePrefix(db, spec_s, feats);
		}
		SageScratch sc;
		const size_t B0 = 256;
		sc.Ensure(B0, params);
		const std::vector<size_t> &all = rows;
		out.assign(all.size(), 0.0);
		std::vector<double> outs;
		for (size_t b = 0; b < all.size(); b += B0) {
			const int B = (int)std::min(B0, all.size() - b);
			if (cancelled) {
				cancelled();
			}
			if (two_hop) {
				SageGather2(spec_s, feats, ce, ds, all, b, B, C, params, vcounts, sc);
			} else {
				SageGather(db, spec_s, feats, prefix, ds, all, b, B, params, vcounts, sc);
			}
			SageForward(params, B, sc, outs);
			for (int i = 0; i < B; i++) {
				out[b + (size_t)i] = ctx.target.Decode(outs[(size_t)i], ds.examples[all[b + (size_t)i]].base);
			}
		}
	}
};

} // namespace pql
