// predict.hpp — PREDICT, EXPLAIN, BACKTEST and the model registry.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/train.hpp"

namespace pql {

// ===========================================================================
// 14. Prediction and registry
// ===========================================================================

struct Prediction {
	uint32_t entity_row = 0;
	double anchor = 0; // the time the prediction is made from
	double value = 0;
};

// Predict with a trained model. The stored training statement supplies the
// anchor and horizon so a PREDICT that omits them still lines up; anything the
// caller does restate must match, or the features would be built differently
// from how they were learned.
inline std::vector<Prediction> RunPredict(const Database &db, const Model &model, const Statement &stmt,
                                          const CancelCheck &cancelled = CancelCheck(),
                                          const std::vector<uint8_t> *only_rows = nullptr) {
	const Frame &entity = db.At(stmt.entity_table);
	Statement eff = stmt;
	if (eff.anchor.name.empty() && !eff.anchor_is_literal) {
		eff.anchor = model.spec_stmt.anchor;
	}
	if (!eff.horizon.present) {
		eff.horizon = model.spec_stmt.horizon;
	}
	// A model trained on a generated grid has no anchor column to read. Unless
	// the caller names a moment, predict from the most recent point in the data:
	// "what happens in the next horizon, starting now".
	if (model.spec_stmt.every.present && eff.anchor.name.empty() && !eff.anchor_is_literal) {
		double latest = -std::numeric_limits<double>::infinity();
		for (size_t l = 0; l < db.links.size(); l++) {
			const Link &lk = db.links[l];
			if (ToUpper(lk.parent_table) != ToUpper(entity.name) || !lk.dated) {
				continue;
			}
			const Frame &cf = db.At(lk.child_table);
			for (size_t r = 0; r < cf.nrows; r++) {
				const double t = lk.Time((uint32_t)r);
				if (std::isfinite(t) && t > latest) {
					latest = t;
				}
			}
		}
		if (std::isfinite(latest)) {
			eff.anchor_literal.kind = Literal::Kind::NUMBER;
			eff.anchor_literal.number = latest;
			eff.anchor_is_literal = true;
		}
	}
	// Restating the target or horizon was previously discarded, so a PREDICT could
	// ask for EXISTS and be answered with a count.
	if (stmt.target.kind != TargetKind::ATTRIBUTE || !stmt.target.ref.name.empty()) {
		const std::string asked = stmt.target.ToString();
		const std::string trained = model.spec_stmt.target.ToString();
		if (ToUpper(asked) != ToUpper(trained)) {
			throw std::runtime_error("pql: model '" + model.name + "' predicts " + trained + ", not " + asked +
			                         "; drop the target or retrain");
		}
	}
	if (stmt.horizon.present && model.spec_stmt.horizon.present &&
	    stmt.horizon.Micros() != model.spec_stmt.horizon.Micros()) {
		throw std::runtime_error("pql: model '" + model.name + "' was trained with HORIZON " +
		                         model.spec_stmt.horizon.ToString() + ", not " + stmt.horizon.ToString());
	}
	eff.target = model.spec_stmt.target;
	const int anchor_col = eff.anchor_is_literal ? -1 : ResolveAnchorColumn(entity, eff);

	ValidateFilterScope(eff.filter.get(), entity, eff.entity_alias, "WHERE");
	FeatureSpec spec = model.features;
	spec.Rebind(db, entity);
	const AggCache cache = BuildAggCache(db, spec);
	Dataset ds = CollectExamples(db, eff, spec, cache, anchor_col, false, only_rows);
	if (cancelled) {
		cancelled();
	}
	std::vector<Prediction> out;
	out.reserve(ds.examples.size());
	// Through the same batched path training scored with, rather than a row at a
	// time.
	{
		std::vector<size_t> all(ds.examples.size());
		for (size_t i = 0; i < all.size(); i++) {
			all[i] = i;
		}
		std::vector<double> vals;
		const ScoreContext sctx {db, entity, spec, ds, model.target, model.name, cancelled};
		model.learner->Score(sctx, all, vals);
		for (size_t i = 0; i < ds.examples.size(); i++) {
			Prediction p;
			p.entity_row = ds.examples[i].entity_row;
			p.anchor = ds.examples[i].anchor;
			p.value = vals[i];
			out.push_back(p);
		}
	}
	return out;
}

// One replayed prediction with the outcome that actually followed.
struct BacktestRow {
	uint32_t entity_row = 0;
	double anchor = 0;
	double predicted = 0;
	double actual = 0;
	double baseline = 0; // persistence, so every row can be judged against it
};

// Replay a trained model across its anchors and pair each prediction with the
// label that actually materialised. This is the same machinery training uses,
// with labels kept rather than discarded, so a backtest cannot silently diverge
// from how the model was scored.
// How much the model leans on each feature.
//
// Permutation importance: shuffle one feature group across rows, leaving every
// other feature and the labels exactly where they were, and see how much worse
// the model scores. A feature it does not use costs nothing to destroy. This is
// measured, not read off the weights, so it accounts for what the network
// actually does with a feature rather than how large its first-layer weights
// happen to be.
//
// Scored over every replayable row, which includes rows the model trained on, so
// these are "what does it use", not "how well does it generalise". `test` and
// BACKTEST answer the second question.
struct FeatureImportance {
	std::string feature;
	int slots = 0;
	double drop = 0; // metric lost when this group is shuffled; higher means more used
};

inline std::vector<FeatureImportance> RunExplain(const Database &db, const Model &model, const Statement &stmt,
                                                 const CancelCheck &cancelled = CancelCheck()) {
	if (!model.learner->Explainable()) {
		throw std::runtime_error("pql: EXPLAIN reads the dense feature vector, and this model was "
		                         "trained with arch='sage', whose inputs are gathered from the "
		                         "graph per batch. Train with the default arch to explain it.");
	}
	Statement eff = model.spec_stmt;
	eff.kind = StmtKind::BACKTEST;
	if (!stmt.entity_table.empty()) {
		eff.entity_table = stmt.entity_table;
	}
	if (stmt.filter) {
		eff.filter = stmt.filter->Copy();
	}
	const Frame &entity = db.At(eff.entity_table);
	ValidateFilterScope(eff.filter.get(), entity, eff.entity_alias, "WHERE");
	const int anchor_col = ResolveAnchorColumn(entity, eff);

	FeatureSpec spec = model.features;
	spec.Rebind(db, entity);
	const AggCache cache = BuildAggCache(db, spec);
	Dataset ds = CollectExamples(db, eff, spec, cache, anchor_col, /*need_labels=*/true);
	// The same fold training held out. Shuffling a feature on rows the model was
	// fitted on rewards memorised noise: a column of random numbers looked as
	// important as a real driver until this used the test fold instead.
	std::vector<size_t> tr, va, all;
	SplitByAnchor(ds, eff, tr, va, all);
	if (all.size() < 8) {
		all.clear();
		for (size_t i = 0; i < ds.examples.size(); i++) {
			all.push_back(i);
		}
	}
	const size_t n = all.size();
	if (n < 8) {
		throw std::runtime_error("pql: too few rows to explain this model");
	}
	std::vector<double> preds;
	const ScoreContext sctx {db, entity, spec, ds, model.target, model.name, cancelled};
	auto measure = [&]() {
		model.learner->Score(sctx, all, preds);
		if (model.target.classification) {
			std::vector<std::pair<double, int>> sc;
			sc.reserve(n);
			for (size_t i = 0; i < n; i++) {
				sc.emplace_back(preds[i], ds.examples[all[i]].label > 0.5 ? 1 : 0);
			}
			return AUROC(sc);
		}
		std::vector<std::pair<double, double>> pt;
		pt.reserve(n);
		for (size_t i = 0; i < n; i++) {
			pt.emplace_back(preds[i], ds.examples[all[i]].label);
		}
		return MAE(pt);
	};
	const double base = measure();

	// One shuffle, reused for every group, so the groups are compared against the
	// same disturbance rather than against different random draws.
	std::vector<size_t> perm(n);
	for (size_t i = 0; i < n; i++) {
		perm[i] = i;
	}
	uint64_t rng = 0x9E3779B97F4A7C15ull;
	for (size_t i = n; i > 1; i--) {
		rng ^= rng << 13;
		rng ^= rng >> 7;
		rng ^= rng << 17;
		std::swap(perm[i - 1], perm[rng % i]);
	}

	const int width = ds.width;
	std::vector<FeatureImportance> out;
	std::vector<float> saved, shuffled;
	size_t k = 0;
	for (const auto &g : spec.Describe()) {
		const int w = g.second;
		if (k + (size_t)w > (size_t)width) {
			break; // the description and the layout disagree; say nothing rather than lie
		}
		saved.assign(n * (size_t)w, 0.0f);
		shuffled.assign(n * (size_t)w, 0.0f);
		for (size_t i = 0; i < n; i++) {
			std::memcpy(&saved[i * (size_t)w], &ds.x[all[i] * (size_t)width + k], sizeof(float) * (size_t)w);
		}
		for (size_t i = 0; i < n; i++) {
			std::memcpy(&shuffled[i * (size_t)w], &saved[perm[i] * (size_t)w], sizeof(float) * (size_t)w);
		}
		for (size_t i = 0; i < n; i++) {
			std::memcpy(&ds.x[all[i] * (size_t)width + k], &shuffled[i * (size_t)w], sizeof(float) * (size_t)w);
		}
		const double m = measure();
		for (size_t i = 0; i < n; i++) {
			std::memcpy(&ds.x[all[i] * (size_t)width + k], &saved[i * (size_t)w], sizeof(float) * (size_t)w);
		}
		FeatureImportance fi;
		fi.feature = g.first;
		fi.slots = w;
		// AUROC is better when larger, MAE when smaller.
		fi.drop = model.target.classification ? base - m : m - base;
		out.push_back(fi);
		k += (size_t)w;
	}
	std::sort(out.begin(), out.end(), [](const FeatureImportance &a, const FeatureImportance &b) {
		if (a.drop != b.drop) {
			return a.drop > b.drop;
		}
		return a.feature < b.feature;
	});
	return out;
}

inline std::vector<BacktestRow> RunBacktest(const Database &db, const Model &model, const Statement &stmt,
                                            const CancelCheck &cancelled = CancelCheck()) {
	Statement eff = model.spec_stmt;
	eff.kind = StmtKind::BACKTEST;
	if (!stmt.entity_table.empty()) {
		eff.entity_table = stmt.entity_table;
	}
	if (stmt.filter) {
		eff.filter = stmt.filter->Copy();
	}
	const Frame &entity = db.At(eff.entity_table);
	ValidateFilterScope(eff.filter.get(), entity, eff.entity_alias, "WHERE");
	const int anchor_col = ResolveAnchorColumn(entity, eff);

	FeatureSpec spec = model.features;
	spec.Rebind(db, entity);
	const AggCache cache = BuildAggCache(db, spec);
	Dataset ds = CollectExamples(db, eff, spec, cache, anchor_col, /*need_labels=*/true);

	std::vector<double> preds;
	{
		std::vector<size_t> all(ds.examples.size());
		for (size_t i = 0; i < all.size(); i++) {
			all[i] = i;
		}
		const ScoreContext sctx {db, entity, spec, ds, model.target, model.name, cancelled};
		model.learner->Score(sctx, all, preds);
	}

	const double lo = stmt.has_from ? stmt.backtest_from.number : -std::numeric_limits<double>::infinity();
	const double hi = stmt.has_to ? stmt.backtest_to.number : std::numeric_limits<double>::infinity();
	std::vector<BacktestRow> out;
	out.reserve(ds.examples.size());
	for (size_t i = 0; i < ds.examples.size(); i++) {
		const Example &ex = ds.examples[i];
		if (ex.anchor < lo || ex.anchor > hi) {
			continue;
		}
		BacktestRow r;
		r.entity_row = ex.entity_row;
		r.anchor = ex.anchor;
		r.predicted = preds[i];
		// CollectExamples returns the raw label here. Training mutates its own copy
		// (label -= base) for the residual head, so truth there is label + base;
		// adding base again on a fresh collection would double-count it.
		r.actual = ex.label;
		r.baseline = ex.base;
		out.push_back(r);
	}
	return out;
}

class Registry {
public:
	// shared_ptr, not a bare pointer: PREDICT reads the model for the whole of a
	// load (hundreds of ms), and a concurrent DROP or re-TRAIN would otherwise
	// destroy it underneath the reader.
	void Put(const std::string &name, Model m) {
		models_[ToUpper(name)] = std::make_shared<Model>(std::move(m));
	}
	std::shared_ptr<const Model> Get(const std::string &name) const {
		auto it = models_.find(ToUpper(name));
		return it == models_.end() ? nullptr : it->second;
	}
	bool Drop(const std::string &name) {
		return models_.erase(ToUpper(name)) > 0;
	}
	std::vector<std::shared_ptr<const Model>> All() const {
		std::vector<std::shared_ptr<const Model>> v;
		v.reserve(models_.size());
		for (auto &kv : models_) {
			v.push_back(kv.second);
		}
		return v;
	}

private:
	std::unordered_map<std::string, std::shared_ptr<Model>> models_;
};

} // namespace pql
