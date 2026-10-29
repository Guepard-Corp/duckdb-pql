// train.hpp — TRAIN MODEL: from a statement and a database to a trained model.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/mlp.hpp"
#include "pql/sage.hpp"
#include "pql/metrics.hpp"

namespace pql {

// ===========================================================================
// 13. Training
// ===========================================================================

inline TrainReport TrainModel(const Database &db, const Statement &stmt, Model &model,
                              const CancelCheck &cancelled = CancelCheck()) {
	TrainReport rep;
	const Frame &entity = db.At(stmt.entity_table);
	const int anchor_col = ResolveAnchorColumn(entity, stmt);
	const bool forecast = IsForecast(stmt.target.kind);
	// Qualifiers are checked once, up front, against the relation each filter is
	// actually evaluated on.
	ValidateFilterScope(stmt.filter.get(), entity, stmt.entity_alias, "WHERE");
	if (forecast && stmt.target.filter && stmt.target.target_table != "*" && db.Find(stmt.target.target_table) >= 0) {
		ValidateFilterScope(stmt.target.filter.get(), db.At(stmt.target.target_table), "", "inner WHERE");
	}
	const int target_col = forecast ? -1 : entity.Find(stmt.target.ref.name);

	// A text target would be regressed as an arbitrary dictionary code.
	if (!forecast && target_col >= 0 && entity.columns[(size_t)target_col].type == ColType::CATEGORY) {
		throw std::runtime_error("pql: '" + stmt.target.ref.ToString() +
		                         "' is text; PQL predicts numeric and boolean attributes only");
	}
	const int max_mean_cols = (int)stmt.options.Num("MEAN_COLS", 3);
	// Work out where training ends before fitting normalisation, so the statistics
	// never see the validation or test folds. Derived from the anchors directly,
	// which is cheap and needs no features.
	double train_cutoff = std::numeric_limits<double>::infinity();
	if (stmt.every.present && forecast) {
		// The grid is uniform, so the 60% point of the child table's span is the
		// same boundary the split will land on.
		int tl = -1;
		for (size_t l = 0; l < db.links.size(); l++) {
			if (ToUpper(db.links[l].parent_table) == ToUpper(entity.name) &&
			    ToUpper(db.links[l].child_table) == ToUpper(stmt.target.target_table)) {
				tl = (int)l;
				break;
			}
		}
		if (tl >= 0 && db.links[(size_t)tl].dated) {
			const Frame &tf = db.At(stmt.target.target_table);
			const Link &tlk = db.links[(size_t)tl];
			double lo = std::numeric_limits<double>::infinity(), hi = -lo;
			for (size_t r = 0; r < tf.nrows; r++) {
				const double t = tlk.Time((uint32_t)r);
				if (std::isfinite(t)) {
					lo = std::min(lo, t);
					hi = std::max(hi, t);
				}
			}
			if (std::isfinite(lo) && hi > lo) {
				train_cutoff = lo + (hi - lo) * 0.6;
			}
		}
	} else if (anchor_col >= 0) {
		if (stmt.split.present) {
			train_cutoff = stmt.split.validate_from.number;
		} else {
			std::vector<double> anchors;
			anchors.reserve(entity.nrows);
			const Column &ac = entity.columns[(size_t)anchor_col];
			for (size_t r = 0; r < entity.nrows; r++) {
				if (ac.valid[r] && EvalFilter(stmt.filter.get(), entity, r)) {
					anchors.push_back(ac.num[r]);
				}
			}
			if (anchors.size() > 10) {
				std::sort(anchors.begin(), anchors.end());
				train_cutoff = anchors[(anchors.size() * 6) / 10]; // matches the 60/20/20 default
			}
		}
	}
	ValidateExcluded(db, entity, stmt);
	model.features = FitFeatureSpec(db, entity, stmt, anchor_col, target_col, max_mean_cols,
	                                (int)stmt.options.Num("MAX_CATEGORIES", 16), train_cutoff);
	const AggCache cache = BuildAggCache(db, model.features);
	Dataset ds = CollectExamples(db, stmt, model.features, cache, anchor_col, true);
	if (ds.examples.empty()) {
		if (ds.censored > 0) {
			throw std::runtime_error("pql: all " + std::to_string(ds.censored) +
			                         " candidate rows were dropped because their label window ends after the last "
			                         "row of '" +
			                         stmt.target.target_table +
			                         "'. Use a shorter HORIZON or anchors further from the end of the data.");
		}
		throw std::runtime_error("pql: no training rows matched; check WHERE and AT");
	}
	if (model.features.width == 0) {
		throw std::runtime_error("pql: no usable features for '" + entity.name +
		                         "'; it has no numeric columns and no linked child tables");
	}
	for (const auto &la : model.features.link_aggs) {
		if (db.links[(size_t)la.link].duplicate_parent_keys) {
			throw std::runtime_error("pql: '" + db.links[(size_t)la.link].parent_table +
			                         "' has duplicate key values, so child rows cannot be attributed to one row. "
			                         "Give it a unique key (a panel needs a synthetic id).");
		}
	}

	// Deliberately NOT decided here: the fold split happens first, so the task
	// type cannot be chosen by test-fold values, and a COUNT whose labels merely
	// happen to be 0/1 must not silently become a probability capped at 1.
	bool binary = false;
	// Residual only helps a real forecast; an attribute has no "previous window".
	model.spec_stmt = stmt;

	// Temporal split. Without an explicit SPLIT, hold out the latest 20% by
	// anchor so validation is always forward in time, never a random shuffle.
	std::vector<size_t> tr, va, te;
	SplitByAnchor(ds, stmt, tr, va, te);
	// Now that the folds exist, decide the task type from training labels alone.
	{
		const bool countish = stmt.target.kind == TargetKind::COUNT || stmt.target.kind == TargetKind::SUM ||
		                      stmt.target.kind == TargetKind::AVG || stmt.target.kind == TargetKind::MIN ||
		                      stmt.target.kind == TargetKind::MAX;
		binary = !countish;
		if (binary) {
			for (size_t i : tr) {
				const double v = ds.examples[i].label;
				if (v != 0.0 && v != 1.0) {
					binary = false;
					break;
				}
			}
		}
	}
	model.target.classification = binary;
	rep.metric_name = binary ? "auroc" : "mae";
	// Residual only helps a real forecast; an attribute has no "previous window".
	model.target.residual = forecast && !binary;
	model.target.nonnegative = stmt.target.kind == TargetKind::COUNT || stmt.target.kind == TargetKind::EXISTS;
	if (model.target.residual) {
		for (auto &e : ds.examples) {
			e.label -= e.base;
		}
	}
	// LOG_TARGET fits a skewed non-negative attribute on log1p(label). The loss
	// is Huber, which fits the median, and a median survives a monotone
	// transform, so the de-transformed prediction is still the median the MAE
	// metric wants. Attribute targets only: a residual over persistence can be
	// negative.
	model.target.log_target = !binary && !model.target.residual && stmt.options.Num("LOG_TARGET", 0) > 0.5;
	if (model.target.log_target) {
		for (const auto &e : ds.examples) {
			if (e.has_label && e.label < 0.0) {
				throw std::runtime_error("pql: LOG_TARGET needs a non-negative target");
			}
		}
	}
	if (tr.empty()) {
		// Falling back to "train on everything" here would report a test metric
		// measured on the training rows, which reads as a great result.
		throw std::runtime_error("pql: SPLIT leaves no training rows (every anchor is at or after VALIDATE FROM). "
		                         "Check the split boundaries against the anchor column's range.");
	}
	if (binary) {
		// One class cannot be separated from nothing: the fit is a constant and
		// every metric is undefined. This used to train silently and report NULLs.
		size_t pos = 0;
		for (size_t i : tr) {
			pos += ds.examples[i].label > 0.5 ? 1 : 0;
		}
		if (pos == 0 || pos == tr.size()) {
			throw std::runtime_error(std::string("pql: the target has ") +
			                         (pos == 0 ? "no positives" : "no negatives") + " in the training fold (" +
			                         std::to_string(tr.size()) +
			                         " rows), so there is nothing to separate. Check the "
			                         "target, its inner WHERE, and the HORIZON.");
		}
	}

	if (!binary) {
		double sum = 0;
		for (size_t i : tr) {
			sum += model.target.Scale(ds.examples[i].label);
		}
		model.target.mean = tr.empty() ? 0 : sum / double(tr.size());
		double var = 0;
		for (size_t i : tr) {
			const double d = model.target.Scale(ds.examples[i].label) - model.target.mean;
			var += d * d;
		}
		const double sd = tr.size() > 1 ? std::sqrt(var / double(tr.size() - 1)) : 1.0;
		model.target.sd = sd > 1e-9 ? sd : 1.0;
	}

	const uint64_t seed = (uint64_t)stmt.options.Num("SEED", 42);
	model.learner = MakeLearner(stmt);
	const FitContext ctx {db, entity,     stmt,       model.features, ds,   tr,           va,
	                      te, anchor_col, target_col, train_cutoff,   seed, model.target, cancelled};
	model.learner->Fit(ctx, rep);
	rep.n_train = tr.size();
	rep.n_val = va.size();
	rep.n_test = te.size();
	rep.n_censored = ds.censored;
	rep.n_no_anchor = ds.no_anchor;
	if (binary) {
		for (const auto &e : ds.examples) {
			rep.n_positives += (e.label > 0.5) ? 1 : 0;
		}
	}
	return rep;
}

inline std::unique_ptr<Learner> MakeLearner(const Statement &stmt) {
	if (ToUpper(stmt.options.Str("ARCH", "mlp")) == "SAGE") {
		return std::unique_ptr<Learner>(new SageLearner());
	}
	return std::unique_ptr<Learner>(new MlpLearner());
}

} // namespace pql
