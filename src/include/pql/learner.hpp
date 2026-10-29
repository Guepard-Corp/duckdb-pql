// learner.hpp — What every learning algorithm looks like to the rest of PQL, and
// the model that holds one.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/features.hpp"
#include "pql/metrics.hpp"

namespace pql {

// ===========================================================================
// 10. Model
// ===========================================================================

// How a label is presented to a learner and how a raw output is read back. The
// same arithmetic for every algorithm, so a prediction means the same thing
// whichever one produced it.
struct Target {
	bool classification = false;
	bool residual = false;    // output is base + f(x)
	bool nonnegative = false; // counts and sums of non-negative columns
	bool log_target = false;  // fitted on log1p(label), read back with expm1
	double mean = 0, sd = 1;  // regression targets are standardized

	// The label as the learner sees it: on the log scale when asked, then standardized.
	double Scale(double y) const {
		return log_target ? std::log1p(y) : y;
	}
	// The value the loss is fitted to.
	double Encode(double y) const {
		return (Scale(y) - mean) / sd;
	}
	// A raw output in the units the caller reports: a probability, or a
	// de-standardised value plus the persistence base.
	double Decode(double raw, double base) const {
		double v = classification ? 1.0 / (1.0 + std::exp(-raw)) : raw * sd + mean;
		if (!classification) {
			if (log_target) {
				v = std::expm1(v);
			}
			if (residual) {
				v += base;
			}
			if (nonnegative && v < 0.0) {
				v = 0.0; // a negative count is not a number anyone can use
			}
		}
		return v;
	}
	// The outcome an example really had, in the same units.
	double Truth(const Example &e) const {
		return residual ? e.label + e.base : e.label;
	}
};

struct TrainReport {
	size_t n_train = 0, n_val = 0, n_test = 0;
	size_t n_positives = 0; // classification only; 0 here explains a NaN metric
	size_t n_censored = 0;  // dropped: label window ran past the data
	size_t n_no_anchor = 0; // dropped: the anchor column was NULL
	// What a model with no information gets, so the metric beside it means
	// something: persistence for a forecast, the positive rate for a classifier.
	double baseline_metric = 0;
	// Undefined until a fold was scored: an empty fold reported 0.0, which for
	// MAE is a perfect model and for AUROC the worst one.
	double val_metric = std::numeric_limits<double>::quiet_NaN();
	double test_metric = std::numeric_limits<double>::quiet_NaN();
	double pr_auc = std::numeric_limits<double>::quiet_NaN(); // classification only
	std::string metric_name = "auroc";
	int epochs_run = 0;
	int width = 0;
};

// Everything TrainModel has prepared before an algorithm takes over. Labels are
// in `ds` already shifted by the persistence base where the target is residual;
// the learner encodes them with `target` for its loss.
struct FitContext {
	const Database &db;
	const Frame &entity;
	const Statement &stmt;
	const FeatureSpec &features;
	const Dataset &ds;
	const std::vector<size_t> &train, &val, &test;
	int anchor_col;
	int target_col;
	double train_cutoff;
	uint64_t seed;
	const Target &target;
	const CancelCheck &cancelled;
};

struct ScoreContext {
	const Database &db;
	const Frame &entity;
	const FeatureSpec &features;
	const Dataset &ds;
	const Target &target;
	const std::string &name; // the model's, for messages
	const CancelCheck &cancelled;
};

// A learning algorithm. Fit trains on the context's folds, keeps its best
// epoch by the validation fold, and fills the report's epochs_run, width,
// val_metric and the test-fold numbers (ReportTest below). Score gives the
// reported value for each listed example of a dataset built from the same
// feature spec.
struct Learner {
	virtual ~Learner() = default;
	virtual const char *Arch() const = 0;
	virtual void Fit(const FitContext &ctx, TrainReport &rep) = 0;
	virtual void Score(const ScoreContext &ctx, const std::vector<size_t> &rows, std::vector<double> &out) const = 0;
	// Whether EXPLAIN can shuffle the dense feature groups it was trained on.
	virtual bool Explainable() const = 0;
	// The entity columns the algorithm reads, for OPTIMIZE to check its lever.
	virtual void EntityInputs(const FeatureSpec &features, std::vector<std::string> &numeric,
	                          const std::vector<FeatureSpec::CatCol> *&cats) const = 0;
	virtual size_t ParamBytes() const = 0;
};

// The validation metric an epoch earns: AUROC, or -MAE so that higher is better.
inline double FoldMetric(const Dataset &ds, const std::vector<size_t> &fold, const std::vector<double> &scores,
                         const Target &t) {
	if (t.classification) {
		std::vector<std::pair<double, int>> sc;
		sc.reserve(fold.size());
		for (size_t i = 0; i < fold.size(); i++) {
			sc.emplace_back(scores[i], ds.examples[fold[i]].label > 0.5 ? 1 : 0);
		}
		return AUROC(sc);
	}
	std::vector<std::pair<double, double>> pt;
	pt.reserve(fold.size());
	for (size_t i = 0; i < fold.size(); i++) {
		const Example &e = ds.examples[fold[i]];
		pt.emplace_back(scores[i], t.Truth(e));
	}
	return -MAE(pt);
}

// The test-fold numbers of the report, from the scores of the test fold.
inline void ReportTest(const Dataset &ds, const std::vector<size_t> &te, const std::vector<double> &scores,
                       const Target &t, TrainReport &rep) {
	if (te.empty()) {
		return;
	}
	if (t.classification) {
		std::vector<std::pair<double, int>> sc;
		int64_t pos = 0;
		for (size_t i = 0; i < te.size(); i++) {
			const int y = ds.examples[te[i]].label > 0.5 ? 1 : 0;
			sc.emplace_back(scores[i], y);
			pos += y;
		}
		rep.test_metric = AUROC(sc);
		rep.pr_auc = AveragePrecision(sc);
		rep.baseline_metric = te.empty() ? 0.0 : double(pos) / double(te.size());
	} else {
		std::vector<std::pair<double, double>> pt;
		double base_err = 0;
		for (size_t i = 0; i < te.size(); i++) {
			const Example &e = ds.examples[te[i]];
			const double truth = t.Truth(e);
			pt.emplace_back(scores[i], truth);
			base_err += std::fabs(e.base - truth);
		}
		rep.test_metric = MAE(pt);
		rep.baseline_metric = te.empty() ? 0.0 : base_err / double(te.size());
	}
}

// A trained model: what it predicts, the features it reads, how its outputs
// are read back, and the algorithm that was fitted.
struct Model {
	std::string name;
	Statement spec_stmt; // the training statement, replayed at predict time
	FeatureSpec features;
	Target target;
	std::unique_ptr<Learner> learner;

	Model() = default;
	Model(Model &&) = default;
	Model &operator=(Model &&) = default;

	// Roughly how much memory this model holds, for the cache accounting DuckDB
	// asks for. Weights only: the training data is never retained.
	size_t ApproxBytes() const {
		size_t bytes = learner ? learner->ParamBytes() : 0;
		// The feature spec is small but not free: a text column keeps one label
		// per category, and a high-cardinality one keeps thousands.
		for (const auto &cc : features.cat_cols) {
			bytes += cc.labels.size() * 24;
			for (const auto &l : cc.labels) {
				bytes += l.size();
			}
			bytes += (cc.slot_by_code.size() + cc.freq_by_code.size()) * 4;
		}
		bytes += features.self_cols.size() * 48 + features.link_aggs.size() * 96;
		return bytes;
	}
};

// The algorithm a statement asks for; defined with the learners, in train.hpp.
inline std::unique_ptr<Learner> MakeLearner(const Statement &stmt);

} // namespace pql
