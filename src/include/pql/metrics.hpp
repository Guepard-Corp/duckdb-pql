// metrics.hpp — AUROC, average precision and MAE.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/common.hpp"

namespace pql {

// ===========================================================================
// 11. Metrics
// ===========================================================================

// A score that is not a number makes the ranking undefined. It also has to be
// caught before the sort: std::sort on NaN keys is undefined behaviour, and the
// tie loop below advances on `==`, which NaN never satisfies, so one NaN score
// used to spin the validation pass forever.
inline bool AnyNaNScore(const std::vector<std::pair<double, int>> &scored) {
	for (const auto &p : scored) {
		if (std::isnan(p.first)) {
			return true;
		}
	}
	return false;
}

inline double AUROC(const std::vector<std::pair<double, int>> &scored) {
	if (AnyNaNScore(scored)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	std::vector<std::pair<double, int>> v(scored);
	std::sort(v.begin(), v.end());
	double rank_sum = 0;
	int64_t npos = 0, nneg = 0;
	size_t i = 0;
	while (i < v.size()) {
		size_t j = i;
		while (j < v.size() && v[j].first == v[i].first) {
			j++;
		}
		const double avg_rank = (double(i) + double(j - 1)) / 2.0 + 1.0;
		for (size_t k = i; k < j; k++) {
			if (v[k].second) {
				rank_sum += avg_rank;
				npos++;
			} else {
				nneg++;
			}
		}
		i = j;
	}
	if (!npos || !nneg) {
		// Undefined, not "chance". Returning 0.5 made every epoch tie, so strict
		// improvement never fired and training silently kept epoch 1.
		return std::numeric_limits<double>::quiet_NaN();
	}
	return (rank_sum - double(npos) * double(npos + 1) / 2.0) / (double(npos) * double(nneg));
}

// Average precision: the area under the precision-recall curve.
//
// AUROC answers "does a positive usually outrank a negative", which stays
// flattering when positives are rare: a model can rank well and still be useless
// at any threshold anyone would deploy. Average precision is what degrades
// visibly there, and the number to beat is the positive rate itself, which is
// what a coin achieves.
inline double AveragePrecision(const std::vector<std::pair<double, int>> &scored) {
	if (AnyNaNScore(scored)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	std::vector<std::pair<double, int>> v(scored);
	// Descending by score. Ties share a threshold, so they are consumed together
	// or the result would depend on the order they happen to arrive in.
	std::sort(v.begin(), v.end(),
	          [](const std::pair<double, int> &a, const std::pair<double, int> &b) { return a.first > b.first; });
	int64_t npos = 0;
	for (const auto &p : v) {
		npos += p.second ? 1 : 0;
	}
	if (npos == 0 || (size_t)npos == v.size()) {
		return std::numeric_limits<double>::quiet_NaN(); // undefined, not zero
	}
	double sum = 0.0;
	int64_t tp = 0, seen = 0;
	size_t i = 0;
	while (i < v.size()) {
		size_t j = i;
		int64_t block_pos = 0;
		while (j < v.size() && v[j].first == v[i].first) {
			block_pos += v[j].second ? 1 : 0;
			j++;
		}
		tp += block_pos;
		seen += (int64_t)(j - i);
		if (block_pos) {
			// Every positive in the tie gets the precision at the end of the block.
			sum += double(block_pos) * (double(tp) / double(seen));
		}
		i = j;
	}
	return sum / double(npos);
}

inline double MAE(const std::vector<std::pair<double, double>> &pred_true) {
	if (pred_true.empty()) {
		// Undefined, not zero. Zero is the score of a perfect model, and nothing
		// was scored at all.
		return std::numeric_limits<double>::quiet_NaN();
	}
	double acc = 0;
	for (auto &p : pred_true) {
		acc += std::fabs(p.first - p.second);
	}
	return acc / double(pred_true.size());
}

} // namespace pql
