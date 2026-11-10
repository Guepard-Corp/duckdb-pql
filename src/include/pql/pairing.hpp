// pairing.hpp — the exact plan for an ASSIGN that is a pure pairing.
//
// When what a task is worth on a resource does not depend on the resource's other
// tasks, and the only limits say how many tasks each resource may take, the best
// plan is the classic assignment problem, which has an exact answer. It is found
// with the Hungarian algorithm: each resource is a column once per task it may
// take, and with ELSE each task has a column of its own for "nobody".
//
// A plan is a pure pairing when
//   - every limit is COUNT(*) PER <lever> <= a cap (no FILTER, no SUM, no floor), and
//   - the tasks on a resource are not sequenced, or each resource takes at most one
//     task and there are no windows, so a placement's setup is just its START.
// Anything else (routes, sums such as hours, floors, windows) goes to the search.
#pragma once

#include "pql/assign.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace pql {

// Fills `sol` with the best plan and returns true, or returns false when the
// problem is not a pure pairing (the search then solves it).
inline bool SolvePairing(const AssignProblem &P, AssignSolution &sol) {
	const size_t n = P.n, m = P.m;

	// 1. Is it a pairing? Each resource's capacity comes from its count limits.
	std::vector<size_t> cap(m, n);
	for (const auto &lim : P.limits) {
		if (!lim.at_most) {
			return false;
		}
		for (size_t o = 0; o < n * m; o++) {
			if (P.allowed[o] && (lim.amount[o] != 1.0 || lim.coef_setup[o] != 0.0 || lim.coef_duration[o] != 0.0)) {
				return false;
			}
		}
		for (size_t r = 0; r < m; r++) {
			const double c = std::floor(lim.cap[r] + 1e-9);
			cap[r] = std::min(cap[r], c <= 0 ? size_t(0) : (size_t)std::min(c, (double)n));
		}
	}
	if (P.sequenced) {
		if (!P.window_from.empty()) {
			return false;
		}
		for (size_t r = 0; r < m; r++) {
			if (cap[r] > 1) {
				return false;
			}
		}
	}

	// 2. What each placement is worth: its value, plus its setup, duration and begin
	// when the tasks are sequenced (with one task per resource they are fixed).
	auto worth = [&](size_t i, size_t r) {
		const size_t o = i * m + r;
		double v = P.value[o];
		if (P.sequenced) {
			v += P.coef_setup[o] * P.start[o] + P.coef_duration[o] * P.duration[o] +
			     P.coef_begin[o] * (P.free_from[r] + P.start[o]);
		}
		return v;
	};

	// 3. Columns: each resource once per task it may take, then "nobody" per task.
	const bool has_else = n > 0 && !std::isnan(P.else_value[0]);
	std::vector<size_t> column_resource;
	for (size_t r = 0; r < m; r++) {
		column_resource.insert(column_resource.end(), cap[r], r);
	}
	const size_t slots = column_resource.size();
	const size_t cols = slots + (has_else ? n : 0);
	sol = AssignSolution();
	sol.usage.assign(P.limits.size(), std::vector<double>(m, 0.0));
	if (cols < n) {
		return true; // more tasks than places and no ELSE: no plan (feasible stays false)
	}
	const double forbidden = 1e15;
	auto cost = [&](size_t i, size_t c) {
		if (c >= slots) {
			return -P.else_value[i];
		}
		const size_t r = column_resource[c];
		return P.allowed[i * m + r] ? -worth(i, r) : forbidden;
	};

	// 4. The Hungarian algorithm (rows <= columns, 1-indexed): minimum total cost.
	const double inf = std::numeric_limits<double>::infinity();
	std::vector<double> u(n + 1, 0), v(cols + 1, 0);
	std::vector<size_t> match(cols + 1, 0), way(cols + 1, 0);
	for (size_t i = 1; i <= n; i++) {
		match[0] = i;
		size_t j0 = 0;
		std::vector<double> minv(cols + 1, inf);
		std::vector<char> used(cols + 1, 0);
		do {
			used[j0] = 1;
			const size_t i0 = match[j0];
			double delta = inf;
			size_t j1 = 0;
			for (size_t j = 1; j <= cols; j++) {
				if (used[j]) {
					continue;
				}
				const double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
				if (cur < minv[j]) {
					minv[j] = cur;
					way[j] = j0;
				}
				if (minv[j] < delta) {
					delta = minv[j];
					j1 = j;
				}
			}
			for (size_t j = 0; j <= cols; j++) {
				if (used[j]) {
					u[match[j]] += delta;
					v[j] -= delta;
				} else {
					minv[j] -= delta;
				}
			}
			j0 = j1;
		} while (match[j0] != 0);
		do {
			const size_t j1 = way[j0];
			match[j0] = match[j1];
			j0 = j1;
		} while (j0);
	}

	// 5. Read the plan back.
	sol.resource.assign(n, -1);
	sol.sequence.assign(n, -1);
	sol.begin.assign(n, 0.0);
	sol.setup.assign(n, 0.0);
	sol.duration.assign(n, 0.0);
	std::vector<int> placed_on(m, 0);
	double total = 0;
	for (size_t c = 1; c <= cols; c++) {
		const size_t i = match[c];
		if (i == 0) {
			continue;
		}
		if (c - 1 >= slots) {
			total += P.else_value[i - 1];
			continue;
		}
		const size_t r = column_resource[c - 1];
		if (!P.allowed[(i - 1) * m + r]) {
			return true; // a task that must be placed has nowhere allowed to go: no plan
		}
		sol.resource[i - 1] = (int)r;
		sol.sequence[i - 1] = placed_on[r]++;
		if (P.sequenced) {
			const size_t o = (i - 1) * m + r;
			sol.setup[i - 1] = P.start[o];
			sol.duration[i - 1] = P.duration[o];
			sol.begin[i - 1] = P.free_from[r] + P.start[o];
		}
		for (auto &u_r : sol.usage) {
			u_r[r] += 1.0;
		}
		total += worth(i - 1, r);
	}
	sol.objective = total;
	sol.bound = total; // exact: nothing can do better
	sol.feasible = true;
	sol.exact = true;
	return true;
}

} // namespace pql
