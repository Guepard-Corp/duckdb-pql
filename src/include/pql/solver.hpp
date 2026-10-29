// solver.hpp — OPTIMIZE's solver: a multiple-choice knapsack, exact where it can be and bounded where it cannot.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/common.hpp"

namespace pql {

// ===========================================================================
// 15. Optimisation
// ===========================================================================
//
// OPTIMIZE turns predictions into a decision. Each candidate row takes one of
// K options, and the plan maximises a total subject to a few linear limits:
//
//     maximise    SUM_i v[i][x_i]
//     subject to  SUM_i a_j[i][x_i] <= b_j      for every limit j
//
// A ">=" limit arrives negated, and MINIMIZE arrives as the maximum of the
// negation. Choosing which rows to act on is the K = 2 case, with option 0,
// "leave it alone", worth nothing and costing nothing.
//
// That is the multiple-choice knapsack problem, which is NP-hard, so the
// solver reports how good its plan is instead of claiming it is the best:
//
//   1. A problem small enough to enumerate is enumerated, and the plan is
//      exact.
//   2. Otherwise the linear relaxation is solved by column generation. Its dual
//      is a price per unit of each limit, and any non-negative prices p bound
//      every plan from above by SUM_i max_k (v - p.a) + p.b. The bound reported
//      is the least of those seen, so it holds whether or not the relaxation
//      converged.
//   3. A plan is read off at those prices, repaired if it breaks a limit, and
//      improved a row at a time while the limits allow.
//
// With one limit the plan lands within about one row's value of the bound,
// which on thousands of rows is a rounding error. With several it is usually
// as close, and the bound shows it either way.

struct OptProblem {
	size_t n = 0;
	int K = 0;
	std::vector<double> v;              // n * K, maximised
	std::vector<std::vector<double>> a; // per limit, n * K
	std::vector<double> b;              // per limit
};

struct OptSettings {
	double exhaustive_plans = 262144.0; // enumerate when K^n is at most this
	int max_rounds = 300;               // column generation
	long search_nodes = 2000000;        // branch and bound, after the relaxation
	double dp_cells = 3.0e7;            // rows x distinct totals, for the dynamic programme
};

struct OptSolution {
	std::vector<int> choice;
	bool feasible = false;
	// Not even a fractional plan meets the limits, so no plan does.
	bool infeasible_proven = false;
	bool exact = false; // proven optimal: objective == bound
	double objective = 0;
	// No plan that meets the limits does better than this.
	double bound = std::numeric_limits<double>::infinity();
	// Per limit, at `choice`. When no plan was found, `choice` is the closest
	// attempt and this is how far it got.
	std::vector<double> usage;
};

// Dense two-phase simplex for the small master problems of column generation:
//
//     maximise c.x  subject to  A x (<= or =) b,  x >= 0
//
// A handful of rows and a few hundred columns, so a full tableau is both the
// simplest form and fast enough. Bland's rule throughout: column generation
// produces degenerate masters, and a cycling pivot rule would hang on them.
struct SmallLP {
	int rows = 0, cols = 0;
	std::vector<double> A;   // rows x cols, row-major
	std::vector<double> b;   // rows
	std::vector<uint8_t> eq; // 1: equality, 0: <=
	std::vector<double> c;   // cols

	enum class Status : uint8_t { OPTIMAL, INFEASIBLE, UNBOUNDED, LIMIT };
	Status status = Status::OPTIMAL;
	std::vector<double> x; // cols
	std::vector<double> y; // rows: the duals, >= 0 on a <= row
	double value = 0;
};

inline void SolveSmallLP(SmallLP &lp, double tol = 1e-10, const CancelCheck &cancelled = CancelCheck()) {
	const int R = lp.rows, n0 = lp.cols;
	// Columns: the originals, a slack per <= row, then an artificial per row.
	std::vector<int> slack_of((size_t)R, -1);
	int ns = 0;
	for (int i = 0; i < R; i++) {
		if (!lp.eq[(size_t)i]) {
			slack_of[(size_t)i] = n0 + ns++;
		}
	}
	const int art0 = n0 + ns;
	const int N = art0 + R;
	const size_t W = (size_t)N + 1; // the right-hand side is the last column
	std::vector<double> T((size_t)R * W, 0.0);
	std::vector<double> sign((size_t)R, 1.0);
	double bmax = 0;
	for (int i = 0; i < R; i++) {
		double *row = &T[(size_t)i * W];
		for (int j = 0; j < n0; j++) {
			row[j] = lp.A[(size_t)i * (size_t)n0 + (size_t)j];
		}
		if (slack_of[(size_t)i] >= 0) {
			row[slack_of[(size_t)i]] = 1.0;
		}
		row[N] = lp.b[(size_t)i];
		// The artificials start as the basis, which needs a non-negative rhs.
		if (row[N] < 0) {
			sign[(size_t)i] = -1.0;
			for (size_t j = 0; j < W; j++) {
				row[j] = -row[j];
			}
		}
		row[art0 + i] = 1.0;
		bmax = std::max(bmax, std::fabs(row[N]));
	}
	std::vector<int> basis((size_t)R);
	for (int i = 0; i < R; i++) {
		basis[(size_t)i] = art0 + i;
	}
	std::vector<double> cost((size_t)N, 0.0), d(W, 0.0);
	std::vector<uint8_t> allowed((size_t)N, 1);
	// d[j] = z_j - c_j for every column, d[N] = the objective value.
	auto reprice = [&]() {
		for (size_t j = 0; j < W; j++) {
			double s = 0;
			for (int i = 0; i < R; i++) {
				s += cost[(size_t)basis[(size_t)i]] * T[(size_t)i * W + j];
			}
			d[j] = j < (size_t)N ? s - cost[j] : s;
		}
	};
	auto pivot = [&](int r, int q) {
		double *pr = &T[(size_t)r * W];
		const double pv = pr[q];
		for (size_t j = 0; j < W; j++) {
			pr[j] /= pv;
		}
		for (int i = 0; i < R; i++) {
			if (i == r) {
				continue;
			}
			double *ri = &T[(size_t)i * W];
			const double f = ri[q];
			if (f != 0.0) {
				for (size_t j = 0; j < W; j++) {
					ri[j] -= f * pr[j];
				}
			}
		}
		const double f = d[(size_t)q];
		if (f != 0.0) {
			for (size_t j = 0; j < W; j++) {
				d[j] -= f * pr[j];
			}
		}
		basis[(size_t)r] = q;
	};
	const int max_iter = 50 * (N + R) + 1000;
	// 0: optimal, 1: unbounded, 2: out of iterations.
	auto run = [&]() -> int {
		for (int it = 0; it < max_iter; it++) {
			if (cancelled && (it & 63) == 63) {
				cancelled();
			}
			int q = -1;
			for (int j = 0; j < N; j++) {
				if (allowed[(size_t)j] && d[(size_t)j] < -tol) {
					q = j; // Bland: the lowest index that improves
					break;
				}
			}
			if (q < 0) {
				return 0;
			}
			int r = -1;
			double best = 0;
			for (int i = 0; i < R; i++) {
				const double aiq = T[(size_t)i * W + (size_t)q];
				if (aiq <= tol) {
					continue;
				}
				const double ratio = T[(size_t)i * W + (size_t)N] / aiq;
				if (r < 0 || ratio < best - tol || (ratio <= best + tol && basis[(size_t)i] < basis[(size_t)r])) {
					r = i;
					best = ratio;
				}
			}
			if (r < 0) {
				return 1;
			}
			pivot(r, q);
		}
		return 2;
	};

	// Phase 1: drive the artificials to zero.
	for (int i = 0; i < R; i++) {
		cost[(size_t)(art0 + i)] = -1.0;
	}
	reprice();
	const int p1 = run();
	lp.x.assign((size_t)n0, 0.0);
	lp.y.assign((size_t)R, 0.0);
	if (p1 == 2) {
		lp.status = SmallLP::Status::LIMIT;
		return;
	}
	if (d[(size_t)N] < -1e-9 * std::max(1.0, bmax)) {
		lp.status = SmallLP::Status::INFEASIBLE;
		return;
	}
	// An artificial still basic at zero is swapped for any real column that
	// can take its place; a row with none left is redundant and stays put.
	for (int i = 0; i < R; i++) {
		if (basis[(size_t)i] < art0) {
			continue;
		}
		for (int j = 0; j < art0; j++) {
			if (std::fabs(T[(size_t)i * W + (size_t)j]) > 1e-9) {
				pivot(i, j);
				break;
			}
		}
	}
	// Phase 2: the real objective, with the artificials barred from entering.
	for (int j = 0; j < N; j++) {
		cost[(size_t)j] = j < n0 ? lp.c[(size_t)j] : 0.0;
		if (j >= art0) {
			allowed[(size_t)j] = 0;
		}
	}
	reprice();
	const int p2 = run();
	if (p2 == 1) {
		lp.status = SmallLP::Status::UNBOUNDED;
		return;
	}
	if (p2 == 2) {
		lp.status = SmallLP::Status::LIMIT;
		return;
	}
	lp.status = SmallLP::Status::OPTIMAL;
	for (int i = 0; i < R; i++) {
		const int bj = basis[(size_t)i];
		if (bj < n0) {
			lp.x[(size_t)bj] = T[(size_t)i * W + (size_t)N];
		}
	}
	// An artificial column starts as e_i, so in the final tableau it holds
	// B^-1 e_i and its reduced cost is (c_B B^-1)_i: the dual of row i, up to
	// the sign the row was flipped by.
	for (int i = 0; i < R; i++) {
		lp.y[(size_t)i] = sign[(size_t)i] * d[(size_t)(art0 + i)];
	}
	lp.value = d[(size_t)N];
}

inline OptSolution SolveOpt(const OptProblem &P, const OptSettings &set = OptSettings(),
                            const CancelCheck &cancelled = CancelCheck()) {
	OptSolution sol;
	const size_t n = P.n;
	const size_t K = (size_t)P.K;
	const size_t m = P.b.size();
	const double inf = std::numeric_limits<double>::infinity();
	if (K == 0 || P.v.size() != n * K || P.a.size() != m) {
		throw std::runtime_error("pql: internal error: malformed optimisation problem");
	}
	for (const auto &aj : P.a) {
		if (aj.size() != n * K) {
			throw std::runtime_error("pql: internal error: malformed optimisation problem");
		}
		for (double x : aj) {
			if (!std::isfinite(x)) {
				throw std::runtime_error("pql: internal error: a limit's coefficient is not a finite number");
			}
		}
	}
	for (double x : P.v) {
		if (!std::isfinite(x)) {
			throw std::runtime_error("pql: internal error: a value is not a finite number");
		}
	}
	for (double x : P.b) {
		if (!std::isfinite(x)) {
			throw std::runtime_error("pql: internal error: a limit is not a finite number");
		}
	}
	auto evaluate = [&](const std::vector<int> &x, double &obj, std::vector<double> &use) {
		obj = 0;
		use.assign(m, 0.0);
		for (size_t i = 0; i < n; i++) {
			const size_t o = i * K + (size_t)x[i];
			obj += P.v[o];
			for (size_t j = 0; j < m; j++) {
				use[j] += P.a[j][o];
			}
		}
	};
	// A limit counts as met within a rounding error of the limit itself. Not of
	// its largest coefficient: one row costing 1e12 would otherwise excuse an
	// overshoot of a thousand on a budget of a hundred.
	std::vector<double> slack_tol(m), scale(m, 0.0);
	for (size_t j = 0; j < m; j++) {
		for (double x : P.a[j]) {
			scale[j] = std::max(scale[j], std::fabs(x));
		}
		slack_tol[j] = 1e-9 * std::max(1.0, std::fabs(P.b[j]));
	}
	auto meets = [&](const std::vector<double> &use) {
		for (size_t j = 0; j < m; j++) {
			if (use[j] > P.b[j] + slack_tol[j]) {
				return false;
			}
		}
		return true;
	};
	auto finish = [&](const std::vector<int> &x) {
		sol.choice = x;
		evaluate(x, sol.objective, sol.usage);
		sol.feasible = meets(sol.usage);
	};

	// Unconstrained, every row takes its best option. When that meets the
	// limits, nothing can beat it.
	std::vector<int> greedy(n, 0);
	for (size_t i = 0; i < n; i++) {
		for (size_t k = 1; k < K; k++) {
			if (P.v[i * K + k] > P.v[i * K + (size_t)greedy[i]]) {
				greedy[i] = (int)k;
			}
		}
	}
	finish(greedy);
	if (sol.feasible) {
		sol.exact = true;
		sol.bound = sol.objective;
		return sol;
	}
	// A single limit that even its own cheapest plan breaks: nothing can work.
	for (size_t j = 0; j < m; j++) {
		double least = 0;
		std::vector<int> x(n, 0);
		for (size_t i = 0; i < n; i++) {
			for (size_t k = 1; k < K; k++) {
				if (P.a[j][i * K + k] < P.a[j][i * K + (size_t)x[i]]) {
					x[i] = (int)k;
				}
			}
			least += P.a[j][i * K + (size_t)x[i]];
		}
		if (least > P.b[j] + slack_tol[j]) {
			finish(x);
			sol.infeasible_proven = true;
			sol.bound = -inf;
			return sol;
		}
	}

	// Small enough to try every plan.
	double plans = 1;
	for (size_t i = 0; i < n && plans <= set.exhaustive_plans; i++) {
		plans *= double(K);
	}
	if (plans <= set.exhaustive_plans) {
		std::vector<int> x(n, 0), best;
		double obj;
		std::vector<double> use;
		evaluate(x, obj, use);
		bool found = false;
		double best_obj = -inf;
		auto shift = [&](size_t i, size_t from, size_t to) {
			obj += P.v[i * K + to] - P.v[i * K + from];
			for (size_t j = 0; j < m; j++) {
				use[j] += P.a[j][i * K + to] - P.a[j][i * K + from];
			}
		};
		for (uint64_t step = 0;; step++) {
			if (cancelled && (step & 0xFFFF) == 0) {
				cancelled();
			}
			if (meets(use) && (!found || obj > best_obj + 1e-12 * std::max(1.0, std::fabs(best_obj)))) {
				found = true;
				best_obj = obj;
				best = x;
			}
			size_t i = 0;
			for (; i < n; i++) {
				const size_t was = (size_t)x[i];
				if (was + 1 < K) {
					x[i] = (int)(was + 1);
					shift(i, was, was + 1);
					break;
				}
				x[i] = 0;
				shift(i, was, 0);
			}
			if (i == n) {
				break;
			}
		}
		if (!found) {
			// Keep the closest attempt for the message: the unconstrained best.
			finish(greedy);
			sol.feasible = false;
			sol.infeasible_proven = true;
			sol.bound = -inf;
			return sol;
		}
		finish(best);
		sol.exact = true;
		sol.bound = sol.objective;
		return sol;
	}

	// ---- dynamic programme over the totals ------------------------------------
	// A budget in whole currency units, or a count, has few distinct totals, and
	// over those the problem is exact by dynamic programming: the best value for
	// every reachable combination of totals, one row at a time. Each limit is
	// shifted so its cheapest option per row costs nothing, which makes totals
	// only grow and caps them at the limit.
	//
	// When the coefficients are whole at some power of ten and the table fits,
	// the answer is exact. Otherwise each cost is rounded onto a grid that does
	// fit, both ways: rounded up, a plan that fits fits the real limits too, so it
	// is feasible and close and the search further down starts from it; rounded
	// down, every real plan fits, so the best value there bounds them all.
	std::vector<int> dp_x;
	double dp_obj = -inf, dp_bound = inf;
	if (set.dp_cells > 0 && K <= 255 && m > 0) {
		std::vector<double> base(m, 0.0), cap(m, 0.0), exact_scale(m, 0.0), dscale(m, 0.0);
		std::vector<size_t> span(m, 1), stride(m, 1);
		std::vector<uint8_t> whole_dim(m, 0);
		std::vector<double> row_lo(n * m, 0.0);
		for (size_t j = 0; j < m; j++) {
			for (size_t i = 0; i < n; i++) {
				double lo = inf;
				for (size_t k = 0; k < K; k++) {
					lo = std::min(lo, P.a[j][i * K + k]);
				}
				row_lo[i * m + j] = lo;
				base[j] += lo;
			}
			// The impossible case returned above, but a limit within rounding of its
			// own least total can still come out a hair below zero.
			cap[j] = std::max(0.0, P.b[j] - base[j]);
			for (double s10 = 1; s10 <= 1000 && exact_scale[j] == 0; s10 *= 10) {
				bool whole = true;
				for (size_t o = 0; o < n * K && whole; o++) {
					const double w = (P.a[j][o] - row_lo[(o / K) * m + j]) * s10;
					whole = std::fabs(w - std::round(w)) <= 1e-9 * std::max(1.0, std::fabs(w));
				}
				if (whole) {
					exact_scale[j] = s10;
				}
			}
		}
		// The exact table, if it fits; else a grid of about the same size per limit.
		double exact_states = 1;
		for (size_t j = 0; j < m; j++) {
			exact_states *= exact_scale[j] > 0 ? std::floor(cap[j] * exact_scale[j] + 1e-9) + 1 : inf;
		}
		// Bounded by rows times states, by states alone (the two value tables held
		// at once), and by the work of a pass.
		const double max_states = 2.0e6;
		const double cells = std::min(set.dp_cells, 1.5e8 / double(K));
		const bool exact_dp = exact_states * double(n) <= cells && exact_states <= max_states;
		// A grid of about a thousand cells per row loses at most a thousandth of the
		// limit to rounding, whatever the costs; finer buys nothing. Below a few
		// dozen rows the search further down is exact and quick, so no grid at all.
		const double per_dim =
		    n < 32 ? 0.0
		           : std::floor(std::min(std::pow(std::min(cells / double(n), max_states), 1.0 / double(m)),
		                                 1024.0 * double(n)));
		bool usable = true;
		double states = 1;
		for (size_t j = 0; j < m; j++) {
			const double whole_span = exact_scale[j] > 0 ? std::floor(cap[j] * exact_scale[j] + 1e-9) + 1 : inf;
			if (exact_dp || whole_span <= per_dim) {
				dscale[j] = exact_scale[j];
				whole_dim[j] = 1;
				span[j] = (size_t)whole_span;
			} else if (per_dim >= 64 && cap[j] > 0) {
				dscale[j] = (per_dim - 1) / cap[j];
				span[j] = (size_t)per_dim;
			} else if (cap[j] == 0) {
				dscale[j] = 1;
				span[j] = 1;
			} else {
				usable = false;
			}
			states *= double(span[j]);
		}
		usable = usable && states * double(n) <= cells && states <= max_states;
		bool all_whole = true;
		for (size_t j = 0; j < m; j++) {
			all_whole = all_whole && whole_dim[j];
		}
		const size_t S = usable ? (size_t)states : 0;
		for (size_t j = 1; j < m; j++) {
			stride[j] = stride[j - 1] * span[j - 1];
		}
		// One pass of the table. Costs rounded up give a plan that fits the real
		// limits; rounded down, a problem every real plan fits, whose best value is
		// therefore a bound. Whole-number costs are neither, and give the answer.
		std::vector<std::vector<size_t>> step(m);
		std::vector<size_t> offset;
		std::vector<uint8_t> took;
		auto pass = [&](bool up, bool keep, std::vector<int> &plan) -> double {
			for (size_t j = 0; j < m; j++) {
				step[j].assign(n * K, 0);
			}
			offset.assign(n * K, 0);
			for (size_t o = 0; o < n * K; o++) {
				for (size_t j = 0; j < m; j++) {
					const double w = (P.a[j][o] - row_lo[(o / K) * m + j]) * dscale[j];
					const double slop = 1e-9 * std::max(1.0, w);
					const double cells = whole_dim[j] ? std::round(w) : up ? std::ceil(w - slop) : std::floor(w + slop);
					// A step of `span` never fits.
					const size_t st = cells >= double(span[j]) ? span[j] : (size_t)std::max(0.0, cells);
					step[j][o] = st;
					offset[o] += std::min(st, span[j]) * stride[j];
				}
			}
			std::vector<double> cur(S, -inf), nxt(S);
			if (keep) {
				took.assign(n * S, 0);
			}
			std::vector<size_t> at(m, 0);
			cur[0] = 0;
			for (size_t i = 0; i < n; i++) {
				if (cancelled && (i & 0xFF) == 0) {
					cancelled();
				}
				std::fill(nxt.begin(), nxt.end(), -inf);
				std::fill(at.begin(), at.end(), 0);
				for (size_t st = 0; st < S; st++) {
					if (st) {
						// The mixed-radix digits of the state, advanced alongside its index.
						for (size_t j = 0; j < m; j++) {
							if (++at[j] < span[j]) {
								break;
							}
							at[j] = 0;
						}
					}
					if (cur[st] == -inf) {
						continue;
					}
					for (size_t k = 0; k < K; k++) {
						const size_t o = i * K + k;
						bool ok = true;
						for (size_t j = 0; j < m && ok; j++) {
							ok = at[j] + step[j][o] < span[j];
						}
						if (!ok) {
							continue;
						}
						const size_t to = st + offset[o];
						const double val = cur[st] + P.v[o];
						if (val > nxt[to]) {
							nxt[to] = val;
							if (keep) {
								took[i * S + to] = (uint8_t)k;
							}
						}
					}
				}
				cur.swap(nxt);
			}
			size_t end = S;
			for (size_t st = 0; st < S; st++) {
				if (cur[st] > -inf && (end == S || cur[st] > cur[end])) {
					end = st;
				}
			}
			if (end == S) {
				return -inf;
			}
			if (keep) {
				plan.assign(n, 0);
				size_t at_state = end;
				for (size_t i = n; i-- > 0;) {
					const int k = took[i * S + at_state];
					plan[i] = k;
					at_state -= offset[i * K + (size_t)k];
				}
			}
			return cur[end];
		};
		if (usable) {
			std::vector<int> x;
			const double best = pass(true, true, x);
			if (best == -inf && all_whole) {
				// Every combination of totals was tried and none fits the limits.
				finish(greedy);
				sol.feasible = false;
				sol.infeasible_proven = true;
				sol.bound = -inf;
				return sol;
			}
			if (best > -inf) {
				double obj;
				std::vector<double> use;
				evaluate(x, obj, use);
				if (meets(use)) {
					if (all_whole) {
						finish(x);
						sol.exact = true;
						sol.bound = sol.objective;
						return sol;
					}
					dp_x = x;
					dp_obj = obj;
				}
				// A whole-number table whose plan fails the limit check means the
				// rounding disagreed with it; the general method below decides.
			}
			if (!all_whole) {
				std::vector<int> unused;
				dp_bound = pass(false, false, unused);
				if (dp_bound == -inf) {
					// Not even the relaxed costs fit: no plan does.
					finish(greedy);
					sol.feasible = false;
					sol.infeasible_proven = true;
					sol.bound = -inf;
					return sol;
				}
			}
		}
	}

	// ---- column generation over the relaxation -------------------------------
	// The master works in normalised units: values divided by the largest |v|,
	// each limit by its largest |a|, so its tolerances mean the same thing
	// whatever the currency. Prices are converted back before pricing.
	double vs = 0;
	for (double x : P.v) {
		vs = std::max(vs, std::fabs(x));
	}
	if (vs == 0) {
		vs = 1;
	}
	std::vector<size_t> act; // limits that depend on the choice at all
	for (size_t j = 0; j < m; j++) {
		if (scale[j] > 0) {
			act.push_back(j);
		}
	}
	const size_t ma = act.size();
	const double tie = 1e-12 * vs;
	// The best option per row at prices p, in original units. `with_value`
	// false prices resources alone, for the feasibility phase. Ties go to the
	// option that uses less of the priced limits, which is the side of a tie
	// that keeps a plan inside them.
	std::vector<double> wt(m, 0.0);
	auto price = [&](const std::vector<double> &p, bool with_value, std::vector<int> &x, double &V,
	                 std::vector<double> &U) {
		for (size_t j = 0; j < m; j++) {
			wt[j] = scale[j] > 0 ? (p[j] * scale[j] / vs + 1e-6) / scale[j] : 0.0;
		}
		x.assign(n, 0);
		V = 0;
		U.assign(m, 0.0);
		for (size_t i = 0; i < n; i++) {
			size_t bk = 0;
			double bval = -inf, bres = inf;
			for (size_t k = 0; k < K; k++) {
				const size_t o = i * K + k;
				double red = with_value ? P.v[o] : 0.0, res = 0;
				for (size_t j = 0; j < m; j++) {
					red -= p[j] * P.a[j][o];
					res += wt[j] * P.a[j][o];
				}
				// Resources alone are priced in normalised units, so their ties are
				// too; the value's scale would swamp them.
				const double t = with_value ? tie : 1e-12 * std::max(1.0, std::fabs(red));
				if (red > bval + t || (red >= bval - t && res < bres)) {
					bk = k;
					bval = red;
					bres = res;
				}
			}
			x[i] = (int)bk;
			const size_t o = i * K + bk;
			V += P.v[o];
			for (size_t j = 0; j < m; j++) {
				U[j] += P.a[j][o];
			}
		}
	};
	struct Column {
		double V;
		std::vector<double> U;
		std::vector<double> p; // the prices that produced it, to rebuild it
		bool with_value;
	};
	std::vector<Column> cols;
	auto add_column = [&](const std::vector<double> &p, bool with_value, double V, const std::vector<double> &U) {
		for (const auto &c : cols) {
			bool same = std::fabs(c.V - V) <= 1e-12 * std::max(1.0, std::fabs(V));
			for (size_t j = 0; same && j < m; j++) {
				same = std::fabs(c.U[j] - U[j]) <= 1e-12 * std::max(1.0, std::fabs(U[j]));
			}
			if (same) {
				return false;
			}
		}
		cols.push_back(Column {V, U, p, with_value});
		return true;
	};
	std::vector<int> x;
	double V;
	std::vector<double> U;
	{
		std::vector<double> zero(m, 0.0);
		price(zero, true, x, V, U);
		add_column(zero, true, V, U);
		for (size_t j : act) {
			std::vector<double> p(m, 0.0);
			p[j] = 1.0;
			price(p, false, x, V, U);
			add_column(p, false, V, U);
		}
	}
	// How many pricing rounds the size of the problem affords.
	const double work = double(n) * double(K) * double(m + 1);
	const int rounds = (int)std::max(20.0, std::min(double(set.max_rounds), 4.0e9 / std::max(1.0, work)));
	auto build_master = [&](bool phase_a, double relax) {
		SmallLP lp;
		const size_t T = cols.size();
		lp.rows = (int)ma + 1;
		lp.cols = (int)(T + (phase_a ? ma : 0));
		lp.A.assign((size_t)lp.rows * (size_t)lp.cols, 0.0);
		lp.b.assign((size_t)lp.rows, 0.0);
		lp.eq.assign((size_t)lp.rows, 0);
		lp.c.assign((size_t)lp.cols, 0.0);
		for (size_t r = 0; r < ma; r++) {
			const size_t j = act[r];
			for (size_t t = 0; t < T; t++) {
				lp.A[r * (size_t)lp.cols + t] = cols[t].U[j] / scale[j];
			}
			if (phase_a) {
				lp.A[r * (size_t)lp.cols + T + r] = -1.0;
			}
			lp.b[r] = P.b[j] / scale[j] + relax;
		}
		for (size_t t = 0; t < T; t++) {
			lp.A[ma * (size_t)lp.cols + t] = 1.0;
			lp.c[t] = phase_a ? 0.0 : cols[t].V / vs;
		}
		for (size_t r = 0; phase_a && r < ma; r++) {
			lp.c[T + r] = -1.0;
		}
		lp.b[ma] = 1.0;
		lp.eq[ma] = 1;
		SolveSmallLP(lp, 1e-10, cancelled);
		return lp;
	};
	// The master is a dense tableau with a row per limit, re-solved every round:
	// cheap for a handful of limits and slow for hundreds, where convergence is
	// slow as well. Past this many the prices come from the subgradient below.
	const bool use_master = ma <= 24;
	// Phase A: is there any fractional plan inside the limits? Minimise the
	// total overshoot over convex combinations of the plans found so far.
	bool lp_feasible = false;
	for (int round = 0; use_master && round < rounds; round++) {
		if (cancelled) {
			cancelled();
		}
		SmallLP lp = build_master(true, 0.0);
		if (lp.status != SmallLP::Status::OPTIMAL) {
			break;
		}
		if (lp.value >= -1e-9 * double(ma + 1)) {
			lp_feasible = true;
			break;
		}
		std::vector<double> p(m, 0.0);
		for (size_t r = 0; r < ma; r++) {
			p[act[r]] = std::max(0.0, lp.y[r]) / scale[act[r]];
		}
		price(p, false, x, V, U);
		double rc = -lp.y[ma];
		for (size_t r = 0; r < ma; r++) {
			rc -= std::max(0.0, lp.y[r]) * U[act[r]] / scale[act[r]];
		}
		if (rc <= 1e-10) {
			// No plan lowers the overshoot: the relaxation itself is infeasible.
			finish(x);
			sol.feasible = false;
			sol.infeasible_proven = true;
			sol.bound = -inf;
			return sol;
		}
		if (!add_column(p, false, V, U)) {
			break; // a column already held: stalled, which proves nothing
		}
	}
	// Phase B: maximise over convex combinations, pricing with the duals.
	double bound = inf;
	std::vector<double> last_p(m, 0.0), best_p(m, 0.0);
	bool have_prices = false, converged = false;
	for (int round = 0; lp_feasible && round < rounds; round++) {
		if (cancelled) {
			cancelled();
		}
		SmallLP lp = build_master(false, 1e-9);
		if (lp.status != SmallLP::Status::OPTIMAL) {
			break;
		}
		std::vector<double> p(m, 0.0);
		for (size_t r = 0; r < ma; r++) {
			p[act[r]] = std::max(0.0, lp.y[r]) * vs / scale[act[r]];
		}
		price(p, true, x, V, U);
		// The Lagrangian bound at these prices, against the true limits.
		double L = V;
		for (size_t j = 0; j < m; j++) {
			L += p[j] * (P.b[j] - U[j]);
		}
		last_p = p;
		have_prices = true;
		if (L < bound) {
			bound = L;
			best_p = p;
		}
		double rc = V / vs - lp.y[ma];
		for (size_t r = 0; r < ma; r++) {
			rc -= std::max(0.0, lp.y[r]) * U[act[r]] / scale[act[r]];
		}
		if (rc <= 1e-9 * std::max(1.0, std::fabs(lp.value))) {
			converged = true;
			break;
		}
		if (!add_column(p, true, V, U)) {
			break;
		}
	}

	// ---- subgradient over the prices ---------------------------------------------
	// When the master did not converge, or was not tried, the prices come from
	// the Lagrangian directly: raise the price of a limit the plan at the current
	// prices overshoots, lower it where there is room, with Polyak's step toward
	// the best plan known. Slower to settle than the master on a few limits and
	// far steadier on many, where every group's price has to find its own level.
	// Worked in normalised units, so a budget in dollars and a count move alike.
	std::vector<double> feasible_p;
	double feasible_V = -inf;
	if (!converged && ma > 0) {
		std::vector<double> q(m, 0.0);
		for (size_t j : act) {
			q[j] = (have_prices ? best_p[j] : 0.0) * scale[j] / vs;
		}
		const int iters = (int)std::max(40.0, std::min(800.0, 2.0e9 / std::max(1.0, work)));
		double alpha = 1.0;
		int stale = 0;
		std::vector<double> p(m, 0.0), g(m, 0.0);
		for (int it = 0; it < iters && alpha > 1e-4; it++) {
			if (cancelled && (it & 7) == 0) {
				cancelled();
			}
			for (size_t j : act) {
				p[j] = q[j] * vs / scale[j];
			}
			price(p, true, x, V, U);
			double L = V;
			for (size_t j = 0; j < m; j++) {
				L += p[j] * (P.b[j] - U[j]);
			}
			if (L < bound) {
				bound = L;
				best_p = p;
				stale = 0;
			} else if (++stale >= 20) {
				alpha *= 0.5;
				stale = 0;
			}
			last_p = p;
			have_prices = true;
			if (meets(U) && V > feasible_V) {
				feasible_V = V;
				feasible_p = p;
			}
			// The projected gradient of L in q: limits at a zero price with room
			// left cannot be lowered further.
			double norm2 = 0;
			for (size_t j : act) {
				g[j] = (P.b[j] - U[j]) / scale[j];
				if (q[j] <= 0 && g[j] > 0) {
					g[j] = 0;
				}
				norm2 += g[j] * g[j];
			}
			if (norm2 <= 1e-24) {
				break; // the plan at these prices meets every limit it prices: optimal
			}
			const double target = feasible_V > -inf ? feasible_V : L - 0.05 * (std::fabs(L) + vs);
			const double t = alpha * std::max(0.0, L - target) / vs / norm2;
			for (size_t j : act) {
				q[j] = std::max(0.0, q[j] - t * g[j]);
			}
		}
	}

	// ---- a plan at those prices ------------------------------------------------
	auto overshoot = [&](const std::vector<double> &use) {
		double s = 0;
		for (size_t j = 0; j < m; j++) {
			if (use[j] > P.b[j] + slack_tol[j]) {
				s += (use[j] - P.b[j]) / scale[j];
			}
		}
		return s;
	};
	std::vector<double> tmp(m);
	auto moved = [&](const std::vector<double> &use, size_t i, size_t from, size_t to) -> const std::vector<double> & {
		for (size_t j = 0; j < m; j++) {
			tmp[j] = use[j] + P.a[j][i * K + to] - P.a[j][i * K + from];
		}
		return tmp;
	};
	struct Move {
		double score;
		uint32_t row;
		uint32_t to;
	};
	auto by_score = [](const Move &l, const Move &r) {
		if (l.score != r.score) {
			return l.score > r.score;
		}
		return l.row != r.row ? l.row < r.row : l.to < r.to;
	};
	// The rows each limit can see at all, for the paired moves below.
	std::vector<std::vector<uint32_t>> support(m);
	for (size_t j = 0; j < m; j++) {
		for (size_t i = 0; i < n; i++) {
			double lo = inf, hi = -inf;
			for (size_t k = 0; k < K; k++) {
				lo = std::min(lo, P.a[j][i * K + k]);
				hi = std::max(hi, P.a[j][i * K + k]);
			}
			if (hi > lo) {
				support[j].push_back((uint32_t)i);
			}
		}
	}
	// Two switches at once, for when no single one lowers the overshoot: every
	// group at its minimum and the budget over, say, where dropping a dear row
	// breaks its group and adding a cheap one breaks the budget, and only the
	// swap helps. The first move eases a limit that is over; the second is looked
	// for only among the rows of the limits the first one broke.
	std::vector<double> u1(m), u2(m);
	auto pair_step = [&](std::vector<int> &x, std::vector<double> &use) {
		const double cur = overshoot(use);
		double best_next = cur - 1e-12, best_loss = inf;
		size_t r1 = n, k1 = 0, r2 = n, k2 = 0;
		std::vector<Move> first;
		std::vector<size_t> broke;
		for (size_t j = 0; j < m; j++) {
			if (use[j] <= P.b[j] + slack_tol[j]) {
				continue;
			}
			first.clear();
			for (uint32_t i : support[j]) {
				const size_t c = (size_t)x[i];
				for (size_t k = 0; k < K; k++) {
					const double ease = (P.a[j][i * K + c] - P.a[j][i * K + k]) / scale[j];
					if (k == c || ease <= 0) {
						continue;
					}
					// Ranked by how much the move eases the limit alone. Weighing in the
					// value lost buries the moves that matter: the dear rows still worth
					// something never reach the top.
					first.push_back(Move {ease, i, (uint32_t)k});
				}
			}
			const size_t top = std::min<size_t>(first.size(), 64);
			std::partial_sort(first.begin(), first.begin() + (long)top, first.end(), by_score);
			for (size_t f = 0; f < top; f++) {
				const Move &a = first[f];
				const size_t ca = (size_t)x[a.row];
				for (size_t jj = 0; jj < m; jj++) {
					u1[jj] = use[jj] + P.a[jj][a.row * K + a.to] - P.a[jj][a.row * K + ca];
				}
				const double loss_a = P.v[a.row * K + ca] - P.v[a.row * K + a.to];
				broke.clear();
				for (size_t jj = 0; jj < m; jj++) {
					if (u1[jj] > P.b[jj] + slack_tol[jj] && u1[jj] > use[jj] && support[jj].size() <= 4096) {
						broke.push_back(jj);
					}
				}
				for (size_t jj : broke) {
					for (uint32_t i2 : support[jj]) {
						if (i2 == a.row) {
							continue;
						}
						const size_t c2 = (size_t)x[i2];
						for (size_t k = 0; k < K; k++) {
							if (k == c2) {
								continue;
							}
							for (size_t q = 0; q < m; q++) {
								u2[q] = u1[q] + P.a[q][i2 * K + k] - P.a[q][i2 * K + c2];
							}
							const double next = overshoot(u2);
							const double loss = loss_a + P.v[i2 * K + c2] - P.v[i2 * K + k];
							if (next < best_next - 1e-12 || (next <= best_next + 1e-12 && loss < best_loss)) {
								best_next = next;
								best_loss = loss;
								r1 = a.row;
								k1 = a.to;
								r2 = i2;
								k2 = k;
							}
						}
					}
				}
			}
		}
		if (r1 == n) {
			return false;
		}
		use = moved(use, r1, (size_t)x[r1], k1);
		x[r1] = (int)k1;
		use = moved(use, r2, (size_t)x[r2], k2);
		x[r2] = (int)k2;
		return true;
	};
	auto improve = [&](std::vector<int> &x, const std::vector<double> &p) {
		double obj;
		std::vector<double> use;
		evaluate(x, obj, use);
		std::vector<uint8_t> touched(n);
		std::vector<Move> moves;
		// Repair: switches that most reduce the overshoot per unit of value lost,
		// and pairs of them when no single one does.
		const int max_pairs = (int)std::min<size_t>(4 * m + 64, 2000);
		int pairs = 0;
		for (int pass = 0; pass < 40 + max_pairs && overshoot(use) > 0; pass++) {
			if (cancelled) {
				cancelled();
			}
			std::vector<uint8_t> over(m, 0);
			for (size_t j = 0; j < m; j++) {
				over[j] = use[j] > P.b[j] + slack_tol[j];
			}
			moves.clear();
			for (size_t i = 0; i < n; i++) {
				const size_t c = (size_t)x[i];
				for (size_t k = 0; k < K; k++) {
					if (k == c) {
						continue;
					}
					double red = 0;
					for (size_t j = 0; j < m; j++) {
						if (over[j]) {
							red += (P.a[j][i * K + c] - P.a[j][i * K + k]) / scale[j];
						}
					}
					if (red <= 1e-15) {
						continue;
					}
					const double loss = (P.v[i * K + c] - P.v[i * K + k]) / vs;
					moves.push_back(Move {red / (std::max(loss, 0.0) + 1e-12), (uint32_t)i, (uint32_t)k});
				}
			}
			std::sort(moves.begin(), moves.end(), by_score);
			std::fill(touched.begin(), touched.end(), 0);
			double cur = overshoot(use);
			bool progress = false;
			for (const auto &mv : moves) {
				if (touched[mv.row]) {
					continue;
				}
				const auto &nu = moved(use, mv.row, (size_t)x[mv.row], mv.to);
				const double next = overshoot(nu);
				if (next < cur - 1e-15) {
					use = nu;
					cur = next;
					x[mv.row] = (int)mv.to;
					touched[mv.row] = 1;
					progress = true;
					if (cur <= 0) {
						break;
					}
				}
			}
			if (!progress) {
				if (pairs++ >= max_pairs || !pair_step(x, use)) {
					break;
				}
			}
		}
		if (overshoot(use) > 0) {
			return false;
		}
		// Fill: improving switches, best value per unit of priced resource first.
		std::vector<double> y(m);
		for (size_t j = 0; j < m; j++) {
			y[j] = scale[j] > 0 ? p[j] * scale[j] / vs + 1e-9 : 0.0;
		}
		for (int pass = 0; pass < 40; pass++) {
			if (cancelled) {
				cancelled();
			}
			moves.clear();
			for (size_t i = 0; i < n; i++) {
				const size_t c = (size_t)x[i];
				for (size_t k = 0; k < K; k++) {
					const double gain = P.v[i * K + k] - P.v[i * K + c];
					if (k == c || gain <= tie) {
						continue;
					}
					double used = 0;
					for (size_t j = 0; j < m; j++) {
						if (scale[j] > 0) {
							used += y[j] * std::max(0.0, (P.a[j][i * K + k] - P.a[j][i * K + c]) / scale[j]);
						}
					}
					moves.push_back(Move {(gain / vs) / (used + 1e-12), (uint32_t)i, (uint32_t)k});
				}
			}
			std::sort(moves.begin(), moves.end(), by_score);
			std::fill(touched.begin(), touched.end(), 0);
			size_t applied = 0;
			for (const auto &mv : moves) {
				if (touched[mv.row]) {
					continue;
				}
				const auto &nu = moved(use, mv.row, (size_t)x[mv.row], mv.to);
				if (meets(nu)) {
					use = nu;
					x[mv.row] = (int)mv.to;
					touched[mv.row] = 1;
					applied++;
				}
			}
			if (applied == 0) {
				break;
			}
		}
		// Polish: the single best improving switch, a bounded number of times.
		const int polish = (int)std::max(3.0, std::min(50.0, 2.0e7 / std::max(1.0, work)));
		for (int it = 0; it < polish; it++) {
			double best_gain = tie;
			size_t bi = n, bk = 0;
			for (size_t i = 0; i < n; i++) {
				const size_t c = (size_t)x[i];
				for (size_t k = 0; k < K; k++) {
					const double gain = P.v[i * K + k] - P.v[i * K + c];
					if (k == c || gain <= best_gain) {
						continue;
					}
					if (meets(moved(use, i, c, k))) {
						best_gain = gain;
						bi = i;
						bk = k;
					}
				}
			}
			if (bi == n) {
				break;
			}
			use = moved(use, bi, (size_t)x[bi], bk);
			x[bi] = (int)bk;
		}
		return true;
	};

	// Where to start a plan from: the last prices, the prices that gave the
	// tightest bound, and the best plan the master holds that already meets
	// the limits, each rebuilt the way it was first priced.
	struct Start {
		std::vector<double> p;
		bool with_value;
	};
	std::vector<Start> starts;
	if (have_prices) {
		starts.push_back(Start {last_p, true});
		if (best_p != last_p) {
			starts.push_back(Start {best_p, true});
		}
	}
	{
		int best_t = -1;
		for (size_t t = 0; t < cols.size(); t++) {
			if (meets(cols[t].U) && (best_t < 0 || cols[t].V > cols[(size_t)best_t].V)) {
				best_t = (int)t;
			}
		}
		if (best_t >= 0) {
			starts.push_back(Start {cols[(size_t)best_t].p, cols[(size_t)best_t].with_value});
		}
	}
	if (!feasible_p.empty()) {
		starts.push_back(Start {feasible_p, true});
	}
	if (starts.empty()) {
		starts.push_back(Start {std::vector<double>(m, 0.0), true});
	}
	bool found = false;
	std::vector<int> best_x;
	double best_obj = -inf;
	std::vector<int> closest;
	double closest_over = inf;
	for (const auto &st : starts) {
		std::vector<int> cand;
		price(st.p, st.with_value, cand, V, U);
		const bool ok = improve(cand, st.p);
		double obj;
		std::vector<double> use;
		evaluate(cand, obj, use);
		if (ok && meets(use)) {
			if (!found || obj > best_obj) {
				found = true;
				best_obj = obj;
				best_x = cand;
			}
		} else if (overshoot(use) < closest_over) {
			closest_over = overshoot(use);
			closest = cand;
		}
	}
	if (!dp_x.empty() && (!found || dp_obj > best_obj)) {
		found = true;
		best_obj = dp_obj;
		best_x = dp_x;
	}

	// ---- branch and bound over what the prices leave open ------------------------
	// At prices p, a partial plan is bounded by L(p) less the margins its choices
	// give up, a margin being how far an option falls below its row's best at p.
	// An option whose margin alone exceeds the gap to the incumbent cannot be in
	// a better plan, so most rows are fixed outright and the search runs over the
	// few near the price line. Finishing it proves the incumbent optimal, or, with
	// no incumbent, that no plan exists; running out of nodes changes nothing.
	const std::vector<double> &bp = have_prices ? best_p : std::vector<double>(m, 0.0);
	std::vector<double> margin(n * K);
	double Lp = 0;
	for (size_t i = 0; i < n; i++) {
		double best = -inf;
		for (size_t k = 0; k < K; k++) {
			const size_t o = i * K + k;
			double r = P.v[o];
			for (size_t j = 0; j < m; j++) {
				r -= bp[j] * P.a[j][o];
			}
			margin[o] = r;
			best = std::max(best, r);
		}
		for (size_t k = 0; k < K; k++) {
			margin[i * K + k] = best - margin[i * K + k];
		}
		Lp += best;
	}
	for (size_t j = 0; j < m; j++) {
		Lp += bp[j] * P.b[j];
	}
	bound = std::min(bound, std::min(Lp, dp_bound));
	const double tolv = 1e-9 * std::max(1.0, std::fabs(Lp));
	bool complete = false;
	if (!found || std::min(Lp, dp_bound) - best_obj > tolv) {
		const double gap0 = found ? Lp - best_obj : inf;
		std::vector<int> fixed(n, 0);
		std::vector<double> use0(m, 0.0);
		double val0 = 0;
		std::vector<uint32_t> free_rows;
		std::vector<std::vector<uint32_t>> opts;
		std::vector<uint32_t> allowed;
		for (size_t i = 0; i < n; i++) {
			allowed.clear();
			for (size_t k = 0; k < K; k++) {
				if (margin[i * K + k] < gap0 - tolv) {
					allowed.push_back((uint32_t)k);
				}
			}
			std::sort(allowed.begin(), allowed.end(), [&](uint32_t l, uint32_t r) {
				const double ml = margin[i * K + l], mr = margin[i * K + r];
				return ml != mr ? ml < mr : l < r;
			});
			if (allowed.size() == 1) {
				const size_t o = i * K + allowed[0];
				fixed[i] = (int)allowed[0];
				val0 += P.v[o];
				for (size_t j = 0; j < m; j++) {
					use0[j] += P.a[j][o];
				}
			} else {
				free_rows.push_back((uint32_t)i);
				opts.push_back(allowed);
			}
		}
		// The rows that move the limits most go first, so infeasible branches die
		// near the root.
		std::vector<double> spread(free_rows.size(), 0.0);
		for (size_t f = 0; f < free_rows.size(); f++) {
			for (size_t j = 0; j < m; j++) {
				if (scale[j] == 0) {
					continue;
				}
				double lo = inf, hi = -inf;
				for (uint32_t k : opts[f]) {
					lo = std::min(lo, P.a[j][free_rows[f] * K + k]);
					hi = std::max(hi, P.a[j][free_rows[f] * K + k]);
				}
				spread[f] += (hi - lo) / scale[j];
			}
		}
		std::vector<size_t> ord(free_rows.size());
		for (size_t f = 0; f < ord.size(); f++) {
			ord[f] = f;
		}
		std::stable_sort(ord.begin(), ord.end(), [&](size_t l, size_t r) { return spread[l] > spread[r]; });
		std::vector<uint32_t> rows_o(ord.size());
		std::vector<std::vector<uint32_t>> opts_o(ord.size());
		for (size_t f = 0; f < ord.size(); f++) {
			rows_o[f] = free_rows[ord[f]];
			opts_o[f].swap(opts[ord[f]]);
		}
		const size_t F = rows_o.size();
		// The least each limit can still be pushed to by the rows not yet decided.
		std::vector<std::vector<double>> least(m, std::vector<double>(F + 1, 0.0));
		for (size_t j = 0; j < m; j++) {
			for (size_t f = F; f-- > 0;) {
				double lo = inf;
				for (uint32_t k : opts_o[f]) {
					lo = std::min(lo, P.a[j][rows_o[f] * K + k]);
				}
				least[j][f] = least[j][f + 1] + lo;
			}
		}
		std::vector<double> use = use0;
		double val = val0, marg = 0;
		std::vector<uint32_t> next(F + 1, 0), cur(F, 0);
		size_t d = 0;
		long nodes = 0;
		complete = true;
		double inc = found ? best_obj : -inf;
		auto undo = [&](size_t depth) {
			const size_t o = rows_o[depth] * K + cur[depth];
			for (size_t j = 0; j < m; j++) {
				use[j] -= P.a[j][o];
			}
			val -= P.v[o];
			marg -= margin[o];
		};
		while (true) {
			if (d == F) {
				if (meets(use) && (!found || val > inc + tolv)) {
					found = true;
					inc = val;
					best_obj = val;
					best_x = fixed;
					for (size_t f = 0; f < F; f++) {
						best_x[rows_o[f]] = (int)cur[f];
					}
				}
				if (d == 0) {
					break;
				}
				undo(--d);
				continue;
			}
			bool down = false;
			const double gap = found ? Lp - inc : inf;
			while (next[d] < opts_o[d].size()) {
				const uint32_t k = opts_o[d][next[d]++];
				const size_t o = rows_o[d] * K + k;
				if (marg + margin[o] >= gap - tolv) {
					next[d] = (uint32_t)opts_o[d].size(); // sorted by margin: the rest are worse
					break;
				}
				bool ok = true;
				for (size_t j = 0; j < m && ok; j++) {
					ok = use[j] + P.a[j][o] + least[j][d + 1] <= P.b[j] + slack_tol[j];
				}
				if (!ok) {
					continue;
				}
				if (++nodes > set.search_nodes) {
					complete = false;
					break;
				}
				if (cancelled && (nodes & 0xFFFF) == 0) {
					cancelled();
				}
				for (size_t j = 0; j < m; j++) {
					use[j] += P.a[j][o];
				}
				val += P.v[o];
				marg += margin[o];
				cur[d] = k;
				next[++d] = 0;
				down = true;
				break;
			}
			if (!complete) {
				break;
			}
			if (down) {
				continue;
			}
			if (d == 0) {
				break;
			}
			undo(--d);
		}
	}

	if (!found) {
		finish(closest.empty() ? greedy : closest);
		sol.feasible = false;
		// A finished search over every option is a proof.
		sol.infeasible_proven = complete;
		sol.bound = complete ? -inf : bound;
		return sol;
	}
	finish(best_x);
	if (complete || std::min(Lp, dp_bound) - sol.objective <= tolv) {
		sol.exact = true;
		sol.bound = sol.objective;
		return sol;
	}
	sol.bound = std::max(bound, sol.objective);
	sol.exact = sol.bound - sol.objective <= 1e-9 * std::max(1.0, std::fabs(sol.objective));
	return sol;
}

// ---------------------------------------------------------------------------
// OPTIMIZE over a model's predictions
// ---------------------------------------------------------------------------

} // namespace pql
