// The assignment and sequencing solver, on its own: against brute force on
// tiny problems, against its own bound and the feasibility rules on random
// ones, and for repeatability.
#include "pql/pql.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace pql;

static int g_failed = 0;
#define CHECK(cond, ...)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			g_failed++;                                                                                                \
			printf("   FAIL: ");                                                                                       \
			printf(__VA_ARGS__);                                                                                       \
			printf("\n");                                                                                              \
		}                                                                                                              \
	} while (0)

// A random problem: n tasks, m resources, hours as the one clock.
static AssignProblem Random(std::mt19937_64 &rng, size_t n, size_t m, bool sequenced, bool windows, bool must_place,
                            int limits) {
	std::uniform_real_distribution<double> value(10, 100), hours(1, 6), gap(0.5, 4), cap(8, 30), coin(0, 1);
	AssignProblem P;
	P.n = n;
	P.m = m;
	P.sequenced = sequenced;
	P.value.resize(n * m);
	P.allowed.resize(n * m);
	P.else_value.resize(n);
	P.coef_setup.assign(n * m, 0.0);
	P.coef_duration.assign(n * m, 0.0);
	P.coef_begin.assign(n * m, 0.0);
	for (size_t i = 0; i < n; i++) {
		P.else_value[i] = must_place ? std::numeric_limits<double>::quiet_NaN() : value(rng) * 0.2;
		for (size_t r = 0; r < m; r++) {
			P.value[i * m + r] = value(rng);
			P.allowed[i * m + r] = coin(rng) < 0.85 ? 1 : 0;
			P.coef_setup[i * m + r] = -5.0; // an hour of setup costs 5
		}
	}
	if (sequenced) {
		P.start.resize(n * m);
		P.setup.resize(n * n * m);
		P.duration.resize(n * m);
		P.free_from.assign(m, 0.0);
		for (size_t r = 0; r < m; r++) {
			for (size_t i = 0; i < n; i++) {
				P.start[i * m + r] = gap(rng);
				P.duration[i * m + r] = hours(rng);
				for (size_t j = 0; j < n; j++) {
					P.setup[(j * n + i) * m + r] = j == i ? 0.0 : gap(rng);
				}
			}
		}
		if (windows) {
			P.window_from.resize(n);
			P.window_to.resize(n);
			for (size_t i = 0; i < n; i++) {
				P.window_from[i] = std::uniform_real_distribution<double>(0, 20)(rng);
				P.window_to[i] = P.window_from[i] + std::uniform_real_distribution<double>(2, 12)(rng);
			}
		}
	}
	for (int l = 0; l < limits; l++) {
		AssignProblem::Limit lim;
		lim.at_most = true;
		lim.amount.assign(n * m, sequenced ? 0.0 : 1.0); // unsequenced: a count; sequenced: hours
		lim.coef_setup.assign(n * m, sequenced ? 1.0 : 0.0);
		lim.coef_duration.assign(n * m, sequenced ? 1.0 : 0.0);
		lim.cap.resize(m);
		for (size_t r = 0; r < m; r++) {
			lim.cap[r] = sequenced ? cap(rng) : std::floor(std::uniform_real_distribution<double>(1, 4)(rng));
		}
		P.limits.push_back(lim);
	}
	return P;
}

// The objective and feasibility of a plan given as (resource, position) per task,
// computed here independently of the solver.
static bool Evaluate(const AssignProblem &P, const std::vector<int> &res, const std::vector<int> &pos, double &obj) {
	const size_t n = P.n, m = P.m;
	obj = 0;
	for (size_t r = 0; r < m; r++) {
		std::vector<int> route;
		for (size_t p = 0; p < n; p++) {
			for (size_t i = 0; i < n; i++) {
				if (res[i] == (int)r && pos[i] == (int)p) {
					route.push_back((int)i);
				}
			}
		}
		std::vector<double> usage(P.limits.size(), 0.0);
		double clock = P.free_from.empty() ? 0.0 : P.free_from[r];
		for (size_t p = 0; p < route.size(); p++) {
			const size_t i = (size_t)route[p];
			const size_t o = i * m + r;
			if (!P.allowed[o]) {
				return false;
			}
			double st = 0, dur = 0, begin = 0;
			if (P.sequenced) {
				st = p == 0 ? P.start[o] : P.setup[((size_t)route[p - 1] * n + i) * m + r];
				dur = P.duration[o];
				begin = clock + st;
				if (!P.window_from.empty()) {
					begin = std::max(begin, P.window_from[i]);
					if (begin > P.window_to[i]) {
						return false;
					}
				}
				clock = begin + dur;
			}
			obj += P.value[o] + P.coef_setup[o] * st + P.coef_duration[o] * dur + P.coef_begin[o] * begin;
			for (size_t l = 0; l < P.limits.size(); l++) {
				usage[l] += P.limits[l].amount[o] + P.limits[l].coef_setup[o] * st + P.limits[l].coef_duration[o] * dur;
			}
		}
		for (size_t l = 0; l < P.limits.size(); l++) {
			if (P.limits[l].at_most ? usage[l] > P.limits[l].cap[r] + 1e-9 : usage[l] < P.limits[l].cap[r] - 1e-9) {
				return false;
			}
		}
	}
	for (size_t i = 0; i < n; i++) {
		if (res[i] < 0) {
			if (std::isnan(P.else_value[i])) {
				return false;
			}
			obj += P.else_value[i];
		}
	}
	return true;
}

// Every plan of a tiny problem: each task on a resource or none, every order.
static double Brute(const AssignProblem &P) {
	const size_t n = P.n, m = P.m;
	std::vector<int> res(n, -1), pos(n, -1);
	double best = -std::numeric_limits<double>::infinity();
	// Enumerate assignments, then for each resource every permutation of its tasks.
	std::vector<int> choice(n, -1);
	std::function<void(size_t)> rec = [&](size_t i) {
		if (i == n) {
			// Permute within each resource.
			std::vector<std::vector<int>> groups(m);
			for (size_t t = 0; t < n; t++) {
				if (choice[t] >= 0) {
					groups[(size_t)choice[t]].push_back((int)t);
				}
			}
			std::function<void(size_t)> perm = [&](size_t r) {
				if (r == m) {
					for (size_t t = 0; t < n; t++) {
						res[t] = choice[t];
						pos[t] = -1;
					}
					for (size_t rr = 0; rr < m; rr++) {
						for (size_t p = 0; p < groups[rr].size(); p++) {
							pos[(size_t)groups[rr][p]] = (int)p;
						}
					}
					double obj;
					if (Evaluate(P, res, pos, obj)) {
						best = std::max(best, obj);
					}
					return;
				}
				std::sort(groups[r].begin(), groups[r].end());
				do {
					perm(r + 1);
				} while (P.sequenced && std::next_permutation(groups[r].begin(), groups[r].end()));
			};
			perm(0);
			return;
		}
		for (int c = -1; c < (int)m; c++) {
			choice[i] = c;
			rec(i + 1);
		}
	};
	rec(0);
	return best;
}

static double Objective(const AssignProblem &P, const AssignSolution &s, bool &feasible) {
	double obj;
	feasible = Evaluate(P, s.resource, s.sequence, obj);
	return obj;
}

int main() {
	std::mt19937_64 rng(7);
	{
		printf("== against brute force on tiny problems\n");
		int exact = 0, total = 0;
		double worst_gap = 0;
		for (int t = 0; t < 60; t++) {
			const size_t n = 3 + (size_t)(t % 4), m = 2 + (size_t)(t % 2);
			const bool seq = t % 3 != 0, win = seq && t % 2 == 0;
			AssignProblem P = Random(rng, n, m, seq, win, t % 5 == 0, t % 4 == 0 ? 1 : 0);
			const double opt = Brute(P);
			if (!std::isfinite(opt)) {
				continue; // no feasible plan exists; the solver reports so below
			}
			AssignSettings st;
			st.iterations = 3000;
			const AssignSolution s = SolveAssign(P, st);
			bool feasible;
			const double obj = Objective(P, s, feasible);
			CHECK(s.feasible && feasible, "plan %d should be feasible", t);
			CHECK(std::fabs(obj - s.objective) < 1e-6, "plan %d: solver says %.4f, independent evaluation %.4f", t,
			      s.objective, obj);
			CHECK(obj <= opt + 1e-6, "plan %d beats brute force?! %.4f > %.4f", t, obj, opt);
			CHECK(s.bound >= opt - 1e-6, "plan %d: bound %.4f below the true optimum %.4f", t, s.bound, opt);
			worst_gap = std::max(worst_gap, (opt - obj) / std::max(1.0, std::fabs(opt)));
			exact += std::fabs(obj - opt) < 1e-6;
			total++;
		}
		printf("   optimum found on %d of %d; worst gap %.3f%%\n", exact, total, 100 * worst_gap);
		CHECK(exact >= total * 9 / 10, "the search should find the optimum on nearly every tiny problem");
	}
	{
		printf("== bound, feasibility and repeatability on larger problems\n");
		for (int t = 0; t < 12; t++) {
			const size_t n = 30 + (size_t)(t * 5), m = 4 + (size_t)(t % 3);
			AssignProblem P = Random(rng, n, m, t % 4 != 0, t % 2 == 0, false, 1 + t % 2);
			AssignSettings st;
			st.iterations = 2000;
			const AssignSolution a = SolveAssign(P, st);
			const AssignSolution b = SolveAssign(P, st);
			bool feasible;
			const double obj = Objective(P, a, feasible);
			CHECK(a.feasible && feasible, "problem %d: plan infeasible by the independent rules", t);
			CHECK(std::fabs(obj - a.objective) < 1e-6, "problem %d: objective disagrees", t);
			CHECK(a.bound >= a.objective - 1e-6, "problem %d: bound %.3f below the plan %.3f", t, a.bound, a.objective);
			CHECK(a.objective == b.objective && a.resource == b.resource && a.sequence == b.sequence,
			      "problem %d: two runs with one seed differ", t);
			printf("   n=%zu m=%zu %s: plan %.1f, bound %.1f (gap %.2f%%), %d placed\n", n, m,
			       P.sequenced ? "sequenced" : "assignment", a.objective, a.bound,
			       100 * (a.bound - a.objective) / std::max(1.0, std::fabs(a.bound)),
			       (int)std::count_if(a.resource.begin(), a.resource.end(), [](int r) { return r >= 0; }));
		}
	}
	{
		printf("== an unsequenced assignment is the knapsack\n");
		// With no sequencing and one count limit per resource, the relaxation is
		// the problem itself, so the bound should be met.
		for (int t = 0; t < 8; t++) {
			AssignProblem P = Random(rng, 12, 3, false, false, false, 1);
			AssignSettings st;
			st.iterations = 2000;
			const AssignSolution s = SolveAssign(P, st);
			CHECK(s.feasible, "unsequenced %d infeasible", t);
			CHECK(s.exact || s.bound - s.objective <= 1e-6 * std::max(1.0, s.bound),
			      "unsequenced %d: plan %.3f, bound %.3f, not exact", t, s.objective, s.bound);
		}
	}
	{
		printf("== what must be placed, is\n");
		AssignProblem P = Random(rng, 6, 3, true, false, true, 0);
		for (size_t o = 0; o < P.allowed.size(); o++) {
			P.allowed[o] = 1;
		}
		const AssignSolution s = SolveAssign(P);
		CHECK(s.feasible, "every task could be placed");
		for (size_t i = 0; i < P.n; i++) {
			CHECK(s.resource[i] >= 0, "task %zu left unplaced although it must be placed", i);
		}
	}
	if (g_failed) {
		printf("%d assignment test(s) FAILED\n", g_failed);
		return 1;
	}
	printf("all assignment tests passed\n");
	return 0;
}
