// assign.hpp — OPTIMIZE's second solver: tasks placed on resources, in
// sequence when asked, under one limit per resource.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include <chrono>

#include "pql/common.hpp"
#include "pql/solver.hpp"

namespace pql {

// ===========================================================================
// 15b. Assignment and sequencing
// ===========================================================================
//
// ASSIGN turns predictions into a plan where the value of a placement depends
// on what else the resource does. Every task takes one resource or none; on a
// sequenced resource the tasks are ordered, each begins when the previous one
// ends plus the setup between them (or when its window opens, if later), and
// a plan is feasible when every task begins inside its window and every
// resource stays within its limits. With windows and setups this is a vehicle
// routing problem with time windows and optional customers, which no method
// solves exactly at scale; the question is how close a search gets.
//
// The search is SISR, "slack induction by string removals" (Christiaens and
// Vanden Berghe, Transportation Science 2020), the simplest of the methods
// that are competitive with the best known results on these problems:
//
//   1. Ruin. Pick a task at random, walk its nearest neighbours, and from each
//      neighbour's resource take out a short string of consecutive tasks
//      around it. Strings rather than scattered tasks are what free up room:
//      the slack they leave lets a reinsertion reorder a whole stretch.
//   2. Recreate. Put the removed tasks back one at a time, each where it adds
//      the most, in an order drawn from a few rules (random, most valuable,
//      earliest window), skipping each position with a small chance ("blinks")
//      so the same ruin does not always rebuild the same plan. A task that is
//      worth more on nothing stays on nothing.
//   3. Accept by simulated annealing on a fixed, cooling schedule, and keep the
//      best plan seen. Deterministic for a seed.
//   4. Polish a copy of every new best by local search: exchange the
//      ends of two resources' sequences (which includes swapping whole
//      sequences), swap two tasks, move one task, swap a placed task for an
//      unplaced one. These are the moves ruin-and-recreate makes only by luck.
//
// An insertion is checked in constant time: every position keeps the latest
// moment its task may begin without making a later one late, so a new task
// fits when it begins inside its window and the task after it can still
// begin in time. Every limit is a sum, so its change is a few additions.
//
// A bound comes with the plan: with sequencing relaxed (every task at its
// cheapest possible setup) the problem is a multiple-choice knapsack over the
// per-resource limits, which the solver next door bounds from above. It is a
// true ceiling, and loose when setups matter.

struct AssignProblem {
	size_t n = 0; // tasks
	size_t m = 0; // resources
	// [i * m + r]: what task i on resource r is worth, before the plan terms.
	std::vector<double> value;
	// [i]: what task i is worth when placed on nothing; NaN when it must be placed.
	std::vector<double> else_value;
	// [i * m + r]: whether task i may go on resource r at all.
	std::vector<uint8_t> allowed;

	bool sequenced = false;
	// Hours, when sequenced. start[i * m + r]: from the resource's own starting
	// state to task i; duration[i * m + r]: how long task i occupies r;
	// setup: from task j to task i, [(j * n + i) * m + r], or [j * n + i] when
	// it is the same on every resource (setup_shared), which is the common case
	// and the one that stays small when there are many resources.
	std::vector<double> start, setup, duration;
	bool setup_shared = false;
	// [i]: when task i may begin, on one clock in hours; ±infinity when open.
	std::vector<double> window_from, window_to;
	// [r]: when resource r becomes free, on the same clock.
	std::vector<double> free_from;
	// [i * m + r]: how much of each placement value the objective counts.
	std::vector<double> coef_setup, coef_duration, coef_begin;

	// One limit per resource: SUM(amount + coefficients * plan values) over the
	// tasks placed on r, at most (or at least) cap[r].
	struct Limit {
		std::vector<double> amount;                    // [i * m + r]
		std::vector<double> coef_setup, coef_duration; // [i * m + r]
		std::vector<double> cap;                       // [r]
		bool at_most = true;
	};
	std::vector<Limit> limits;

	double Setup(size_t j, size_t i, size_t r) const {
		return setup_shared ? setup[j * n + i] : setup[(j * n + i) * m + r];
	}
};

struct AssignSettings {
	uint64_t seed = 42;
	// Ruin-and-recreate rounds. 0 means a default from the problem's size.
	int iterations = 0;
	int remove = 0;      // tasks taken out per round, on average; 0 means an eighth of the tasks, between 3 and 10
	int max_string = 10; // the longest string taken from one resource
	// The annealing temperature, as multiples of a typical task's worth over
	// nothing, from the first round to the last.
	double temp_start = 1.0, temp_end = 0.01;
	double seconds = 120; // a safety cap on wall time; the iteration count is what makes a run repeatable
};

struct AssignSolution {
	std::vector<int> resource;                  // [i]: resource index, or -1 when placed on nothing
	std::vector<int> sequence;                  // [i]: position on its resource, from 0; -1 when unplaced
	std::vector<double> begin, setup, duration; // [i]: the placement values; 0 when unplaced
	std::vector<std::vector<double>> usage;     // [limit][r]
	double objective = 0;
	double bound = std::numeric_limits<double>::infinity();
	bool feasible = false;
	bool exact = false;
	int iterations = 0;
};

namespace assign_detail {

constexpr double kInf = std::numeric_limits<double>::infinity();
// A task that must be placed is worth this much on nothing: finite, so the
// arithmetic stays finite, and large, so no plan leaves it unplaced by choice.
constexpr double kMustPlace = -1e12;

struct Rng {
	uint64_t st;
	explicit Rng(uint64_t seed) : st(seed ? seed : 0x9E3779B97F4A7C15ULL) {
	}
	uint64_t Next() {
		st ^= st << 13;
		st ^= st >> 7;
		st ^= st << 17;
		return st;
	}
	size_t Below(size_t n) {
		return n ? (size_t)(Next() % n) : 0;
	}
	double Unit() { // in (0, 1]
		return (double(Next() >> 11) + 1.0) / double(1ULL << 53);
	}
};

// One resource's sequence and what walking it gives. `latest[p]` is the latest
// moment task p may begin without making any later task begin after its window.
struct Route {
	std::vector<int> seq;
	std::vector<double> begin, end, setup_in, latest;
	std::vector<double> usage; // per limit
	double value = 0;          // the objective of the tasks on it
	bool feasible = true;
};

// The plan being searched: one route per resource, where each task sits, and
// the objective of it all.
struct Plan {
	std::vector<Route> routes;
	std::vector<int> where, pos; // per task: resource and position, or -1
	double objective = 0;
};

class Search {
public:
	Search(const AssignProblem &p, const AssignSettings &s) : P(p), S(s), n(p.n), m(p.m), rng(s.seed) {
		slow_path = false;
		for (double c : P.coef_begin) {
			slow_path = slow_path || c != 0.0; // a begin time in the objective moves with every later task
		}
		BuildNeighbours();
	}

	const AssignProblem &P;
	const AssignSettings &S;
	const size_t n, m;
	Rng rng;
	bool slow_path;
	std::vector<std::vector<int>> neighbours; // per task: the closest other tasks, nearest first
	std::vector<std::vector<int>> near_res;   // per task: the resources it is cheapest to start, when there are many

	double ElseValue(size_t i) const {
		return std::isnan(P.else_value[i]) ? kMustPlace : P.else_value[i];
	}
	double Start(size_t i, size_t r) const {
		return P.sequenced ? P.start[i * m + r] : 0.0;
	}
	double Dur(size_t i, size_t r) const {
		return P.sequenced ? P.duration[i * m + r] : 0.0;
	}
	double SetupTo(size_t j, size_t i, size_t r) const {
		return P.sequenced ? P.Setup(j, i, r) : 0.0;
	}
	double From(size_t i) const {
		return P.sequenced && P.window_from.size() == n ? P.window_from[i] : -kInf;
	}
	double To(size_t i) const {
		return P.sequenced && P.window_to.size() == n ? P.window_to[i] : kInf;
	}
	double FreeFrom(size_t r) const {
		return P.sequenced && P.free_from.size() == m && std::isfinite(P.free_from[r]) ? P.free_from[r] : 0.0;
	}
	// The objective of task i on r given the setup it pays and when it begins.
	double PlacedValue(size_t i, size_t r, double setup, double begin) const {
		const size_t o = i * m + r;
		return P.value[o] + P.coef_setup[o] * setup + P.coef_duration[o] * Dur(i, r) + P.coef_begin[o] * begin;
	}

	// ---- walking a route --------------------------------------------------

	void Rebuild(Plan &pl, size_t r) const {
		Route &rt = pl.routes[r];
		Walk(r, rt);
		for (size_t p = 0; p < rt.seq.size(); p++) {
			pl.where[(size_t)rt.seq[p]] = (int)r;
			pl.pos[(size_t)rt.seq[p]] = (int)p;
		}
	}

	// Everything about route r that follows from its sequence: begin times,
	// setups, latest begins, limit usage, value, and whether it is feasible.
	void Walk(size_t r, Route &rt) const {
		const size_t L = rt.seq.size();
		rt.begin.assign(L, 0.0);
		rt.end.assign(L, 0.0);
		rt.setup_in.assign(L, 0.0);
		rt.latest.assign(L, 0.0);
		rt.usage.assign(P.limits.size(), 0.0);
		rt.value = 0;
		rt.feasible = true;
		double clock = FreeFrom(r);
		for (size_t p = 0; p < L; p++) {
			const size_t i = (size_t)rt.seq[p];
			const size_t o = i * m + r;
			if (!P.allowed[o]) {
				rt.feasible = false;
			}
			const double st = p == 0 ? Start(i, r) : SetupTo((size_t)rt.seq[p - 1], i, r);
			double b = clock + st;
			if (b < From(i)) {
				b = From(i); // wait for the window to open
			}
			if (b > To(i)) {
				rt.feasible = false;
			}
			rt.begin[p] = b;
			rt.setup_in[p] = st;
			rt.end[p] = clock = b + Dur(i, r);
			rt.value += PlacedValue(i, r, st, b);
			for (size_t l = 0; l < P.limits.size(); l++) {
				const auto &lim = P.limits[l];
				rt.usage[l] += lim.amount[o] + lim.coef_setup[o] * st + lim.coef_duration[o] * Dur(i, r);
			}
		}
		for (size_t p = L; p-- > 0;) {
			const size_t i = (size_t)rt.seq[p];
			double lat = To(i);
			if (p + 1 < L) {
				lat = std::min(lat, rt.latest[p + 1] - Dur(i, r) - SetupTo(i, (size_t)rt.seq[p + 1], r));
			}
			rt.latest[p] = lat;
		}
		for (size_t l = 0; l < P.limits.size(); l++) {
			const auto &lim = P.limits[l];
			if (lim.at_most && rt.usage[l] > lim.cap[r] + 1e-9 * std::max(1.0, std::fabs(lim.cap[r]))) {
				rt.feasible = false;
			}
		}
	}

	// The change in the plan's objective from putting task i on r at position p,
	// or -infinity when it does not fit there.
	double InsertionGain(const Plan &pl, size_t i, size_t r, size_t p) const {
		const size_t o = i * m + r;
		if (!P.allowed[o]) {
			return -kInf;
		}
		const Route &rt = pl.routes[r];
		const size_t L = rt.seq.size();
		if (slow_path) {
			// A begin time in the objective: every later task may move, so walk it.
			Plan trial;
			trial.routes.assign(m, Route());
			trial.where.assign(n, -1);
			trial.pos.assign(n, -1);
			trial.routes[r].seq = rt.seq;
			trial.routes[r].seq.insert(trial.routes[r].seq.begin() + (long)p, (int)i);
			Rebuild(trial, r);
			if (!trial.routes[r].feasible) {
				return -kInf;
			}
			return trial.routes[r].value - rt.value - ElseValue(i);
		}
		const bool first = p == 0;
		const size_t prev = first ? 0 : (size_t)rt.seq[p - 1];
		const double s_in = first ? Start(i, r) : SetupTo(prev, i, r);
		const double prev_end = first ? FreeFrom(r) : rt.end[p - 1];
		double b = prev_end + s_in;
		if (b < From(i)) {
			b = From(i);
		}
		if (b > To(i)) {
			return -kInf;
		}
		double delta = PlacedValue(i, r, s_in, b);
		double s_out = 0, s_old = 0;
		size_t next = 0;
		if (p < L) {
			next = (size_t)rt.seq[p];
			s_out = SetupTo(i, next, r);
			s_old = rt.setup_in[p];
			if (b + Dur(i, r) + s_out > rt.latest[p] + 1e-9) {
				return -kInf; // the next task, or one after it, would be late
			}
			delta += P.coef_setup[next * m + r] * (s_out - s_old);
		}
		for (size_t l = 0; l < P.limits.size(); l++) {
			const auto &lim = P.limits[l];
			if (!lim.at_most) {
				continue; // a floor is judged on the finished plan
			}
			double d = lim.amount[o] + lim.coef_setup[o] * s_in + lim.coef_duration[o] * Dur(i, r);
			if (p < L) {
				d += lim.coef_setup[next * m + r] * (s_out - s_old);
			}
			if (rt.usage[l] + d > lim.cap[r] + 1e-9 * std::max(1.0, std::fabs(lim.cap[r]))) {
				return -kInf;
			}
		}
		return delta - ElseValue(i);
	}

	struct Spot {
		double gain = -kInf;
		size_t r = 0, p = 0;
	};

	// The best place for task i, each position skipped with a small chance.
	Spot BestSpot(const Plan &pl, size_t i, double blink) {
		Spot best;
		const double latest_begin = To(i);
		auto consider = [&](size_t r) {
			if (!P.allowed[i * m + r]) {
				return;
			}
			const Route &rt = pl.routes[r];
			const size_t L = rt.seq.size();
			const size_t positions = P.sequenced ? L + 1 : 1;
			for (size_t q = 0; q < positions; q++) {
				const size_t p = P.sequenced ? q : L;
				// Ends only grow along a sequence: once the task before this
				// position ends after the window closes, no later position fits.
				if (P.sequenced && p > 0 && rt.end[p - 1] > latest_begin) {
					break;
				}
				if (blink > 0 && rng.Unit() < blink) {
					continue;
				}
				const double g = InsertionGain(pl, i, r, p);
				if (g > best.gain) {
					best.gain = g;
					best.r = r;
					best.p = p;
				}
			}
		};
		if (near_res.empty()) {
			for (size_t r = 0; r < m; r++) {
				consider(r);
			}
			return best;
		}
		// Many resources: the ones the task's neighbours are on, and the ones it is cheapest to start.
		seen_stamp++;
		auto once = [&](size_t r) {
			if (seen[r] != seen_stamp) {
				seen[r] = seen_stamp;
				consider(r);
			}
		};
		for (int j : neighbours[i]) {
			if (pl.where[(size_t)j] >= 0) {
				once((size_t)pl.where[(size_t)j]);
			}
		}
		for (int r : near_res[i]) {
			once((size_t)r);
		}
		return best;
	}

	void Insert(Plan &pl, size_t i, size_t r, size_t p) {
		Touch(pl, r);
		auto &seq = pl.routes[r].seq;
		const double before = pl.routes[r].value;
		seq.insert(seq.begin() + (long)p, (int)i);
		Rebuild(pl, r);
		pl.objective += pl.routes[r].value - before - ElseValue(i);
	}

	// ---- ruin and recreate -------------------------------------------------

	// Take out a string of `len` tasks around position `at` of route r; with
	// `keep` > 0, a string of len + keep with `keep` consecutive tasks left in
	// its middle. Returns the tasks taken out.
	void RemoveString(Plan &pl, size_t r, size_t at, size_t len, size_t keep, std::vector<int> &out) {
		Touch(pl, r);
		Route &rt = pl.routes[r];
		const size_t L = rt.seq.size();
		const size_t span = std::min(L, len + keep);
		const size_t lo = at + 1 >= span ? at + 1 - span : 0;
		const size_t hi = std::min(at, L - span);
		const size_t s = lo + rng.Below(hi - lo + 1);
		const size_t keep_at = keep ? s + rng.Below(span - keep + 1) : L; // first kept position
		const double before = rt.value;
		std::vector<int> stay;
		for (size_t p = 0; p < L; p++) {
			const bool in_span = p >= s && p < s + span;
			const bool kept = keep && p >= keep_at && p < keep_at + keep;
			if (in_span && !kept) {
				const size_t i = (size_t)rt.seq[p];
				out.push_back((int)i);
				pl.where[i] = -1;
				pl.pos[i] = -1;
				pl.objective += ElseValue(i);
			} else {
				stay.push_back(rt.seq[p]);
			}
		}
		rt.seq.swap(stay);
		Rebuild(pl, r);
		pl.objective += rt.value - before;
	}

	// SISR's ruin: strings from the resources of a random task's neighbours.
	std::vector<int> Ruin(Plan &pl) {
		std::vector<int> pool;
		size_t placed = 0, busy = 0;
		for (const auto &rt : pl.routes) {
			placed += rt.seq.size();
			busy += rt.seq.empty() ? 0 : 1;
		}
		const double avg_len = busy ? double(placed) / double(busy) : 1.0;
		const double cbar = S.remove > 0 ? S.remove : std::min(10.0, std::max(3.0, std::round(double(n) / 8.0)));
		const double ls_max = std::max(1.0, std::min(double(std::max(1, S.max_string)), avg_len));
		const double ks_max = std::max(1.0, 4.0 * cbar / (1.0 + ls_max) - 1.0);
		const size_t ks = 1 + (size_t)std::floor(rng.Unit() * ks_max);
		const size_t seed_task = rng.Below(n);
		std::vector<uint8_t> ruined(m, 0);
		size_t done = 0, pooled = 0;
		auto visit = [&](size_t t) {
			if (pl.where[t] < 0) {
				// An unplaced task near the ruin: a chance to bring it in.
				if (pooled < (size_t)cbar) {
					pool.push_back((int)t);
					pooled++;
				}
				return;
			}
			const size_t r = (size_t)pl.where[t];
			if (ruined[r] || done >= ks) {
				return;
			}
			const size_t L = pl.routes[r].seq.size();
			const double lmax = std::min(double(L), ls_max);
			const size_t len = 1 + (size_t)std::floor(rng.Unit() * lmax - 1e-12);
			size_t keep = 0;
			if (L > len && rng.Unit() < 0.5) {
				// The split string: some of the middle stays.
				keep = 1;
				while (len + keep < L && rng.Unit() < 0.5) {
					keep++;
				}
			}
			RemoveString(pl, r, (size_t)pl.pos[t], std::min(len, L), keep, pool);
			ruined[r] = 1;
			done++;
		};
		visit(seed_task);
		for (int t : neighbours[seed_task]) {
			if (done >= ks && pooled >= (size_t)cbar) {
				break;
			}
			visit((size_t)t);
		}
		std::sort(pool.begin(), pool.end());
		pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
		return pool;
	}

	// Put back every task in `pool` where it adds the most, in one of a few orders.
	void Recreate(Plan &pl, std::vector<int> &pool, double blink) {
		for (size_t k = pool.size(); k > 1; k--) {
			std::swap(pool[k - 1], pool[rng.Below(k)]);
		}
		const size_t rule = rng.Below(11);
		if (rule < 4) {
			// random, as shuffled
		} else if (rule < 8) {
			std::stable_sort(pool.begin(), pool.end(),
			                 [&](int a, int b) { return best_gain[(size_t)a] > best_gain[(size_t)b]; });
		} else if (rule < 10) {
			std::stable_sort(pool.begin(), pool.end(), [&](int a, int b) { return From((size_t)a) < From((size_t)b); });
		} else {
			std::stable_sort(pool.begin(), pool.end(), [&](int a, int b) { return To((size_t)a) < To((size_t)b); });
		}
		for (int t : pool) {
			const size_t i = (size_t)t;
			const Spot sp = BestSpot(pl, i, blink);
			const bool must = std::isnan(P.else_value[i]);
			if (sp.gain == -kInf || (!must && sp.gain <= 1e-9)) {
				continue; // stays on nothing
			}
			Insert(pl, i, sp.r, sp.p);
		}
	}

	// ---- polish -------------------------------------------------------------

	// Local search to a local optimum, first improvement, over four moves:
	// exchange the ends of two resources' sequences (which includes swapping
	// two whole sequences), swap two tasks, move one task anywhere, and swap a
	// placed task for one on nothing. Each move is judged by walking the routes
	// it changes. Returns whether the plan improved.
	bool Polish(Plan &pl, const CancelCheck &cancelled) {
		bool any = false;
		Route ta, tb;
		const double eps = 1e-9;
		auto try_pair = [&](size_t a, size_t b, std::vector<int> &&sa, std::vector<int> &&sb, double else_delta) {
			ta.seq = std::move(sa);
			Walk(a, ta);
			if (!ta.feasible) {
				return false;
			}
			double gain = ta.value - pl.routes[a].value + else_delta;
			if (b != a) {
				tb.seq = std::move(sb);
				Walk(b, tb);
				if (!tb.feasible) {
					return false;
				}
				gain += tb.value - pl.routes[b].value;
			}
			if (gain <= eps * std::max(1.0, std::fabs(pl.objective))) {
				return false;
			}
			const std::vector<int> leaving_a = pl.routes[a].seq,
			                       leaving_b = b != a ? pl.routes[b].seq : std::vector<int>();
			for (int t : leaving_a) {
				pl.where[(size_t)t] = -1;
				pl.pos[(size_t)t] = -1;
			}
			for (int t : leaving_b) {
				pl.where[(size_t)t] = -1;
				pl.pos[(size_t)t] = -1;
			}
			pl.routes[a] = ta;
			Rebuild(pl, a);
			if (b != a) {
				pl.routes[b] = tb;
				Rebuild(pl, b);
			}
			pl.objective += gain;
			return true;
		};
		auto related = [&](size_t a, size_t b) {
			if (m <= 40) {
				return true;
			}
			for (int t : pl.routes[a].seq) {
				for (size_t k = 0; k < std::min<size_t>(10, neighbours[(size_t)t].size()); k++) {
					if (pl.where[(size_t)neighbours[(size_t)t][k]] == (int)b) {
						return true;
					}
				}
			}
			return pl.routes[a].seq.empty() || pl.routes[b].seq.empty();
		};
		bool improved = true;
		while (improved) {
			improved = false;
			if (cancelled) {
				cancelled();
			}
			// 1. Exchange tails: a = A[:i] + B[j:], b = B[:j] + A[i:].
			for (size_t a = 0; a < m && !improved; a++) {
				for (size_t b = a + 1; b < m && !improved; b++) {
					if (!P.sequenced || !related(a, b)) {
						continue;
					}
					const std::vector<int> A = pl.routes[a].seq, B = pl.routes[b].seq;
					for (size_t i = 0; i <= A.size() && !improved; i++) {
						for (size_t j = 0; j <= B.size() && !improved; j++) {
							if (i == A.size() && j == B.size()) {
								continue;
							}
							std::vector<int> na(A.begin(), A.begin() + (long)i), nb(B.begin(), B.begin() + (long)j);
							na.insert(na.end(), B.begin() + (long)j, B.end());
							nb.insert(nb.end(), A.begin() + (long)i, A.end());
							improved = try_pair(a, b, std::move(na), std::move(nb), 0.0);
						}
					}
				}
			}
			// 2. Swap two placed tasks, on one resource or two.
			for (size_t t = 0; t < n && !improved; t++) {
				if (pl.where[t] < 0) {
					continue;
				}
				for (int u_ : neighbours[t]) {
					const size_t u = (size_t)u_;
					if (pl.where[u] < 0 || u < t) {
						continue;
					}
					const size_t a = (size_t)pl.where[t], b = (size_t)pl.where[u];
					std::vector<int> na = pl.routes[a].seq;
					if (a == b) {
						std::swap(na[(size_t)pl.pos[t]], na[(size_t)pl.pos[u]]);
						improved = try_pair(a, a, std::move(na), {}, 0.0);
					} else {
						std::vector<int> nb = pl.routes[b].seq;
						na[(size_t)pl.pos[t]] = (int)u;
						nb[(size_t)pl.pos[u]] = (int)t;
						improved = try_pair(a, b, std::move(na), std::move(nb), 0.0);
					}
					if (improved) {
						break;
					}
				}
			}
			// 3. Move one placed task to any position on any resource.
			for (size_t t = 0; t < n && !improved; t++) {
				if (pl.where[t] < 0) {
					continue;
				}
				const size_t a = (size_t)pl.where[t];
				std::vector<int> without = pl.routes[a].seq;
				without.erase(without.begin() + pl.pos[t]);
				for (size_t b = 0; b < m && !improved; b++) {
					if (!P.allowed[t * m + b] || (b != a && !related(a, b))) {
						continue;
					}
					const std::vector<int> &base = b == a ? without : pl.routes[b].seq;
					const size_t positions = P.sequenced ? base.size() + 1 : 1;
					for (size_t q = 0; q < positions && !improved; q++) {
						std::vector<int> nb = base;
						nb.insert(nb.begin() + (long)(P.sequenced ? q : base.size()), (int)t);
						if (b == a) {
							if (nb == pl.routes[a].seq) {
								continue;
							}
							improved = try_pair(a, a, std::move(nb), {}, 0.0);
						} else {
							improved = try_pair(a, b, std::vector<int>(without), std::move(nb), 0.0);
						}
					}
				}
			}
			// 4. Swap a placed task for a task on nothing, in its place.
			for (size_t t = 0; t < n && !improved; t++) {
				if (pl.where[t] < 0) {
					continue;
				}
				const size_t a = (size_t)pl.where[t];
				for (int u_ : neighbours[t]) {
					const size_t u = (size_t)u_;
					if (pl.where[u] >= 0 || !P.allowed[u * m + a]) {
						continue;
					}
					std::vector<int> na = pl.routes[a].seq;
					na[(size_t)pl.pos[t]] = (int)u;
					improved = try_pair(a, a, std::move(na), {}, ElseValue(t) - ElseValue(u));
					if (improved) {
						break;
					}
				}
			}
			any = any || improved;
		}
		return any;
	}

	// ---- undo --------------------------------------------------------------

	// A round changes a few routes. They are saved the first time they are
	// touched, so a rejected round is undone by putting them back.
	std::vector<int> touched;
	std::vector<Route> saved;
	std::vector<uint8_t> is_touched;
	double saved_objective = 0;

	void BeginRound(const Plan &pl) {
		touched.clear();
		saved.clear();
		is_touched.assign(m, 0);
		saved_objective = pl.objective;
	}
	void Touch(const Plan &pl, size_t r) {
		if (!is_touched.empty() && !is_touched[r]) {
			is_touched[r] = 1;
			touched.push_back((int)r);
			saved.push_back(pl.routes[r]);
		}
	}
	void Undo(Plan &pl) {
		for (size_t k = 0; k < touched.size(); k++) {
			const size_t r = (size_t)touched[k];
			for (int i : pl.routes[r].seq) {
				pl.where[(size_t)i] = -1;
				pl.pos[(size_t)i] = -1;
			}
		}
		for (size_t k = 0; k < touched.size(); k++) {
			const size_t r = (size_t)touched[k];
			pl.routes[r] = saved[k];
			for (size_t p = 0; p < pl.routes[r].seq.size(); p++) {
				pl.where[(size_t)pl.routes[r].seq[p]] = (int)r;
				pl.pos[(size_t)pl.routes[r].seq[p]] = (int)p;
			}
		}
		pl.objective = saved_objective;
	}

	// ---- setup -------------------------------------------------------------

	std::vector<double> best_gain; // per task: what its best single placement is worth over nothing
	std::vector<uint32_t> seen;
	uint32_t seen_stamp = 0;

	// How far apart two tasks are, for the ruin's neighbourhoods: the cheaper of
	// the two setups between them, or without sequencing, how differently they
	// are valued.
	void BuildNeighbours() {
		best_gain.assign(n, -kInf);
		std::vector<int> best_r(n, -1);
		for (size_t i = 0; i < n; i++) {
			for (size_t r = 0; r < m; r++) {
				if (P.allowed[i * m + r]) {
					const double g = PlacedValue(i, r, Start(i, r), 0.0) - ElseValue(i);
					if (g > best_gain[i]) {
						best_gain[i] = g;
						best_r[i] = (int)r;
					}
				}
			}
		}
		const size_t K = std::min<size_t>(n ? n - 1 : 0, 60);
		const size_t probe = P.setup_shared ? 1 : std::min<size_t>(m, 8);
		neighbours.assign(n, {});
		std::vector<std::pair<double, int>> d;
		for (size_t i = 0; i < n; i++) {
			d.clear();
			for (size_t j = 0; j < n; j++) {
				if (j == i) {
					continue;
				}
				double dist;
				if (P.sequenced) {
					dist = kInf;
					for (size_t k = 0; k < probe; k++) {
						const size_t r = P.setup_shared ? 0 : (k * m) / probe;
						dist = std::min(dist, std::min(P.Setup(i, j, r), P.Setup(j, i, r)));
					}
				} else {
					dist = (best_r[i] != best_r[j] ? 1e6 : 0.0) + std::fabs(best_gain[i] - best_gain[j]);
				}
				d.emplace_back(dist, (int)j);
			}
			const size_t take = std::min(K, d.size());
			std::partial_sort(d.begin(), d.begin() + (long)take, d.end());
			for (size_t k = 0; k < take; k++) {
				neighbours[i].push_back(d[k].second);
			}
		}
		if (n > 400 && m > 20) {
			near_res.assign(n, {});
			std::vector<std::pair<double, int>> s;
			for (size_t i = 0; i < n; i++) {
				s.clear();
				for (size_t r = 0; r < m; r++) {
					if (P.allowed[i * m + r]) {
						s.emplace_back(Start(i, r), (int)r);
					}
				}
				const size_t take = std::min<size_t>(12, s.size());
				std::partial_sort(s.begin(), s.begin() + (long)take, s.end());
				for (size_t k = 0; k < take; k++) {
					near_res[i].push_back(s[k].second);
				}
			}
			seen.assign(m, 0);
		}
	}

	// ---- the search ---------------------------------------------------------

	Plan Empty() const {
		Plan pl;
		pl.routes.assign(m, Route());
		pl.where.assign(n, -1);
		pl.pos.assign(n, -1);
		for (size_t r = 0; r < m; r++) {
			pl.routes[r].usage.assign(P.limits.size(), 0.0);
		}
		for (size_t i = 0; i < n; i++) {
			pl.objective += ElseValue(i);
		}
		return pl;
	}

	Plan Run(int &iterations_done, const CancelCheck &cancelled) {
		Plan cur = Empty();
		// The first plan: every task offered once, most valuable first, no blinks.
		std::vector<int> all(n);
		for (size_t i = 0; i < n; i++) {
			all[i] = (int)i;
		}
		std::stable_sort(all.begin(), all.end(),
		                 [&](int a, int b) { return best_gain[(size_t)a] > best_gain[(size_t)b]; });
		for (int t : all) {
			const Spot sp = BestSpot(cur, (size_t)t, 0.0);
			if (sp.gain != -kInf && (sp.gain > 1e-9 || std::isnan(P.else_value[(size_t)t]))) {
				Insert(cur, (size_t)t, sp.r, sp.p);
			}
		}
		Plan best = cur;

		// The temperature, in the units of the objective: from a tenth of a
		// typical task's worth down to a thousandth of it.
		std::vector<double> worth;
		for (double g : best_gain) {
			if (std::isfinite(g) && g > 0) {
				worth.push_back(g);
			}
		}
		double typical = 1.0;
		if (!worth.empty()) {
			std::nth_element(worth.begin(), worth.begin() + (long)(worth.size() / 2), worth.end());
			typical = std::max(1e-9, worth[worth.size() / 2]);
		}
		const double t0 = S.temp_start * typical, tf = S.temp_end * typical;
		// Measured on dispatch weeks: 25,000 rounds per task find the best plan
		// known for 40 loads on every seed tried, and keep improving past that.
		const int iters =
		    S.iterations > 0 ? S.iterations : (int)std::min<size_t>(3000000, std::max<size_t>(200000, 25000 * n));
		const auto clock0 = std::chrono::steady_clock::now();
		iterations_done = 0;
		for (int it = 0; it < iters; it++) {
			if ((it & 255) == 0) {
				if (cancelled) {
					cancelled();
				}
				const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count();
				if (elapsed > S.seconds) {
					break;
				}
			}
			iterations_done = it + 1;
			const double T = t0 * std::pow(tf / t0, double(it) / double(iters));
			BeginRound(cur);
			std::vector<int> pool = Ruin(cur);
			Recreate(cur, pool, 0.01);
			if (cur.objective > saved_objective + T * std::log(rng.Unit())) {
				if (cur.objective > best.objective + 1e-9) {
					// The polished copy is the answer; the search carries on from
					// where it was, since a polished plan is a basin it rarely leaves.
					best = cur;
					Polish(best, cancelled);
				}
			} else {
				Undo(cur);
			}
		}
		Polish(best, cancelled);
		return best;
	}
};

} // namespace assign_detail

// The relaxation bound: every task at its cheapest conceivable setup and begin,
// then a multiple-choice knapsack over the per-resource limits.
inline double AssignBound(const AssignProblem &P, const OptSettings &set, const CancelCheck &cancelled) {
	using namespace assign_detail;
	const size_t n = P.n, m = P.m;
	OptProblem Q;
	Q.n = n;
	Q.K = (int)m + 1; // resources, then "nothing"
	const size_t K = m + 1;
	Q.v.assign(n * K, -kInf);
	bool open_ended = false;
	auto least_setup = [&](size_t i, size_t r) {
		double st = P.start[i * m + r];
		for (size_t j = 0; j < n; j++) {
			if (j != i && P.allowed[j * m + r]) {
				st = std::min(st, P.Setup(j, i, r));
			}
		}
		return st;
	};
	for (size_t i = 0; i < n; i++) {
		for (size_t r = 0; r < m; r++) {
			const size_t o = i * m + r;
			if (!P.allowed[o]) {
				continue;
			}
			double st = 0.0, dur = 0.0, begin_lo = 0.0, begin_hi = 0.0;
			if (P.sequenced) {
				st = least_setup(i, r);
				dur = P.duration[o];
				const double free = P.free_from.empty() || !std::isfinite(P.free_from[r]) ? 0.0 : P.free_from[r];
				begin_lo = std::max(free + st, P.window_from.size() == n ? P.window_from[i] : -kInf);
				begin_hi = P.window_to.size() == n ? P.window_to[i] : kInf;
			}
			// The most favourable value of each term, independently: an upper bound.
			// Setup is at its minimum here; a positive weight on it would need its
			// maximum instead, which the relaxation does not know, so it is treated
			// as open-ended.
			double v = P.value[o] + P.coef_duration[o] * dur;
			if (P.coef_setup[o] > 0 && st > 0) {
				open_ended = true;
			} else {
				v += P.coef_setup[o] * st;
			}
			if (P.coef_begin[o] != 0.0) {
				const double b = P.coef_begin[o] > 0 ? begin_hi : begin_lo;
				if (!std::isfinite(b)) {
					open_ended = true;
				} else {
					v += P.coef_begin[o] * b;
				}
			}
			Q.v[i * K + r] = v;
		}
		Q.v[i * K + m] = std::isnan(P.else_value[i]) ? -kInf : P.else_value[i];
	}
	if (open_ended) {
		return kInf;
	}
	// One knapsack limit per (limit, resource): a task on another resource uses nothing of it.
	for (size_t l = 0; l < P.limits.size(); l++) {
		const AssignProblem::Limit &lim = P.limits[l];
		const double cs = lim.at_most ? 1.0 : -1.0;
		for (size_t r = 0; r < m; r++) {
			std::vector<double> a(n * K, 0.0);
			for (size_t i = 0; i < n; i++) {
				const size_t o = i * m + r;
				if (!P.allowed[o]) {
					continue;
				}
				double st = 0.0, dur = 0.0;
				if (P.sequenced) {
					st = least_setup(i, r);
					dur = P.duration[o];
				}
				// The least a placement can use of a ceiling, the most of a floor.
				a[i * K + r] = cs * (lim.amount[o] + lim.coef_setup[o] * st + lim.coef_duration[o] * dur);
			}
			Q.a.push_back(a);
			Q.b.push_back(cs * lim.cap[r]);
		}
	}
	// Forbidden placements carry -infinity, and the knapsack wants finite
	// numbers of a sane size: a placeholder like -1e300 swamps its bound
	// arithmetic and it reported an "exact" optimum far below the true one.
	// A penalty larger than every value together keeps them unattractive and
	// leaves the bound valid (any real plan is still a plan here, worth the same).
	double total = 1.0;
	for (double v : Q.v) {
		if (std::isfinite(v)) {
			total += std::fabs(v);
		}
	}
	for (double &v : Q.v) {
		if (v == -kInf) {
			v = -1000.0 * total;
		}
	}
	const OptSolution sol = SolveOpt(Q, set, cancelled);
	return sol.bound;
}

inline AssignSolution SolveAssign(const AssignProblem &P, const AssignSettings &settings = AssignSettings(),
                                  const CancelCheck &cancelled = CancelCheck()) {
	using namespace assign_detail;
	const size_t n = P.n, m = P.m;
	const size_t setup_size = P.setup_shared ? n * n : n * n * m;
	if (P.value.size() != n * m || P.else_value.size() != n || P.allowed.size() != n * m ||
	    P.coef_setup.size() != n * m || P.coef_duration.size() != n * m || P.coef_begin.size() != n * m ||
	    (P.sequenced && (P.start.size() != n * m || P.setup.size() != setup_size || P.duration.size() != n * m))) {
		throw std::runtime_error("pql: internal error: malformed assignment problem");
	}
	for (const auto &lim : P.limits) {
		if (lim.amount.size() != n * m || lim.coef_setup.size() != n * m || lim.coef_duration.size() != n * m ||
		    lim.cap.size() != m) {
			throw std::runtime_error("pql: internal error: malformed assignment limit");
		}
	}

	AssignSolution out;
	out.resource.assign(n, -1);
	out.sequence.assign(n, -1);
	out.begin.assign(n, 0.0);
	out.setup.assign(n, 0.0);
	out.duration.assign(n, 0.0);
	out.usage.assign(P.limits.size(), std::vector<double>(m, 0.0));
	if (n == 0 || m == 0) {
		out.feasible = true;
		for (size_t i = 0; i < n; i++) {
			if (std::isnan(P.else_value[i])) {
				out.feasible = false;
			} else {
				out.objective += P.else_value[i];
			}
		}
		out.bound = out.objective;
		out.exact = out.feasible;
		return out;
	}

	Search search(P, settings);
	const Plan best = search.Run(out.iterations, cancelled);

	// Read the plan back, and check the floors it could not enforce while building.
	out.feasible = true;
	out.objective = 0;
	for (size_t r = 0; r < m; r++) {
		const Route &rt = best.routes[r];
		out.feasible = out.feasible && rt.feasible;
		out.objective += rt.value;
		for (size_t p = 0; p < rt.seq.size(); p++) {
			const size_t i = (size_t)rt.seq[p];
			out.resource[i] = (int)r;
			out.sequence[i] = (int)p;
			out.begin[i] = rt.begin[p];
			out.setup[i] = rt.setup_in[p];
			out.duration[i] = P.sequenced ? P.duration[i * m + r] : 0.0;
		}
		for (size_t l = 0; l < P.limits.size(); l++) {
			out.usage[l][r] = rt.usage[l];
			const AssignProblem::Limit &lim = P.limits[l];
			if (!lim.at_most && rt.usage[l] < lim.cap[r] - 1e-9 * std::max(1.0, std::fabs(lim.cap[r]))) {
				out.feasible = false;
			}
		}
	}
	for (size_t i = 0; i < n; i++) {
		if (out.resource[i] < 0) {
			if (std::isnan(P.else_value[i])) {
				out.feasible = false;
			} else {
				out.objective += P.else_value[i];
			}
		}
	}
	out.bound = AssignBound(P, OptSettings(), cancelled);
	if (out.feasible) {
		out.bound = std::max(out.bound, out.objective);
		out.exact = out.bound - out.objective <= 1e-9 * std::max(1.0, std::fabs(out.objective));
	}
	return out;
}

} // namespace pql
