// Randomized property test for the link index and the temporal windows built
// on it. Every check is against a brute-force reference that scans the child
// table row by row, so a disagreement is a bug in the CSR, the sort, a binary
// search, or a prefix sum, never in the model.
//
// Covered: orphan children, NULL keys, NULL child timestamps, ties and
// near-ties (closer than float resolution at 1e15 micros), unsorted input,
// buckets wider than the radix path's 64-row threshold and than 2048 rows,
// self-referencing tables, two foreign keys between the same pair of tables,
// empty parent and child tables, integer keys at the edge of int64, and
// non-integral double keys.
//
//   clang++ -std=c++17 -O2 -I extras/pql -o /tmp/l extras/pql/test_links.cpp && /tmp/l [seeds]
//   clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -I extras/pql -o /tmp/l extras/pql/test_links.cpp
#include "pql/pql.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>
using namespace pql;

static int fails = 0;
static long checks = 0;
#define CHECK(cond, ...)                                                                                               \
	do {                                                                                                               \
		checks++;                                                                                                      \
		if (!(cond)) {                                                                                                 \
			if (fails < g_maxfail) {                                                                                   \
				printf("  FAIL seed=%u: ", g_seed);                                                                    \
				printf(__VA_ARGS__);                                                                                   \
				printf("\n");                                                                                          \
			}                                                                                                          \
			fails++;                                                                                                   \
		}                                                                                                              \
	} while (0)

static unsigned g_seed = 0;
static int g_maxfail = 40;
static const double DAY = 86400.0 * 1e6;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

static Column MC(const char *n, ColType t, size_t s) {
	Column c;
	c.name = n;
	c.type = t;
	c.num.assign(s, 0.0);
	c.code.assign(s, 0u);
	c.valid.assign(s, 1);
	return c;
}

// A raw child table, before it becomes a Frame.
struct RawChild {
	std::vector<double> key;  // NaN = NULL
	std::vector<double> time; // NaN = NULL
	std::vector<double> amt;  // NaN = NULL
};

static Frame MakeParent(const std::string &name, const std::vector<double> &keys, ColType kt) {
	Frame f;
	f.name = name;
	f.nrows = keys.size();
	Column id = MC("id", kt, keys.size());
	Column x = MC("x", ColType::DOUBLE, keys.size());
	for (size_t i = 0; i < keys.size(); i++) {
		if (std::isnan(keys[i])) {
			id.valid[i] = 0;
		} else {
			id.num[i] = keys[i];
		}
		x.num[i] = double(i);
	}
	f.columns = {id, x};
	return f;
}

static Frame MakeChild(const std::string &name, const RawChild &r, ColType kt, bool with_time) {
	Frame f;
	f.name = name;
	f.nrows = r.key.size();
	Column id = MC("id", ColType::INT64, f.nrows);
	Column pk = MC("parent_id", kt, f.nrows);
	Column ts = MC("ts", ColType::TIMESTAMP, f.nrows);
	Column amt = MC("amount", ColType::DOUBLE, f.nrows);
	for (size_t i = 0; i < f.nrows; i++) {
		id.num[i] = double(i);
		if (std::isnan(r.key[i])) {
			pk.valid[i] = 0;
		} else {
			pk.num[i] = r.key[i];
		}
		if (std::isnan(r.time[i])) {
			ts.valid[i] = 0;
		} else {
			ts.num[i] = r.time[i];
		}
		if (std::isnan(r.amt[i])) {
			amt.valid[i] = 0;
		} else {
			amt.num[i] = r.amt[i];
		}
	}
	f.columns = with_time ? std::vector<Column> {id, pk, ts, amt} : std::vector<Column> {id, pk, amt};
	return f;
}

// Brute-force: which parent row does child i point at? -1 if none. When
// parent keys repeat, any of the matching rows is acceptable; the caller checks
// membership instead.
static std::vector<int> BruteParents(const std::vector<double> &pkeys, const std::vector<double> &ckeys,
                                     std::vector<std::vector<int>> *all = nullptr) {
	std::vector<int> out(ckeys.size(), -1);
	if (all) {
		all->assign(ckeys.size(), {});
	}
	for (size_t c = 0; c < ckeys.size(); c++) {
		if (std::isnan(ckeys[c])) {
			continue;
		}
		for (size_t p = 0; p < pkeys.size(); p++) {
			if (std::isnan(pkeys[p])) {
				continue;
			}
			if (pkeys[p] == ckeys[c]) {
				if (out[c] < 0) {
					out[c] = (int)p;
				}
				if (all) {
					(*all)[c].push_back((int)p);
				}
			}
		}
	}
	return out;
}

struct Brute {
	size_t n = 0, nval = 0;
	double sum = 0, mn = 0, mx = 0;
	bool have = false;
};
// Children of parent p with lo < t <= hi (t NULL never qualifies).
static Brute BruteWindow(const std::vector<int> &parent_of, const RawChild &r, int p, double lo, double hi,
                         bool timed) {
	Brute b;
	for (size_t c = 0; c < r.key.size(); c++) {
		if (parent_of[c] != p) {
			continue;
		}
		if (timed) {
			const double t = r.time[c];
			if (std::isnan(t) || !(t > lo) || !(t <= hi)) {
				continue;
			}
		}
		b.n++;
		if (!std::isnan(r.amt[c])) {
			b.sum += r.amt[c];
			if (!b.have || r.amt[c] < b.mn) {
				b.mn = r.amt[c];
			}
			if (!b.have || r.amt[c] > b.mx) {
				b.mx = r.amt[c];
			}
			b.have = true;
			b.nval++;
		}
	}
	return b;
}

// Verify one built link against the raw tables.
static void CheckLink(const Database &db, const Link &lk, const std::vector<double> &pkeys, const RawChild &r,
                      bool timed, std::mt19937_64 &rng, bool dup_parent_keys) {
	const size_t P = pkeys.size(), C = r.key.size();
	CHECK(lk.off.size() == P + 1, "off size %zu vs %zu", lk.off.size(), P + 1);
	if (lk.off.size() != P + 1) {
		return;
	}
	for (size_t p = 0; p < P; p++) {
		CHECK(lk.off[p] <= lk.off[p + 1], "off not monotone at %zu", p);
	}
	CHECK(lk.off[P] == lk.flat.size(), "off[P]=%u vs flat=%zu", lk.off[P], lk.flat.size());
	CHECK(lk.dated == timed, "dated=%d timed=%d", (int)lk.dated, (int)timed);
	std::vector<std::vector<int>> all;
	std::vector<int> parent_of = BruteParents(pkeys, r.key, &all);
	// membership: every flat entry is a child whose key resolves to that parent
	std::vector<int> seen(C, 0);
	std::vector<int> csr_parent(C, -1);
	for (size_t p = 0; p < P; p++) {
		for (uint32_t i = lk.off[p]; i < lk.off[p + 1]; i++) {
			const uint32_t c = lk.flat[i];
			CHECK(c < C, "flat entry %u out of range", c);
			if (c >= C) {
				continue;
			}
			seen[c]++;
			csr_parent[c] = (int)p;
			bool ok = false;
			for (int q : all[c]) {
				ok |= q == (int)p;
			}
			CHECK(ok, "child %u sits under parent %zu but keys differ (%g vs %g)", c, p, r.key[c], pkeys[p]);
		}
	}
	for (size_t c = 0; c < C; c++) {
		const bool should = parent_of[c] >= 0;
		const bool timed_null = timed && std::isnan(r.time[c]);
		if (should && !timed_null) {
			CHECK(seen[c] == 1, "child %zu (key %g) appears %d times, expected 1", c, r.key[c], seen[c]);
		} else if (!should) {
			CHECK(seen[c] == 0, "orphan/NULL child %zu appears %d times", c, seen[c]);
		} else {
			// A child whose timestamp is NULL cannot be placed in any window. It
			// may be absent or present, but if present it must not break the
			// sorted order of the finite entries (checked below).
			CHECK(seen[c] <= 1, "child %zu appears %d times", c, seen[c]);
		}
	}
	if (!dup_parent_keys) {
		// with unique parent keys the mapping is exact
		for (size_t c = 0; c < C; c++) {
			if (parent_of[c] >= 0 && seen[c] == 1) {
				CHECK(csr_parent[c] == parent_of[c], "child %zu under %d, expected %d", c, csr_parent[c], parent_of[c]);
			}
		}
	} else {
		// with duplicates use whatever the CSR chose as the reference
		parent_of = csr_parent;
	}
	// sorted by time (double), finite entries only; NULL-time entries, if kept,
	// must all sit after every finite one
	if (timed) {
		for (size_t p = 0; p < P; p++) {
			bool seen_nan = false;
			for (uint32_t i = lk.off[p]; i < lk.off[p + 1]; i++) {
				const double t = lk.Time(lk.flat[i]);
				CHECK(t == r.time[lk.flat[i]] || (std::isnan(t) && std::isnan(r.time[lk.flat[i]])),
				      "ctime differs from column for child %u", lk.flat[i]);
				if (std::isnan(t)) {
					seen_nan = true;
					continue;
				}
				CHECK(!seen_nan, "parent %zu: finite time after a NULL time at slot %u", p, i);
				if (i > lk.off[p]) {
					const double prev = lk.Time(lk.flat[i - 1]);
					if (!std::isnan(prev)) {
						CHECK(prev <= t, "parent %zu: bucket not sorted at slot %u (%.0f > %.0f)", p, i, prev, t);
					}
				}
			}
		}
	}

	// Windowed aggregates against brute force. Anchors are drawn from actual
	// child timestamps (ties), just beside them, and at random.
	std::vector<double> times;
	for (size_t c = 0; c < C; c++) {
		if (!std::isnan(r.time[c])) {
			times.push_back(r.time[c]);
		}
	}
	const Frame &child = db.At(lk.child_table);
	const int amt_col = child.Find("amount");
	for (int trial = 0; trial < 40; trial++) {
		const uint32_t p = P ? (uint32_t)(rng() % P) : 0;
		if (!P) {
			break;
		}
		auto pick = [&]() -> double {
			if (times.empty() || rng() % 3 == 0) {
				return 1.6e15 + double(rng() % 2000) * DAY * 0.05;
			}
			const double t = times[rng() % times.size()];
			switch (rng() % 4) {
			case 0:
				return t;
			case 1:
				return t + 1; // one micro after
			case 2:
				return t - 1;
			default:
				return t + double(rng() % 100000) - 50000.0;
			}
		};
		const double anchor = pick();
		// VisiblePrefix: (-inf, anchor]
		const size_t vis = Database::VisiblePrefix(lk, child, p, anchor);
		const Brute bv = BruteWindow(parent_of, r, (int)p, -INFINITY, anchor, timed);
		CHECK(vis == bv.n, "VisiblePrefix parent %u anchor %.0f: got %zu want %zu", p, anchor, vis, bv.n);
		// AggregateWindow over (lo, hi], including hi < lo and hi == lo
		double lo = pick(), hi = pick();
		if (rng() % 5 == 0) {
			hi = lo;
		}
		if (rng() % 7 == 0) {
			std::swap(lo, hi); // may make hi < lo
		}
		const Brute bw = BruteWindow(parent_of, r, (int)p, lo, hi, timed);
		bool any = false;
		const double cnt = AggregateWindow(db, lk, child, p, lo, hi, TargetKind::COUNT, -1, nullptr, any);
		CHECK(cnt == double(bw.n), "COUNT parent %u (%.0f,%.0f]: got %g want %zu", p, lo, hi, cnt, bw.n);
		CHECK(any == (bw.n > 0), "EXISTS mismatch");
		const double sum = AggregateWindow(db, lk, child, p, lo, hi, TargetKind::SUM, amt_col, nullptr, any);
		CHECK(std::fabs(sum - bw.sum) <= 1e-9 * (1 + std::fabs(bw.sum)), "SUM parent %u: got %g want %g", p, sum,
		      bw.sum);
		const double avg = AggregateWindow(db, lk, child, p, lo, hi, TargetKind::AVG, amt_col, nullptr, any);
		const double want_avg = bw.nval ? bw.sum / double(bw.nval) : 0.0;
		CHECK(std::fabs(avg - want_avg) <= 1e-9 * (1 + std::fabs(want_avg)), "AVG parent %u: got %g want %g", p, avg,
		      want_avg);
		const double mn = AggregateWindow(db, lk, child, p, lo, hi, TargetKind::MIN, amt_col, nullptr, any);
		const double mx = AggregateWindow(db, lk, child, p, lo, hi, TargetKind::MAX, amt_col, nullptr, any);
		CHECK(mn == (bw.have ? bw.mn : 0.0), "MIN parent %u: got %g want %g", p, mn, bw.have ? bw.mn : 0.0);
		CHECK(mx == (bw.have ? bw.mx : 0.0), "MAX parent %u: got %g want %g", p, mx, bw.have ? bw.mx : 0.0);
	}

	// BuildFeatures windows and the AggCache prefix sums against brute force.
	if (P > 0 && amt_col >= 0) {
		FeatureSpec spec;
		FeatureSpec::LinkAgg la;
		int li = -1;
		for (size_t l = 0; l < db.links.size(); l++) {
			if (&db.links[l] == &lk) {
				li = (int)l;
			}
		}
		la.link = li;
		la.child_table = lk.child_table;
		la.mean_cols = {amt_col};
		la.mean_names = {"amount"};
		la.mean_mu = {0.0};
		la.mean_sd = {1.0};
		spec.link_aggs.push_back(la);
		spec.ComputeWidth();
		const AggCache cache = BuildAggCache(db, spec);
		const Frame &parent = db.At(lk.parent_table);
		std::vector<float> out;
		for (int trial = 0; trial < 20; trial++) {
			const uint32_t p = (uint32_t)(rng() % P);
			double anchor;
			if (times.empty() || rng() % 3 == 0) {
				anchor = 1.6e15 + double(rng() % 2000) * DAY * 0.05;
			} else {
				anchor = times[rng() % times.size()] + (rng() % 2 ? 0.0 : double(rng() % 3) - 1.0);
			}
			BuildFeatures(db, parent, spec, cache, p, anchor, out);
			size_t k = 0;
			const Brute all_b = BruteWindow(parent_of, r, (int)p, -INFINITY, anchor, timed);
			const float never = out[k++];
			if (timed) {
				CHECK(never == (all_b.n == 0 ? 1.0f : 0.0f), "never-happened flag parent %u: %g with %zu visible", p,
				      never, all_b.n);
			}
			for (int w = 0; w < kNumWindows; w++) {
				const double lo = timed && w < kNumWindows - 1 ? anchor - kWindowDays[w] * DAY : -INFINITY;
				const Brute bw = BruteWindow(parent_of, r, (int)p, lo, anchor, timed);
				const double n_got = std::round(std::expm1((double)out[k]));
				CHECK(n_got == double(bw.n), "window %d count parent %u anchor %.0f: got %g want %zu", w, p, anchor,
				      n_got, bw.n);
				k++;
				const float rec = out[k++];
				if (timed) {
					// recency within window: log1p(days since last visible child)*0.25
					double last = -INFINITY;
					for (size_t c = 0; c < C; c++) {
						if (parent_of[c] == (int)p && !std::isnan(r.time[c]) && r.time[c] <= anchor && r.time[c] > lo &&
						    r.time[c] > last) {
							last = r.time[c];
						}
					}
					if (bw.n > 0) {
						const double since = (anchor - last) / DAY;
						const float want = float(std::log1p(since) * 0.25);
						CHECK(std::fabs(rec - want) <= 1e-5f + 1e-5f * std::fabs(want),
						      "window %d recency parent %u: got %g want %g", w, p, rec, want);
					} else {
						CHECK(rec == 0.0f, "window %d recency should be 0 when empty", w);
					}
				}
				k++; // spacing cv: not checked exactly
				const float mean = out[k++];
				const double want_mean = bw.nval ? bw.sum / double(bw.nval) : 0.0;
				// float prefix sums: allow a tolerance that scales with the running
				// sum of the whole bucket, and record the worst case
				double bucket_abs = 0;
				for (size_t c = 0; c < C; c++) {
					if (parent_of[c] == (int)p && !std::isnan(r.amt[c])) {
						bucket_abs += std::fabs(r.amt[c]);
					}
				}
				const double tol =
				    1e-5 * (1.0 + std::fabs(want_mean)) + 2e-7 * bucket_abs / double(bw.nval ? bw.nval : 1);
				CHECK(std::fabs((double)mean - want_mean) <= tol, "window %d mean parent %u: got %g want %g (n=%zu)", w,
				      p, mean, want_mean, bw.nval);
			}
			CHECK(k == (size_t)spec.width, "feature width %zu vs %d", k, spec.width);
		}

		// SageMeanAt with visible counts
		SageSpec ss;
		SageSpec::NodeType ent, ch;
		ent.table = db.Find(lk.parent_table);
		ent.dim = 0;
		ch.table = db.Find(lk.child_table);
		ch.cols = {amt_col};
		ch.mu = {0.0};
		ch.sd = {1.0};
		ch.dim = 1;
		ss.nodes = {ent, ch};
		SageSpec::EdgeType et;
		et.link = li;
		et.src_node = 1;
		et.dst_node = 0;
		ss.edges = {et};
		ss.entity_node = 0;
		const SageFeatures feat = BuildSageFeatures(db, ss);
		const SagePrefix pre = BuildSagePrefix(db, ss, feat);
		for (int trial = 0; trial < 20; trial++) {
			const uint32_t p = (uint32_t)(rng() % P);
			const double anchor = times.empty() ? 1.6e15 : times[rng() % times.size()] + double(rng() % 3) - 1.0;
			const size_t vis = Database::VisiblePrefix(lk, child, p, anchor);
			float o[2] = {0, 0};
			SageMeanAt(pre, 0, p, vis, o, 1);
			// brute: mean over visible children of amount (NULL -> 0), divided by visible count
			double s = 0;
			size_t n = 0;
			for (size_t c = 0; c < C; c++) {
				if (parent_of[c] != (int)p) {
					continue;
				}
				if (timed && (std::isnan(r.time[c]) || r.time[c] > anchor)) {
					continue;
				}
				s += std::isnan(r.amt[c]) ? 0.0 : r.amt[c];
				n++;
			}
			CHECK(n == vis, "sage visible %zu vs brute %zu", vis, n);
			const double want = n ? s / double(n) : 0.0;
			double bucket_abs = 0;
			for (size_t c = 0; c < C; c++) {
				if (parent_of[c] == (int)p && !std::isnan(r.amt[c])) {
					bucket_abs += std::fabs(r.amt[c]);
				}
			}
			const double tol = 1e-5 * (1.0 + std::fabs(want)) + 2e-7 * bucket_abs / double(n ? n : 1);
			CHECK(std::fabs((double)o[0] - want) <= tol, "SageMeanAt parent %u: got %g want %g (n=%zu)", p, o[0], want,
			      n);
			CHECK(o[1] == float(std::log1p(double(n))), "SageMeanAt degree slot");
		}
	}
}

// One random schema: a parent, a child with a foreign key, sometimes a second
// foreign key from the same child, sometimes a self-reference.
static void RandomCase(unsigned seed) {
	g_seed = seed;
	std::mt19937_64 rng(seed);
	const int mode = (int)(rng() % 8);
	const size_t P = mode == 0 ? 0 : (rng() % 4 == 0 ? 1 + rng() % 3 : 1 + rng() % 40);
	size_t C;
	switch (rng() % 6) {
	case 0:
		C = 0;
		break;
	case 1:
		C = 1 + rng() % 10;
		break;
	case 2:
		C = 64 + rng() % 200; // radix path
		break;
	case 3:
		C = 2100 + rng() % 400; // more than 2048 rows in one bucket is possible
		break;
	default:
		C = rng() % 500;
	}
	const bool timed = rng() % 5 != 0;
	const bool dup_parent = rng() % 6 == 0;
	const int key_mode = (int)(rng() % 6); // 0..3 dense ints, 4 sparse/huge ints, 5 non-integral doubles
	const ColType kt = key_mode == 5 ? ColType::DOUBLE : ColType::INT64;
	std::vector<double> pkeys(P);
	std::set<double> used;
	for (size_t p = 0; p < P; p++) {
		double k;
		for (int attempt = 0;; attempt++) {
			if (key_mode <= 3) {
				k = double((int64_t)(rng() % (P * 3 + 5)) - 3);
			} else if (key_mode == 4) {
				switch (rng() % 5) {
				case 0:
					k = double(INT64_MAX);
					break;
				case 1:
					k = double(INT64_MIN);
					break;
				case 2:
					k = double((int64_t)(rng() % 1000)) * 1e12;
					break;
				default:
					k = double((int64_t)rng() >> (rng() % 40));
				}
			} else {
				k = double(rng() % 50) * 0.5 + (rng() % 3 == 0 ? 1e300 : 0.0);
			}
			if (dup_parent || !used.count(k) || attempt > 20) {
				break;
			}
		}
		used.insert(k);
		pkeys[p] = k;
		if (rng() % 20 == 0) {
			pkeys[p] = NaN; // NULL parent key
		}
	}
	RawChild r;
	r.key.resize(C);
	r.time.resize(C);
	r.amt.resize(C);
	const double t0 = 1.6e15 + double(rng() % 100) * DAY;
	const int time_mode = (int)(rng() % 4); // 0 sorted, 1 unsorted wide, 2 unsorted tight (ties), 3 clustered
	for (size_t c = 0; c < C; c++) {
		if (P == 0 || rng() % 10 == 0) {
			r.key[c] = rng() % 7 == 0 ? NaN : (key_mode == 5 ? 12345.25 : 1e9 + double(rng() % 100)); // orphan/NULL
		} else {
			r.key[c] = pkeys[rng() % P];
			if (std::isnan(r.key[c])) {
				r.key[c] = 42;
			}
		}
		double t;
		switch (time_mode) {
		case 0:
			t = t0 + double(c) * DAY * 0.1;
			break;
		case 1:
			t = t0 + double(rng() % 100000) * DAY * 0.01;
			break;
		case 2:
			t = t0 + double(rng() % 400) * 1e6; // within 400 seconds: float ties
			break;
		default:
			t = t0 + double(rng() % 20) * DAY + double(rng() % 5);
		}
		if (rng() % 15 == 0) {
			t = NaN;
		}
		r.time[c] = t;
		r.amt[c] = rng() % 12 == 0 ? NaN : (double)(rng() % 1000) * 0.25 - 100.0;
	}

	Database db;
	db.tables.push_back(MakeParent("users", pkeys, kt));
	db.tables.push_back(MakeChild("events", r, kt, timed));
	db.fks.push_back({"events", "parent_id", "users", "id"});
	// second FK between the same pair: events.id -> users.id (mode 5)
	const bool second_fk = mode == 5 && kt == ColType::INT64;
	if (second_fk) {
		db.fks.push_back({"events", "id", "users", "id"});
	}
	// self reference: users.id -> users.id (mode 6)
	const bool self_ref = mode == 6;
	if (self_ref) {
		db.fks.push_back({"users", "x", "users", "id"});
	}
	db.BuildLinks({});
	const size_t want_links = 1 + (second_fk ? 1 : 0) + (self_ref ? 1 : 0);
	CHECK(db.links.size() == want_links, "links %zu vs %zu", db.links.size(), want_links);
	if (db.links.empty()) {
		return;
	}
	CheckLink(db, db.links[0], pkeys, r, timed, rng, dup_parent);
	if (second_fk && db.links.size() > 1) {
		RawChild r2 = r;
		for (size_t c = 0; c < C; c++) {
			r2.key[c] = double(c);
		}
		CheckLink(db, db.links[1], pkeys, r2, timed, rng, dup_parent);
	}
	if (self_ref && db.links.size() > 1) {
		const Link &sl = db.links.back();
		CHECK(sl.off.size() == P + 1, "self link off size");
		CHECK(sl.parent_table == "users" && sl.child_table == "users", "self link tables");
	}
}

// Hand-built cases that must hold regardless of the seed.
static void FixedCases() {
	g_seed = 0;
	// NULL child timestamp in the middle of an otherwise unsorted bucket: the
	// binary searches must still count exactly the finite children.
	{
		std::vector<double> pk = {1};
		RawChild r;
		r.key = {1, 1, 1, 1, 1};
		r.time = {100 * DAY, NaN, 50 * DAY, NaN, 75 * DAY};
		r.amt = {1, 2, 3, 4, 5};
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::INT64));
		db.tables.push_back(MakeChild("events", r, ColType::INT64, true));
		db.fks.push_back({"events", "parent_id", "users", "id"});
		db.BuildLinks({});
		const Link &lk = db.links[0];
		const Frame &ch = db.At("events");
		CHECK(Database::VisiblePrefix(lk, ch, 0, 60 * DAY) == 1, "NULL-time bucket: visible at 60d = %zu, want 1",
		      Database::VisiblePrefix(lk, ch, 0, 60 * DAY));
		CHECK(Database::VisiblePrefix(lk, ch, 0, 80 * DAY) == 2, "NULL-time bucket: visible at 80d = %zu, want 2",
		      Database::VisiblePrefix(lk, ch, 0, 80 * DAY));
		CHECK(Database::VisiblePrefix(lk, ch, 0, 1e18) == 3, "NULL-time bucket: visible at inf = %zu, want 3",
		      Database::VisiblePrefix(lk, ch, 0, 1e18));
		bool any = false;
		const double c = AggregateWindow(db, lk, ch, 0, 60 * DAY, 90 * DAY, TargetKind::COUNT, -1, nullptr, any);
		CHECK(c == 1, "NULL-time bucket: COUNT (60d,90d] = %g, want 1", c);
	}
	// Near-ties below float resolution at 1e15 micros, unsorted, wide bucket
	// (radix path): the bucket must be sorted on the double.
	{
		std::vector<double> pk = {7};
		RawChild r;
		const size_t n = 300;
		std::mt19937_64 rng(99);
		for (size_t i = 0; i < n; i++) {
			r.key.push_back(7);
			r.time.push_back(1.7e15 + double(rng() % 120) * 1e6); // within 120 s
			r.amt.push_back(1);
		}
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::INT64));
		db.tables.push_back(MakeChild("events", r, ColType::INT64, true));
		db.fks.push_back({"events", "parent_id", "users", "id"});
		db.BuildLinks({});
		const Link &lk = db.links[0];
		bool sorted = true;
		for (uint32_t i = 1; i < lk.flat.size(); i++) {
			sorted &= lk.Time(lk.flat[i - 1]) <= lk.Time(lk.flat[i]);
		}
		CHECK(sorted, "near-tie bucket of %zu rows is not sorted on the double timestamp", n);
		const Frame &ch = db.At("events");
		const double cutoff = 1.7e15 + 60e6;
		size_t want = 0;
		for (double t : r.time) {
			want += t <= cutoff;
		}
		const size_t got = Database::VisiblePrefix(lk, ch, 0, cutoff);
		CHECK(got == want, "near-tie bucket: visible %zu want %zu", got, want);
	}
	// Integer keys at both ends of int64 in the same parent column.
	{
		std::vector<double> pk = {double(INT64_MIN), double(INT64_MAX), 5};
		RawChild r;
		r.key = {double(INT64_MAX), 5, double(INT64_MIN), 6};
		r.time = {DAY, 2 * DAY, 3 * DAY, 4 * DAY};
		r.amt = {1, 1, 1, 1};
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::INT64));
		db.tables.push_back(MakeChild("events", r, ColType::INT64, true));
		db.fks.push_back({"events", "parent_id", "users", "id"});
		db.BuildLinks({});
		const Link &lk = db.links[0];
		CHECK(lk.off.size() == 4, "int64 extremes: off size %zu", lk.off.size());
		if (lk.off.size() == 4) {
			CHECK(lk.Count(0) == 1 && lk.Count(1) == 1 && lk.Count(2) == 1, "int64 extremes: counts %u %u %u",
			      lk.Count(0), lk.Count(1), lk.Count(2));
		}
	}
	// Double keys far outside int64.
	{
		std::vector<double> pk = {1e300, -1e300, 2.5};
		RawChild r;
		r.key = {1e300, 2.5, -1e300, 2.5, 3e300};
		r.time = {DAY, 2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY};
		r.amt = {1, 1, 1, 1, 1};
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::DOUBLE));
		db.tables.push_back(MakeChild("events", r, ColType::DOUBLE, true));
		db.fks.push_back({"events", "parent_id", "users", "id"});
		db.BuildLinks({});
		const Link &lk = db.links[0];
		CHECK(lk.off.size() == 4, "huge double keys: off size %zu", lk.off.size());
		if (lk.off.size() == 4) {
			CHECK(lk.Count(0) == 1 && lk.Count(1) == 1 && lk.Count(2) == 2, "huge double keys: counts %u %u %u",
			      lk.Count(0), lk.Count(1), lk.Count(2));
		}
	}
	// Two foreign keys between the same pair of tables: the feature spec fitted
	// at TRAIN carries one aggregate block per link, and Rebind at PREDICT must
	// put each block back on the link it was fitted on, not on the first match.
	{
		std::vector<double> pk = {1, 2, 3};
		RawChild r;
		r.key = {1, 1, 2, 3, 3, 3};
		r.time = {DAY, 2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY, 6 * DAY};
		r.amt = {1, 2, 3, 4, 5, 6};
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::INT64));
		Frame ch = MakeChild("msgs", r, ColType::INT64, true);
		ch.columns[1].name = "sender_id";
		Column recv = MC("receiver_id", ColType::INT64, 6);
		for (size_t i = 0; i < 6; i++) {
			recv.num[i] = 3 - (double)(i % 3); // a different fan-in from the sender key
		}
		ch.columns.push_back(recv);
		db.tables.push_back(ch);
		db.fks.push_back({"msgs", "sender_id", "users", "id"});
		db.fks.push_back({"msgs", "receiver_id", "users", "id"});
		db.BuildLinks({});
		CHECK(db.links.size() == 2, "two-FK: %zu links", db.links.size());
		const Statement st = Parse("TRAIN MODEL m PREDICT users.x FOR users");
		const Frame &users = db.At("users");
		FeatureSpec fitted = FitFeatureSpec(db, users, st, -1, -1, 4);
		CHECK(fitted.link_aggs.size() == 2, "two-FK: %zu link aggs", fitted.link_aggs.size());
		FeatureSpec rebound = fitted;
		rebound.Rebind(db, users);
		for (size_t i = 0; i < fitted.link_aggs.size() && i < rebound.link_aggs.size(); i++) {
			CHECK(fitted.link_aggs[i].link == rebound.link_aggs[i].link,
			      "two-FK: aggregate %zu fitted on link %d but rebound to link %d", i, fitted.link_aggs[i].link,
			      rebound.link_aggs[i].link);
		}
		// and the features must agree before and after the rebind
		const AggCache c1 = BuildAggCache(db, fitted), c2 = BuildAggCache(db, rebound);
		std::vector<float> f1, f2;
		for (uint32_t row = 0; row < 3; row++) {
			BuildFeatures(db, users, fitted, c1, row, 7 * DAY, f1);
			BuildFeatures(db, users, rebound, c2, row, 7 * DAY, f2);
			CHECK(f1 == f2, "two-FK: features for row %u differ after Rebind", row);
		}
	}
	// Parent table with zero rows, child with rows: every child is an orphan.
	{
		std::vector<double> pk;
		RawChild r;
		r.key = {1, 2};
		r.time = {DAY, DAY};
		r.amt = {1, 1};
		Database db;
		db.tables.push_back(MakeParent("users", pk, ColType::INT64));
		db.tables.push_back(MakeChild("events", r, ColType::INT64, true));
		db.fks.push_back({"events", "parent_id", "users", "id"});
		db.BuildLinks({});
		CHECK(db.links.size() == 1 && db.links[0].off.size() == 1 && db.links[0].flat.empty(),
		      "empty parent: link shape");
	}
}

int main(int argc, char **argv) {
	const long seeds = argc > 1 ? std::atol(argv[1]) : 3000;
	g_maxfail = argc > 2 ? std::atoi(argv[2]) : 40;
	FixedCases();
	printf("fixed cases: %ld checks, %d failures\n", checks, fails);
	for (long s = 1; s <= seeds; s++) {
		RandomCase((unsigned)s);
	}
	printf("random cases: %ld seeds, %ld checks, %d failures\n", seeds, checks, fails);
	printf("%s\n", fails ? "LINK/WINDOW FAILURES" : "all link and window properties hold");
	return fails ? 1 : 0;
}
