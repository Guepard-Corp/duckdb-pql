// Text columns: dictionary code 0 is the reserved unknown, slots are fitted on
// the training fold only, an unseen value at prediction time sets no slot (or
// the spare one when training saw NULLs), the empty string is a value and NULL
// is not, comparison is case-sensitive, and a vocabulary past 65535 codes is
// still addressed exactly.
//
//   clang++ -std=c++17 -O2 -I extras/pql -o /tmp/c extras/pql/test_categorical.cpp && /tmp/c
#include "src/pql.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace pql;

static int fails = 0, oks = 0;
#define CHECK(cond, ...)                                                                           \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			printf("  FAIL: ");                                                                    \
			printf(__VA_ARGS__);                                                                   \
			printf("\n");                                                                          \
			fails++;                                                                               \
		} else {                                                                                   \
			oks++;                                                                                 \
		}                                                                                          \
	} while (0)

static const double DAY = 86400.0 * 1e6;

static Column Num(const char *n, ColType t, const std::vector<double> &v) {
	Column c;
	c.name = n;
	c.type = t;
	c.num = v;
	c.code.assign(v.size(), 0u);
	c.valid.assign(v.size(), 1);
	return c;
}
// A text column; nullptr means NULL.
static Column Txt(const char *n, const std::vector<const char *> &v) {
	Column c;
	c.name = n;
	c.type = ColType::CATEGORY;
	c.num.assign(v.size(), 0.0);
	c.code.assign(v.size(), 0u);
	c.valid.assign(v.size(), 1);
	for (size_t i = 0; i < v.size(); i++) {
		if (v[i]) {
			c.code[i] = c.dict.Intern(v[i]);
		} else {
			c.valid[i] = 0;
		}
	}
	return c;
}

int main() {
	printf("== dictionary\n");
	{
		Dictionary d;
		CHECK(d.size() == 1 && d.Lookup("anything") == 0, "fresh dictionary: code 0 is the only entry");
		const uint32_t a = d.Intern("a"), e = d.Intern("");
		CHECK(a == 1 && e == 2, "codes are assigned in order, the empty string is a real value (%u, %u)", a, e);
		CHECK(d.Lookup("") == e, "empty string looks up to its own code, not to 0");
		CHECK(d.Lookup("A") == 0, "lookup is case-sensitive: 'A' is not 'a'");
		CHECK(d.Intern("a") == a, "interning twice returns the same code");
	}

	printf("== filter on a category\n");
	{
		Frame f;
		f.name = "t";
		f.columns = {Txt("s", {"EU", "eu", "", nullptr})};
		f.nrows = 4;
		auto eq = [&](const char *lit, size_t row) {
			Statement st = Parse((std::string("PREDICT t.x FOR t WHERE s = '") + lit + "' USING MODEL m").c_str());
			return EvalFilter(st.filter.get(), f, row);
		};
		CHECK(eq("EU", 0) && !eq("EU", 1), "'EU' matches only the exact spelling");
		CHECK(eq("", 2) && !eq("", 3), "'' matches the empty string, not NULL");
		CHECK(!eq("typo", 0) && !eq("typo", 1) && !eq("typo", 2) && !eq("typo", 3),
		      "an unseen literal matches nothing");
		CHECK(f.columns[0].dict.size() == 4, "filtering did not grow the vocabulary (%zu)", f.columns[0].dict.size());
		Statement ne = Parse("PREDICT t.x FOR t WHERE s != 'typo' USING MODEL m");
		CHECK(EvalFilter(ne.filter.get(), f, 0) && !EvalFilter(ne.filter.get(), f, 3),
		      "!= unseen is true for values and false for NULL");
		Statement isn = Parse("PREDICT t.x FOR t WHERE s IS NULL USING MODEL m");
		CHECK(!EvalFilter(isn.filter.get(), f, 2) && EvalFilter(isn.filter.get(), f, 3),
		      "IS NULL: the empty string is not NULL");
	}

	printf("== FitCatCol on the training fold only\n");
	{
		// rows 0..5 are the training era (anchor <= cutoff), 6..9 are later and
		// carry a value never seen in training
		Frame f;
		f.name = "u";
		f.columns = {Num("ts", ColType::TIMESTAMP, {1, 2, 3, 4, 5, 6, 100, 101, 102, 103}),
		             Txt("plan", {"basic", "basic", "pro", "pro", "pro", nullptr, "late", "late", "basic", "late"})};
		f.nrows = 10;
		FeatureSpec::CatCol cc;
		const bool ok = FitCatCol(f, 1, &f.columns[0], 50.0, 16, cc);
		CHECK(ok, "a two-valued column is a feature");
		CHECK(cc.labels.size() == 2 && cc.labels[0] == "pro" && cc.labels[1] == "basic",
		      "labels are the training-fold values, most frequent first (%zu)", cc.labels.size());
		CHECK(cc.has_other, "training saw a NULL, so a spare slot exists");
		CHECK(cc.Width() == 3, "width = 2 labels + other (%d)", cc.Width());
		CHECK(std::fabs(cc.freqs[0] - 0.6) < 1e-12 && std::fabs(cc.freqs[1] - 0.4) < 1e-12,
		      "frequencies are shares of the training fold (%g, %g)", cc.freqs[0], cc.freqs[1]);
		// encode rows through BuildFeatures
		FeatureSpec spec;
		spec.cat_cols.push_back(cc);
		spec.ComputeWidth();
		Database db;
		db.tables.push_back(f);
		AggCache cache;
		std::vector<float> out;
		BuildFeatures(db, f, spec, cache, 2, 3.0, out);
		CHECK(out.size() == 3 && out[0] == 1 && out[1] == 0 && out[2] == 0, "'pro' -> slot 0");
		BuildFeatures(db, f, spec, cache, 0, 1.0, out);
		CHECK(out[0] == 0 && out[1] == 1 && out[2] == 0, "'basic' -> slot 1");
		BuildFeatures(db, f, spec, cache, 5, 6.0, out);
		CHECK(out[0] == 0 && out[1] == 0 && out[2] == 1, "NULL -> other slot");
		BuildFeatures(db, f, spec, cache, 6, 100.0, out);
		CHECK(out[0] == 0 && out[1] == 0 && out[2] == 1, "'late' (unseen in training) -> other slot");
		// without any NULL in training there is no spare slot and an unseen value sets nothing
		Frame g = f;
		g.columns[1] = Txt("plan", {"basic", "basic", "pro", "pro", "pro", "pro", "late", "late", "basic", "late"});
		FeatureSpec::CatCol cc2;
		CHECK(FitCatCol(g, 1, &g.columns[0], 50.0, 16, cc2) && !cc2.has_other && cc2.Width() == 2,
		      "no NULL in training: no spare slot (width %d)", cc2.Width());
		FeatureSpec spec2;
		spec2.cat_cols.push_back(cc2);
		spec2.ComputeWidth();
		Database db2;
		db2.tables.push_back(g);
		BuildFeatures(db2, g, spec2, cache, 6, 100.0, out);
		CHECK(out.size() == 2 && out[0] == 0 && out[1] == 0, "unseen value with no spare slot sets nothing");
	}

	printf("== Rebind against a different dictionary\n");
	{
		// The same column loaded in a different row order gets different codes;
		// the slots must follow the text.
		Frame f;
		f.name = "u";
		f.columns = {Num("ts", ColType::TIMESTAMP, {1, 2, 3, 4}), Txt("plan", {"a", "a", "b", "b"})};
		f.nrows = 4;
		FeatureSpec spec;
		FeatureSpec::CatCol cc;
		CHECK(FitCatCol(f, 1, nullptr, INFINITY, 16, cc), "fit");
		spec.cat_cols.push_back(cc);
		spec.ComputeWidth();
		Frame g;
		g.name = "u";
		g.columns = {Num("ts", ColType::TIMESTAMP, {1, 2, 3, 4}), Txt("plan", {"b", "b", "a", "zzz"})};
		g.nrows = 4;
		Database db;
		db.tables.push_back(g);
		spec.Rebind(db, g);
		AggCache cache;
		std::vector<float> out;
		BuildFeatures(db, g, spec, cache, 0, 1.0, out);
		const size_t slot_b = cc.labels[0] == "b" ? 0 : 1;
		CHECK(out[slot_b] == 1.0f, "'b' lands on the slot labelled 'b' after a rebind");
		BuildFeatures(db, g, spec, cache, 3, 4.0, out);
		CHECK(out[0] == 0 && out[1] == 0, "'zzz' unseen at training sets nothing");
	}

	printf("== identifier-like and single-valued columns are not features\n");
	{
		Frame f;
		f.name = "u";
		std::vector<const char *> ids, same;
		std::vector<std::string> store;
		for (int i = 0; i < 200; i++) {
			store.push_back("id" + std::to_string(i));
		}
		for (int i = 0; i < 200; i++) {
			ids.push_back(store[(size_t)i].c_str());
			same.push_back("only");
		}
		f.columns = {Txt("email", ids), Txt("k", same)};
		f.nrows = 200;
		FeatureSpec::CatCol cc;
		CHECK(!FitCatCol(f, 0, nullptr, INFINITY, 16, cc), "one value per row is an identifier");
		CHECK(!FitCatCol(f, 1, nullptr, INFINITY, 16, cc), "one value everywhere is a constant");
	}

	printf("== high cardinality (more than 65535 codes)\n");
	{
		// 67000 distinct values over 350000 rows: value k appears 5 times, and
		// the first 1000 values appear many more times than that, so the column
		// is a wide vocabulary rather than an identifier (which would be refused).
		const size_t nv = 67000, n = nv * 5 + 4000;
		std::vector<std::string> store(n);
		std::vector<const char *> v(n);
		for (size_t i = 0; i < n; i++) {
			const size_t k = i < 4000 ? i % 1000 : (i - 4000) / 5;
			store[i] = "v" + std::to_string(k);
			v[i] = store[i].c_str();
		}
		Frame f;
		f.name = "u";
		f.columns = {Txt("sku", v)};
		f.nrows = n;
		const Column &c = f.columns[0];
		CHECK(c.dict.size() == nv + 1, "dictionary holds every value (%zu)", c.dict.size());
		CHECK(c.code[n - 1] == nv, "the last code is past 65535 (%u)", c.code[n - 1]);
		FeatureSpec::CatCol cc;
		const bool ok = FitCatCol(f, 0, nullptr, INFINITY, 16, cc);
		CHECK(ok && !cc.one_hot, "frequency encoding for a wide vocabulary (ok=%d one_hot=%d)", (int)ok, (int)cc.one_hot);
		if (ok) {
			CHECK(cc.slot_by_code.size() == nv + 1 && cc.freq_by_code.size() == nv + 1,
			      "direct lookups cover every code (%zu)", cc.slot_by_code.size());
			CHECK(cc.freq_by_code[c.code[0]] > cc.freq_by_code[c.code[n - 1]],
			      "a frequent value has a higher frequency than a singleton");
			FeatureSpec spec;
			spec.cat_cols.push_back(cc);
			spec.ComputeWidth();
			Database db;
			db.tables.push_back(f);
			AggCache cache;
			std::vector<float> a, b;
			BuildFeatures(db, f, spec, cache, 0, 1.0, a);
			BuildFeatures(db, f, spec, cache, (uint32_t)n - 1, 1.0, b);
			CHECK(a.size() == 1 && std::isfinite(a[0]) && std::isfinite(b[0]) && a[0] > b[0],
			      "frequency feature is finite and ordered (%g vs %g)", a[0], b[0]);
		}
	}

	printf("== end to end: unseen category at prediction time\n");
	{
		// Train on a two-table database, then predict on one where the entity's
		// text column carries a value never seen (and a NULL).
		auto make = [](bool predict_time) {
			Database db;
			Frame u;
			u.name = "users";
			const size_t N = 300;
			std::vector<double> id(N), ts(N);
			std::vector<const char *> plan(N);
			for (size_t i = 0; i < N; i++) {
				id[i] = double(i);
				ts[i] = (1000.0 + double(i)) * DAY;
				plan[i] = predict_time && i % 7 == 0 ? "never_seen" : (i % 11 == 0 ? nullptr : (i % 3 ? "pro" : "basic"));
			}
			u.columns = {Num("id", ColType::INT64, id), Num("ts", ColType::TIMESTAMP, ts), Txt("plan", plan)};
			u.nrows = N;
			Frame e;
			e.name = "events";
			std::vector<double> eid, uid, ets;
			for (size_t i = 0; i < N; i++) {
				const int k = (i % 3) ? 3 : 1;
				for (int j = 0; j < k; j++) {
					eid.push_back(double(eid.size()));
					uid.push_back(double(i));
					ets.push_back(ts[i] + double(j + 1) * DAY);
				}
			}
			e.columns = {Num("id", ColType::INT64, eid), Num("user_id", ColType::INT64, uid),
			             Num("ts", ColType::TIMESTAMP, ets)};
			e.nrows = eid.size();
			db.tables = {u, e};
			db.fks.push_back({"events", "user_id", "users", "id"});
			db.BuildLinks({});
			return db;
		};
		try {
			Database train = make(false);
			Statement st = Parse("TRAIN MODEL m PREDICT COUNT(events) FOR users AT ts HORIZON 30 DAYS "
			                     "OPTIONS (epochs = 5, hidden = 8)");
			Model m;
			TrainReport rep = TrainModel(train, st, m);
			CHECK(std::isfinite(rep.test_metric), "trained (metric %g)", rep.test_metric);
			Database pred = make(true);
			Statement ps = Parse("PREDICT COUNT(events) FOR users USING MODEL m");
			auto out = RunPredict(pred, m, ps);
			CHECK(out.size() == 300, "one prediction per row (%zu)", out.size());
			bool finite = true;
			for (const auto &p : out) {
				finite &= std::isfinite(p.value);
			}
			CHECK(finite, "every prediction is finite with an unseen category present");
		} catch (const std::exception &ex) {
			CHECK(false, "end to end threw: %s", ex.what());
		}
	}

	printf("\n%d passed, %d failed\n", oks, fails);
	return fails ? 1 : 0;
}
