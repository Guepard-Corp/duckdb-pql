// OPTIMIZE: the grammar, the expression evaluator, the solver, and the whole
// path from a trained model to a plan.
//
// The solver is the part that can be wrong quietly: a plan that breaks a limit,
// a bound that is not a bound, "infeasible" for a problem that has an answer.
// So it is checked against brute force on thousands of small random problems,
// against exact dynamic programmes on larger ones, and the small simplex under
// it against optimality certificates. The end-to-end sections build data whose
// best decision is known by construction and check that OPTIMIZE finds it.
#include "pql/pql.hpp"
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
using namespace pql;

// Timing checks are for an optimised build; a sanitised one runs many times slower.
#if defined(__SANITIZE_ADDRESS__)
static const double kSlow = 25;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
static const double kSlow = 25;
#else
static const double kSlow = 1;
#endif
#else
static const double kSlow = 1;
#endif

static int fails = 0, oks = 0;
#define CHECK(cond, ...)                                                                                               \
	do {                                                                                                               \
		if (cond) {                                                                                                    \
			oks++;                                                                                                     \
		} else {                                                                                                       \
			fails++;                                                                                                   \
			std::printf("   FAIL %s:%d: ", __FILE__, __LINE__);                                                        \
			std::printf(__VA_ARGS__);                                                                                  \
			std::printf("\n");                                                                                         \
		}                                                                                                              \
	} while (0)

// ---------------------------------------------------------------------------
// grammar
// ---------------------------------------------------------------------------

// Parses, echoes as `want` when given, and the echo parses back to itself.
static void Ok(const char *sql, const char *want = nullptr) {
	try {
		auto st = Parse(sql);
		const std::string once = st.ToString();
		const std::string again = Parse(once).ToString();
		CHECK(once == again, "echo does not read back the same\n     in:    %s\n     once:  %s\n     again: %s", sql,
		      once.c_str(), again.c_str());
		if (want) {
			CHECK(once == want, "echo\n     in:   %s\n     got:  %s\n     want: %s", sql, once.c_str(), want);
		}
	} catch (const std::exception &e) {
		CHECK(false, "parse: %s\n     error: %s", sql, e.what());
	}
}

static void Err(const std::string &sql, const char *want) {
	try {
		Parse(sql);
		CHECK(false, "expected an error: %s", sql.c_str());
	} catch (const ParseError &e) {
		const std::string m = e.what();
		CHECK(m.find(want) != std::string::npos, "wrong error for: %s\n     got:  %s\n     want: %s",
		      sql.substr(0, 200).c_str(), m.c_str(), want);
		CHECK(e.position <= sql.size(), "error position %u outside the input", e.position);
	} catch (const std::exception &e) {
		CHECK(false, "not a ParseError for %s: %s", sql.c_str(), e.what());
	}
}

static void Grammar() {
	std::printf("== grammar: accepts, and echoes as written\n");
	Ok("OPTIMIZE trucks MAXIMIZE SUM(prediction) USING MODEL m",
	   "OPTIMIZE trucks MAXIMIZE SUM(prediction) USING MODEL m");
	Ok("optimize trucks maximize sum(prediction * 4200 - service_cost) subject to count(*) <= 5 using model breakdown",
	   "OPTIMIZE trucks MAXIMIZE SUM(prediction * 4200 - service_cost) SUBJECT TO COUNT(*) <= 5 USING MODEL breakdown");
	Ok("OPTIMIZE trucks AS t WHERE t.status = 'active' MINIMIZE SUM(t.cost) SUBJECT TO COUNT(*) >= 3 USING MODEL m",
	   "OPTIMIZE trucks AS t WHERE t.status = 'active' MINIMIZE SUM(t.cost) SUBJECT TO COUNT(*) >= 3 USING MODEL m");
	Ok("OPTIMIZE trucks t MAXIMIZE SUM(prediction) USING MODEL m",
	   "OPTIMIZE trucks AS t MAXIMIZE SUM(prediction) USING MODEL m");
	// clauses in any order, the way TRAIN and PREDICT take them
	Ok("OPTIMIZE c USING MODEL m SUBJECT TO SUM(x) <= 10 MAXIMIZE SUM(prediction) WHERE a = 1 CHOOSE o IN ('p', 'q')",
	   "OPTIMIZE c WHERE a = 1 CHOOSE o IN ('p', 'q') MAXIMIZE SUM(prediction) SUBJECT TO SUM(x) <= 10 USING MODEL m");
	Ok("OPTIMIZE c CHOOSE c.offer IN ('none', '10%', '20%') MAXIMIZE SUM(prediction) USING MODEL m",
	   "OPTIMIZE c CHOOSE c.offer IN ('none', '10%', '20%') MAXIMIZE SUM(prediction) USING MODEL m");
	Ok("OPTIMIZE p CHOOSE price IN (9.99, 12, -1, +3) MAXIMIZE SUM(prediction * price) USING MODEL d",
	   "OPTIMIZE p CHOOSE price IN (9.99, 12, -1, 3) MAXIMIZE SUM(prediction * price) USING MODEL d");
	Ok("OPTIMIZE c AT seen MAXIMIZE SUM(prediction) USING MODEL m",
	   "OPTIMIZE c AT seen MAXIMIZE SUM(prediction) USING MODEL m");
	Ok("OPTIMIZE c AT '2024-01-01' MAXIMIZE SUM(prediction) USING MODEL m",
	   "OPTIMIZE c AT '2024-01-01' MAXIMIZE SUM(prediction) USING MODEL m");
	Ok("EXPLAIN OPTIMIZE c MAXIMIZE SUM(prediction) USING MODEL m",
	   "EXPLAIN OPTIMIZE c MAXIMIZE SUM(prediction) USING MODEL m");
	Ok("OPTIMIZE c MAXIMIZE SUM(prediction) SUBJECT TO SUM(cost) <= 500 AND COUNT(*) FILTER (WHERE region = 'EU') <= 3 "
	   "AND SUM(cost) FILTER (WHERE tier IN (1, 2) OR vip IS NOT NULL) >= -2.5 USING MODEL m",
	   "OPTIMIZE c MAXIMIZE SUM(prediction) SUBJECT TO SUM(cost) <= 500 AND COUNT(*) FILTER (WHERE region = 'EU') <= 3 "
	   "AND SUM(cost) FILTER (WHERE (tier IN (1, 2) OR vip IS NOT NULL)) >= -2.5 USING MODEL m");
	Ok("OPTIMIZE c MAXIMIZE COUNT(*) FILTER (WHERE a = 1) USING MODEL m",
	   "OPTIMIZE c MAXIMIZE COUNT(*) FILTER (WHERE a = 1) USING MODEL m");

	std::printf("== grammar: arithmetic keeps its meaning through an echo\n");
	auto expr = [](const char *e, const char *want) {
		const std::string sql = std::string("OPTIMIZE t MAXIMIZE SUM(") + e + ") USING MODEL m";
		const std::string w = std::string("OPTIMIZE t MAXIMIZE SUM(") + want + ") USING MODEL m";
		Ok(sql.c_str(), w.c_str());
	};
	expr("a + b * c", "a + b * c");
	expr("(a + b) * c", "(a + b) * c");
	expr("a - (b - c)", "a - (b - c)");
	expr("a - b - c", "a - b - c");
	expr("a / (b * c)", "a / (b * c)");
	expr("a / b / c", "a / b / c");
	expr("-(a + b)", "-(a + b)");
	expr("- -a", "-(-a)");
	expr("2 * -a", "2 * -a");
	expr("+a", "a");
	expr("((a))", "a");
	expr("a - -b", "a - -b");
	expr("CASE WHEN x = 'p' THEN 1 WHEN x = 'q' THEN 2 ELSE 0 END",
	     "CASE WHEN x = 'p' THEN 1 WHEN x = 'q' THEN 2 ELSE 0 END");
	expr("CASE x WHEN 'p' THEN 1 WHEN 'q' THEN 2 END", "CASE x WHEN 'p' THEN 1 WHEN 'q' THEN 2 END");
	expr("CASE WHEN a > 1 AND b < 2 THEN CASE c WHEN 1 THEN 2 END END * 3",
	     "CASE WHEN (a > 1 AND b < 2) THEN CASE c WHEN 1 THEN 2 END END * 3");
	expr("prediction * 4200 - CASE offer WHEN '10%' THEN 5 ELSE 0 END",
	     "prediction * 4200 - CASE offer WHEN '10%' THEN 5 ELSE 0 END");
	// The bare word is the model's answer; quoted or qualified, it is a column.
	expr("\"prediction\" + t.prediction", "\"prediction\" + t.\"prediction\"");
	expr("\"end\" + \"case\" * \"filter\"", "\"end\" + \"case\" * \"filter\"");
	expr(".5 * 1.25", ".5 * 1.25");
	// Inside the totals a filter may test the model's answer, bare; quoted it is a column.
	Ok("OPTIMIZE t MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) FILTER (WHERE prediction > 0.5) <= 3 USING MODEL m",
	   "OPTIMIZE t MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) FILTER (WHERE prediction > 0.5) <= 3 USING MODEL m");
	expr("CASE WHEN prediction >= 0.5 AND x = 1 THEN prediction ELSE 0 END",
	     "CASE WHEN (prediction >= 0.5 AND x = 1) THEN prediction ELSE 0 END");
	expr("CASE WHEN \"prediction\" > 1 THEN 1 END", "CASE WHEN \"prediction\" > 1 THEN 1 END");
	Ok("OPTIMIZE t WHERE \"prediction\" > 0.5 MAXIMIZE SUM(x) USING MODEL m",
	   "OPTIMIZE t WHERE \"prediction\" > 0.5 MAXIMIZE SUM(x) USING MODEL m");
	// Outside OPTIMIZE nothing changes: a filter's `prediction` is a column.
	Ok("PREDICT t.x FOR t WHERE prediction > 1 USING MODEL m",
	   "PREDICT t.x FOR t WHERE \"prediction\" > 1 USING MODEL m");

	std::printf("== grammar: rejects\n");
	Err("OPTIMIZE t USING MODEL m", "needs a goal");
	Err("OPTIMIZE t MAXIMIZE SUM(x)", "needs USING MODEL");
	Err("OPTIMIZE t MAXIMIZE SUM(x) MINIMIZE SUM(y) USING MODEL m", "duplicate MAXIMIZE/MINIMIZE");
	Err("OPTIMIZE t MAXIMIZE AVG(x) USING MODEL m", "expected SUM(<expression>) or COUNT(*)");
	Err("OPTIMIZE t MAXIMIZE x USING MODEL m", "expected SUM(<expression>) or COUNT(*)");
	Err("OPTIMIZE t MAXIMIZE COUNT(x) USING MODEL m", "write COUNT(*)");
	Err("OPTIMIZE t MAXIMIZE SUM(SUM(x)) USING MODEL m", "aggregates do not nest");
	Err("OPTIMIZE t MAXIMIZE SUM(abs(x)) USING MODEL m", "not something PQL can compute");
	Err("OPTIMIZE t MAXIMIZE SUM('a') USING MODEL m", "is text");
	Err("OPTIMIZE t MAXIMIZE SUM(t.*) USING MODEL m", "is not a value");
	Err("OPTIMIZE t MAXIMIZE SUM() USING MODEL m", "expected a number, a column");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) < 5 USING MODEL m", "written with <= or >=");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) = 5 USING MODEL m", "written with <= or >=");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) <= 'a' USING MODEL m", "limit is a number");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO SUM(x) USING MODEL m", "expected <= or >=");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) <= 1 AND USING MODEL m", "expected SUM(<expression>)");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT COUNT(*) <= 1 USING MODEL m", "expected TO");
	Err("OPTIMIZE t CHOOSE x MAXIMIZE SUM(x) USING MODEL m", "CHOOSE names a column and the values");
	Err("OPTIMIZE t CHOOSE x IN ('a', 'a') MAXIMIZE SUM(x) USING MODEL m", "lists 'a' twice");
	Err("OPTIMIZE t CHOOSE x IN (1, 1.0) MAXIMIZE SUM(x) USING MODEL m", "twice");
	Err("OPTIMIZE t CHOOSE x IN ('a', 1) MAXIMIZE SUM(x) USING MODEL m", "all be text or all be numbers");
	Err("OPTIMIZE t CHOOSE x IN (NULL) MAXIMIZE SUM(x) USING MODEL m", "cannot choose NULL");
	Err("OPTIMIZE t CHOOSE x IN () MAXIMIZE SUM(x) USING MODEL m", "expected a literal");
	Err("OPTIMIZE t MAXIMIZE SUM(CASE WHEN a = 1 THEN 2) USING MODEL m", "expected END");
	Err("OPTIMIZE t MAXIMIZE SUM(CASE END) USING MODEL m", "at least one WHEN");
	Err("OPTIMIZE t MAXIMIZE SUM(CASE x WHEN y THEN 1 END) USING MODEL m", "single quotes");
	Err("OPTIMIZE t WHERE a = 1 WHERE a = 2 MAXIMIZE SUM(x) USING MODEL m", "duplicate WHERE");
	Err("OPTIMIZE t CHOOSE x IN ('a') CHOOSE y IN ('b') MAXIMIZE SUM(x) USING MODEL m", "duplicate CHOOSE");
	Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) <= 1 SUBJECT TO COUNT(*) <= 2 USING MODEL m",
	    "duplicate SUBJECT TO");
	Err("OPTIMIZE t MAXIMIZE SUM(x) USING MODEL m USING MODEL n", "duplicate USING MODEL");
	Err("OPTIMIZE t MAXIMIZE SUM(x) USING MODEL m HORIZON 3 DAYS", "in OPTIMIZE statement");
	Err("OPTIMIZE t MAXIMIZE SUM(x) USING MODEL m OPTIONS (epochs = 1)", "in OPTIMIZE statement");
	Err("OPTIMIZE t MAXIMIZE SUM(x) USING MODEL m; x", "after OPTIMIZE");
	Err("OPTIMIZE t MAXIMIZE SUM(--a\n) USING MODEL m", "expected a number, a column");
	Err("EXPLAIN OPTIMIZE", "expected table to optimize over");
	Err("OPTIMIZE MAXIMIZE SUM(x) USING MODEL m", "unexpected token 'SUM'");
	Err("OPTIMIZE t WHERE prediction > 0.5 MAXIMIZE SUM(x) USING MODEL m",
	    "WHERE picks the rows before anything is predicted");
	Err("OPTIMIZE t WHERE a = 1 AND NOT (prediction < 2) MAXIMIZE SUM(x) USING MODEL m", "cannot test prediction");
	Err("OPTIMIZE t MAXIMIZE SUM(CASE prediction WHEN 1 THEN 2 END) USING MODEL m", "tests a prediction for equality");
	{
		// A number of 400 digits is infinity once read; refused rather than let
		// an infinite limit compare as met.
		const std::string huge(400, '9');
		Err("OPTIMIZE t MAXIMIZE SUM(" + huge + ") USING MODEL m", "too large to represent");
		Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) >= " + huge + " USING MODEL m", "limit is too large");
		Err("OPTIMIZE t MAXIMIZE SUM(x) SUBJECT TO COUNT(*) <= -" + huge + " USING MODEL m", "limit is too large");
		Err("OPTIMIZE t CHOOSE x IN (1, " + huge + ") MAXIMIZE SUM(x) USING MODEL m", "too large to represent");
	}
	{
		std::string many = "OPTIMIZE t MAXIMIZE SUM(1";
		for (int i = 0; i < 1001; i++) {
			many += " + 1";
		}
		Err(many + ") USING MODEL m", "too long");
		std::string deep =
		    "OPTIMIZE t MAXIMIZE SUM(" + std::string(300, '(') + "1" + std::string(300, ')') + ") USING MODEL m";
		Err(deep, "nested too deeply");
		std::string negs = "OPTIMIZE t MAXIMIZE SUM(";
		for (int i = 0; i < 300; i++) {
			negs += "- ";
		}
		Err(negs + "1) USING MODEL m", "nested too deeply");
		std::string cases = "OPTIMIZE t MAXIMIZE SUM(";
		for (int i = 0; i < 300; i++) {
			cases += "CASE WHEN a = 1 THEN ";
		}
		Err(cases + "1) USING MODEL m", "nested too deeply");
	}

	std::printf("== a copied statement keeps every OPTIMIZE clause, deeply\n");
	{
		const char *sql = "EXPLAIN OPTIMIZE c AS cc WHERE cc.a = 1 CHOOSE o IN ('p', 'q') "
		                  "MINIMIZE SUM(prediction - CASE o WHEN 'p' THEN 1 ELSE 2 END) "
		                  "SUBJECT TO COUNT(*) FILTER (WHERE o = 'q') <= 3 AND SUM(x) >= 1 USING MODEL m";
		Statement a = Parse(sql);
		Statement b = a;
		Statement c;
		c = a;
		CHECK(a.ToString() == b.ToString() && a.ToString() == c.ToString(), "a copy differs:\n     %s\n     %s",
		      a.ToString().c_str(), b.ToString().c_str());
		// Deep, not shared: changing the copy leaves the original alone.
		b.optimize.objective.expr->args[0]->kind = Expr::Kind::NUMBER;
		b.optimize.objective.expr->args[0]->text = "7";
		b.optimize.constraints[0].total.filter->values[0].text = "zzz";
		CHECK(a.ToString() == c.ToString(), "changing a copy changed the original:\n     %s", a.ToString().c_str());
		CHECK(b.ToString() != a.ToString(), "the copy did not change");
		CHECK(a.optimize.explain && !a.optimize.maximize && a.optimize.Choosing() && a.optimize.choices.size() == 2 &&
		          a.optimize.constraints.size() == 2 && !a.optimize.constraints[1].at_most,
		      "fields lost in the parse");
	}
}

// ---------------------------------------------------------------------------
// expressions
// ---------------------------------------------------------------------------

static Column MakeCol(const char *name, ColType t, size_t n) {
	Column c;
	c.name = name;
	c.type = t;
	c.num.assign(n, 0.0);
	c.code.assign(n, 0u);
	c.valid.assign(n, 1);
	return c;
}

static Column TextCol(const char *name, const std::vector<std::string> &vals) {
	Column c = MakeCol(name, ColType::CATEGORY, vals.size());
	for (size_t i = 0; i < vals.size(); i++) {
		c.code[i] = c.dict.Intern(vals[i]);
	}
	return c;
}

static void Expressions() {
	std::printf("== expressions: values, CASE, FILTER, and what has no value\n");
	Frame f;
	f.name = "t";
	f.nrows = 1;
	Column x = MakeCol("x", ColType::DOUBLE, 1), y = MakeCol("y", ColType::INT64, 1),
	       z = MakeCol("z", ColType::DOUBLE, 1);
	x.num[0] = 2;
	y.num[0] = 0;
	z.valid[0] = 0;
	Column big = MakeCol("big", ColType::DOUBLE, 1);
	big.num[0] = 1e300;
	Column pc = MakeCol("prediction", ColType::DOUBLE, 1); // a column that happens to share the name
	pc.num[0] = 3;
	f.columns = {x, y, z, TextCol("s", {"u"}), big, pc};
	auto eval = [&](const std::string &agg, double want, const char *missing = nullptr) {
		Statement st = Parse("OPTIMIZE t MAXIMIZE " + agg + " USING MODEL m");
		OptimizeSpec spec = st.optimize;
		BindAggregate(spec.objective, f, "", "objective");
		ExprEval ev;
		const double got = EvalAggregateRow(spec.objective, f, 0, 10.0, ev);
		if (missing) {
			CHECK(ev.missing && ev.why.find(missing) != std::string::npos, "%s: expected no value (%s), got %g (%s)",
			      agg.c_str(), missing, got, ev.why.c_str());
		} else {
			CHECK(!ev.missing && got == want, "%s = %g, want %g%s%s", agg.c_str(), got, want, ev.missing ? ": " : "",
			      ev.why.c_str());
		}
	};
	eval("SUM(1 + 2 * 3)", 7);
	eval("SUM((1 + 2) * 3)", 9);
	eval("SUM(10 - 4 - 3)", 3);
	eval("SUM(10 - (4 - 3))", 9);
	eval("SUM(2 * -x)", -4);
	eval("SUM(-x * 3)", -6);
	eval("SUM(x / 4 * 2)", 1);
	eval("SUM(8 / x / 2)", 2);
	eval("SUM(prediction * x)", 20);
	eval("SUM(t.x + y)", 2);
	eval("SUM(CASE WHEN x > 1 THEN 5 ELSE 6 END)", 5);
	eval("SUM(CASE WHEN x > 5 THEN 5 WHEN s = 'u' THEN 4 ELSE 6 END)", 4);
	eval("SUM(CASE s WHEN 'u' THEN 1 WHEN 'v' THEN 2 END)", 1);
	eval("SUM(CASE s WHEN 'v' THEN 2 END)", 0); // no ELSE inside a SUM: nothing
	eval("SUM(CASE WHEN z IS NULL THEN 3 END)", 3);
	eval("SUM(CASE WHEN x > 5 THEN z ELSE 1 END)", 1); // the NULL branch is never taken
	eval("COUNT(*)", 1);
	eval("COUNT(*) FILTER (WHERE x > 1)", 1);
	eval("COUNT(*) FILTER (WHERE x > 5)", 0);
	eval("SUM(z) FILTER (WHERE x > 5)", 0); // filtered out before it is read
	eval("SUM(x / y)", 0, "divides by zero");
	eval("SUM(z + 1)", 0, "z is NULL");
	eval("SUM(big * big)", 0, "too large");
	eval("SUM(prediction)", 10);
	eval("SUM(\"prediction\")", 3);
	eval("SUM(t.prediction)", 3);
	eval("SUM(CASE WHEN prediction > 5 THEN 1 ELSE 2 END)", 1);
	eval("SUM(CASE WHEN \"prediction\" > 5 THEN 1 ELSE 2 END)", 2);
	eval("COUNT(*) FILTER (WHERE prediction < 5)", 0);
	eval("COUNT(*) FILTER (WHERE prediction IN (10, 11))", 1);
	eval("COUNT(*) FILTER (WHERE prediction != 10)", 0);
	eval("COUNT(*) FILTER (WHERE prediction IS NOT NULL AND NOT (prediction <= 9))", 1);
	auto refuse = [&](const std::string &agg, const char *want) {
		try {
			Statement st = Parse("OPTIMIZE t MAXIMIZE " + agg + " USING MODEL m");
			OptimizeSpec spec = st.optimize;
			BindAggregate(spec.objective, f, "tt", "objective");
			CHECK(false, "%s: expected a refusal", agg.c_str());
		} catch (const std::exception &e) {
			CHECK(std::string(e.what()).find(want) != std::string::npos, "%s: got '%s', want '%s'", agg.c_str(),
			      e.what(), want);
		}
	};
	refuse("SUM(s * 2)", "is text");
	refuse("SUM(nope)", "does not exist");
	refuse("SUM(other.x)", "but it is evaluated on 't' (aliased tt)");
	refuse("SUM(CASE WHEN nope = 1 THEN 1 END)", "does not exist");
	refuse("COUNT(*) FILTER (WHERE nope = 1)", "does not exist");
}

// ---------------------------------------------------------------------------
// the small simplex
// ---------------------------------------------------------------------------

// Primal feasible, dual feasible, and the two objectives agree: a certificate
// that the answer is optimal, whatever path the pivots took.
static bool Certified(const SmallLP &lp, double tol, std::string &why) {
	const int R = lp.rows, C = lp.cols;
	double cx = 0, by = 0;
	for (int j = 0; j < C; j++) {
		if (lp.x[(size_t)j] < -tol) {
			why = "x < 0";
			return false;
		}
		cx += lp.c[(size_t)j] * lp.x[(size_t)j];
	}
	for (int i = 0; i < R; i++) {
		double ax = 0;
		for (int j = 0; j < C; j++) {
			ax += lp.A[(size_t)i * (size_t)C + (size_t)j] * lp.x[(size_t)j];
		}
		const double scale = 1 + std::fabs(lp.b[(size_t)i]);
		if (lp.eq[(size_t)i] ? std::fabs(ax - lp.b[(size_t)i]) > tol * scale : ax > lp.b[(size_t)i] + tol * scale) {
			why = "row " + std::to_string(i) + " violated";
			return false;
		}
		if (!lp.eq[(size_t)i] && lp.y[(size_t)i] < -tol) {
			why = "negative dual on a <= row";
			return false;
		}
		by += lp.b[(size_t)i] * lp.y[(size_t)i];
	}
	for (int j = 0; j < C; j++) {
		double ay = 0;
		for (int i = 0; i < R; i++) {
			ay += lp.A[(size_t)i * (size_t)C + (size_t)j] * lp.y[(size_t)i];
		}
		if (ay < lp.c[(size_t)j] - tol * (1 + std::fabs(lp.c[(size_t)j]))) {
			why = "dual infeasible in column " + std::to_string(j);
			return false;
		}
	}
	if (std::fabs(cx - by) > tol * (1 + std::fabs(cx)) || std::fabs(cx - lp.value) > tol * (1 + std::fabs(cx))) {
		why = "duality gap";
		return false;
	}
	return true;
}

static SmallLP MakeLP(int rows, int cols, std::vector<double> A, std::vector<double> b, std::vector<uint8_t> eq,
                      std::vector<double> c) {
	SmallLP lp;
	lp.rows = rows;
	lp.cols = cols;
	lp.A = std::move(A);
	lp.b = std::move(b);
	lp.eq = std::move(eq);
	lp.c = std::move(c);
	return lp;
}

static void Simplex() {
	std::printf("== the small simplex\n");
	std::string why;
	{
		// max 3x + 5y st x <= 4, 2y <= 12, 3x + 2y <= 18: (2, 6), 36, duals (0, 1.5, 1)
		SmallLP lp = MakeLP(3, 2, {1, 0, 0, 2, 3, 2}, {4, 12, 18}, {0, 0, 0}, {3, 5});
		SolveSmallLP(lp);
		CHECK(lp.status == SmallLP::Status::OPTIMAL && std::fabs(lp.value - 36) < 1e-9 &&
		          std::fabs(lp.x[0] - 2) < 1e-9 && std::fabs(lp.x[1] - 6) < 1e-9,
		      "textbook LP: status %d value %g", (int)lp.status, lp.value);
		CHECK(std::fabs(lp.y[0]) < 1e-9 && std::fabs(lp.y[1] - 1.5) < 1e-9 && std::fabs(lp.y[2] - 1) < 1e-9,
		      "textbook duals %g %g %g", lp.y[0], lp.y[1], lp.y[2]);
	}
	{
		// Beale's example: Dantzig's rule cycles on it forever; Bland's must not.
		SmallLP lp =
		    MakeLP(3, 4, {0.25, -8, -1, 9, 0.5, -12, -0.5, 3, 0, 0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0.75, -20, 0.5, -6});
		SolveSmallLP(lp);
		CHECK(lp.status == SmallLP::Status::OPTIMAL && std::fabs(lp.value - 1.25) < 1e-9, "Beale: status %d value %g",
		      (int)lp.status, lp.value);
		CHECK(Certified(lp, 1e-9, why), "Beale: %s", why.c_str());
	}
	{
		// x >= 0.25 written as -x <= -0.25, an equality, and a negative right-hand side.
		SmallLP lp = MakeLP(3, 2, {1, 1, -1, 0, 0, 1}, {1, -0.25, 0.5}, {1, 0, 0}, {1, 2});
		SolveSmallLP(lp);
		CHECK(lp.status == SmallLP::Status::OPTIMAL && std::fabs(lp.value - 1.5) < 1e-9, "eq + negative rhs: %g",
		      lp.value);
		CHECK(Certified(lp, 1e-9, why), "eq + negative rhs: %s", why.c_str());
	}
	{
		SmallLP lp = MakeLP(2, 1, {1, -1}, {1, -2}, {0, 0}, {1});
		SolveSmallLP(lp);
		CHECK(lp.status == SmallLP::Status::INFEASIBLE, "x <= 1 and x >= 2 should be infeasible");
		SmallLP un = MakeLP(1, 1, {-1}, {1}, {0}, {1});
		SolveSmallLP(un);
		CHECK(un.status == SmallLP::Status::UNBOUNDED, "max x st -x <= 1 should be unbounded");
	}
	{
		// Random feasible, bounded LPs, many of them degenerate: every answer must
		// carry its own certificate.
		std::mt19937 rng(99);
		std::uniform_real_distribution<double> U(-1, 1);
		int certified = 0, total = 0;
		for (int t = 0; t < 3000; t++) {
			const int R = 1 + (int)(rng() % 6), C = 1 + (int)(rng() % 12);
			std::vector<double> A((size_t)(R + 1) * (size_t)C), b((size_t)R + 1), c((size_t)C), x0((size_t)C);
			std::vector<uint8_t> eq((size_t)R + 1, 0);
			for (auto &v : x0) {
				v = (rng() % 3 == 0) ? 0.0 : std::fabs(U(rng));
			}
			for (int i = 0; i < R; i++) {
				eq[(size_t)i] = rng() % 4 == 0;
				double ax = 0;
				for (int j = 0; j < C; j++) {
					double a = (rng() % 3 == 0) ? 0.0 : std::round(U(rng) * 4) / 2;
					A[(size_t)i * (size_t)C + (size_t)j] = a;
					ax += a * x0[(size_t)j];
				}
				b[(size_t)i] = eq[(size_t)i] ? ax : ax + ((rng() % 2) ? 0.0 : std::fabs(U(rng)));
			}
			// Keep it bounded: SUM x <= 10.
			double s0 = 0;
			for (int j = 0; j < C; j++) {
				A[(size_t)R * (size_t)C + (size_t)j] = 1;
				s0 += x0[(size_t)j];
			}
			b[(size_t)R] = std::max(10.0, s0);
			for (auto &v : c) {
				v = std::round(U(rng) * 6) / 2;
			}
			SmallLP lp = MakeLP(R + 1, C, A, b, eq, c);
			SolveSmallLP(lp);
			total++;
			if (lp.status == SmallLP::Status::OPTIMAL && Certified(lp, 1e-7, why)) {
				certified++;
			} else if (fails < 20) {
				std::printf("   random LP %d: status %d %s\n", t, (int)lp.status, why.c_str());
			}
		}
		CHECK(certified == total, "%d of %d random LPs certified optimal", certified, total);
		std::printf("   %d random LPs, every one certified optimal\n", total);
	}
}

// ---------------------------------------------------------------------------
// the solver against brute force and exact programmes
// ---------------------------------------------------------------------------

// The solver's own rule: a limit is met within a rounding error of its size.
static double SlackTol(const OptProblem &P, size_t j) {
	return 1e-9 * std::max(1.0, std::fabs(P.b[j]));
}

static bool Meets(const OptProblem &P, const std::vector<int> &x, double *obj = nullptr) {
	const size_t K = (size_t)P.K;
	double v = 0;
	for (size_t i = 0; i < P.n; i++) {
		v += P.v[i * K + (size_t)x[i]];
	}
	if (obj) {
		*obj = v;
	}
	for (size_t j = 0; j < P.b.size(); j++) {
		double u = 0;
		for (size_t i = 0; i < P.n; i++) {
			u += P.a[j][i * K + (size_t)x[i]];
		}
		if (u > P.b[j] + SlackTol(P, j)) {
			return false;
		}
	}
	return true;
}

static bool Brute(const OptProblem &P, double &best) {
	std::vector<int> x(P.n, 0);
	bool found = false;
	best = -std::numeric_limits<double>::infinity();
	while (true) {
		double v;
		if (Meets(P, x, &v) && v > best) {
			best = v;
			found = true;
		}
		size_t i = 0;
		for (; i < P.n; i++) {
			if (++x[i] < P.K) {
				break;
			}
			x[i] = 0;
		}
		if (i == P.n) {
			break;
		}
	}
	return found;
}

static OptProblem RandomProblem(std::mt19937 &rng, size_t n, int K, size_t m, bool integers, bool subset) {
	std::uniform_real_distribution<double> U(0, 1);
	OptProblem P;
	P.n = n;
	P.K = K;
	P.v.resize(n * (size_t)K);
	P.a.assign(m, std::vector<double>(n * (size_t)K));
	auto draw = [&](double lo, double hi) {
		const double v = lo + (hi - lo) * U(rng);
		return integers ? std::round(v) : v;
	};
	for (size_t i = 0; i < n; i++) {
		for (int k = 0; k < K; k++) {
			const size_t o = i * (size_t)K + (size_t)k;
			const bool idle = subset && k == 0;
			P.v[o] = idle ? 0.0 : draw(-4, 10);
			for (size_t j = 0; j < m; j++) {
				P.a[j][o] = idle ? 0.0 : draw(0, 6);
			}
		}
	}
	for (size_t j = 0; j < m; j++) {
		// Between the least and the most this limit can see, so it binds, and
		// sometimes just outside, so a plan cannot exist.
		double lo = 0, hi = 0;
		for (size_t i = 0; i < n; i++) {
			double mn = 1e300, mx = -1e300;
			for (int k = 0; k < K; k++) {
				mn = std::min(mn, P.a[j][i * (size_t)K + (size_t)k]);
				mx = std::max(mx, P.a[j][i * (size_t)K + (size_t)k]);
			}
			lo += mn;
			hi += mx;
		}
		double b = lo + (hi - lo) * (U(rng) * 1.1 - 0.05);
		if (integers) {
			b = std::round(b);
		}
		if (U(rng) < 0.4) {
			// a >= limit, negated into the <= form the solver takes
			for (auto &x : P.a[j]) {
				x = -x;
			}
			b = -(lo + (hi - lo) * (U(rng) * 1.1 - 0.05));
			if (integers) {
				b = std::round(b);
			}
		}
		P.b.push_back(b);
	}
	return P;
}

static void SolverVsBrute() {
	std::printf("== the solver against brute force\n");
	std::mt19937 rng(20261026);
	struct Tally {
		const char *name;
		OptSettings set;
		bool must_be_exact;
		int instances = 0, feasible = 0, optimal = 0, missed = 0, proven = 0;
		double gap = 0;
	};
	// Each path with the shortcuts before it turned off, so every method is
	// checked on problems brute force can answer.
	std::vector<Tally> paths(4);
	paths[0].name = "default";
	paths[0].must_be_exact = true;
	paths[1].name = "no enumeration";
	paths[1].set.exhaustive_plans = 0;
	paths[1].must_be_exact = true;
	paths[2].name = "relaxation + search";
	paths[2].set.exhaustive_plans = 0;
	paths[2].set.dp_cells = 0;
	paths[2].must_be_exact = true;
	paths[3].name = "relaxation alone";
	paths[3].set.exhaustive_plans = 0;
	paths[3].set.dp_cells = 0;
	paths[3].set.search_nodes = 0;
	paths[3].must_be_exact = false;
	for (int t = 0; t < 6000; t++) {
		const size_t n = 1 + rng() % 9;
		const int K = 2 + (int)(rng() % 2);
		const size_t m = rng() % 4;
		const bool integers = rng() % 3 == 0;
		const bool subset = rng() % 2 == 0;
		OptProblem P = RandomProblem(rng, n, K, m, integers, subset);
		double opt;
		const bool has = Brute(P, opt);
		double range = 0; // the most any one row's choice can move the total
		for (size_t i = 0; i < n; i++) {
			double mn = 1e300, mx = -1e300;
			for (int k = 0; k < K; k++) {
				mn = std::min(mn, P.v[i * (size_t)K + (size_t)k]);
				mx = std::max(mx, P.v[i * (size_t)K + (size_t)k]);
			}
			range = std::max(range, mx - mn);
		}
		const double tol = 1e-7 * (1 + std::fabs(opt));
		for (auto &ta : paths) {
			const OptSolution s = SolveOpt(P, ta.set);
			const char *name = ta.name;
			ta.instances++;
			if (s.feasible) {
				CHECK(Meets(P, s.choice), "%s, instance %d: the plan breaks a limit", name, t);
				CHECK(has, "%s, instance %d: a plan for a problem brute force says has none", name, t);
				double got;
				Meets(P, s.choice, &got);
				CHECK(std::fabs(got - s.objective) <= tol, "%s: reported objective %g, plan is worth %g", name,
				      s.objective, got);
				if (has) {
					CHECK(got <= opt + tol, "%s, instance %d: better than optimal? %g > %g", name, t, got, opt);
					CHECK(s.bound >= opt - tol, "%s, instance %d: bound %g below the optimum %g", name, t, s.bound,
					      opt);
					CHECK(s.bound >= got - tol, "%s: bound %g below its own plan %g", name, s.bound, got);
					if (s.exact) {
						CHECK(std::fabs(got - opt) <= tol, "%s, instance %d: claims exact at %g, optimum %g", name, t,
						      got, opt);
					}
					ta.feasible++;
					ta.optimal += std::fabs(got - opt) <= tol;
					ta.gap += opt - got;
					if (m == 1 && !integers) {
						// One limit: within one row's swing of the best.
						CHECK(opt - got <= range + tol, "%s, instance %d: one limit, gap %g over a row's range %g",
						      name, t, opt - got, range);
					}
				}
			} else {
				CHECK(!s.infeasible_proven || !has, "%s, instance %d: proved infeasible, brute force found %g", name, t,
				      opt);
				if (has) {
					ta.missed++;
					if (ta.must_be_exact || m <= 1) {
						CHECK(false, "%s, instance %d (m=%zu): found no plan, brute force found %g", name, t, m, opt);
					}
				} else {
					ta.proven += s.infeasible_proven;
				}
			}
		}
	}
	for (const auto &ta : paths) {
		std::printf(
		    "   %-20s %d problems, %d with a plan: %d optimal, %d missed, mean gap %.4f; %d proven impossible\n",
		    ta.name, ta.instances, ta.feasible, ta.optimal, ta.missed, ta.feasible ? ta.gap / ta.feasible : 0.0,
		    ta.proven);
		if (ta.must_be_exact) {
			CHECK(ta.optimal == ta.feasible && ta.missed == 0, "%s should be exact on problems this small", ta.name);
		}
	}
	CHECK(paths[3].optimal >= 0.9 * paths[3].feasible, "the relaxation alone found the optimum in only %d of %d",
	      paths[3].optimal, paths[3].feasible);
	CHECK(paths[3].missed <= paths[3].feasible / 100, "the relaxation alone missed %d plans that exist",
	      paths[3].missed);
}

static double KnapsackDP(const std::vector<double> &val, const std::vector<int> &w, int cap) {
	std::vector<double> best((size_t)cap + 1, 0.0);
	for (size_t i = 0; i < val.size(); i++) {
		if (val[i] <= 0) {
			continue;
		}
		for (int c = cap; c >= w[i]; c--) {
			best[(size_t)c] = std::max(best[(size_t)c], best[(size_t)(c - w[i])] + val[i]);
		}
	}
	return best[(size_t)cap];
}

// Every row takes exactly one option; option costs are whole numbers and one
// option per row is free, so the budget is always reachable.
static double ChoiceDP(const OptProblem &P, const std::vector<int> &cost, int cap) {
	const size_t K = (size_t)P.K;
	std::vector<double> dp((size_t)cap + 1, 0.0), next((size_t)cap + 1);
	for (size_t i = 0; i < P.n; i++) {
		for (int c = 0; c <= cap; c++) {
			double best = -1e300;
			for (size_t k = 0; k < K; k++) {
				const int w = cost[i * K + k];
				if (w <= c) {
					best = std::max(best, dp[(size_t)(c - w)] + P.v[i * K + k]);
				}
			}
			next[(size_t)c] = best;
		}
		dp.swap(next);
	}
	return dp[(size_t)cap];
}

static void SolverAtScale() {
	std::printf("== the solver against exact programmes, and at scale\n");
	std::mt19937 rng(7);
	std::uniform_real_distribution<double> U(0, 1);
	{
		// Top k: one count limit is solved exactly.
		const size_t n = 5000, k = 37;
		OptProblem P;
		P.n = n;
		P.K = 2;
		P.v.assign(n * 2, 0.0);
		P.a.assign(1, std::vector<double>(n * 2, 0.0));
		std::vector<double> vals(n);
		for (size_t i = 0; i < n; i++) {
			vals[i] = U(rng) * 100 - 20;
			P.v[i * 2 + 1] = vals[i];
			P.a[0][i * 2 + 1] = 1;
		}
		P.b = {double(k)};
		const OptSolution s = SolveOpt(P);
		std::vector<double> sorted = vals;
		std::sort(sorted.rbegin(), sorted.rend());
		double want = 0;
		for (size_t i = 0; i < k; i++) {
			want += std::max(0.0, sorted[i]);
		}
		CHECK(s.feasible && std::fabs(s.objective - want) < 1e-6, "top %zu: got %g, want %g", k, s.objective, want);
		CHECK(s.exact, "top k should be proven optimal (bound %g, objective %g)", s.bound, s.objective);
	}
	{
		// 0/1 knapsack against its dynamic programme.
		double worst = 0;
		for (int t = 0; t < 40; t++) {
			const size_t n = 50 + rng() % 400;
			const int cap = 100 + (int)(rng() % 1500);
			OptProblem P;
			P.n = n;
			P.K = 2;
			P.v.assign(n * 2, 0.0);
			P.a.assign(1, std::vector<double>(n * 2, 0.0));
			std::vector<double> val(n);
			std::vector<int> w(n);
			double vmax = 0;
			for (size_t i = 0; i < n; i++) {
				w[i] = 1 + (int)(rng() % 60);
				val[i] = U(rng) * 50 - 5;
				vmax = std::max(vmax, val[i]);
				P.v[i * 2 + 1] = val[i];
				P.a[0][i * 2 + 1] = w[i];
			}
			P.b = {double(cap)};
			const double opt = KnapsackDP(val, w, cap);
			const OptSolution dp = SolveOpt(P);
			CHECK(dp.exact && std::fabs(dp.objective - opt) < 1e-6, "knapsack %d by DP: %g, optimum %g", t,
			      dp.objective, opt);
			OptSettings general;
			general.dp_cells = 0;
			const OptSolution s = SolveOpt(P, general);
			CHECK(s.feasible && Meets(P, s.choice), "knapsack %d: no feasible plan", t);
			CHECK(s.objective <= opt + 1e-6 && s.bound >= opt - 1e-6, "knapsack %d: plan %g, optimum %g, bound %g", t,
			      s.objective, opt, s.bound);
			CHECK(opt - s.objective <= vmax + 1e-6, "knapsack %d: gap %g over one item's value %g", t,
			      opt - s.objective, vmax);
			worst = std::max(worst, (opt - s.objective) / std::max(1.0, opt));
		}
		std::printf("   0/1 knapsack, 40 problems: exact by DP; without it, worst gap %.4f%%\n", worst * 100);
		CHECK(worst < 0.01, "knapsack worst relative gap %g", worst);
	}
	{
		// One option per row, one budget, against its dynamic programme.
		double worst = 0;
		for (int t = 0; t < 30; t++) {
			const size_t n = 30 + rng() % 200;
			const int K = 2 + (int)(rng() % 4);
			const int cap = 50 + (int)(rng() % 800);
			OptProblem P;
			P.n = n;
			P.K = K;
			P.v.resize(n * (size_t)K);
			P.a.assign(1, std::vector<double>(n * (size_t)K));
			std::vector<int> cost(n * (size_t)K);
			for (size_t i = 0; i < n; i++) {
				for (int k = 0; k < K; k++) {
					const size_t o = i * (size_t)K + (size_t)k;
					cost[o] = k == 0 ? 0 : 1 + (int)(rng() % 25);
					P.v[o] = U(rng) * 20 + cost[o] * U(rng) * 0.8;
					P.a[0][o] = cost[o];
				}
			}
			P.b = {double(cap)};
			const double opt = ChoiceDP(P, cost, cap);
			const OptSolution dp = SolveOpt(P);
			CHECK(dp.exact && std::fabs(dp.objective - opt) < 1e-6, "choice %d by DP: %g, optimum %g", t, dp.objective,
			      opt);
			OptSettings general;
			general.dp_cells = 0;
			const OptSolution s = SolveOpt(P, general);
			CHECK(s.feasible && Meets(P, s.choice), "choice %d: no feasible plan", t);
			CHECK(s.objective <= opt + 1e-6 && s.bound >= opt - 1e-6, "choice %d: plan %g, optimum %g, bound %g", t,
			      s.objective, opt, s.bound);
			worst = std::max(worst, (opt - s.objective) / std::max(1.0, opt));
		}
		std::printf("   one-of-K with a budget, 30 problems: exact by DP; without it, worst gap %.4f%%\n", worst * 100);
		CHECK(worst < 0.01, "multiple-choice worst relative gap %g", worst);
	}
	{
		// Near ties at the price line with costs that are not whole: hundreds of
		// trucks almost certain to break down, so each is worth about the same,
		// and a budget that fits a few dozen. Rounding every cost down to a whole
		// number relaxes the problem, so that problem's exact optimum is at least
		// the true one: the plan has to come within a hair of it.
		std::mt19937 r2(41);
		const size_t n = 800;
		OptProblem P;
		P.n = n;
		P.K = 2;
		P.v.assign(n * 2, 0.0);
		P.a.assign(1, std::vector<double>(n * 2, 0.0));
		std::vector<double> val(n);
		std::vector<int> w(n);
		for (size_t i = 0; i < n; i++) {
			const double risk = (i % 3 == 0) ? 0.999 + 0.001 * U(r2) : 0.05 * U(r2);
			w[i] = 50 + (int)(i * 37 % 150);
			val[i] = risk * 4200 - w[i] * 1.0001;
			P.v[i * 2 + 1] = val[i];
			P.a[0][i * 2 + 1] = w[i] * 1.00001;
		}
		P.b = {2000};
		const OptSolution s = SolveOpt(P);
		const double relaxed = KnapsackDP(val, w, 2000);
		std::printf("   near ties, costs not whole: plan %.2f, relaxed optimum %.2f, bound %.2f\n", s.objective,
		            relaxed, s.bound);
		CHECK(s.feasible && Meets(P, s.choice), "near ties: no feasible plan");
		CHECK(s.objective >= relaxed * (1 - 1e-3), "near ties: plan %g is more than 0.1%% below %g", s.objective,
		      relaxed);
		CHECK(s.bound >= s.objective, "near ties: bound below plan");
	}
	{
		// Ties everywhere: every row the same. The tie-break must not fill the
		// plan with one side of the tie and leave it there.
		const size_t n = 20000;
		OptProblem P;
		P.n = n;
		P.K = 2;
		P.v.assign(n * 2, 0.0);
		P.a.assign(1, std::vector<double>(n * 2, 0.0));
		for (size_t i = 0; i < n; i++) {
			P.v[i * 2 + 1] = -3;    // MINIMIZE cost 3 ...
			P.a[0][i * 2 + 1] = -1; // ... SUBJECT TO COUNT(*) >= 50
		}
		P.b = {-50};
		const auto t0 = std::chrono::steady_clock::now();
		const OptSolution s = SolveOpt(P);
		const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		CHECK(s.feasible && std::fabs(s.objective + 150) < 1e-9, "identical rows, at least 50: objective %g",
		      s.objective);
		CHECK(secs < 5 * kSlow, "identical rows took %.2fs", secs);
	}
	{
		// Large: 100k rows, 3 options, a budget and a cap on one option.
		const size_t n = 100000;
		const int K = 3;
		OptProblem P;
		P.n = n;
		P.K = K;
		P.v.resize(n * K);
		P.a.assign(2, std::vector<double>(n * K, 0.0));
		double total_cost = 0;
		for (size_t i = 0; i < n; i++) {
			const double base = U(rng) * 10;
			for (int k = 0; k < K; k++) {
				const size_t o = i * K + (size_t)k;
				P.v[o] = base + k * U(rng) * 6;
				P.a[0][o] = k * 2.0 + (k ? U(rng) : 0.0);
				P.a[1][o] = k == 2 ? 1.0 : 0.0;
				total_cost += P.a[0][o] / K;
			}
		}
		P.b = {total_cost * 0.3, 5000.0};
		const auto t0 = std::chrono::steady_clock::now();
		const OptSolution s = SolveOpt(P);
		const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		const double gap = (s.bound - s.objective) / std::fabs(s.bound);
		std::printf("   100000 rows x 3 options, 2 limits: %.2fs, within %.5f%% of the bound\n", secs, gap * 100);
		CHECK(s.feasible && Meets(P, s.choice), "large: no feasible plan");
		CHECK(gap < 1e-3, "large: gap %g", gap);
		CHECK(secs < 30 * kSlow, "large: %.1fs", secs);
	}
	{
		// Degenerate shapes.
		OptProblem empty;
		empty.K = 2;
		OptSolution s = SolveOpt(empty);
		CHECK(s.feasible && s.objective == 0 && s.exact, "no rows: an empty plan");
		empty.a.assign(1, {});
		empty.b = {-1}; // COUNT(*) >= 1 over nothing
		s = SolveOpt(empty);
		CHECK(!s.feasible && s.infeasible_proven, "no rows and a positive minimum: proven impossible");
		OptProblem flat;
		flat.n = 3;
		flat.K = 2;
		flat.v = {0, 1, 0, 2, 0, 3};
		flat.a.assign(1, std::vector<double>(6, 0.0)); // a limit nothing touches
		flat.b = {5};
		s = SolveOpt(flat);
		CHECK(s.feasible && s.objective == 6, "a limit that nothing uses changes nothing");
		flat.b = {-5};
		s = SolveOpt(flat);
		CHECK(!s.feasible && s.infeasible_proven, "0 <= -5 cannot hold");
		// A limit a hair below its own least total, inside the tolerance: the
		// cheapest plan meets it, and the table over totals has one cell, not none.
		OptProblem hair;
		hair.n = 40;
		hair.K = 2;
		hair.v.resize(80);
		hair.a.assign(1, std::vector<double>(80));
		for (size_t i = 0; i < 40; i++) {
			hair.v[i * 2] = double(i % 7);
			hair.v[i * 2 + 1] = double(i % 5) + 3;
			hair.a[0][i * 2] = 1000;
			hair.a[0][i * 2 + 1] = 2000;
		}
		hair.b = {40000 - 1e-6};
		s = SolveOpt(hair);
		bool cheapest = s.feasible;
		for (size_t i = 0; cheapest && i < 40; i++) {
			cheapest = s.choice[i] == 0;
		}
		CHECK(cheapest && s.exact, "a limit within rounding of its least total: the cheapest plan, exactly");
	}
}

// ---------------------------------------------------------------------------
// what an independent review found, kept found
// ---------------------------------------------------------------------------

// Groups that each need a minimum, under one budget: the shape where a limit's
// price has to be found per group.
static OptProblem Groups(std::mt19937 &rng, size_t n, size_t G, size_t at_least, double slack, double value_scale,
                         double value_shift) {
	std::uniform_real_distribution<double> U(0, 1);
	OptProblem P;
	P.n = n;
	P.K = 2;
	P.v.assign(n * 2, 0.0);
	P.a.assign(G + 1, std::vector<double>(n * 2, 0.0));
	std::vector<std::vector<double>> by(G);
	for (size_t i = 0; i < n; i++) {
		const double cost = std::round((20 + U(rng) * 80) * 100) / 100 + 0.001;
		P.v[i * 2 + 1] = (U(rng) * 100 + value_shift) * value_scale;
		P.a[i % G][i * 2 + 1] = -1; // COUNT(*) FILTER (WHERE grp = g) >= at_least, negated
		P.a[G][i * 2 + 1] = cost;   // SUM(cost) <= budget
		by[i % G].push_back(cost);
	}
	double least = 0;
	for (auto &c : by) {
		std::sort(c.begin(), c.end());
		for (size_t k = 0; k < at_least; k++) {
			least += c[k];
		}
	}
	for (size_t g = 0; g < G; g++) {
		P.b.push_back(-double(at_least));
	}
	P.b.push_back(least + slack);
	return P;
}

static void Regressions() {
	std::printf("== what an independent review found\n");
	std::mt19937 rng(5150);
	{
		// One row costing 1e12 used to widen the tolerance of the whole budget to
		// a thousand, so a plan spending 1000 of 100 passed as met, and exact.
		OptProblem P;
		P.n = 101;
		P.K = 2;
		P.v.assign(202, 0.0);
		P.a.assign(1, std::vector<double>(202, 0.0));
		for (size_t i = 0; i < 101; i++) {
			const double cost = i < 100 ? 10 : 1e12;
			P.v[i * 2 + 1] = 4000 - cost;
			P.a[0][i * 2 + 1] = cost;
		}
		P.b = {100};
		const OptSolution s = SolveOpt(P);
		size_t taken = 0;
		for (int c : s.choice) {
			taken += c;
		}
		CHECK(s.feasible && s.usage[0] <= 100 + 1e-6 && taken == 10, "an outlier cost: %zu rows, usage %g of 100",
		      taken, s.usage[0]);
	}
	{
		// The same problem with its values scaled up: pricing ties used to grow
		// with the values, and a feasible problem was proved impossible.
		for (double vs : {1.0, 1e9, 1e12, 1e15}) {
			std::mt19937 r(7);
			OptProblem P = Groups(r, 200, 3, 2, 5, vs, -50);
			OptSettings general;
			general.dp_cells = 0;
			general.exhaustive_plans = 0;
			for (const OptSettings &set : {OptSettings(), general}) {
				const OptSolution s = SolveOpt(P, set);
				CHECK(s.feasible && Meets(P, s.choice), "values x%g: no plan (proven impossible: %d)", vs,
				      (int)s.infeasible_proven);
			}
		}
	}
	{
		// Many minimums. The master LP stalls on this shape and the plan needs
		// swaps within a group, which no single switch makes.
		for (size_t G : {40, 100, 200}) {
			std::mt19937 r(3);
			OptProblem P = Groups(r, 2000, G, 2, 500, 0.2, -260); // every row costs value: take only what is needed
			const auto t0 = std::chrono::steady_clock::now();
			const OptSolution s = SolveOpt(P);
			const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			std::printf("   %zu groups, a minimum each and a budget: %s in %.2fs, gap %.2f%%\n", G,
			            s.feasible ? "a plan" : "NO PLAN", secs,
			            s.feasible ? 100 * (s.bound - s.objective) / std::fabs(s.bound) : 0.0);
			CHECK(s.feasible && Meets(P, s.choice), "%zu groups with minimums: no plan, though one exists", G);
			CHECK(!s.infeasible_proven, "%zu groups: proven impossible, though a plan exists", G);
			CHECK(secs < 15 * kSlow, "%zu groups took %.1fs", G, secs);
		}
	}
	{
		// Hundreds of caps used to take minutes: the master was a dense tableau
		// of one row per limit, solved from scratch each round.
		const size_t n = 5000, G = 200;
		OptProblem P;
		P.n = n;
		P.K = 2;
		P.v.assign(n * 2, 0.0);
		P.a.assign(G + 1, std::vector<double>(n * 2, 0.0));
		std::uniform_real_distribution<double> U(0, 1);
		double total = 0;
		for (size_t i = 0; i < n; i++) {
			P.v[i * 2 + 1] = U(rng) * 10;
			P.a[i % G][i * 2 + 1] = 1;
			P.a[G][i * 2 + 1] = 1 + U(rng) * 9;
			total += P.a[G][i * 2 + 1];
		}
		for (size_t g = 0; g < G; g++) {
			P.b.push_back(3);
		}
		P.b.push_back(total * 0.05);
		const auto t0 = std::chrono::steady_clock::now();
		const OptSolution s = SolveOpt(P);
		const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		std::printf("   %zu caps and a budget over %zu rows: %.2fs, gap %.3f%%\n", G, n, secs,
		            100 * (s.bound - s.objective) / std::fabs(s.bound));
		CHECK(s.feasible && Meets(P, s.choice), "many caps: no plan");
		CHECK(secs < 20 * kSlow, "many caps took %.1fs", secs);
		CHECK((s.bound - s.objective) / std::fabs(s.bound) < 0.01, "many caps: gap over 1%%");
	}
	{
		// The solver refuses what it cannot compare, rather than call NaN met.
		OptProblem P;
		P.n = 2;
		P.K = 2;
		P.v = {0, 1, 0, 2};
		P.a = {{0, 1, 0, std::numeric_limits<double>::quiet_NaN()}};
		P.b = {5};
		bool threw = false;
		try {
			SolveOpt(P);
		} catch (const std::exception &) {
			threw = true;
		}
		CHECK(threw, "a NaN coefficient must be refused");
		P.a = {{0, 1, 0, 1}};
		P.b = {-std::numeric_limits<double>::infinity()};
		threw = false;
		try {
			SolveOpt(P);
		} catch (const std::exception &) {
			threw = true;
		}
		CHECK(threw, "an infinite limit must be refused");
	}
}

// ---------------------------------------------------------------------------
// end to end: a trained model, a decision with a known best answer
// ---------------------------------------------------------------------------

static const double DAY = 86400.0 * 1e6;

static Model Train(const Database &db, const char *sql) {
	Statement st = Parse(sql);
	Model m;
	m.name = st.model;
	TrainModel(db, st, m);
	return m;
}

static OptimizeOutcome Opt(Database &db, const Model &m, const std::string &sql) {
	return RunOptimize(db, m, Parse(sql));
}

static void Refused(const std::function<void()> &fn, const char *want, const char *what) {
	try {
		fn();
		CHECK(false, "%s: expected a refusal containing '%s'", what, want);
	} catch (const std::exception &e) {
		CHECK(std::string(e.what()).find(want) != std::string::npos, "%s:\n     got:  %s\n     want: %s", what,
		      e.what(), want);
	}
}

// Customers whose spend responds to an offer, by segment: segment a buys a lot
// more with any offer, segment b barely notices. Offers were handed out at
// random, so what the model learns about them is causal.
static const double kEffect[2][3] = {{0, 30, 32}, {0, 3, 4}};
static const double kCost[3] = {0, 10, 20};
static const char *kOffer[3] = {"none", "10%", "20%"};

static Database Uplift(uint32_t seed, size_t n, std::vector<int> &seg, std::vector<int> &offer) {
	std::mt19937 rng(seed);
	std::normal_distribution<double> N(0, 3);
	std::uniform_real_distribution<double> U(0, 1);
	Frame f;
	f.name = "customers";
	f.nrows = n;
	Column id = MakeCol("id", ColType::INT64, n), spend = MakeCol("spend", ColType::DOUBLE, n);
	Column tenure = MakeCol("tenure", ColType::DOUBLE, n), tier = MakeCol("tier", ColType::INT64, n);
	std::vector<std::string> segs(n), offers(n);
	seg.assign(n, 0);
	offer.assign(n, 0);
	for (size_t i = 0; i < n; i++) {
		seg[i] = U(rng) < 0.5 ? 0 : 1;
		offer[i] = (int)(rng() % 3);
		id.num[i] = double(i);
		tenure.num[i] = U(rng) * 10;
		tier.num[i] = double(1 + rng() % 3);
		spend.num[i] = (seg[i] ? 40 : 50) + kEffect[seg[i]][offer[i]] + N(rng);
		segs[i] = seg[i] ? "b" : "a";
		offers[i] = kOffer[offer[i]];
	}
	f.columns = {id, TextCol("segment", segs), TextCol("offer", offers), tenure, tier, spend};
	Database db;
	db.tables.push_back(f);
	db.BuildLinks({});
	return db;
}

static void EndToEndChoose() {
	std::printf("== end to end: CHOOSE recovers the best offer per segment\n");
	std::vector<int> seg, offer;
	Database db = Uplift(5, 3000, seg, offer);
	const Model m = Train(db, "TRAIN MODEL spend PREDICT customers.spend FOR customers EXCLUDE (id) "
	                          "OPTIONS (epochs = 80, hidden = 32)");
	const std::string cost = "CASE offer WHEN '10%' THEN 10 WHEN '20%' THEN 20 ELSE 0 END";
	const std::string head = "OPTIMIZE customers CHOOSE offer IN ('none', '10%', '20%') ";
	const Column before = db.At("customers").At("offer");
	OptimizeOutcome o = Opt(db, m, head + "MAXIMIZE SUM(prediction - " + cost + ") USING MODEL spend");
	const size_t n = seg.size();
	CHECK(o.plan.size() == n && o.candidates == n, "every customer gets a choice: %zu of %zu", o.plan.size(), n);
	// Truth: segment a is best served by 10% (30 - 10 beats 32 - 20), b by nothing.
	size_t right = 0;
	double plan_true = 0, best_true = 0, now_true = 0;
	for (const auto &r : o.plan) {
		const int s = seg[r.entity_row];
		const int want = s == 0 ? 1 : 0;
		right += r.choice == want;
		plan_true += kEffect[s][r.choice] - kCost[r.choice];
		best_true += kEffect[s][want] - kCost[want];
		now_true += kEffect[s][offer[r.entity_row]] - kCost[offer[r.entity_row]];
	}
	std::printf("   right offer for %zu of %zu; true gain %.0f against a best of %.0f and %.0f as handed out\n", right,
	            n, plan_true, best_true, now_true);
	CHECK(right >= n * 9 / 10, "right offer for only %zu of %zu", right, n);
	CHECK(plan_true >= 0.9 * best_true, "plan's true gain %g, best %g", plan_true, best_true);
	CHECK(plan_true > now_true, "the plan should beat the offers as handed out");
	CHECK(o.exact && o.bound == o.objective, "no limits: the plan is each row's best, which is exact");
	CHECK(o.has_status_quo, "status quo is reported when CHOOSE is used");
	{
		// The status quo is the same model on the data as it stands.
		auto now = RunPredict(db, m, Parse("PREDICT customers.spend FOR customers USING MODEL spend"));
		double sq = 0;
		const Column &oc = db.At("customers").At("offer");
		for (const auto &p : now) {
			const std::string &v = oc.dict.values[oc.code[p.entity_row]];
			sq += p.value - (v == "10%" ? 10 : v == "20%" ? 20 : 0);
		}
		CHECK(std::fabs(sq - o.status_quo) < 1e-6 * (1 + std::fabs(sq)), "status quo %g, PREDICT says %g", o.status_quo,
		      sq);
		CHECK(o.objective >= o.status_quo, "the plan is predicted to beat the status quo");
	}
	{
		// The lever is put back.
		const Column &after = db.At("customers").At("offer");
		bool same = after.valid == before.valid && after.code.size() == before.code.size();
		for (size_t i = 0; same && i < after.code.size(); i++) {
			same = after.dict.values[after.code[i]] == before.dict.values[before.code[i]];
		}
		CHECK(same, "OPTIMIZE left the lever column changed");
		CHECK(after.dict.size() == before.dict.size(), "OPTIMIZE grew the lever's dictionary from %zu to %zu",
		      before.dict.size(), after.dict.size());
	}
	{
		// Each row's predicted value under its choice equals a PREDICT with the
		// column set to that choice.
		Frame &f = db.At("customers");
		Column &oc = f.columns[(size_t)f.Find("offer")];
		const Column saved = oc;
		for (size_t r = 0; r < n; r++) {
			oc.code[r] = oc.dict.Intern(kOffer[o.plan[r].choice]);
		}
		auto p = RunPredict(db, m, Parse("PREDICT customers.spend FOR customers USING MODEL spend"));
		oc = saved;
		double worst = 0;
		for (size_t r = 0; r < n; r++) {
			worst = std::max(worst, std::fabs(p[r].value - o.plan[r].prediction));
		}
		CHECK(worst < 1e-9, "plan prediction differs from PREDICT under the same choice by %g", worst);
	}
	{
		// A budget: half of segment a can have 10%.
		size_t na = 0;
		for (int s : seg) {
			na += s == 0;
		}
		const double budget = 10.0 * double(na / 2);
		OptimizeOutcome b = Opt(db, m,
		                        head + "MAXIMIZE SUM(prediction - " + cost + ") SUBJECT TO SUM(" + cost +
		                            ") <= " + TidyNumber(budget) + " USING MODEL spend");
		size_t ten = 0, ten_a = 0, twenty = 0;
		for (const auto &r : b.plan) {
			ten += r.choice == 1;
			ten_a += r.choice == 1 && seg[r.entity_row] == 0;
			twenty += r.choice == 2;
		}
		std::printf(
		    "   budget %.0f: %zu at 10%% (%zu in segment a), %zu at 20%%, spent %.0f; objective %.1f, bound %.1f\n",
		    budget, ten, ten_a, twenty, b.usage[0], b.objective, b.bound);
		CHECK(b.usage.size() == 1 && b.usage[0] <= budget + 1e-6, "budget broken: %g > %g", b.usage[0], budget);
		CHECK(b.usage[0] >= budget - 20, "budget left unused: %g of %g", b.usage[0], budget);
		CHECK(ten_a >= ten * 95 / 100, "the 10%% offers should go to segment a: %zu of %zu", ten_a, ten);
		CHECK(b.objective <= b.bound + 1e-9 && b.bound - b.objective <= 30, "budget gap %g", b.bound - b.objective);
		CHECK(b.objective < o.objective, "a binding budget cannot beat no budget");
		OptimizeOutcome e = Opt(db, m,
		                        "EXPLAIN " + head + "MAXIMIZE SUM(prediction - " + cost + ") SUBJECT TO SUM(" + cost +
		                            ") <= " + TidyNumber(budget) + " USING MODEL spend");
		bool same = e.plan.size() == b.plan.size() && e.objective == b.objective;
		for (size_t i = 0; same && i < e.plan.size(); i++) {
			same = e.plan[i].choice == b.plan[i].choice && e.plan[i].entity_row == b.plan[i].entity_row;
		}
		CHECK(same, "EXPLAIN OPTIMIZE and OPTIMIZE solve the same plan, deterministically");
	}
	{
		// A cap on how many get any offer, written with FILTER.
		OptimizeOutcome c = Opt(db, m,
		                        head + "MAXIMIZE SUM(prediction - " + cost +
		                            ") SUBJECT TO COUNT(*) FILTER (WHERE offer != 'none') <= 100 USING MODEL spend");
		size_t offered = 0;
		for (const auto &r : c.plan) {
			offered += r.choice != 0;
		}
		CHECK(offered == 100 && c.usage[0] == 100, "at most 100 offers: %zu (usage %g)", offered, c.usage[0]);
		CHECK(c.counts.size() == 3 && c.counts[0] == n - 100, "counts per choice");
	}
	{
		// WHERE reads the lever as it stands: only customers on no offer today.
		OptimizeOutcome w =
		    Opt(db, m, head + "WHERE offer = 'none' MAXIMIZE SUM(prediction - " + cost + ") USING MODEL spend");
		size_t none_now = 0;
		for (int x : offer) {
			none_now += x == 0;
		}
		bool all_none = true;
		for (const auto &r : w.plan) {
			all_none = all_none && offer[r.entity_row] == 0;
		}
		CHECK(w.plan.size() == none_now && all_none, "WHERE offer = 'none': %zu rows, %zu today", w.plan.size(),
		      none_now);
		size_t switched = 0;
		for (const auto &r : w.plan) {
			switched += r.choice != 0;
		}
		CHECK(switched > 0, "a WHERE on the lever must not pin the choice to the current value");
	}
	{
		// MINIMIZE is the same decision from the other side.
		OptimizeOutcome mn = Opt(db, m, head + "MINIMIZE SUM(" + cost + " - prediction) USING MODEL spend");
		bool same = mn.plan.size() == o.plan.size();
		for (size_t i = 0; same && i < mn.plan.size(); i++) {
			same = mn.plan[i].choice == o.plan[i].choice;
		}
		CHECK(same && std::fabs(mn.objective + o.objective) < 1e-6 * (1 + std::fabs(o.objective)),
		      "MINIMIZE of the negation is the same plan");
	}

	std::printf("== end to end: what CHOOSE refuses\n");
	{
		// Other numeric columns take part in the arithmetic as they stand.
		OptimizeOutcome t =
		    Opt(db, m, head + "MAXIMIZE SUM(prediction - " + cost + " - tenure / tier) USING MODEL spend");
		const Frame &cf = db.At("customers");
		double shift = 0;
		for (size_t r = 0; r < n; r++) {
			shift += cf.At("tenure").num[r] / cf.At("tier").num[r];
		}
		CHECK(std::fabs((o.objective - shift) - t.objective) < 1e-6 * (1 + std::fabs(o.objective)),
		      "a per-row constant shifts the total and leaves the plan: %g vs %g", o.objective - shift, t.objective);
	}
	{
		const Column snap = db.At("customers").At("offer");
		Frame &f = db.At("customers");
		Column &ten = f.columns[(size_t)f.Find("tenure")];
		ten.valid[7] = 0;
		Refused([&] { Opt(db, m, head + "MAXIMIZE SUM(prediction - tenure) USING MODEL spend"); },
		        "1 of 3000 rows have no value for the objective: tenure is NULL", "a NULL in the objective");
		ten.valid[7] = 1;
		const Column &after = db.At("customers").At("offer");
		bool same = after.code.size() == snap.code.size();
		for (size_t i = 0; same && i < after.code.size(); i++) {
			same = after.dict.values[after.code[i]] == snap.dict.values[snap.code[i]];
		}
		CHECK(same, "a refusal after the lever was tried still puts it back");
	}
	Refused(
	    [&] {
		    Opt(db, m, "OPTIMIZE customers CHOOSE offer IN ('none', '50%') MAXIMIZE SUM(prediction) USING MODEL spend");
	    },
	    "never saw offer = '50%'", "an offer the model never saw");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE offer IN (1, 2) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "is text, so CHOOSE lists text values", "numbers for a text lever");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE spend IN (1, 2) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "is what model 'spend' predicts", "the target as the lever");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE id IN (1, 2) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "does not read 'id'", "an EXCLUDEd column as the lever");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE tier IN (1.5) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "holds whole numbers", "a fraction for a whole-number lever");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE tier IN (1, 9) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "tier = 9 is outside every value 'tier' has in 'customers' (1 to 3)", "a lever value outside the data");
	Refused([&] { Opt(db, m, "OPTIMIZE customers CHOOSE nope IN (1) MAXIMIZE SUM(prediction) USING MODEL spend"); },
	        "not a column of 'customers'", "a lever that does not exist");
	Refused(
	    [&] {
		    Opt(db, m, "OPTIMIZE customers AS c CHOOSE x.offer IN ('none') MAXIMIZE SUM(prediction) USING MODEL spend");
	    },
	    "but the plan is over 'customers'", "a lever on another relation");
	Refused([&] { Opt(db, m, head + "MAXIMIZE SUM(prediction * offer) USING MODEL spend"); }, "is text",
	        "arithmetic on a text lever");
	Refused(
	    [&] {
		    Opt(db, m,
		        head + "MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) FILTER (WHERE offer = 'none') >= 3001 USING MODEL "
		               "spend");
	    },
	    "no plan can meet COUNT(*) FILTER (WHERE offer = 'none') >= 3001: the most any plan reaches is 3000",
	    "a minimum larger than the table");
	Refused(
	    [&] {
		    Opt(db, m,
		        head + "MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) FILTER (WHERE offer = 'none') <= 10 AND "
		               "COUNT(*) FILTER (WHERE offer != 'none') <= 10 USING MODEL spend");
	    },
	    "no plan meets all of these limits together", "limits that contradict each other");
	{
		// A model that never read the lever.
		const Model blind = Train(db, "TRAIN MODEL blind PREDICT customers.spend FOR customers EXCLUDE (id, offer) "
		                              "OPTIONS (epochs = 5, hidden = 8)");
		Refused([&] { Opt(db, blind, head + "MAXIMIZE SUM(prediction) USING MODEL blind"); },
		        "does not read 'offer', so every choice would get the same prediction", "a model blind to the lever");
	}
	{
		// A text lever with more values than MAX_CATEGORIES is only a frequency.
		const Model freq = Train(db, "TRAIN MODEL freq PREDICT customers.spend FOR customers EXCLUDE (id) "
		                             "OPTIONS (epochs = 5, hidden = 8, max_categories = 2)");
		Refused([&] { Opt(db, freq, head + "MAXIMIZE SUM(prediction) USING MODEL freq"); },
		        "OPTIONS (max_categories = 3)", "a frequency-encoded lever");
	}
}

// Units sold fall with price, faster in segment b: a should be priced high and
// b low. Prices were set at random.
static void EndToEndPrice() {
	std::printf("== end to end: a numeric lever, priced for revenue\n");
	std::mt19937 rng(11);
	std::normal_distribution<double> N(0, 2);
	const size_t n = 4000;
	const double prices[4] = {5, 7, 9, 11};
	Frame f;
	f.name = "deals";
	f.nrows = n;
	Column id = MakeCol("id", ColType::INT64, n), price = MakeCol("price", ColType::DOUBLE, n);
	Column units = MakeCol("units", ColType::DOUBLE, n);
	std::vector<std::string> segs(n);
	std::vector<int> seg(n);
	for (size_t i = 0; i < n; i++) {
		seg[i] = (int)(rng() % 2);
		segs[i] = seg[i] ? "b" : "a";
		id.num[i] = double(i);
		price.num[i] = prices[rng() % 4];
		units.num[i] = 100 - (seg[i] ? 9.0 : 4.0) * price.num[i] + N(rng);
	}
	f.columns = {id, TextCol("segment", segs), price, units};
	Database db;
	db.tables.push_back(f);
	db.BuildLinks({});
	const Model m = Train(db, "TRAIN MODEL demand PREDICT deals.units FOR deals EXCLUDE (id) "
	                          "OPTIONS (epochs = 100, hidden = 32)");
	OptimizeOutcome o = Opt(db, m,
	                        "OPTIMIZE deals CHOOSE price IN (5, 7, 9, 11) MAXIMIZE SUM(prediction * price) "
	                        "USING MODEL demand");
	// Revenue p(100 - 4p) peaks at 11 of these for a; p(100 - 9p) at 5 for b.
	size_t right = 0;
	for (const auto &r : o.plan) {
		right += prices[r.choice] == (seg[r.entity_row] ? 5.0 : 11.0);
	}
	std::printf("   right price for %zu of %zu\n", right, n);
	CHECK(right >= n * 8 / 10, "right price for only %zu of %zu", right, n);
	Refused(
	    [&] {
		    Opt(db, m, "OPTIMIZE deals CHOOSE price IN (5, 13) MAXIMIZE SUM(prediction * price) USING MODEL demand");
	    },
	    "price = 13 is outside every value 'price' has in 'deals' (5 to 11)", "a price never tried");
	Refused([&] { Opt(db, m, "OPTIMIZE deals CHOOSE price IN ('high') MAXIMIZE SUM(prediction) USING MODEL demand"); },
	        "is a number, so CHOOSE lists numbers", "text for a numeric lever");
}

// Trucks with a breakdown history: the model forecasts the next 30 days, and
// OPTIMIZE picks which to service.
static Database Fleet(uint32_t seed, size_t n, bool with_offer = false) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> U(0, 1);
	Frame trucks;
	trucks.name = "trucks";
	trucks.nrows = n;
	Column id = MakeCol("id", ColType::INT64, n), ts = MakeCol("ts", ColType::TIMESTAMP, n);
	Column age = MakeCol("age", ColType::DOUBLE, n), cost = MakeCol("service_cost", ColType::DOUBLE, n);
	std::vector<double> rate(n);
	std::vector<std::string> plan(n);
	for (size_t i = 0; i < n; i++) {
		id.num[i] = double(i);
		ts.num[i] = (500.0 + U(rng) * 500.0) * DAY;
		rate[i] = U(rng);
		age.num[i] = std::floor(U(rng) * 12);
		cost.num[i] = 20 + (double)(rng() % 100);
		plan[i] = U(rng) < 0.5 ? "basic" : "care";
	}
	std::vector<double> r_tid, r_ts;
	for (size_t i = 0; i < n; i++) {
		const int past = int(rate[i] * 20);
		for (int k = 0; k < past; k++) {
			r_tid.push_back(double(i));
			r_ts.push_back(ts.num[i] - U(rng) * 300 * DAY);
		}
		const double p = with_offer ? rate[i] * (plan[i] == "care" ? 0.3 : 1.0) : rate[i];
		if (U(rng) < p) {
			r_tid.push_back(double(i));
			r_ts.push_back(ts.num[i] + (1 + U(rng) * 25) * DAY);
		}
	}
	trucks.columns = {id, ts, age, cost};
	if (with_offer) {
		trucks.columns.push_back(TextCol("plan", plan));
	}
	Frame rep;
	rep.name = "repairs";
	rep.nrows = r_tid.size();
	Column rid = MakeCol("id", ColType::INT64, rep.nrows), rt = MakeCol("truck_id", ColType::INT64, rep.nrows);
	Column rts = MakeCol("ts", ColType::TIMESTAMP, rep.nrows), hours = MakeCol("hours", ColType::DOUBLE, rep.nrows);
	for (size_t i = 0; i < rep.nrows; i++) {
		rid.num[i] = double(i);
		rt.num[i] = r_tid[i];
		rts.num[i] = r_ts[i];
		hours.num[i] = 1 + U(rng) * 4;
	}
	rep.columns = {rid, rt, rts, hours};
	Database db;
	db.tables = {trucks, rep};
	db.fks.push_back({"repairs", "truck_id", "trucks", "id"});
	db.BuildLinks({});
	return db;
}

static void EndToEndSelect() {
	std::printf("== end to end: choosing which rows to act on\n");
	Database db = Fleet(3, 600);
	const Model m = Train(db, "TRAIN MODEL breakdown PREDICT EXISTS(repairs) FOR trucks AT ts HORIZON 30 DAYS "
	                          "OPTIONS (epochs = 60, hidden = 32)");
	auto preds = RunPredict(db, m, Parse("PREDICT EXISTS(repairs) FOR trucks USING MODEL breakdown"));
	const Column &cost = db.At("trucks").At("service_cost");
	const Column &age = db.At("trucks").At("age");
	std::vector<std::pair<double, uint32_t>> value;
	for (const auto &p : preds) {
		value.emplace_back(p.value * 1000 - cost.num[p.entity_row], p.entity_row);
	}
	std::sort(
	    value.begin(), value.end(),
	    [](const std::pair<double, uint32_t> &a, const std::pair<double, uint32_t> &b) { return a.first > b.first; });
	{
		OptimizeOutcome o = Opt(db, m,
		                        "OPTIMIZE trucks MAXIMIZE SUM(prediction * 1000 - service_cost) "
		                        "SUBJECT TO COUNT(*) <= 5 USING MODEL breakdown");
		bool same = o.plan.size() == 5;
		for (size_t i = 0; same && i < 5; i++) {
			same = o.plan[i].entity_row == value[i].second && std::fabs(o.plan[i].objective - value[i].first) < 1e-9;
		}
		CHECK(same, "five service bays: the five most valuable trucks, best first");
		CHECK(o.exact, "a count limit is solved exactly");
		CHECK(o.counts.size() == 2 && o.counts[1] == 5 && o.counts[0] == preds.size() - 5, "acted on / left alone");
		bool anchored = true;
		for (const auto &r : o.plan) {
			anchored = anchored && r.anchor == db.At("trucks").At("ts").num[r.entity_row];
		}
		CHECK(anchored, "each planned row carries the anchor its prediction was made from");
	}
	{
		// No limit: every truck worth servicing, and none that is not.
		OptimizeOutcome o =
		    Opt(db, m, "OPTIMIZE trucks MAXIMIZE SUM(prediction * 1000 - service_cost) USING MODEL breakdown");
		size_t positive = 0;
		for (const auto &v : value) {
			positive += v.first > 0;
		}
		CHECK(o.plan.size() == positive, "no limit: %zu planned, %zu worth it", o.plan.size(), positive);
	}
	{
		// A budget: a knapsack, against its dynamic programme.
		OptimizeOutcome o = Opt(db, m,
		                        "OPTIMIZE trucks MAXIMIZE SUM(prediction * 1000 - service_cost) "
		                        "SUBJECT TO SUM(service_cost) <= 600 USING MODEL breakdown");
		std::vector<double> val;
		std::vector<int> w;
		double vmax = 0;
		for (const auto &p : preds) {
			val.push_back(p.value * 1000 - cost.num[p.entity_row]);
			w.push_back((int)cost.num[p.entity_row]);
			vmax = std::max(vmax, val.back());
		}
		const double opt = KnapsackDP(val, w, 600);
		std::printf("   budget 600: plan %.2f, optimum %.2f, bound %.2f, spent %.0f\n", o.objective, opt, o.bound,
		            o.usage[0]);
		CHECK(o.usage[0] <= 600, "budget broken: %g", o.usage[0]);
		CHECK(o.objective <= opt + 1e-6 && o.bound >= opt - 1e-6, "plan %g, optimum %g, bound %g", o.objective, opt,
		      o.bound);
		CHECK(opt - o.objective <= vmax, "gap %g over one truck's value %g", opt - o.objective, vmax);
	}
	{
		// MINIMIZE with a minimum: the three cheapest.
		OptimizeOutcome o =
		    Opt(db, m, "OPTIMIZE trucks MINIMIZE SUM(service_cost) SUBJECT TO COUNT(*) >= 3 USING MODEL breakdown");
		std::vector<double> costs;
		for (const auto &p : preds) {
			costs.push_back(cost.num[p.entity_row]);
		}
		std::sort(costs.begin(), costs.end());
		CHECK(o.plan.size() == 3 && std::fabs(o.objective - (costs[0] + costs[1] + costs[2])) < 1e-9,
		      "three cheapest: %zu rows costing %g", o.plan.size(), o.objective);
		bool ascending = true;
		for (size_t i = 1; i < o.plan.size(); i++) {
			ascending = ascending && o.plan[i - 1].objective <= o.plan[i].objective;
		}
		CHECK(ascending && o.bound <= o.objective + 1e-9, "MINIMIZE: cheapest first, and the bound is a floor");
	}
	{
		// A limit on the model's own answer: no truck below even odds.
		OptimizeOutcome o = Opt(db, m,
		                        "OPTIMIZE trucks MAXIMIZE SUM(prediction * 1000 - service_cost) "
		                        "SUBJECT TO COUNT(*) FILTER (WHERE prediction < 0.5) <= 0 USING MODEL breakdown");
		bool none_low = true;
		for (const auto &r : o.plan) {
			none_low = none_low && r.prediction >= 0.5;
		}
		CHECK(none_low && !o.plan.empty(), "FILTER (WHERE prediction < 0.5) <= 0 kept low-risk trucks out");
	}
	{
		// A FILTERed limit and a WHERE together.
		OptimizeOutcome o =
		    Opt(db, m,
		        "OPTIMIZE trucks WHERE age >= 2 MAXIMIZE SUM(prediction) "
		        "SUBJECT TO COUNT(*) <= 10 AND COUNT(*) FILTER (WHERE age > 8) <= 2 USING MODEL breakdown");
		size_t old = 0;
		bool in_where = true;
		for (const auto &r : o.plan) {
			old += age.num[r.entity_row] > 8;
			in_where = in_where && age.num[r.entity_row] >= 2;
		}
		CHECK(o.plan.size() == 10 && old <= 2 && in_where, "WHERE and FILTER respected: %zu rows, %zu old",
		      o.plan.size(), old);
		CHECK(o.usage.size() == 2 && o.usage[0] == 10 && o.usage[1] == (double)old, "usage per limit");
	}
	{
		Database copy = db;
		copy.At("trucks").columns[(size_t)copy.At("trucks").Find("service_cost")].valid[4] = 0;
		Refused([&] { Opt(copy, m, "OPTIMIZE trucks MAXIMIZE SUM(prediction - service_cost) USING MODEL breakdown"); },
		        "service_cost is NULL", "a NULL cost");
		// ... and leaving the row out is the way past it.
		OptimizeOutcome o =
		    Opt(copy, m,
		        "OPTIMIZE trucks WHERE service_cost IS NOT NULL MAXIMIZE SUM(prediction - service_cost) "
		        "USING MODEL breakdown");
		CHECK(o.candidates == preds.size() - 1, "WHERE ... IS NOT NULL leaves exactly that row out");
		Refused([&] { Opt(db, m, "OPTIMIZE trucks MAXIMIZE SUM(1 / (age - age)) USING MODEL breakdown"); },
		        "divides by zero", "a division by zero");
		Refused(
		    [&] {
			    Opt(db, m, "OPTIMIZE trucks MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) >= 601 USING MODEL breakdown");
		    },
		    "the most any plan reaches is 600", "more rows than there are");
		Refused(
		    [&] {
			    Opt(db, m,
			        "OPTIMIZE trucks MAXIMIZE SUM(prediction) SUBJECT TO SUM(service_cost) <= -1 USING MODEL "
			        "breakdown");
		    },
		    "no plan can meet SUM(service_cost) <= -1: the least any plan reaches is 0", "a negative budget");
		Refused(
		    [&] {
			    Opt(db, m,
			        "OPTIMIZE trucks MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) <= 2 AND COUNT(*) >= 3 USING MODEL "
			        "breakdown");
		    },
		    "no plan meets all of these limits together", "contradicting counts");
		Refused([&] { Opt(db, m, "OPTIMIZE trucks MAXIMIZE SUM(ts) USING MODEL breakdown"); }, "is a timestamp",
		        "arithmetic on a timestamp");
		Refused([&] { Opt(db, m, "OPTIMIZE trucks CHOOSE ts IN (1) MAXIMIZE SUM(prediction) USING MODEL breakdown"); },
		        "is the anchor", "the anchor as the lever");
		Refused([&] { Opt(db, m, "OPTIMIZE trucks WHERE nope = 1 MAXIMIZE SUM(prediction) USING MODEL breakdown"); },
		        "does not exist on 'trucks'", "a WHERE on a missing column");
	}
	{
		// Nothing matches: an empty plan, not an error, unless a minimum says otherwise.
		OptimizeOutcome o =
		    Opt(db, m, "OPTIMIZE trucks WHERE age > 100 MAXIMIZE SUM(prediction) USING MODEL breakdown");
		CHECK(o.plan.empty() && o.candidates == 0 && o.objective == 0, "no candidates: an empty plan");
		Refused(
		    [&] {
			    Opt(db, m,
			        "OPTIMIZE trucks WHERE age > 100 MAXIMIZE SUM(prediction) SUBJECT TO COUNT(*) >= 1 USING MODEL "
			        "breakdown");
		    },
		    "the most any plan reaches is 0", "a minimum over no candidates");
	}
	{
		// The same statement twice is the same plan.
		const std::string sql =
		    "OPTIMIZE trucks MAXIMIZE SUM(prediction * 1000 - service_cost) "
		    "SUBJECT TO SUM(service_cost) <= 900 AND COUNT(*) FILTER (WHERE age > 6) <= 3 USING MODEL breakdown";
		OptimizeOutcome a = Opt(db, m, sql), b = Opt(db, m, sql);
		bool same = a.plan.size() == b.plan.size() && a.objective == b.objective && a.bound == b.bound;
		for (size_t i = 0; same && i < a.plan.size(); i++) {
			same = a.plan[i].entity_row == b.plan[i].entity_row;
		}
		CHECK(same, "deterministic");
	}
}

// A graph model choosing a lever on the entity, with the effect it learned from
// the history: trucks on the care plan break down far less.
static void EndToEndSage() {
	std::printf("== end to end: a GraphSAGE model behind CHOOSE\n");
	Database db = Fleet(17, 1500, true);
	const Model m = Train(db, "TRAIN MODEL risk PREDICT EXISTS(repairs) FOR trucks AT ts HORIZON 30 DAYS "
	                          "OPTIONS (arch = 'sage', epochs = 60, hidden = 32)");
	OptimizeOutcome o = Opt(db, m,
	                        "OPTIMIZE trucks CHOOSE plan IN ('basic', 'care') "
	                        "MINIMIZE SUM(prediction * 1000 + CASE plan WHEN 'care' THEN 150 ELSE 0 END) "
	                        "USING MODEL risk");
	CHECK(o.plan.size() == 1500, "every truck gets a plan");
	// The model must have learned that care lowers the risk, on average.
	Frame &f = db.At("trucks");
	Column &pc = f.columns[(size_t)f.Find("plan")];
	const Column saved = pc;
	double mean[2] = {0, 0};
	for (int k = 0; k < 2; k++) {
		const uint32_t code = pc.dict.Intern(k ? "care" : "basic");
		for (auto &c : pc.code) {
			c = code;
		}
		auto p = RunPredict(db, m, Parse("PREDICT EXISTS(repairs) FOR trucks USING MODEL risk"));
		for (const auto &x : p) {
			mean[k] += x.value / double(p.size());
		}
	}
	pc = saved;
	std::printf("   mean risk on basic %.3f, on care %.3f\n", mean[0], mean[1]);
	CHECK(mean[1] < mean[0] - 0.1, "the graph model should see care lowering the risk");
	// No limit: each row's own better option.
	size_t care = 0;
	bool argmax = true;
	for (const auto &r : o.plan) {
		care += r.choice == 1;
		argmax = argmax && std::isfinite(r.prediction);
	}
	CHECK(care > 0 && care < o.plan.size() && argmax, "care for the risky, basic for the rest: %zu on care", care);
	CHECK(o.exact, "no limits: exact");
}

int main() {
	Grammar();
	Expressions();
	Simplex();
	SolverVsBrute();
	SolverAtScale();
	Regressions();
	EndToEndChoose();
	EndToEndPrice();
	EndToEndSelect();
	EndToEndSage();
	std::printf("%d passed, %d failed\n", oks, fails);
	return fails ? 1 : 0;
}
