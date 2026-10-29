// optimize.hpp — OPTIMIZE over a model's predictions: objectives, limits, the lever, the plan.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/predict.hpp"
#include "pql/solver.hpp"

namespace pql {

// A number as a person would write it, for messages.
inline std::string TidyNumber(double d) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.10g", d);
	return buf;
}

// Resolve the columns an expression names against the entity frame, and refuse
// what arithmetic cannot use: text, timestamps, and a qualifier that names a
// different relation.
inline void BindExpr(Expr *e, const Frame &entity, const std::string &alias, const std::string &where) {
	if (!e) {
		return;
	}
	if (e->kind == Expr::Kind::COLUMN) {
		if (!e->ref.qualifier.empty()) {
			const std::string q = ToUpper(e->ref.qualifier);
			if (q != ToUpper(entity.name) && (alias.empty() || q != ToUpper(alias))) {
				throw std::runtime_error("pql: the " + where + " refers to '" + e->ref.ToString() +
				                         "', but it is evaluated on '" + entity.name + "'" +
				                         (alias.empty() ? "" : " (aliased " + alias + ")"));
			}
		}
		const int c = entity.Find(e->ref.name);
		if (c < 0) {
			throw std::runtime_error("pql: the " + where + " refers to column '" + e->ref.name +
			                         "', which does not exist on '" + entity.name + "'");
		}
		const Column &col = entity.columns[(size_t)c];
		if (col.type == ColType::CATEGORY) {
			throw std::runtime_error("pql: '" + col.name + "' is text, so the " + where +
			                         " cannot do arithmetic with it; compare it inside CASE WHEN instead");
		}
		if (col.type == ColType::TIMESTAMP) {
			throw std::runtime_error("pql: '" + col.name + "' is a timestamp, which the " + where +
			                         " cannot do arithmetic with");
		}
		e->col = c;
		return;
	}
	for (const auto &w : e->when) {
		ValidateFilterScope(w.get(), entity, alias, where.c_str());
	}
	for (auto &a : e->args) {
		BindExpr(a.get(), entity, alias, where);
	}
}

// Why a value did not exist, for the message that refuses it.
struct ExprEval {
	bool missing = false;
	std::string why;
	void Miss(const std::string &reason) {
		if (!missing) {
			missing = true;
			why = reason;
		}
	}
};

// The value of an expression on one row. A NULL column or a division by zero
// leaves it undefined. SQL's SUM would quietly skip such a row, which in a plan
// means treating an unknown cost as free, so the caller refuses it instead.
//
// One departure from SQL: a CASE with no ELSE and no matching WHEN is 0, not
// NULL. Every expression here sits inside a SUM, where the two mean the same.
inline double EvalExpr(const Expr *e, const Frame &f, size_t row, double prediction, ExprEval &ev) {
	switch (e->kind) {
	case Expr::Kind::NUMBER:
		return e->number;
	case Expr::Kind::PREDICTION:
		if (!std::isfinite(prediction)) {
			ev.Miss("the model returned no prediction");
			return 0.0;
		}
		return prediction;
	case Expr::Kind::COLUMN: {
		const Column &c = f.columns[(size_t)e->col];
		if (!c.valid[row]) {
			ev.Miss(c.name + " is NULL");
			return 0.0;
		}
		return c.num[row];
	}
	case Expr::Kind::NEG:
		return -EvalExpr(e->args[0].get(), f, row, prediction, ev);
	case Expr::Kind::CASE:
		for (size_t i = 0; i < e->when.size(); i++) {
			if (EvalFilter(e->when[i].get(), f, row, prediction)) {
				return EvalExpr(e->args[i].get(), f, row, prediction, ev);
			}
		}
		return e->has_else ? EvalExpr(e->args.back().get(), f, row, prediction, ev) : 0.0;
	default:
		break;
	}
	const double l = EvalExpr(e->args[0].get(), f, row, prediction, ev);
	const double r = EvalExpr(e->args[1].get(), f, row, prediction, ev);
	double v = 0;
	switch (e->kind) {
	case Expr::Kind::ADD:
		v = l + r;
		break;
	case Expr::Kind::SUB:
		v = l - r;
		break;
	case Expr::Kind::MUL:
		v = l * r;
		break;
	default:
		if (r == 0.0) {
			ev.Miss("it divides by zero");
			return 0.0;
		}
		v = l / r;
		break;
	}
	if (!std::isfinite(v)) {
		ev.Miss("the value is too large to represent");
		return 0.0;
	}
	return v;
}

inline void BindAggregate(OptAggregate &a, const Frame &entity, const std::string &alias, const std::string &where) {
	ValidateFilterScope(a.filter.get(), entity, alias, where.c_str());
	BindExpr(a.expr.get(), entity, alias, where);
}

// One row's contribution to a total: zero when its FILTER leaves it out.
inline double EvalAggregateRow(const OptAggregate &a, const Frame &f, size_t row, double prediction, ExprEval &ev) {
	if (a.filter && !EvalFilter(a.filter.get(), f, row, prediction)) {
		return 0.0;
	}
	return a.count ? 1.0 : EvalExpr(a.expr.get(), f, row, prediction, ev);
}

// Check the column CHOOSE names, and its values, against what the model can
// actually answer for. Returns its position on the entity frame.
//
// OPTIMIZE asks the model "what if this row's lever were X?" by setting the
// column to X and predicting again. That question only has an answer when the
// model reads the column, and only for values it saw while training: an unseen
// category sets no input at all, and a number outside the data is an
// extrapolation the model was never fitted on.
inline int ResolveLever(const Frame &entity, const Model &model, const Statement &stmt, const OptimizeSpec &spec) {
	const Ref &lv = spec.lever;
	if (!lv.qualifier.empty()) {
		const std::string q = ToUpper(lv.qualifier);
		if (q != ToUpper(entity.name) && (stmt.entity_alias.empty() || q != ToUpper(stmt.entity_alias))) {
			throw std::runtime_error("pql: CHOOSE names '" + lv.ToString() + "', but the plan is over '" + entity.name +
			                         "'");
		}
	}
	const int c = entity.Find(lv.name);
	if (c < 0) {
		throw std::runtime_error("pql: CHOOSE names '" + lv.name + "', which is not a column of '" + entity.name + "'");
	}
	const Column &col = entity.columns[(size_t)c];
	const PredictExpr &tgt = model.spec_stmt.target;
	if (!IsForecast(tgt.kind) && ToUpper(tgt.ref.name) == ToUpper(col.name)) {
		throw std::runtime_error("pql: '" + col.name + "' is what model '" + model.name +
		                         "' predicts, so it cannot also be what OPTIMIZE chooses");
	}
	if (!stmt.anchor_is_literal) {
		Statement eff = model.spec_stmt;
		if (!stmt.anchor.name.empty()) {
			eff.anchor = stmt.anchor;
		}
		if (ResolveAnchorColumn(entity, eff) == c) {
			throw std::runtime_error("pql: '" + col.name +
			                         "' is the anchor, the moment each prediction is made from, not a decision");
		}
	}
	if (col.type == ColType::TIMESTAMP) {
		throw std::runtime_error("pql: '" + col.name +
		                         "' is a timestamp; CHOOSE picks a value for a column the model "
		                         "reads as an input, and timestamps are not inputs");
	}
	const FeatureSpec::CatCol *cat = nullptr;
	bool numeric_input = false;
	auto same = [&](const std::string &name) {
		return ToUpper(name) == ToUpper(col.name);
	};
	{
		std::vector<std::string> numeric;
		const std::vector<FeatureSpec::CatCol> *cats = nullptr;
		model.learner->EntityInputs(model.features, numeric, cats);
		for (const auto &name : numeric) {
			numeric_input = numeric_input || same(name);
		}
		if (cats) {
			for (const auto &cc : *cats) {
				if (same(cc.name)) {
					cat = &cc;
				}
			}
		}
	}
	if (!numeric_input && !cat) {
		throw std::runtime_error("pql: model '" + model.name + "' does not read '" + col.name +
		                         "', so every choice would get the same prediction. A column is left out of a "
		                         "model when it is a key, constant, EXCLUDEd, or nearly unique per row; retrain "
		                         "with it as an input to choose it");
	}
	if (cat) {
		for (const auto &v : spec.choices) {
			if (v.kind != Literal::Kind::STRING) {
				throw std::runtime_error("pql: '" + col.name + "' is text, so CHOOSE lists text values, e.g. CHOOSE " +
				                         col.name + " IN ('a', 'b')");
			}
		}
		if (!cat->one_hot) {
			throw std::runtime_error("pql: '" + col.name +
			                         "' has more distinct values than MAX_CATEGORIES, so model '" + model.name +
			                         "' only knows how common each value is, not what it does. Retrain "
			                         "with OPTIONS (max_categories = " +
			                         std::to_string(cat->labels.size()) + ") to choose it");
		}
		for (const auto &v : spec.choices) {
			bool seen = false;
			for (const auto &l : cat->labels) {
				seen = seen || l == v.text;
			}
			if (!seen) {
				throw std::runtime_error("pql: model '" + model.name + "' never saw " + col.name + " = " +
				                         v.ToString() + " while training, so it has no idea what that choice does");
			}
		}
		return c;
	}
	for (const auto &v : spec.choices) {
		if (v.kind != Literal::Kind::NUMBER) {
			throw std::runtime_error("pql: '" + col.name + "' is a number, so CHOOSE lists numbers, e.g. CHOOSE " +
			                         col.name + " IN (10, 20)");
		}
	}
	double lo = std::numeric_limits<double>::infinity(), hi = -lo;
	for (size_t r = 0; r < entity.nrows; r++) {
		if (col.valid[r]) {
			lo = std::min(lo, col.num[r]);
			hi = std::max(hi, col.num[r]);
		}
	}
	for (const auto &v : spec.choices) {
		if ((col.type == ColType::INT64 || col.type == ColType::BOOL) && v.number != std::floor(v.number)) {
			throw std::runtime_error("pql: '" + col.name + "' holds whole numbers, so " + v.ToString() +
			                         " is not a value it can take");
		}
		if (!(v.number >= lo && v.number <= hi)) {
			throw std::runtime_error(
			    "pql: " + col.name + " = " + v.ToString() + " is outside every value '" + col.name + "' has in '" +
			    entity.name + "'" +
			    (std::isfinite(lo) ? " (" + TidyNumber(lo) + " to " + TidyNumber(hi) + ")" : std::string(" (none)")) +
			    "; a model cannot know what happens there");
		}
	}
	return c;
}

struct OptimizeOutcome {
	bool choosing = false;
	bool maximize = true;
	int lever_col = -1;           // CHOOSE: the column on the entity frame
	std::vector<Literal> choices; // CHOOSE: as written
	struct Row {
		uint32_t entity_row = 0;
		double anchor = 0;
		int choice = 0; // CHOOSE: index into `choices`; otherwise 1, "act on it"
		double prediction = 0;
		double objective = 0; // the row's share of the total
	};
	// CHOOSE: every candidate, in table order. Otherwise the rows to act on, the
	// most valuable first.
	std::vector<Row> plan;
	size_t candidates = 0;
	double objective = 0; // the plan's total, in the direction the statement asked
	double bound = 0;     // no plan that meets the limits does better
	bool exact = false;
	bool has_status_quo = false; // CHOOSE: the total with every lever as it stands
	double status_quo = 0;
	std::vector<double> usage;  // per constraint, as written
	std::vector<size_t> counts; // CHOOSE: rows per choice; otherwise {left alone, acted on}
};

inline OptimizeOutcome RunOptimize(Database &db, const Model &model, const Statement &stmt,
                                   const CancelCheck &cancelled = CancelCheck(),
                                   const OptSettings &settings = OptSettings()) {
	if (stmt.kind != StmtKind::OPTIMIZE) {
		throw std::runtime_error("pql: internal error: RunOptimize needs an OPTIMIZE statement");
	}
	Frame &entity = db.At(stmt.entity_table);
	const std::string &alias = stmt.entity_alias;
	ValidateFilterScope(stmt.filter.get(), entity, alias, "WHERE");
	// A copy, so its columns can be bound to this frame.
	OptimizeSpec spec = stmt.optimize;
	BindAggregate(spec.objective, entity, alias, "objective");
	std::vector<std::string> labels;
	labels.push_back("objective");
	for (auto &c : spec.constraints) {
		labels.push_back("constraint " + c.ToString());
		BindAggregate(c.total, entity, alias, labels.back());
	}

	OptimizeOutcome out;
	out.choosing = spec.Choosing();
	out.maximize = spec.maximize;
	out.choices = spec.choices;
	const int lever = out.choosing ? ResolveLever(entity, model, stmt, spec) : -1;
	out.lever_col = lever;

	// The rows a plan may act on, decided on the data as it stands: a WHERE that
	// names the lever reads its current value, not the one being tried.
	std::vector<uint8_t> candidate(entity.nrows, 0);
	for (size_t r = 0; r < entity.nrows; r++) {
		candidate[r] = EvalFilter(stmt.filter.get(), entity, r) ? 1 : 0;
	}
	Statement ps;
	ps.kind = StmtKind::PREDICT;
	ps.model = model.name;
	ps.entity_table = stmt.entity_table;
	ps.entity_alias = alias;
	ps.anchor = stmt.anchor;
	ps.anchor_literal = stmt.anchor_literal;
	ps.anchor_is_literal = stmt.anchor_is_literal;

	// The lever is overwritten once per choice and put back however this
	// returns, so the frame leaves as it came.
	struct Restore {
		Column *col = nullptr;
		std::vector<double> num;
		std::vector<uint32_t> code;
		std::vector<uint8_t> valid;
		Dictionary dict; // a value tried may have been added to it
		void Put() {
			if (col) {
				col->num = num;
				col->code = code;
				col->valid = valid;
				col->dict = dict;
			}
		}
		~Restore() {
			Put();
		}
	} restore;
	if (lever >= 0) {
		Column &lc = entity.columns[(size_t)lever];
		restore.col = &lc;
		restore.num = lc.num;
		restore.code = lc.code;
		restore.valid = lc.valid;
		restore.dict = lc.dict;
	}

	const size_t K = out.choosing ? spec.choices.size() : 1;
	const size_t m = spec.constraints.size();
	std::vector<Prediction> rows;
	std::vector<double> pred, obj;           // n * K
	std::vector<std::vector<double>> con(m); // per constraint, n * K
	size_t missing_rows = 0;
	std::string missing_why, missing_where;
	std::vector<uint8_t> row_missing;
	for (size_t k = 0; k < K; k++) {
		if (lever >= 0) {
			Column &lc = entity.columns[(size_t)lever];
			const Literal &v = spec.choices[k];
			const uint32_t code = lc.type == ColType::CATEGORY ? lc.dict.Intern(v.text) : 0u;
			for (size_t r = 0; r < entity.nrows; r++) {
				if (!candidate[r]) {
					continue;
				}
				lc.valid[r] = 1;
				if (lc.type == ColType::CATEGORY) {
					lc.code[r] = code;
				} else {
					lc.num[r] = v.number;
				}
			}
		}
		std::vector<Prediction> ps_k = RunPredict(db, model, ps, cancelled, &candidate);
		if (k == 0) {
			rows = ps_k;
			pred.assign(rows.size() * K, 0.0);
			obj.assign(rows.size() * K, 0.0);
			for (auto &c : con) {
				c.assign(rows.size() * K, 0.0);
			}
			row_missing.assign(rows.size(), 0);
		} else if (ps_k.size() != rows.size()) {
			throw std::runtime_error("pql: internal error: choices produced different candidate rows");
		}
		for (size_t i = 0; i < rows.size(); i++) {
			if (ps_k[i].entity_row != rows[i].entity_row) {
				throw std::runtime_error("pql: internal error: choices produced different candidate rows");
			}
			const size_t r = rows[i].entity_row;
			const double p = ps_k[i].value;
			pred[i * K + k] = p;
			for (size_t j = 0; j <= m; j++) {
				ExprEval ev;
				const OptAggregate &agg = j == 0 ? spec.objective : spec.constraints[j - 1].total;
				const double val = EvalAggregateRow(agg, entity, r, p, ev);
				if (ev.missing) {
					if (!row_missing[i]) {
						row_missing[i] = 1;
						missing_rows++;
					}
					if (missing_why.empty()) {
						missing_why = ev.why;
						missing_where = labels[j];
					}
				}
				(j == 0 ? obj : con[j - 1])[i * K + k] = val;
			}
		}
	}
	if (missing_rows > 0) {
		throw std::runtime_error("pql: " + std::to_string(missing_rows) + " of " + std::to_string(rows.size()) +
		                         " rows have no value for the " + missing_where + ": " + missing_why +
		                         " there. Leave them out with WHERE, or give them a value");
	}
	const size_t n = rows.size();
	out.candidates = n;

	// What the rows are worth with every lever left as it is, from the same model.
	if (lever >= 0) {
		restore.Put();
		std::vector<Prediction> now = RunPredict(db, model, ps, cancelled, &candidate);
		bool ok = now.size() == n;
		double total = 0;
		for (size_t i = 0; ok && i < n; i++) {
			ExprEval ev;
			ok = now[i].entity_row == rows[i].entity_row;
			total += EvalAggregateRow(spec.objective, entity, rows[i].entity_row, now[i].value, ev);
			ok = ok && !ev.missing;
		}
		out.has_status_quo = ok;
		out.status_quo = ok ? total : 0.0;
	}

	OptProblem P;
	P.n = n;
	const double sgn = spec.maximize ? 1.0 : -1.0;
	if (out.choosing) {
		P.K = (int)K;
		P.v.resize(n * K);
		for (size_t o = 0; o < n * K; o++) {
			P.v[o] = sgn * obj[o];
		}
		P.a.assign(m, std::vector<double>(n * K));
		for (size_t j = 0; j < m; j++) {
			const double cs = spec.constraints[j].at_most ? 1.0 : -1.0;
			for (size_t o = 0; o < n * K; o++) {
				P.a[j][o] = cs * con[j][o];
			}
		}
	} else {
		// Option 0 leaves the row alone and is worth nothing; option 1 acts on it.
		P.K = 2;
		P.v.assign(n * 2, 0.0);
		for (size_t i = 0; i < n; i++) {
			P.v[i * 2 + 1] = sgn * obj[i];
		}
		P.a.assign(m, std::vector<double>(n * 2, 0.0));
		for (size_t j = 0; j < m; j++) {
			const double cs = spec.constraints[j].at_most ? 1.0 : -1.0;
			for (size_t i = 0; i < n; i++) {
				P.a[j][i * 2 + 1] = cs * con[j][i];
			}
		}
	}
	for (const auto &c : spec.constraints) {
		P.b.push_back((c.at_most ? 1.0 : -1.0) * c.limit.number);
	}
	const OptSolution sol = SolveOpt(P, settings, cancelled);
	const size_t KP = (size_t)P.K;
	auto total_of = [&](size_t j, const std::vector<int> &choice) {
		double s = 0;
		for (size_t i = 0; i < n; i++) {
			if (out.choosing) {
				s += con[j][i * K + (size_t)choice[i]];
			} else if (choice[i] == 1) {
				s += con[j][i];
			}
		}
		return s;
	};
	if (!sol.feasible) {
		if (sol.infeasible_proven) {
			// One limit that no plan can meet even on its own is the clearest thing
			// to say. Otherwise it is the combination.
			for (size_t j = 0; j < m; j++) {
				const OptConstraint &c = spec.constraints[j];
				double extreme = 0;
				for (size_t i = 0; i < n; i++) {
					double e =
					    c.at_most ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
					for (size_t k = 0; k < KP; k++) {
						const double v = out.choosing ? con[j][i * K + k] : (k == 1 ? con[j][i] : 0.0);
						e = c.at_most ? std::min(e, v) : std::max(e, v);
					}
					extreme += e;
				}
				const bool impossible = c.at_most ? extreme > c.limit.number + 1e-9 * std::max(1.0, std::fabs(extreme))
				                                  : extreme < c.limit.number - 1e-9 * std::max(1.0, std::fabs(extreme));
				if (impossible) {
					throw std::runtime_error(
					    "pql: no plan can meet " + c.ToString() + ": " +
					    (c.at_most ? "the least any plan reaches is " : "the most any plan reaches is ") +
					    TidyNumber(extreme));
				}
			}
			std::string all;
			for (size_t j = 0; j < m; j++) {
				all += (j ? ", " : "") + spec.constraints[j].ToString();
			}
			throw std::runtime_error("pql: no plan meets all of these limits together: " + all);
		}
		std::string worst;
		double worst_by = 0;
		for (size_t j = 0; j < m; j++) {
			const OptConstraint &c = spec.constraints[j];
			const double used = total_of(j, sol.choice);
			const double miss = c.at_most ? used - c.limit.number : c.limit.number - used;
			if (miss > worst_by) {
				worst_by = miss;
				worst = c.ToString();
			}
		}
		// Not a proof: the search stopped short, so say so rather than call the
		// limits impossible.
		throw std::runtime_error("pql: found no plan that meets every limit, though one may exist: the search stopped "
		                         "before it could tell. The closest misses " +
		                         worst + " by " + TidyNumber(worst_by) +
		                         ". Loosening that limit, or planning fewer groups per statement, may help");
	}

	out.counts.assign(out.choosing ? K : 2, 0);
	out.objective = 0;
	for (size_t i = 0; i < n; i++) {
		const int c = sol.choice[i];
		out.counts[(size_t)c]++;
		if (!out.choosing && c == 0) {
			continue;
		}
		OptimizeOutcome::Row row;
		row.entity_row = rows[i].entity_row;
		row.anchor = rows[i].anchor;
		row.choice = c;
		const size_t o = out.choosing ? i * K + (size_t)c : i;
		row.prediction = pred[o];
		row.objective = obj[o];
		out.objective += row.objective;
		out.plan.push_back(row);
	}
	if (!out.choosing) {
		// The most valuable first; ties stay in table order.
		std::stable_sort(out.plan.begin(), out.plan.end(),
		                 [&](const OptimizeOutcome::Row &l, const OptimizeOutcome::Row &r) {
			                 return spec.maximize ? l.objective > r.objective : l.objective < r.objective;
		                 });
	}
	for (size_t j = 0; j < m; j++) {
		out.usage.push_back(total_of(j, sol.choice));
	}
	out.bound = sgn * sol.bound;
	out.exact = sol.exact;
	return out;
}

} // namespace pql
