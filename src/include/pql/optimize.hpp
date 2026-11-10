// optimize.hpp — OPTIMIZE over a model's predictions: objectives, limits, the lever, the plan.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/assign.hpp"
#include "pql/pairing.hpp"
#include "pql/predict.hpp"
#include "pql/solver.hpp"

namespace pql {

// A number as a person would write it, for messages.
inline std::string TidyNumber(double d) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.10g", d);
	return buf;
}

// The rows an ASSIGN expression or filter may read besides the task's own:
// the resource it is placed on and, in SETUP, the task placed before it.
struct Frames {
	const Frame &task;
	const std::string &alias;
	const Frame *resource; // resource.<column>
	bool prev;             // prev.<column>, a row of the task frame
	Frames(const Frame &t, const std::string &a, const Frame *r, bool p) : task(t), alias(a), resource(r), prev(p) {
	}
};

// Which frame a reference names, by its qualifier: 0 the task, 1 the previous
// task, 2 the resource. Refuses what the context has no row for.
inline int FrameOf(const Ref &r, const Frames &fr, const std::string &where) {
	const std::string q = ToUpper(r.qualifier);
	if (q == "RESOURCE") {
		if (!fr.resource) {
			throw std::runtime_error("pql: the " + where + " refers to '" + r.ToString() +
			                         "', but there is no resource row there");
		}
		return 2;
	}
	if (q == "PREV") {
		if (!fr.prev) {
			throw std::runtime_error("pql: the " + where + " refers to '" + r.ToString() +
			                         "', but only SETUP has a previous task");
		}
		return 1;
	}
	if (!q.empty() && q != ToUpper(fr.task.name) && (fr.alias.empty() || q != ToUpper(fr.alias))) {
		throw std::runtime_error("pql: the " + where + " refers to '" + r.ToString() + "', but it is evaluated on '" +
		                         fr.task.name + "'" + (fr.alias.empty() ? "" : " (aliased " + fr.alias + ")"));
	}
	return 0;
}

inline const Frame &FrameAt(int which, const Frames &fr) {
	return which == 2 ? *fr.resource : fr.task;
}

// A filter over a task row and, where the context has them, the resource row
// and the previous task: every column it names must exist on its frame.
inline void ValidateFilterFrames(const FilterNode *f, const Frames &fr, const std::string &where) {
	if (!f) {
		return;
	}
	if (f->kind != FilterNode::Kind::CMP) {
		for (const auto &c : f->children) {
			ValidateFilterFrames(c.get(), fr, where);
		}
		return;
	}
	if (f->prediction) {
		return;
	}
	for (const Ref *r : {&f->ref, f->rhs_is_ref ? &f->rhs : nullptr}) {
		if (!r) {
			continue;
		}
		const Frame &frame = FrameAt(FrameOf(*r, fr, where), fr);
		if (frame.Find(r->name) < 0) {
			throw std::runtime_error("pql: the " + where + " refers to column '" + r->name +
			                         "', which does not exist on '" + frame.name + "'");
		}
	}
}

// Resolve the columns an expression names, and refuse what arithmetic cannot
// use: text, timestamps, and a qualifier that names a relation the context
// has no row for.
inline void BindExprFrames(Expr *e, const Frames &fr, const std::string &where) {
	if (!e) {
		return;
	}
	if (e->kind == Expr::Kind::COLUMN) {
		e->frame = FrameOf(e->ref, fr, where);
		const Frame &frame = FrameAt(e->frame, fr);
		const int c = frame.Find(e->ref.name);
		if (c < 0) {
			throw std::runtime_error("pql: the " + where + " refers to column '" + e->ref.name +
			                         "', which does not exist on '" + frame.name + "'");
		}
		const Column &col = frame.columns[(size_t)c];
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
		ValidateFilterFrames(w.get(), fr, where);
	}
	for (auto &a : e->args) {
		BindExprFrames(a.get(), fr, where);
	}
}

// The task row alone: what CHOOSE and the plain OPTIMIZE evaluate on.
inline void BindExpr(Expr *e, const Frame &entity, const std::string &alias, const std::string &where) {
	BindExprFrames(e, Frames {entity, alias, nullptr, false}, where);
}

// Why a value did not exist, for the message that refuses it; and, for an
// ASSIGN expression, the other rows it may read and the plan values of the
// placement being evaluated.
struct ExprEval {
	bool missing = false;
	std::string why;
	const Frame *resource = nullptr;
	size_t resource_row = 0;
	const Frame *prev = nullptr;
	size_t prev_row = 0;
	double plan_setup = 0, plan_duration = 0, plan_begin = 0;
	void Miss(const std::string &reason) {
		if (!missing) {
			missing = true;
			why = reason;
		}
	}
};

// Great-circle distance in miles.
inline double Haversine(double lat1, double lon1, double lat2, double lon2) {
	const double rad = 3.14159265358979323846 / 180.0;
	const double dlat = (lat2 - lat1) * rad, dlon = (lon2 - lon1) * rad;
	const double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
	                 std::cos(lat1 * rad) * std::cos(lat2 * rad) * std::sin(dlon / 2) * std::sin(dlon / 2);
	return 2.0 * 3958.8 * std::asin(std::sqrt(std::max(0.0, std::min(1.0, a))));
}

// One side of a comparison: a number, or a text code resolved to its string.
struct CmpValue {
	bool present = false, text = false;
	double num = 0;
	std::string str;
};

inline CmpValue ColumnValue(const Frame &f, const std::string &name, size_t row) {
	CmpValue v;
	const int ci = f.Find(name);
	if (ci < 0) {
		throw std::runtime_error("pql: filter references unknown column '" + name + "'");
	}
	const Column &c = f.columns[(size_t)ci];
	v.present = c.valid[row] != 0;
	if (!v.present) {
		return v;
	}
	if (c.type == ColType::CATEGORY) {
		v.text = true;
		v.str = c.code[row] < c.dict.values.size() ? c.dict.values[c.code[row]] : std::string();
	} else {
		v.num = c.num[row];
	}
	return v;
}

// A filter that may name the resource row and the previous task (ASSIGN's
// ONLY IF and the CASE conditions of its expressions). A comparison that names
// neither, and compares with a literal, is evaluated exactly as everywhere
// else.
inline bool EvalFilterFrames(const FilterNode *f, const Frame &task, size_t row, double prediction,
                             const ExprEval &ev) {
	if (!f) {
		return true;
	}
	switch (f->kind) {
	case FilterNode::Kind::AND:
		for (auto &c : f->children) {
			if (!EvalFilterFrames(c.get(), task, row, prediction, ev)) {
				return false;
			}
		}
		return true;
	case FilterNode::Kind::OR:
		for (auto &c : f->children) {
			if (EvalFilterFrames(c.get(), task, row, prediction, ev)) {
				return true;
			}
		}
		return false;
	case FilterNode::Kind::NOT:
		return f->children.empty() ? true : !EvalFilterFrames(f->children[0].get(), task, row, prediction, ev);
	default:
		break;
	}
	const std::string q = f->prediction ? std::string() : ToUpper(f->ref.qualifier);
	const bool special = f->rhs_is_ref || q == "RESOURCE" || q == "PREV";
	if (!special) {
		return EvalFilter(f, task, row, prediction);
	}
	auto side = [&](const Ref &r) -> CmpValue {
		const std::string rq = ToUpper(r.qualifier);
		if (rq == "RESOURCE") {
			return ev.resource ? ColumnValue(*ev.resource, r.name, ev.resource_row) : CmpValue();
		}
		if (rq == "PREV") {
			return ev.prev ? ColumnValue(*ev.prev, r.name, ev.prev_row) : CmpValue();
		}
		return ColumnValue(task, r.name, row);
	};
	const CmpValue l = side(f->ref);
	if (f->op == CmpOp::IS_NULL) {
		return !l.present;
	}
	if (f->op == CmpOp::IS_NOT_NULL) {
		return l.present;
	}
	CmpValue r;
	if (f->rhs_is_ref) {
		r = side(f->rhs);
	} else if (!f->values.empty()) {
		const Literal &lit = f->values[0];
		r.present = lit.kind != Literal::Kind::NUL;
		r.text = lit.kind == Literal::Kind::STRING;
		r.num = lit.number;
		r.str = lit.text;
	}
	if (!l.present || !r.present) {
		return false;
	}
	if (l.text != r.text) {
		throw std::runtime_error("pql: '" + f->ref.ToString() + "' and " +
		                         (f->rhs_is_ref ? f->rhs.ToString() : f->values[0].ToString()) +
		                         " are not the same kind of value, so they cannot be compared");
	}
	const int cmp = l.text ? (l.str < r.str   ? -1
	                          : l.str > r.str ? 1
	                                          : 0)
	                       : (l.num < r.num   ? -1
	                          : l.num > r.num ? 1
	                                          : 0);
	switch (f->op) {
	case CmpOp::EQ:
		return cmp == 0;
	case CmpOp::NE:
		return cmp != 0;
	case CmpOp::LT:
		return cmp < 0;
	case CmpOp::LE:
		return cmp <= 0;
	case CmpOp::GT:
		return cmp > 0;
	case CmpOp::GE:
		return cmp >= 0;
	default:
		throw std::runtime_error("pql: IN is compared with values, not with another column");
	}
}

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
		const Frame *frame = &f;
		size_t at = row;
		if (e->frame == 1) {
			if (!ev.prev) {
				ev.Miss("there is no previous task");
				return 0.0;
			}
			frame = ev.prev;
			at = ev.prev_row;
		} else if (e->frame == 2) {
			if (!ev.resource) {
				ev.Miss("there is no resource row");
				return 0.0;
			}
			frame = ev.resource;
			at = ev.resource_row;
		}
		const Column &c = frame->columns[(size_t)e->col];
		if (!c.valid[at]) {
			ev.Miss(c.name + " is NULL");
			return 0.0;
		}
		return c.num[at];
	}
	case Expr::Kind::PLAN:
		return e->text == "setup" ? ev.plan_setup : e->text == "duration" ? ev.plan_duration : ev.plan_begin;
	case Expr::Kind::DISTANCE: {
		double a[4];
		for (int i = 0; i < 4; i++) {
			a[i] = EvalExpr(e->args[(size_t)i].get(), f, row, prediction, ev);
		}
		return Haversine(a[0], a[1], a[2], a[3]);
	}
	case Expr::Kind::NEG:
		return -EvalExpr(e->args[0].get(), f, row, prediction, ev);
	case Expr::Kind::CASE:
		for (size_t i = 0; i < e->when.size(); i++) {
			if (EvalFilterFrames(e->when[i].get(), f, row, prediction, ev)) {
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

// What an ASSIGN clause resolves to on the loaded tables.
struct AssignBinding {
	const Frame *resource = nullptr;
	int key_col = -1;
	std::vector<Literal> keys; // one per resource row, in row order
	int free_from_col = -1;
	int window_from_col = -1, window_to_col = -1;
	bool clock_is_timestamp = false; // windows and FROM are timestamps; hours are converted
	std::vector<int> cap_cols;       // per constraint: the resource column of the limit, or -1
};

inline const Frame &ResourceFrame(const Database &db, const AssignSpec &a) {
	const int t = db.Find(a.resources.qualifier);
	if (t < 0) {
		throw std::runtime_error("pql: ASSIGN ... OVER names '" + a.resources.qualifier +
		                         "', which is not a table that was loaded");
	}
	return db.tables[(size_t)t];
}

inline AssignBinding ResolveAssign(const Database &db, const Frame &entity, const std::string &alias,
                                   OptimizeSpec &spec) {
	AssignSpec &a = spec.assign;
	AssignBinding b;
	b.resource = &ResourceFrame(db, a);
	const Frame &res = *b.resource;
	b.key_col = res.Find(a.resources.name);
	if (b.key_col < 0) {
		throw std::runtime_error("pql: '" + a.resources.name + "' is not a column of '" + res.name + "'");
	}
	const Column &key = res.columns[(size_t)b.key_col];
	if (key.type == ColType::TIMESTAMP) {
		throw std::runtime_error("pql: the resource key '" + key.name +
		                         "' is a timestamp, which cannot name a resource");
	}
	for (size_t r = 0; r < res.nrows; r++) {
		if (!key.valid[r]) {
			throw std::runtime_error("pql: a row of '" + res.name + "' has no '" + key.name +
			                         "'; every resource needs a key");
		}
		Literal k;
		if (key.type == ColType::CATEGORY) {
			k.kind = Literal::Kind::STRING;
			k.text = key.code[r] < key.dict.values.size() ? key.dict.values[key.code[r]] : std::string();
		} else {
			k.kind = Literal::Kind::NUMBER;
			k.number = key.num[r];
			k.text = TidyNumber(k.number);
		}
		for (const auto &prev : b.keys) {
			if ((k.kind == Literal::Kind::STRING && prev.text == k.text) ||
			    (k.kind == Literal::Kind::NUMBER && prev.number == k.number)) {
				throw std::runtime_error("pql: '" + res.name + "' has two rows with " + key.name + " = " +
				                         k.ToString() + "; a resource key names one resource");
			}
		}
		b.keys.push_back(k);
	}
	if (b.keys.empty()) {
		throw std::runtime_error("pql: '" + res.name + "' has no rows, so there is nothing to assign to");
	}
	if (a.has_else && !b.keys.empty() && a.else_value.kind != b.keys[0].kind) {
		throw std::runtime_error("pql: ELSE " + a.else_value.ToString() +
		                         " is not the same kind of value as the keys of '" + res.name + "'");
	}
	if (a.has_else) {
		for (const auto &k : b.keys) {
			if ((k.kind == Literal::Kind::STRING && k.text == a.else_value.text) ||
			    (k.kind == Literal::Kind::NUMBER && k.number == a.else_value.number)) {
				throw std::runtime_error("pql: ELSE " + a.else_value.ToString() + " is also a key of '" + res.name +
				                         "'; the value for nothing must not name a resource");
			}
		}
	}
	// The clause's columns and expressions, each on the frames it may read.
	const Frames pair {entity, alias, &res, false};
	const Frames with_prev {entity, alias, &res, true};
	ValidateFilterFrames(a.only_if.get(), pair, "ONLY IF");
	if (a.sequenced) {
		BindExprFrames(a.start.get(), pair, "START");
		BindExprFrames(a.setup.get(), with_prev, "SETUP");
		BindExprFrames(a.duration.get(), pair, "DURATION");
		if (!a.free_from.name.empty()) {
			b.free_from_col = res.Find(a.free_from.name);
			if (b.free_from_col < 0) {
				throw std::runtime_error("pql: SEQUENCE FROM names '" + a.free_from.name +
				                         "', which is not a column of '" + res.name + "'");
			}
		}
		if (a.has_window) {
			for (const Ref *w : {&a.window_from, &a.window_to}) {
				if (FrameOf(*w, Frames {entity, alias, nullptr, false}, "WINDOW") != 0) {
					throw std::runtime_error("pql: WINDOW names columns of the task row");
				}
			}
			b.window_from_col = entity.Find(a.window_from.name);
			b.window_to_col = entity.Find(a.window_to.name);
			if (b.window_from_col < 0 || b.window_to_col < 0) {
				throw std::runtime_error("pql: WINDOW names '" +
				                         (b.window_from_col < 0 ? a.window_from.name : a.window_to.name) +
				                         "', which is not a column of '" + entity.name + "'");
			}
			const ColType tf = entity.columns[(size_t)b.window_from_col].type;
			const ColType tt = entity.columns[(size_t)b.window_to_col].type;
			if ((tf == ColType::TIMESTAMP) != (tt == ColType::TIMESTAMP)) {
				throw std::runtime_error("pql: WINDOW's two columns must both be timestamps, or both numbers");
			}
			b.clock_is_timestamp = tf == ColType::TIMESTAMP;
		}
		if (b.free_from_col >= 0) {
			const bool ts = res.columns[(size_t)b.free_from_col].type == ColType::TIMESTAMP;
			if (a.has_window && ts != b.clock_is_timestamp) {
				throw std::runtime_error("pql: SEQUENCE FROM and WINDOW must be on one clock: both timestamps, or "
				                         "both numbers");
			}
			b.clock_is_timestamp = ts;
		}
	}
	for (auto &c : spec.constraints) {
		int col = -1;
		if (c.limit_is_ref) {
			col = res.Find(c.limit_ref.name);
			if (col < 0) {
				throw std::runtime_error("pql: the limit '" + c.limit_ref.ToString() + "' is not a column of '" +
				                         res.name + "'");
			}
			if (res.columns[(size_t)col].type == ColType::CATEGORY ||
			    res.columns[(size_t)col].type == ColType::TIMESTAMP) {
				throw std::runtime_error("pql: the limit '" + c.limit_ref.ToString() + "' must be a number");
			}
		}
		b.cap_cols.push_back(col);
	}
	return b;
}

// Whether a filter or an expression can come out differently on two
// resources: it names a resource column or the model's answer, which is asked
// per resource.
inline bool DependsOnResource(const FilterNode *f) {
	if (!f) {
		return false;
	}
	if (f->kind != FilterNode::Kind::CMP) {
		for (const auto &c : f->children) {
			if (DependsOnResource(c.get())) {
				return true;
			}
		}
		return false;
	}
	return f->prediction || ToUpper(f->ref.qualifier) == "RESOURCE" ||
	       (f->rhs_is_ref && ToUpper(f->rhs.qualifier) == "RESOURCE");
}
inline bool DependsOnResource(const Expr *e) {
	if (!e) {
		return false;
	}
	if (e->kind == Expr::Kind::PREDICTION || (e->kind == Expr::Kind::COLUMN && e->frame == 2)) {
		return true;
	}
	for (const auto &w : e->when) {
		if (DependsOnResource(w.get())) {
			return true;
		}
	}
	for (const auto &a : e->args) {
		if (DependsOnResource(a.get())) {
			return true;
		}
	}
	return false;
}

// Whether an expression reads any plan value.
inline bool UsesPlan(const Expr *e) {
	if (!e) {
		return false;
	}
	if (e->kind == Expr::Kind::PLAN) {
		return true;
	}
	for (const auto &a : e->args) {
		if (UsesPlan(a.get())) {
			return true;
		}
	}
	return false;
}

struct OptimizeOutcome {
	bool choosing = false;
	bool maximize = true;
	int lever_col = -1;           // CHOOSE: the column on the entity frame
	std::vector<Literal> choices; // CHOOSE: as written
	struct Row {
		uint32_t entity_row = 0;
		double anchor = 0;
		int choice = 0; // CHOOSE/ASSIGN: index into `choices`; otherwise 1, "act on it"
		double prediction = 0;
		double objective = 0; // the row's share of the total
		// ASSIGN: where the task sits on its resource, and its plan values. A task
		// placed on nothing has sequence -1.
		int sequence = -1;
		double begin = 0, setup = 0, duration = 0;
	};
	// ASSIGN: the resources in `choices` order (the ELSE value last when there
	// is one), each limit's usage per resource, and how `begin` is reported.
	bool assigning = false;
	size_t resources = 0;
	std::vector<std::vector<double>> usage_per; // [constraint][resource]
	std::vector<double> caps_per;               // [constraint * resources + r], as written
	bool begin_is_timestamp = false;
	double begin_epoch = 0; // microseconds at hour 0 of the plan's clock
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

// The ASSIGN half of RunOptimize: from the predictions per (task, resource) to
// a plan. `pred`, `obj` and `con` are indexed [i * K + k] over the candidate
// rows and the choices (resources, then ELSE), as the shared loop filled them
// with the plan values at zero.
inline OptimizeOutcome RunAssign(Database &db, const Frame &entity, const std::string &alias, const Model *model,
                                 const Statement &stmt, OptimizeSpec &spec, const AssignBinding &ab,
                                 const std::vector<Prediction> &rows, const std::vector<double> &pred,
                                 const std::vector<double> &obj, const std::vector<std::vector<double>> &con,
                                 const CancelCheck &cancelled, OptimizeOutcome out) {
	(void)db;
	(void)model;
	(void)stmt;
	const AssignSpec &a = spec.assign;
	const Frame &res = *ab.resource;
	const size_t n = rows.size(), m = ab.keys.size(), K = spec.choices.size();
	const size_t L = spec.constraints.size();
	const double sgn = spec.maximize ? 1.0 : -1.0;
	const double kHour = 3600.0 * 1000000.0; // microseconds
	const double inf = std::numeric_limits<double>::infinity();
	out.resources = m;

	AssignProblem P;
	P.n = n;
	P.m = m;
	P.sequenced = a.sequenced;
	P.value.assign(n * m, 0.0);
	P.else_value.assign(n, std::numeric_limits<double>::quiet_NaN());
	P.allowed.assign(n * m, 1);
	P.coef_setup.assign(n * m, 0.0);
	P.coef_duration.assign(n * m, 0.0);
	P.coef_begin.assign(n * m, 0.0);
	for (size_t j = 0; j < L; j++) {
		AssignProblem::Limit lim;
		lim.at_most = spec.constraints[j].at_most;
		lim.amount.assign(n * m, 0.0);
		lim.coef_setup.assign(n * m, 0.0);
		lim.coef_duration.assign(n * m, 0.0);
		lim.cap.assign(m, 0.0);
		P.limits.push_back(lim);
	}

	// Values and their plan coefficients. A total is linear in the plan values by
	// rule, so three evaluations give the constant and the slopes; a fourth checks.
	auto eval_total = [&](const OptAggregate &agg, size_t i, size_t r, double st, double dur, double begin,
	                      const char *what) {
		ExprEval ev;
		ev.resource = &res;
		ev.resource_row = r;
		ev.plan_setup = st;
		ev.plan_duration = dur;
		ev.plan_begin = begin;
		const double v = EvalAggregateRow(agg, entity, rows[i].entity_row, pred[i * K + r], ev);
		if (ev.missing) {
			throw std::runtime_error("pql: a row has no value for the " + std::string(what) + ": " + ev.why);
		}
		return v;
	};
	auto slopes = [&](const OptAggregate &agg, bool uses_plan, size_t i, size_t r, double base, double &cs, double &cd,
	                  double &cb, const char *what) {
		cs = cd = cb = 0.0;
		if (!uses_plan) {
			return;
		}
		cs = eval_total(agg, i, r, 1.0, 0.0, 0.0, what) - base;
		cd = eval_total(agg, i, r, 0.0, 1.0, 0.0, what) - base;
		cb = eval_total(agg, i, r, 0.0, 0.0, 1.0, what) - base;
		const double twice = eval_total(agg, i, r, 2.0, 2.0, 2.0, what);
		if (std::fabs(twice - (base + 2 * (cs + cd + cb))) > 1e-6 * std::max(1.0, std::fabs(twice))) {
			throw std::runtime_error(std::string("pql: plan.setup, plan.duration and plan.begin must enter the ") +
			                         what +
			                         " linearly (a weight times the value), so the solver can price a placement");
		}
	};
	const bool obj_plan = !spec.objective.count && UsesPlan(spec.objective.expr.get());
	std::vector<uint8_t> lim_plan(L, 0);
	for (size_t j = 0; j < L; j++) {
		lim_plan[j] = !spec.constraints[j].total.count && UsesPlan(spec.constraints[j].total.expr.get());
	}
	for (size_t i = 0; i < n; i++) {
		for (size_t r = 0; r < m; r++) {
			const size_t o = i * m + r;
			P.value[o] = sgn * obj[i * K + r];
			double cs, cd, cb;
			slopes(spec.objective, obj_plan, i, r, obj[i * K + r], cs, cd, cb, "objective");
			P.coef_setup[o] = sgn * cs;
			P.coef_duration[o] = sgn * cd;
			P.coef_begin[o] = sgn * cb;
			for (size_t j = 0; j < L; j++) {
				const double csgn = spec.constraints[j].at_most ? 1.0 : -1.0;
				AssignProblem::Limit &lim = P.limits[j];
				lim.amount[o] = csgn * con[j][i * K + r];
				double ls, ld, lb;
				slopes(spec.constraints[j].total, lim_plan[j] != 0, i, r, con[j][i * K + r], ls, ld, lb, "limit");
				if (lb != 0.0) {
					throw std::runtime_error(
					    "pql: a limit cannot depend on plan.begin; use plan.setup and plan.duration");
				}
				lim.coef_setup[o] = csgn * ls;
				lim.coef_duration[o] = csgn * ld;
			}
		}
		if (a.has_else) {
			P.else_value[i] = sgn * obj[i * K + m];
		}
	}
	// A limit's cap per resource; a floor arrives negated like everything else.
	for (size_t j = 0; j < L; j++) {
		const OptConstraint &c = spec.constraints[j];
		const double csgn = c.at_most ? 1.0 : -1.0;
		for (size_t r = 0; r < m; r++) {
			double cap = c.limit.number;
			if (ab.cap_cols[j] >= 0) {
				const Column &cc = res.columns[(size_t)ab.cap_cols[j]];
				if (!cc.valid[r]) {
					throw std::runtime_error("pql: '" + res.name + "' row with " + ab.keys[r].ToString() + " has no " +
					                         cc.name + ", so its limit is unknown");
				}
				cap = cc.num[r];
			}
			P.limits[j].cap[r] = csgn * cap;
			out.caps_per.push_back(cap);
		}
	}
	// Which pairs are allowed, and when sequenced, the geometry of time.
	for (size_t i = 0; i < n; i++) {
		for (size_t r = 0; r < m; r++) {
			ExprEval ev;
			ev.resource = &res;
			ev.resource_row = r;
			P.allowed[i * m + r] = EvalFilterFrames(a.only_if.get(), entity, rows[i].entity_row, inf, ev) ? 1 : 0;
		}
	}
	if (a.sequenced) {
		P.start.assign(n * m, 0.0);
		// A SETUP that names nothing about the resource is the same on all of
		// them: one number per pair of tasks instead of one per resource too.
		P.setup_shared = !DependsOnResource(a.setup.get());
		P.setup.assign(P.setup_shared ? n * n : n * n * m, 0.0);
		P.duration.assign(n * m, 0.0);
		auto hours = [&](const Expr *e, size_t i, size_t r, const Frame *prev, size_t prev_row, const char *what) {
			if (!e) {
				return 0.0;
			}
			ExprEval ev;
			ev.resource = &res;
			ev.resource_row = r;
			ev.prev = prev;
			ev.prev_row = prev_row;
			const double v = EvalExpr(e, entity, rows[i].entity_row, pred[i * K + r], ev);
			if (ev.missing) {
				throw std::runtime_error("pql: a row has no value for " + std::string(what) + ": " + ev.why);
			}
			if (v < 0) {
				throw std::runtime_error(std::string("pql: ") + what + " came out negative (" + TidyNumber(v) +
				                         "); hours cannot be");
			}
			return v;
		};
		for (size_t i = 0; i < n; i++) {
			if (P.setup_shared) {
				for (size_t j = 0; j < n; j++) {
					if (j != i) {
						P.setup[j * n + i] = hours(a.setup.get(), i, 0, &entity, rows[j].entity_row, "SETUP");
					}
				}
			}
			for (size_t r = 0; r < m; r++) {
				if (!P.allowed[i * m + r]) {
					continue;
				}
				P.start[i * m + r] = hours(a.start.get(), i, r, nullptr, 0, "START");
				P.duration[i * m + r] = hours(a.duration.get(), i, r, nullptr, 0, "DURATION");
				if (P.setup_shared) {
					continue;
				}
				for (size_t j = 0; j < n; j++) {
					if (j != i && P.allowed[j * m + r]) {
						P.setup[(j * n + i) * m + r] = hours(a.setup.get(), i, r, &entity, rows[j].entity_row, "SETUP");
					}
				}
			}
			if (cancelled && (i & 15) == 0) {
				cancelled();
			}
		}
		// One clock in hours. Timestamps are measured from the earliest moment
		// named, so the numbers stay small; numbers are taken as hours already.
		double epoch = inf;
		auto raw = [&](const Column &c, size_t r) {
			return c.valid[r] ? c.num[r] : std::numeric_limits<double>::quiet_NaN();
		};
		if (ab.clock_is_timestamp) {
			if (a.has_window) {
				for (size_t i = 0; i < n; i++) {
					const double t = raw(entity.columns[(size_t)ab.window_from_col], rows[i].entity_row);
					if (std::isfinite(t)) {
						epoch = std::min(epoch, t);
					}
				}
			}
			if (ab.free_from_col >= 0) {
				for (size_t r = 0; r < m; r++) {
					const double t = raw(res.columns[(size_t)ab.free_from_col], r);
					if (std::isfinite(t)) {
						epoch = std::min(epoch, t);
					}
				}
			}
			if (!std::isfinite(epoch)) {
				epoch = 0;
			}
		} else {
			epoch = 0;
		}
		out.begin_is_timestamp = ab.clock_is_timestamp;
		out.begin_epoch = epoch;
		auto to_hours = [&](double t, double when_missing) {
			if (!std::isfinite(t)) {
				return when_missing;
			}
			return ab.clock_is_timestamp ? (t - epoch) / kHour : t;
		};
		if (a.has_window) {
			P.window_from.assign(n, -inf);
			P.window_to.assign(n, inf);
			for (size_t i = 0; i < n; i++) {
				P.window_from[i] = to_hours(raw(entity.columns[(size_t)ab.window_from_col], rows[i].entity_row), -inf);
				P.window_to[i] = to_hours(raw(entity.columns[(size_t)ab.window_to_col], rows[i].entity_row), inf);
			}
		}
		P.free_from.assign(m, 0.0);
		if (ab.free_from_col >= 0) {
			for (size_t r = 0; r < m; r++) {
				P.free_from[r] = to_hours(raw(res.columns[(size_t)ab.free_from_col], r), 0.0);
			}
		}
	}

	// A pure pairing has an exact answer; anything else is searched.
	AssignSolution sol;
	if (!SolvePairing(P, sol)) {
		AssignSettings settings;
		sol = SolveAssign(P, settings, cancelled);
	}
	if (!sol.feasible) {
		throw std::runtime_error("pql: found no plan that places every task that must be placed inside every "
		                         "window and limit. Loosen a limit, widen a window, or give the plan an ELSE");
	}
	// Read the plan back in the units the statement asked for.
	out.counts.assign(K, 0);
	out.objective = 0;
	for (size_t i = 0; i < n; i++) {
		OptimizeOutcome::Row row;
		row.entity_row = rows[i].entity_row;
		row.anchor = rows[i].anchor;
		const int r = sol.resource[i];
		const size_t k = r >= 0 ? (size_t)r : m;
		row.choice = (int)k;
		row.prediction = pred[i * K + k];
		if (r >= 0) {
			row.sequence = sol.sequence[i];
			row.setup = sol.setup[i];
			row.duration = sol.duration[i];
			row.begin = ab.clock_is_timestamp ? out.begin_epoch + sol.begin[i] * kHour : sol.begin[i];
			row.objective =
			    eval_total(spec.objective, i, (size_t)r, sol.setup[i], sol.duration[i], sol.begin[i], "objective");
		} else {
			row.objective = obj[i * K + m];
		}
		out.counts[k]++;
		out.objective += row.objective;
		out.plan.push_back(row);
	}
	out.usage_per.assign(L, std::vector<double>(m, 0.0));
	for (size_t j = 0; j < L; j++) {
		const double csgn = spec.constraints[j].at_most ? 1.0 : -1.0;
		for (size_t r = 0; r < m; r++) {
			out.usage_per[j][r] = csgn * sol.usage[j][r];
		}
		double total = 0;
		for (size_t r = 0; r < m; r++) {
			total += out.usage_per[j][r];
		}
		out.usage.push_back(total);
	}
	out.bound = sgn * sol.bound;
	out.exact = sol.exact;
	return out;
}

// `model` is null when the statement has no USING MODEL: the plan then stands on
// the table's own columns (the parser has refused prediction and CHOOSE), every
// row the WHERE keeps is a candidate, and a task left on ASSIGN's ELSE adds
// nothing to the totals, as a row not picked adds nothing without ASSIGN.
inline OptimizeOutcome RunOptimize(Database &db, const Model *model, const Statement &stmt,
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

	// ASSIGN decides the lever the way CHOOSE does, with one choice per resource
	// and one more for nothing; the predictions are gathered by the same loop.
	AssignBinding ab;
	if (spec.Assigning()) {
		ab = ResolveAssign(db, entity, alias, spec);
		spec.lever = spec.assign.lever;
		spec.choices = ab.keys;
		if (spec.assign.has_else) {
			spec.choices.push_back(spec.assign.else_value);
		}
	}
	OptimizeOutcome out;
	out.choosing = spec.Choosing();
	out.assigning = spec.Assigning();
	out.maximize = spec.maximize;
	out.choices = spec.choices;
	// Without a model the lever is only written to the plan; a column of that name
	// is used when the table has one, so CASE and FILTER can read the value tried.
	const int lever = !out.choosing ? -1 : model ? ResolveLever(entity, *model, stmt, spec)
	                                             : entity.Find(spec.lever.name);
	out.lever_col = lever;

	// The rows a plan may act on, decided on the data as it stands: a WHERE that
	// names the lever reads its current value, not the one being tried.
	std::vector<uint8_t> candidate(entity.nrows, 0);
	for (size_t r = 0; r < entity.nrows; r++) {
		candidate[r] = EvalFilter(stmt.filter.get(), entity, r) ? 1 : 0;
	}
	Statement ps;
	ps.kind = StmtKind::PREDICT;
	ps.model = model ? model->name : std::string();
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
		std::vector<Prediction> ps_k;
		if (model) {
			ps_k = RunPredict(db, *model, ps, cancelled, &candidate);
		} else {
			for (size_t r = 0; r < entity.nrows; r++) {
				if (candidate[r]) {
					Prediction p;
					p.entity_row = (uint32_t)r;
					ps_k.push_back(p);
				}
			}
		}
		// Without a model, nothing is known about a task left to ELSE: it adds nothing.
		const bool counts = model || !(out.assigning && spec.assign.has_else && k + 1 == K);
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
				const double val = counts ? EvalAggregateRow(agg, entity, r, p, ev) : 0.0;
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

	if (out.assigning) {
		return RunAssign(db, entity, alias, model, stmt, spec, ab, rows, pred, obj, con, cancelled, out);
	}

	// What the rows are worth with every lever left as it is, from the same model.
	if (lever >= 0 && model) {
		restore.Put();
		std::vector<Prediction> now = RunPredict(db, *model, ps, cancelled, &candidate);
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

inline OptimizeOutcome RunOptimize(Database &db, const Model &model, const Statement &stmt,
                                   const CancelCheck &cancelled = CancelCheck(),
                                   const OptSettings &settings = OptSettings()) {
	return RunOptimize(db, &model, stmt, cancelled, settings);
}

} // namespace pql
