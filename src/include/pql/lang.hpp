// lang.hpp — The language: column types, tokenizer, AST, options and the parser.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/common.hpp"

namespace pql {

// ===========================================================================
// 1. Column types
// ===========================================================================

enum class ColType : uint8_t { INVALID = 0, INT64, DOUBLE, BOOL, TIMESTAMP, CATEGORY };

// ===========================================================================
// 2. Tokenizer
// ===========================================================================

enum class Tok : uint8_t { END = 0, IDENT, NUMBER, STRING, PUNCT, KEYWORD };

struct Token {
	Tok kind = Tok::END;
	std::string text;  // identifiers keep original case
	std::string upper; // uppercased, for keyword comparison
	double number = 0;
	uint32_t pos = 0; // byte offset in the source, for error messages
};

// The reserved words PQL itself dispatches on. Anything else that arrives as
// an identifier stays an identifier, so a column called `horizon` is still
// addressable as "horizon" when quoted.
inline bool IsPqlKeyword(const std::string &up) {
	static const char *kw[] = {
	    "TRAIN",   "MODEL",    "PREDICT",  "FOR",      "WHERE",    "AT",         "HORIZON",  "USING",   "GRAPH",
	    "SPLIT",   "TEMPORAL", "VALIDATE", "TEST",     "FROM",     "OPTIONS",    "AS",       "AND",     "OR",
	    "NOT",     "IN",       "IS",       "NULL",     "COUNT",    "EXISTS",     "SUM",      "AVG",     "MIN",
	    "MAX",     "DAY",      "DAYS",     "WEEK",     "WEEKS",    "MONTH",      "MONTHS",   "YEAR",    "YEARS",
	    "HOUR",    "HOURS",    "DROP",     "SHOW",     "MODELS",   "EVERY",      "BACKTEST", "TO",      "EXPLAIN",
	    "EXCLUDE", "REPLACE",  "IF",       "OPTIMIZE", "MAXIMIZE", "MINIMIZE",   "CHOOSE",   "SUBJECT", "CASE",
	    "WHEN",    "THEN",     "ELSE",     "END",      "FILTER",   "PREDICTION", nullptr};
	for (int i = 0; kw[i]; i++) {
		if (up == kw[i]) {
			return true;
		}
	}
	return false;
}

struct ParseError : std::runtime_error {
	uint32_t position;
	ParseError(const std::string &msg, uint32_t pos) : std::runtime_error(msg), position(pos) {
	}
};

inline std::vector<Token> Tokenize(const std::string &src) {
	std::vector<Token> out;
	uint32_t i = 0;
	const uint32_t n = (uint32_t)src.size();
	while (i < n) {
		unsigned char c = (unsigned char)src[i];
		if (std::isspace(c)) {
			i++;
			continue;
		}
		// line comment
		if (c == '-' && i + 1 < n && src[i + 1] == '-') {
			while (i < n && src[i] != '\n') {
				i++;
			}
			continue;
		}
		// block comment. DuckDB hands the statement over verbatim, comments and
		// all, once its own parser has rejected it; an unclosed one is an error
		// rather than a comment that swallows the rest of the statement.
		if (c == '/' && i + 1 < n && src[i + 1] == '*') {
			const uint32_t open = i;
			i += 2;
			while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) {
				i++;
			}
			if (i + 1 >= n) {
				throw ParseError("unterminated block comment; a /* is never closed", open);
			}
			i += 2;
			continue;
		}
		Token t;
		t.pos = i;
		// single-quoted string literal, '' escapes a quote
		if (c == '\'') {
			const uint32_t open = i;
			i++;
			std::string v;
			bool closed = false;
			while (i < n) {
				if (src[i] == '\'') {
					if (i + 1 < n && src[i + 1] == '\'') {
						v.push_back('\'');
						i += 2;
						continue;
					}
					i++;
					closed = true;
					break;
				}
				v.push_back(src[i++]);
			}
			// Running off the end used to be accepted, so `region = 'US` quietly
			// became `region = 'US'`: a missing quote changed the meaning of a
			// statement instead of failing it.
			if (!closed) {
				throw ParseError("unterminated string literal; a ' is never closed", open);
			}
			t.kind = Tok::STRING;
			t.text = v;
			t.upper = ToUpper(v);
			out.push_back(std::move(t));
			continue;
		}
		// double-quoted identifier keeps its exact spelling and never becomes a keyword
		if (c == '"') {
			const uint32_t open = i;
			i++;
			std::string v;
			bool closed = false;
			while (i < n) {
				if (src[i] == '"') {
					if (i + 1 < n && src[i + 1] == '"') { // "" is one literal quote, as in SQL
						v.push_back('"');
						i += 2;
						continue;
					}
					i++;
					closed = true;
					break;
				}
				v.push_back(src[i++]);
			}
			if (!closed) {
				throw ParseError("unterminated quoted identifier; a \" is never closed", open);
			}
			if (v.empty()) {
				throw ParseError("empty quoted identifier", open);
			}
			t.kind = Tok::IDENT;
			t.text = v;
			t.upper = ToUpper(v);
			out.push_back(std::move(t));
			continue;
		}
		if (std::isdigit(c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)src[i + 1]))) {
			uint32_t s = i;
			while (i < n && (std::isdigit((unsigned char)src[i]) || src[i] == '.')) {
				i++;
			}
			t.kind = Tok::NUMBER;
			t.text = src.substr(s, i - s);
			t.upper = t.text;
			char *end = nullptr;
			t.number = std::strtod(t.text.c_str(), &end);
			if (!end || *end != '\0') {
				throw ParseError("malformed number '" + t.text + "'", s);
			}
			out.push_back(std::move(t));
			continue;
		}
		if (std::isalpha(c) || c == '_') {
			uint32_t s = i;
			while (i < n && (std::isalnum((unsigned char)src[i]) || src[i] == '_')) {
				i++;
			}
			t.text = src.substr(s, i - s);
			t.upper = ToUpper(t.text);
			t.kind = IsPqlKeyword(t.upper) ? Tok::KEYWORD : Tok::IDENT;
			out.push_back(std::move(t));
			continue;
		}
		// multi-character operators first, so <= and <> do not split
		static const char *ops2[] = {"<=", ">=", "!=", "<>", nullptr};
		bool matched = false;
		for (int k = 0; ops2[k]; k++) {
			if (i + 1 < n && src[i] == ops2[k][0] && src[i + 1] == ops2[k][1]) {
				t.kind = Tok::PUNCT;
				t.text = ops2[k];
				t.upper = t.text;
				out.push_back(std::move(t));
				i += 2;
				matched = true;
				break;
			}
		}
		if (matched) {
			continue;
		}
		t.kind = Tok::PUNCT;
		t.text = std::string(1, (char)c);
		t.upper = t.text;
		out.push_back(std::move(t));
		i++;
	}
	out.push_back(Token {});
	return out;
}

// ===========================================================================
// 3. AST
// ===========================================================================

// Echo an identifier so it reads back as the same identifier.
//
// Statements are echoed in SHOW MODELS and kept as a model's record of itself,
// so a name that needed quotes going in needs them coming out. A table called
// "my table" was emitted bare, and the statement could not be run again.
inline std::string QuoteIdent(const std::string &name) {
	bool plain = !name.empty() && !std::isdigit((unsigned char)name[0]);
	for (char c : name) {
		if (!(std::isalnum((unsigned char)c) || c == '_')) {
			plain = false;
			break;
		}
	}
	if (plain && IsPqlKeyword(ToUpper(name))) {
		plain = false; // a column called `horizon` has to come back quoted
	}
	if (plain) {
		return name;
	}
	std::string out = "\"";
	for (char c : name) {
		if (c == '"') {
			out += "\"\"";
		} else {
			out.push_back(c);
		}
	}
	return out + "\"";
}

// A qualified reference: [table.]column, where table may be an alias.
struct Ref {
	std::string qualifier; // may be empty
	std::string name;

	std::string ToString() const {
		return qualifier.empty() ? QuoteIdent(name) : QuoteIdent(qualifier) + "." + QuoteIdent(name);
	}
};

enum class CmpOp : uint8_t { EQ, NE, LT, LE, GT, GE, IN, NOT_IN, IS_NULL, IS_NOT_NULL };

struct Literal {
	enum class Kind : uint8_t { NUMBER, STRING, NUL } kind = Kind::NUL;
	double number = 0;
	std::string text;

	std::string ToString() const {
		switch (kind) {
		case Kind::NUMBER: {
			// A date literal is resolved to microseconds but keeps its original
			// spelling, so `'2025-08-01'` must echo with its quotes or the statement
			// a model records of itself will not parse: the tokenizer reads 2025,
			// then a minus.
			const char *begin = text.c_str();
			char *end = nullptr;
			std::strtod(begin, &end);
			const bool plain = end && end != begin && *end == '\0';
			return plain ? text : ("'" + text + "'");
		}
		case Kind::STRING: {
			// A quote inside the text is written doubled, the way it was read.
			std::string q = "'";
			for (char c : text) {
				if (c == '\'') {
					q += "''";
				} else {
					q.push_back(c);
				}
			}
			return q + "'";
		}
		default:
			return "NULL";
		}
	}
};

// Filters form a small boolean tree. Keeping OR rather than flattening to a
// conjunction matters: `status='yes' OR status='maybe'` is the single most
// common shape in event-style tasks, and rewriting it as IN would lose the
// user's spelling when we echo the plan back.
struct FilterNode {
	enum class Kind : uint8_t { CMP, AND, OR, NOT } kind = Kind::CMP;
	// CMP
	Ref ref;
	// Inside OPTIMIZE's totals a comparison may be on the model's answer rather
	// than a column: `FILTER (WHERE prediction > 0.5)`.
	bool prediction = false;
	CmpOp op = CmpOp::EQ;
	std::vector<Literal> values;
	// AND / OR / NOT
	std::vector<std::unique_ptr<FilterNode>> children;

	std::unique_ptr<FilterNode> Copy() const {
		auto o = std::unique_ptr<FilterNode>(new FilterNode());
		o->kind = kind;
		o->ref = ref;
		o->prediction = prediction;
		o->op = op;
		o->values = values;
		for (auto &c : children) {
			o->children.push_back(c->Copy());
		}
		return o;
	}
	std::string ToString() const {
		switch (kind) {
		case Kind::AND:
		case Kind::OR: {
			const char *j = kind == Kind::AND ? " AND " : " OR ";
			std::string s = "(";
			for (size_t i = 0; i < children.size(); i++) {
				if (i) {
					s += j;
				}
				s += children[i]->ToString();
			}
			return s + ")";
		}
		case Kind::NOT:
			return "NOT " + (children.empty() ? std::string("?") : children[0]->ToString());
		default:
			break;
		}
		std::string s = prediction ? std::string("prediction") : ref.ToString();
		switch (op) {
		case CmpOp::EQ:
			s += " = ";
			break;
		case CmpOp::NE:
			s += " != ";
			break;
		case CmpOp::LT:
			s += " < ";
			break;
		case CmpOp::LE:
			s += " <= ";
			break;
		case CmpOp::GT:
			s += " > ";
			break;
		case CmpOp::GE:
			s += " >= ";
			break;
		case CmpOp::IS_NULL:
			return s + " IS NULL";
		case CmpOp::IS_NOT_NULL:
			return s + " IS NOT NULL";
		case CmpOp::IN:
		case CmpOp::NOT_IN: {
			s += (op == CmpOp::IN ? " IN (" : " NOT IN (");
			for (size_t i = 0; i < values.size(); i++) {
				if (i) {
					s += ", ";
				}
				s += values[i].ToString();
			}
			return s + ")";
		}
		}
		return s + (values.empty() ? std::string("?") : values[0].ToString());
	}
};

enum class TargetKind : uint8_t { ATTRIBUTE, COUNT, EXISTS, SUM, AVG, MIN, MAX };

inline const char *TargetKindName(TargetKind k) {
	switch (k) {
	case TargetKind::ATTRIBUTE:
		return "ATTRIBUTE";
	case TargetKind::COUNT:
		return "COUNT";
	case TargetKind::EXISTS:
		return "EXISTS";
	case TargetKind::SUM:
		return "SUM";
	case TargetKind::AVG:
		return "AVG";
	case TargetKind::MIN:
		return "MIN";
	default:
		return "MAX";
	}
}

// Is the target computed from future rows of another table (forecast), or read
// off the entity row itself (imputation)? Everything downstream branches here.
inline bool IsForecast(TargetKind k) {
	return k != TargetKind::ATTRIBUTE;
}

struct PredictExpr {
	TargetKind kind = TargetKind::ATTRIBUTE;
	// ATTRIBUTE: ref is the entity column. Forecast: target_table is the
	// related table, ref is the aggregated column (empty for COUNT/EXISTS).
	Ref ref;
	std::string target_table;
	std::unique_ptr<FilterNode> filter; // inner WHERE, restricts the target rows

	PredictExpr() = default;
	PredictExpr(PredictExpr &&) = default;
	PredictExpr &operator=(PredictExpr &&) = default;
	PredictExpr(const PredictExpr &o) {
		*this = o;
	}
	PredictExpr &operator=(const PredictExpr &o) {
		if (this == &o) {
			return *this;
		}
		kind = o.kind;
		ref = o.ref;
		target_table = o.target_table;
		filter = o.filter ? o.filter->Copy() : nullptr;
		return *this;
	}

	std::string ToString() const {
		if (kind == TargetKind::ATTRIBUTE) {
			return ref.ToString();
		}
		std::string s = std::string(TargetKindName(kind)) + "(";
		// `*` is the wildcard in COUNT(*), not a table anyone named.
		const std::string tbl = target_table == "*" ? target_table : QuoteIdent(target_table);
		s += ref.name.empty() ? tbl : (tbl + "." + QuoteIdent(ref.name));
		if (filter) {
			s += " WHERE " + filter->ToString();
		}
		return s + ")";
	}
};

struct Interval {
	int64_t amount = 0;
	std::string unit; // normalized singular: DAY, WEEK, MONTH, YEAR, HOUR
	bool present = false;

	// Micros, matching DuckDB's TIMESTAMP unit. Months and years are resolved
	// with calendar-average lengths; a task whose horizon must respect calendar
	// month boundaries should say so in days.
	int64_t Micros() const {
		int64_t day = 86400LL * 1000000LL;
		if (unit == "HOUR") {
			return amount * 3600LL * 1000000LL;
		}
		if (unit == "DAY") {
			return amount * day;
		}
		if (unit == "WEEK") {
			return amount * 7 * day;
		}
		if (unit == "MONTH") {
			return int64_t(amount * 30.436875 * double(day));
		}
		if (unit == "YEAR") {
			return int64_t(amount * 365.2425 * double(day));
		}
		return 0;
	}
	std::string ToString() const {
		return std::to_string(amount) + " " + unit + (amount == 1 ? "" : "S");
	}
};

// The learning algorithms, by the name ARCH takes; each is one header behind
// the Learner interface, and MakeLearner picks it.
static const char *const kArchNames[] = {"MLP", "SAGE"};
inline bool IsKnownArch(const std::string &upper) {
	for (const char *a : kArchNames) {
		if (upper == a) {
			return true;
		}
	}
	return false;
}

struct Options {
	std::unordered_map<std::string, Literal> kv;

	// Sorted, so a statement echoed back reads the same way twice. The map is
	// unordered and iterating it directly would shuffle between runs.
	std::string ToString() const {
		if (kv.empty()) {
			return std::string();
		}
		std::vector<std::string> keys;
		keys.reserve(kv.size());
		for (const auto &p : kv) {
			keys.push_back(p.first);
		}
		std::sort(keys.begin(), keys.end());
		std::string s = " OPTIONS (";
		for (size_t i = 0; i < keys.size(); i++) {
			s += (i ? ", " : "") + keys[i] + " = " + kv.at(keys[i]).ToString();
		}
		return s + ")";
	}

	// A silently ignored option reads as "my setting had no effect", which is the
	// most common way to waste an afternoon.
	static bool Known(const std::string &k) {
		static const char *ok[] = {"EPOCHS", "HIDDEN",         "LR", "BATCH",      "SEED", "MEAN_COLS", "ARCH",
		                           "LAYERS", "MAX_CATEGORIES", "L2", "LOG_TARGET", nullptr};
		for (int i = 0; ok[i]; i++) {
			if (k == ok[i]) {
				return true;
			}
		}
		return false;
	}
	static std::string KnownList() {
		return "EPOCHS, HIDDEN, LR, BATCH, SEED, MEAN_COLS, ARCH, LAYERS, MAX_CATEGORIES, L2, LOG_TARGET";
	}

	double Num(const std::string &k, double dflt) const {
		auto it = kv.find(k);
		if (it == kv.end() || it->second.kind != Literal::Kind::NUMBER) {
			return dflt;
		}
		return it->second.number;
	}
	std::string Str(const std::string &k, const std::string &dflt) const {
		auto it = kv.find(k);
		if (it == kv.end()) {
			return dflt;
		}
		return it->second.kind == Literal::Kind::STRING ? it->second.text : it->second.ToString();
	}
	bool Has(const std::string &k) const {
		return kv.count(k) > 0;
	}
};

struct TemporalSplit {
	bool present = false;
	Literal validate_from;
	Literal test_from;
};

// Arithmetic over one entity row, for OPTIMIZE's objective and constraints:
// numbers, numeric columns, the model's `prediction`, + - * /, and CASE. Kept
// as written so a statement echoes the way it was typed.
struct Expr {
	enum class Kind : uint8_t { NUMBER, COLUMN, PREDICTION, NEG, ADD, SUB, MUL, DIV, CASE } kind = Kind::NUMBER;
	double number = 0;
	std::string text; // NUMBER: the spelling it was read with
	Ref ref;          // COLUMN
	int col = -1;     // COLUMN: resolved against a frame by BindExpr
	// NEG: one operand. ADD..DIV: two. CASE: one THEN value per condition, then
	// the ELSE value when there is one.
	std::vector<std::unique_ptr<Expr>> args;
	std::vector<std::unique_ptr<FilterNode>> when; // CASE: one condition per THEN
	bool has_else = false;
	// `CASE offer WHEN '10%' THEN ...` is stored as the equalities it means, with
	// the operand kept so it echoes in the short form.
	bool simple_case = false;
	Ref operand;

	std::unique_ptr<Expr> Copy() const {
		auto o = std::unique_ptr<Expr>(new Expr());
		o->kind = kind;
		o->number = number;
		o->text = text;
		o->ref = ref;
		o->col = col;
		for (const auto &a : args) {
			o->args.push_back(a->Copy());
		}
		for (const auto &w : when) {
			o->when.push_back(w->Copy());
		}
		o->has_else = has_else;
		o->simple_case = simple_case;
		o->operand = operand;
		return o;
	}

	// Binding strength, for parenthesising on the way out: a child is wrapped
	// when it binds more loosely than its parent, or as loosely on the right, so
	// `a - (b - c)` keeps its parentheses and reads back as the same tree.
	int Precedence() const {
		switch (kind) {
		case Kind::ADD:
		case Kind::SUB:
			return 1;
		case Kind::MUL:
		case Kind::DIV:
			return 2;
		case Kind::NEG:
			return 3;
		default:
			return 4;
		}
	}

	std::string ToString() const {
		switch (kind) {
		case Kind::NUMBER:
			return text.empty() ? std::to_string(number) : text;
		case Kind::COLUMN:
			return ref.ToString();
		case Kind::PREDICTION:
			return "prediction";
		case Kind::NEG: {
			// Two minus signs in a row would read back as a line comment.
			const Expr &a = *args[0];
			const bool wrap = a.Precedence() < 3 || a.kind == Kind::NEG;
			return wrap ? "-(" + a.ToString() + ")" : "-" + a.ToString();
		}
		case Kind::CASE: {
			std::string s = "CASE";
			if (simple_case) {
				s += " " + operand.ToString();
			}
			for (size_t i = 0; i < when.size(); i++) {
				s += " WHEN ";
				s += simple_case ? when[i]->values[0].ToString() : when[i]->ToString();
				s += " THEN " + args[i]->ToString();
			}
			if (has_else) {
				s += " ELSE " + args.back()->ToString();
			}
			return s + " END";
		}
		default:
			break;
		}
		const char *op = kind == Kind::ADD ? " + " : kind == Kind::SUB ? " - " : kind == Kind::MUL ? " * " : " / ";
		const int p = Precedence();
		const Expr &l = *args[0], &r = *args[1];
		const std::string ls = l.Precedence() < p ? "(" + l.ToString() + ")" : l.ToString();
		const std::string rs = r.Precedence() <= p ? "(" + r.ToString() + ")" : r.ToString();
		return ls + op + rs;
	}
};

// SUM(<expr>) or COUNT(*), over the rows a plan acts on, optionally narrowed
// by FILTER (WHERE ...) as in SQL.
struct OptAggregate {
	bool count = false;
	std::unique_ptr<Expr> expr; // SUM only
	std::unique_ptr<FilterNode> filter;

	OptAggregate() = default;
	OptAggregate(OptAggregate &&) = default;
	OptAggregate &operator=(OptAggregate &&) = default;
	OptAggregate(const OptAggregate &o) {
		*this = o;
	}
	OptAggregate &operator=(const OptAggregate &o) {
		if (this != &o) {
			count = o.count;
			expr = o.expr ? o.expr->Copy() : nullptr;
			filter = o.filter ? o.filter->Copy() : nullptr;
		}
		return *this;
	}

	std::string ToString() const {
		std::string s = count ? std::string("COUNT(*)") : "SUM(" + (expr ? expr->ToString() : std::string()) + ")";
		if (filter) {
			s += " FILTER (WHERE " + filter->ToString() + ")";
		}
		return s;
	}
};

struct OptConstraint {
	OptAggregate total;
	bool at_most = true; // <= when true, >= otherwise
	Literal limit;

	std::string ToString() const {
		return total.ToString() + (at_most ? " <= " : " >= ") + limit.ToString();
	}
};

// What OPTIMIZE asks for. Without CHOOSE the decision is which rows to act on;
// with it, which of the listed values each row's lever column should take.
struct OptimizeSpec {
	bool explain = false; // EXPLAIN OPTIMIZE: report the totals, not the plan
	bool maximize = true;
	OptAggregate objective;
	std::vector<OptConstraint> constraints;
	Ref lever;                    // CHOOSE <lever>; empty name means "which rows"
	std::vector<Literal> choices; // CHOOSE ... IN (<choices>)

	bool Choosing() const {
		return !lever.name.empty();
	}
};

enum class StmtKind : uint8_t { TRAIN, PREDICT, BACKTEST, EXPLAIN, DROP_MODEL, SHOW_MODELS, OPTIMIZE };

// Everything a statement holds that copies itself.
//
// The one member that cannot is the outer filter, which is a unique_ptr and
// needs a deep copy. Keeping the rest here means the hand-written copy below
// touches exactly one field, so adding a clause cannot silently be left out of
// it. Enumerating every member by hand is how EXCLUDE and OR REPLACE were both
// dropped from a copied statement: the clause parsed, took effect during
// training, and then vanished from the model's own record of itself.
struct StatementData {
	StmtKind kind = StmtKind::PREDICT;
	// TRAIN OR REPLACE. Without it, training over a name that already holds a
	// model is refused: rerunning a statement and silently discarding the model
	// that was there is not a thing anyone asks for.
	bool or_replace = false;
	// DROP MODEL IF EXISTS. Without it, dropping a name that holds nothing is an
	// error, as it is in SQL.
	bool if_exists = false;
	std::string model; // TRAIN MODEL <model> / USING MODEL <model>
	PredictExpr target;
	std::string entity_table;
	std::string entity_alias;
	// Columns the model must not look at. A leaky field, a free-text note, or
	// anything EXPLAIN showed to be worthless: naming it here is cheaper than
	// building a view without it.
	std::vector<Ref> excluded;
	Ref anchor;             // AT <column>
	Literal anchor_literal; // AT '2024-01-01' (prediction only)
	bool anchor_is_literal = false;
	Interval horizon;
	// EVERY <interval>: build the anchors instead of requiring a column. One
	// entity row then yields one example per generated date, which is what a
	// forecast over a catalog table (products, customers) actually needs.
	Interval every;
	std::vector<std::string> graph_tables; // USING GRAPH (...); empty = auto
	TemporalSplit split;
	Options options;
	bool saw_using_model = false;
	// BACKTEST window. Empty means "everything the model can be replayed over".
	Literal backtest_from, backtest_to;
	bool has_from = false, has_to = false;
	OptimizeSpec optimize; // OPTIMIZE only; deep-copies itself
};

struct Statement : StatementData {
	std::unique_ptr<FilterNode> filter; // outer WHERE, restricts entities

	Statement() = default;
	Statement(Statement &&) = default;
	Statement &operator=(Statement &&) = default;
	Statement(const Statement &o) : StatementData(o), filter(o.filter ? o.filter->Copy() : nullptr) {
	}
	Statement &operator=(const Statement &o) {
		if (this != &o) {
			StatementData::operator=(o);
			filter = o.filter ? o.filter->Copy() : nullptr;
		}
		return *this;
	}

	// The alias a filter may use for the entity; falls back to the table name.
	const std::string &EntityName() const {
		return entity_alias.empty() ? entity_table : entity_alias;
	}

	std::string ToString() const {
		std::string s;
		if (kind == StmtKind::DROP_MODEL) {
			return std::string("DROP MODEL ") + (if_exists ? "IF EXISTS " : "") + QuoteIdent(model);
		}
		if (kind == StmtKind::SHOW_MODELS) {
			return "SHOW MODELS";
		}
		if (kind == StmtKind::BACKTEST) {
			return "BACKTEST MODEL " + QuoteIdent(model);
		}
		if (kind == StmtKind::EXPLAIN) {
			return "EXPLAIN MODEL " + QuoteIdent(model);
		}
		if (kind == StmtKind::OPTIMIZE) {
			const OptimizeSpec &o = optimize;
			s = std::string(o.explain ? "EXPLAIN " : "") + "OPTIMIZE " + QuoteIdent(entity_table);
			if (!entity_alias.empty()) {
				s += " AS " + QuoteIdent(entity_alias);
			}
			if (filter) {
				s += " WHERE " + filter->ToString();
			}
			if (anchor_is_literal) {
				s += " AT " + anchor_literal.ToString();
			} else if (!anchor.name.empty()) {
				s += " AT " + anchor.ToString();
			}
			if (o.Choosing()) {
				s += " CHOOSE " + o.lever.ToString() + " IN (";
				for (size_t i = 0; i < o.choices.size(); i++) {
					s += (i ? ", " : "") + o.choices[i].ToString();
				}
				s += ")";
			}
			s += (o.maximize ? " MAXIMIZE " : " MINIMIZE ") + o.objective.ToString();
			for (size_t i = 0; i < o.constraints.size(); i++) {
				s += (i ? " AND " : " SUBJECT TO ") + o.constraints[i].ToString();
			}
			return s + " USING MODEL " + QuoteIdent(model);
		}
		s = (kind == StmtKind::TRAIN)
		        ? ((or_replace ? "TRAIN OR REPLACE MODEL " : "TRAIN MODEL ") + QuoteIdent(model) + " PREDICT ")
		        : std::string("PREDICT ");
		s += target.ToString();
		s += " FOR " + QuoteIdent(entity_table);
		if (!entity_alias.empty()) {
			s += " AS " + QuoteIdent(entity_alias);
		}
		if (!excluded.empty()) {
			s += " EXCLUDE (";
			for (size_t i = 0; i < excluded.size(); i++) {
				s += (i ? ", " : "") + excluded[i].ToString();
			}
			s += ")";
		}
		if (filter) {
			s += " WHERE " + filter->ToString();
		}
		if (anchor_is_literal) {
			s += " AT " + anchor_literal.ToString();
		} else if (!anchor.name.empty()) {
			s += " AT " + anchor.ToString();
		}
		if (horizon.present) {
			s += " HORIZON " + horizon.ToString();
		}
		if (every.present) {
			s += " EVERY " + every.ToString();
		}
		if (!graph_tables.empty()) {
			s += " USING GRAPH (";
			for (size_t i = 0; i < graph_tables.size(); i++) {
				s += (i ? ", " : "") + QuoteIdent(graph_tables[i]);
			}
			s += ")";
		}
		if (split.present) {
			s += " SPLIT TEMPORAL VALIDATE FROM " + split.validate_from.ToString() + " TEST FROM " +
			     split.test_from.ToString();
		}
		if (kind == StmtKind::PREDICT) {
			s += " USING MODEL " + QuoteIdent(model);
		}
		if (kind == StmtKind::TRAIN) {
			s += options.ToString();
		}
		return s;
	}
};

// ===========================================================================
// 4. Parser
// ===========================================================================

class Parser {
public:
	explicit Parser(const std::string &src) : src_(src), toks_(Tokenize(src)) {
	}

	// Does this input begin with a PQL statement? Used by the DuckDB parser
	// extension to decide whether to claim the tokens at all. Deliberately
	// cheap and non-throwing: a plain SELECT must fall through untouched.
	static bool Looks(const std::string &src) {
		auto t = Tokenize(src);
		if (t.empty() || t[0].kind != Tok::KEYWORD) {
			return false;
		}
		const std::string &u = t[0].upper;
		if (u == "TRAIN" || u == "PREDICT" || u == "OPTIMIZE") {
			return true;
		}
		if (u == "EXPLAIN" && t.size() > 1 && (t[1].upper == "MODEL" || t[1].upper == "OPTIMIZE")) {
			return true;
		}
		if (u == "BACKTEST" && t.size() > 1 && t[1].upper == "MODEL") {
			return true;
		}
		if (u == "DROP" && t.size() > 1 && t[1].upper == "MODEL") {
			return true;
		}
		if (u == "SHOW" && t.size() > 1 && t[1].upper == "MODELS") {
			return true;
		}
		return false;
	}

	Statement ParseStatement() {
		Statement st;
		if (AcceptKw("SHOW")) {
			ExpectKw("MODELS");
			st.kind = StmtKind::SHOW_MODELS;
			EndOfStatement("SHOW MODELS");
			return st;
		}
		if (AcceptKw("DROP")) {
			ExpectKw("MODEL");
			st.kind = StmtKind::DROP_MODEL;
			if (PeekKw("IF")) {
				Next();
				ExpectKw("EXISTS");
				st.if_exists = true;
			}
			st.model = ExpectIdent("model name");
			EndOfStatement("DROP MODEL");
			return st;
		}
		if (AcceptKw("OPTIMIZE")) {
			return ParseOptimize(false);
		}
		if (AcceptKw("EXPLAIN")) {
			if (AcceptKw("OPTIMIZE")) {
				return ParseOptimize(true);
			}
			ExpectKw("MODEL");
			st.kind = StmtKind::EXPLAIN;
			st.model = ExpectIdent("model name");
			bool seen_for = false, seen_w = false;
			while (Peek().kind != Tok::END && !(Peek().kind == Tok::PUNCT && Peek().text == ";")) {
				if (PeekKw("FOR")) {
					Once(seen_for, "FOR");
					Next();
					st.entity_table = ExpectIdent("entity table");
				} else if (PeekKw("WHERE")) {
					Once(seen_w, "WHERE");
					Next();
					st.filter = ParseFilter();
				} else {
					throw ParseError("unexpected token '" + Peek().text + "' in EXPLAIN", Peek().pos);
				}
			}
			AcceptPunct(";");
			return st;
		}
		if (AcceptKw("BACKTEST")) {
			ExpectKw("MODEL");
			st.kind = StmtKind::BACKTEST;
			st.model = ExpectIdent("model name");
			bool seen_w = false, seen_f = false, seen_t = false, seen_for = false;
			while (Peek().kind != Tok::END && !(Peek().kind == Tok::PUNCT && Peek().text == ";")) {
				if (PeekKw("FOR")) {
					Once(seen_for, "FOR");
					Next();
					st.entity_table = ExpectIdent("entity table");
				} else if (PeekKw("WHERE")) {
					Once(seen_w, "WHERE");
					Next();
					st.filter = ParseFilter();
				} else if (PeekKw("FROM")) {
					Once(seen_f, "FROM");
					Next();
					st.backtest_from = ParseLiteral();
					st.has_from = true;
				} else if (PeekKw("TO")) {
					Once(seen_t, "TO");
					Next();
					st.backtest_to = ParseLiteral();
					st.has_to = true;
				} else {
					throw ParseError("unexpected token '" + Peek().text + "' in BACKTEST", Peek().pos);
				}
			}
			AcceptPunct(";");
			return st;
		}
		if (AcceptKw("TRAIN")) {
			if (AcceptKw("OR")) {
				ExpectKw("REPLACE");
				st.or_replace = true;
			}
			ExpectKw("MODEL");
			st.kind = StmtKind::TRAIN;
			st.model = ExpectIdent("model name");
			ExpectKw("PREDICT");
		} else {
			ExpectKw("PREDICT");
			st.kind = StmtKind::PREDICT;
		}

		st.target = ParsePredictExpr();

		ExpectKw("FOR");
		st.entity_table = ExpectIdent("entity table");
		if (AcceptKw("AS")) {
			st.entity_alias = ExpectIdent("alias");
		} else if (Peek().kind == Tok::IDENT && !IsClauseStart(Peek().upper)) {
			// `FOR customers c` — the AS is optional, as in SQL.
			st.entity_alias = Next().text;
		}

		// Clauses may arrive in any order. Accepting them order-independently
		// avoids the usual frustration of a rigid statement grammar, and each
		// is rejected on a second appearance so typos surface rather than
		// silently overwriting.
		bool seen_where = false, seen_at = false, seen_h = false, seen_graph = false;
		bool seen_split = false, seen_opts = false, seen_using = false, seen_every = false;
		bool seen_excl = false;
		while (Peek().kind != Tok::END && !(Peek().kind == Tok::PUNCT && Peek().text == ";")) {
			if (PeekKw("WHERE")) {
				Once(seen_where, "WHERE");
				Next();
				st.filter = ParseFilter();
			} else if (PeekKw("AT")) {
				Once(seen_at, "AT");
				Next();
				if (Peek().kind == Tok::STRING || Peek().kind == Tok::NUMBER) {
					st.anchor_literal = ParseLiteral();
					st.anchor_is_literal = true;
				} else {
					st.anchor = ParseRef();
				}
			} else if (PeekKw("HORIZON")) {
				Once(seen_h, "HORIZON");
				Next();
				st.horizon = ParseInterval();
			} else if (PeekKw("EVERY")) {
				Once(seen_every, "EVERY");
				Next();
				st.every = ParseInterval();
			} else if (PeekKw("USING")) {
				Next();
				if (AcceptKw("GRAPH")) {
					Once(seen_graph, "USING GRAPH");
					ExpectPunct("(");
					while (true) {
						st.graph_tables.push_back(ExpectIdent("table name"));
						if (!AcceptPunct(",")) {
							break;
						}
					}
					ExpectPunct(")");
				} else {
					ExpectKw("MODEL");
					Once(seen_using, "USING MODEL");
					st.saw_using_model = true;
					const std::string named = ExpectIdent("model name");
					if (st.kind != StmtKind::TRAIN) {
						st.model = named;
					}
				}
			} else if (PeekKw("SPLIT")) {
				Once(seen_split, "SPLIT");
				Next();
				ExpectKw("TEMPORAL");
				st.split.present = true;
				ExpectKw("VALIDATE");
				ExpectKw("FROM");
				st.split.validate_from = ParseLiteral();
				ExpectKw("TEST");
				ExpectKw("FROM");
				st.split.test_from = ParseLiteral();
			} else if (PeekKw("EXCLUDE")) {
				Once(seen_excl, "EXCLUDE");
				Next();
				ExpectPunct("(");
				while (true) {
					st.excluded.push_back(ParseRef());
					if (!AcceptPunct(",")) {
						break;
					}
				}
				ExpectPunct(")");
			} else if (PeekKw("OPTIONS")) {
				Once(seen_opts, "OPTIONS");
				Next();
				ExpectPunct("(");
				while (true) {
					const Token &opt_tok = Peek();
					std::string k = ToUpper(ExpectIdent("option name"));
					if (!Options::Known(k)) {
						throw ParseError("unknown option '" + opt_tok.text + "'; expected one of " +
						                     Options::KnownList(),
						                 opt_tok.pos);
					}
					ExpectPunct("=");
					st.options.kv[k] = ParseLiteral();
					if (!AcceptPunct(",")) {
						break;
					}
				}
				ExpectPunct(")");
			} else {
				throw ParseError("unexpected token '" + Peek().text + "' in " +
				                     (st.kind == StmtKind::TRAIN ? "TRAIN" : "PREDICT") + " statement",
				                 Peek().pos);
			}
		}
		AcceptPunct(";");
		Validate(st);
		return st;
	}

private:
	// -- semantic checks that are cheap and catch the common mistakes early --
	static void Validate(Statement &st) {
		if (st.kind == StmtKind::PREDICT && st.model.empty()) {
			throw ParseError("PREDICT requires USING MODEL <name>", 0);
		}
		if (st.kind == StmtKind::TRAIN && st.saw_using_model) {
			throw ParseError("USING MODEL belongs to PREDICT; TRAIN names its model after "
			                 "TRAIN MODEL",
			                 0);
		}
		if (st.kind == StmtKind::PREDICT && st.split.present) {
			throw ParseError("SPLIT applies to TRAIN, not PREDICT", 0);
		}
		if (st.kind == StmtKind::PREDICT && !st.options.kv.empty()) {
			throw ParseError("OPTIONS apply to TRAIN, not PREDICT", 0);
		}
		if (st.kind == StmtKind::PREDICT && !st.graph_tables.empty()) {
			throw ParseError("USING GRAPH applies to TRAIN; a model already fixed its graph", 0);
		}
		if (st.horizon.present && st.horizon.amount <= 0) {
			throw ParseError("HORIZON must be a positive number of time units", 0);
		}
		// Unbounded options are a denial of service: HIDDEN squares into the weight
		// matrices, so a stray zero allocates tens of gigabytes.
		struct Bound {
			const char *key;
			double lo, hi;
		};
		if (st.options.Has("ARCH")) {
			const std::string a = ToUpper(st.options.Str("ARCH", "mlp"));
			if (!IsKnownArch(a)) {
				throw ParseError("ARCH must be 'mlp' or 'sage'", 0);
			}
		}
		static const Bound bounds[] = {{"EPOCHS", 1, 100000},
		                               {"HIDDEN", 1, 1024},
		                               {"LAYERS", 1, 2},
		                               {"LR", 1e-6, 1.0},
		                               {"BATCH", 1, 65536},
		                               {"MEAN_COLS", 0, 64},
		                               {"MAX_CATEGORIES", 0, 4096},
		                               {"L2", 0.0, 1.0},
		                               {"LOG_TARGET", 0, 1},
		                               {nullptr, 0, 0}};
		for (int i = 0; bounds[i].key; i++) {
			if (!st.options.Has(bounds[i].key)) {
				continue;
			}
			const double v = st.options.Num(bounds[i].key, bounds[i].lo);
			if (!(v >= bounds[i].lo && v <= bounds[i].hi)) {
				auto tidy = [](double d) {
					std::string t = std::to_string(d);
					if (t.find('.') != std::string::npos) {
						while (!t.empty() && t.back() == '0') {
							t.pop_back();
						}
						if (!t.empty() && t.back() == '.') {
							t.pop_back();
						}
					}
					return t;
				};
				throw ParseError(std::string(bounds[i].key) + " must be between " + tidy(bounds[i].lo) + " and " +
				                     tidy(bounds[i].hi) + " (got " + tidy(v) + ")",
				                 0);
			}
		}
		const bool forecast = IsForecast(st.target.kind);
		// PREDICT inherits AT and HORIZON from the trained model, so only TRAIN
		// has to state them.
		if (forecast && !st.horizon.present && st.kind == StmtKind::TRAIN) {
			throw ParseError(std::string(TargetKindName(st.target.kind)) +
			                     "(...) is a forecast over future rows and needs a HORIZON "
			                     "(e.g. HORIZON 30 DAYS)",
			                 0);
		}
		if (st.every.present) {
			if (!forecast) {
				throw ParseError("EVERY generates forecast anchors; it needs an aggregate target "
				                 "such as COUNT(<table>)",
				                 0);
			}
			if (!st.anchor.name.empty() || st.anchor_is_literal) {
				throw ParseError("use AT for an existing anchor column, or EVERY to generate "
				                 "anchors, not both",
				                 0);
			}
			if (st.every.amount <= 0) {
				throw ParseError("EVERY must be a positive interval", 0);
			}
		}
		if (forecast && st.anchor.name.empty() && !st.anchor_is_literal && !st.every.present &&
		    st.kind == StmtKind::TRAIN) {
			throw ParseError("a forecast needs an anchor time: add AT <column> so every feature is "
			                 "taken at or before it and the label strictly after it",
			                 0);
		}
		if (!forecast && st.horizon.present) {
			throw ParseError("HORIZON is meaningless when predicting an attribute; drop it, or "
			                 "predict an aggregate such as COUNT(<table>)",
			                 0);
		}
		if (st.kind == StmtKind::TRAIN && st.anchor_is_literal) {
			throw ParseError("TRAIN needs AT <column> (a per-row anchor), not a single timestamp", 0);
		}
		if (st.target.kind != TargetKind::COUNT && st.target.kind != TargetKind::EXISTS && forecast &&
		    st.target.ref.name.empty()) {
			throw ParseError(std::string(TargetKindName(st.target.kind)) + " needs a column, e.g. " +
			                     TargetKindName(st.target.kind) + "(orders.amount)",
			                 0);
		}
	}

	static bool IsClauseStart(const std::string &up) {
		return up == "WHERE" || up == "AT" || up == "HORIZON" || up == "USING" || up == "SPLIT" || up == "OPTIONS" ||
		       up == "EVERY" || up == "FROM" || up == "TO";
	}

	// [EXPLAIN] OPTIMIZE <table> [AS <alias>] [WHERE ...] [AT ...]
	//   [CHOOSE <column> IN (<literal>, ...)]
	//   MAXIMIZE | MINIMIZE <aggregate>
	//   [SUBJECT TO <aggregate> (<= | >=) <number> [AND ...]]
	//   USING MODEL <name>
	Statement ParseOptimize(bool explain) {
		Statement st;
		st.kind = StmtKind::OPTIMIZE;
		st.optimize.explain = explain;
		st.entity_table = ExpectIdent("table to optimize over");
		if (AcceptKw("AS")) {
			st.entity_alias = ExpectIdent("alias");
		} else if (Peek().kind == Tok::IDENT && !IsClauseStart(Peek().upper)) {
			st.entity_alias = Next().text;
		}
		bool seen_where = false, seen_at = false, seen_choose = false, seen_goal = false, seen_subject = false;
		bool seen_using = false;
		while (Peek().kind != Tok::END && !(Peek().kind == Tok::PUNCT && Peek().text == ";")) {
			if (PeekKw("WHERE")) {
				Once(seen_where, "WHERE");
				Next();
				PredScope scope(pred_ctx_, PredCtx::REFUSED);
				st.filter = ParseFilter();
			} else if (PeekKw("AT")) {
				Once(seen_at, "AT");
				Next();
				if (Peek().kind == Tok::STRING || Peek().kind == Tok::NUMBER) {
					st.anchor_literal = ParseLiteral();
					st.anchor_is_literal = true;
				} else {
					st.anchor = ParseRef();
				}
			} else if (PeekKw("CHOOSE")) {
				Once(seen_choose, "CHOOSE");
				Next();
				st.optimize.lever = ParseRef();
				if (!PeekKw("IN")) {
					throw ParseError("CHOOSE names a column and the values it may take: CHOOSE " +
					                     st.optimize.lever.ToString() + " IN ('a', 'b')",
					                 Peek().pos);
				}
				Next();
				ExpectPunct("(");
				while (true) {
					const Token &vt = Peek();
					Literal v = ParseLiteral();
					if (v.kind == Literal::Kind::NUL) {
						throw ParseError("CHOOSE cannot choose NULL; list the values a row may take", vt.pos);
					}
					if (v.kind == Literal::Kind::NUMBER && !std::isfinite(v.number)) {
						throw ParseError("a CHOOSE value is too large to represent", vt.pos);
					}
					for (const auto &prev : st.optimize.choices) {
						const bool same =
						    prev.kind == v.kind &&
						    (v.kind == Literal::Kind::NUMBER ? prev.number == v.number : prev.text == v.text);
						if (same) {
							throw ParseError("CHOOSE lists " + v.ToString() + " twice", vt.pos);
						}
					}
					if (!st.optimize.choices.empty() && st.optimize.choices[0].kind != v.kind) {
						throw ParseError("CHOOSE values must all be text or all be numbers", vt.pos);
					}
					st.optimize.choices.push_back(v);
					if (!AcceptPunct(",")) {
						break;
					}
				}
				ExpectPunct(")");
			} else if (PeekKw("MAXIMIZE") || PeekKw("MINIMIZE")) {
				Once(seen_goal, "MAXIMIZE/MINIMIZE");
				st.optimize.maximize = Next().upper == "MAXIMIZE";
				st.optimize.objective = ParseAggregate();
			} else if (PeekKw("SUBJECT")) {
				Once(seen_subject, "SUBJECT TO");
				Next();
				ExpectKw("TO");
				while (true) {
					st.optimize.constraints.push_back(ParseConstraint());
					if (!AcceptKw("AND")) {
						break;
					}
				}
			} else if (PeekKw("USING")) {
				Once(seen_using, "USING MODEL");
				Next();
				ExpectKw("MODEL");
				st.model = ExpectIdent("model name");
			} else {
				throw ParseError("unexpected token '" + Peek().text + "' in OPTIMIZE statement", Peek().pos);
			}
		}
		EndOfStatement("OPTIMIZE");
		if (!seen_goal) {
			throw ParseError("OPTIMIZE needs a goal: MAXIMIZE SUM(<expression>) or MINIMIZE SUM(<expression>)", 0);
		}
		if (st.model.empty()) {
			throw ParseError("OPTIMIZE needs USING MODEL <name>: the model whose predictions it plans with", 0);
		}
		return st;
	}

	// SUM(<expr>) | COUNT(*), then an optional FILTER (WHERE <filter>).
	OptAggregate ParseAggregate() {
		PredScope scope(pred_ctx_, PredCtx::ALLOWED);
		OptAggregate agg;
		const Token &t = Peek();
		if (PeekKw("COUNT")) {
			Next();
			ExpectPunct("(");
			if (!AcceptPunct("*")) {
				throw ParseError("COUNT counts the rows a plan acts on; write COUNT(*), and narrow it with "
				                 "FILTER (WHERE ...)",
				                 Peek().pos);
			}
			ExpectPunct(")");
			agg.count = true;
		} else if (PeekKw("SUM")) {
			Next();
			ExpectPunct("(");
			agg.expr = ParseExpr();
			ExpectPunct(")");
		} else {
			throw ParseError("expected SUM(<expression>) or COUNT(*) but found '" + t.text +
			                     "'; a plan is judged by a total over the rows it acts on",
			                 t.pos);
		}
		if (AcceptKw("FILTER")) {
			ExpectPunct("(");
			ExpectKw("WHERE");
			agg.filter = ParseFilter();
			ExpectPunct(")");
		}
		return agg;
	}

	OptConstraint ParseConstraint() {
		OptConstraint c;
		c.total = ParseAggregate();
		const Token &o = Peek();
		if (o.kind == Tok::PUNCT && (o.text == "<=" || o.text == ">=")) {
			c.at_most = o.text == "<=";
			Next();
		} else if (o.kind == Tok::PUNCT && (o.text == "<" || o.text == ">" || o.text == "=")) {
			throw ParseError("a constraint is a limit, written with <= or >=; an exact total is rarely "
			                 "reachable with whole choices, so bound it from both sides instead",
			                 o.pos);
		} else {
			throw ParseError("expected <= or >= after " + c.total.ToString(), o.pos);
		}
		const Token &lt = Peek();
		c.limit = ParseLiteral();
		if (c.limit.kind != Literal::Kind::NUMBER) {
			throw ParseError("a constraint's limit is a number", lt.pos);
		}
		if (!std::isfinite(c.limit.number)) {
			throw ParseError("a constraint's limit is too large to represent", lt.pos);
		}
		return c;
	}

	// expr := term { (+|-) term } ; term := unary { (*|/) unary } ;
	// unary := - unary | + unary | primary
	// A chain `a + b + c ...` nests one level per operator, and everything that
	// walks an expression recurses, so its length is bounded the way filter
	// nesting is.
	static const int kMaxExprOps = 1000;
	void CountOp(uint32_t pos) {
		if (++expr_ops_ > kMaxExprOps) {
			throw ParseError("expression is too long (more than 1000 operators)", pos);
		}
	}
	std::unique_ptr<Expr> ParseExpr() {
		auto lhs = ParseExprTerm();
		while (PeekPunct("+") || PeekPunct("-")) {
			CountOp(Peek().pos);
			auto node = std::unique_ptr<Expr>(new Expr());
			node->kind = Next().text == "+" ? Expr::Kind::ADD : Expr::Kind::SUB;
			node->args.push_back(std::move(lhs));
			node->args.push_back(ParseExprTerm());
			lhs = std::move(node);
		}
		return lhs;
	}
	std::unique_ptr<Expr> ParseExprTerm() {
		auto lhs = ParseExprUnary();
		while (PeekPunct("*") || PeekPunct("/")) {
			CountOp(Peek().pos);
			auto node = std::unique_ptr<Expr>(new Expr());
			node->kind = Next().text == "*" ? Expr::Kind::MUL : Expr::Kind::DIV;
			node->args.push_back(std::move(lhs));
			node->args.push_back(ParseExprUnary());
			lhs = std::move(node);
		}
		return lhs;
	}
	std::unique_ptr<Expr> ParseExprUnary() {
		if (PeekPunct("-") || PeekPunct("+")) {
			Descend(Peek().pos);
			const bool neg = Next().text == "-";
			auto inner = ParseExprUnary();
			depth_--;
			if (!neg) {
				return inner;
			}
			auto node = std::unique_ptr<Expr>(new Expr());
			node->kind = Expr::Kind::NEG;
			node->args.push_back(std::move(inner));
			return node;
		}
		return ParseExprAtom();
	}
	std::unique_ptr<Expr> ParseExprAtom() {
		const Token &t = Peek();
		auto node = std::unique_ptr<Expr>(new Expr());
		if (t.kind == Tok::NUMBER) {
			if (!std::isfinite(t.number)) {
				throw ParseError("number '" + t.text.substr(0, 40) + (t.text.size() > 40 ? "...'" : "'") +
				                     " is too large to represent",
				                 t.pos);
			}
			node->kind = Expr::Kind::NUMBER;
			node->number = t.number;
			node->text = t.text;
			Next();
			return node;
		}
		if (PeekPunct("(")) {
			Descend(t.pos);
			Next();
			auto inner = ParseExpr();
			ExpectPunct(")");
			depth_--;
			return inner;
		}
		if (PeekKw("CASE")) {
			Descend(t.pos);
			Next();
			node->kind = Expr::Kind::CASE;
			if (!PeekKw("WHEN")) {
				if (PeekKw("PREDICTION") && !(Peek(1).kind == Tok::PUNCT && Peek(1).text == ".")) {
					throw ParseError("CASE prediction WHEN ... tests a prediction for equality, which it almost never "
					                 "is exactly; write CASE WHEN prediction > ... THEN ... instead",
					                 Peek().pos);
				}
				node->simple_case = true;
				node->operand = ParseRef();
			}
			if (!PeekKw("WHEN")) {
				throw ParseError("CASE needs at least one WHEN ... THEN ...", Peek().pos);
			}
			while (AcceptKw("WHEN")) {
				if (node->simple_case) {
					auto eq = std::unique_ptr<FilterNode>(new FilterNode());
					eq->kind = FilterNode::Kind::CMP;
					eq->ref = node->operand;
					eq->op = CmpOp::EQ;
					eq->values.push_back(ParseLiteral());
					node->when.push_back(std::move(eq));
				} else {
					node->when.push_back(ParseFilter());
				}
				ExpectKw("THEN");
				node->args.push_back(ParseExpr());
			}
			if (AcceptKw("ELSE")) {
				node->has_else = true;
				node->args.push_back(ParseExpr());
			}
			ExpectKw("END");
			depth_--;
			return node;
		}
		if (t.kind == Tok::STRING) {
			throw ParseError("'" + t.text + "' is text; arithmetic needs numbers. Compare text inside CASE WHEN",
			                 t.pos);
		}
		if (t.kind == Tok::IDENT || t.kind == Tok::KEYWORD) {
			if (Peek(1).kind == Tok::PUNCT && Peek(1).text == "(") {
				const bool agg =
				    t.upper == "SUM" || t.upper == "COUNT" || t.upper == "AVG" || t.upper == "MIN" || t.upper == "MAX";
				throw ParseError(agg ? std::string("aggregates do not nest: SUM(...) takes a value per row")
				                     : "'" + t.text + "(' is not something PQL can compute; use + - * / and CASE",
				                 t.pos);
			}
			// The bare word is the model's answer for the row. A column that happens
			// to be called prediction is still reachable quoted, or qualified.
			if (t.kind == Tok::KEYWORD && t.upper == "PREDICTION" &&
			    !(Peek(1).kind == Tok::PUNCT && Peek(1).text == ".")) {
				Next();
				node->kind = Expr::Kind::PREDICTION;
				return node;
			}
			node->kind = Expr::Kind::COLUMN;
			node->ref = ParseRef();
			if (node->ref.name == "*") {
				throw ParseError("'" + node->ref.ToString() + "' is not a value", t.pos);
			}
			return node;
		}
		throw ParseError("expected a number, a column, prediction, or CASE but found '" + t.text + "'", t.pos);
	}

	void Once(bool &flag, const char *what) {
		if (flag) {
			throw ParseError(std::string("duplicate ") + what + " clause", Peek().pos);
		}
		flag = true;
	}

	PredictExpr ParsePredictExpr() {
		PredictExpr e;
		const Token &t = Peek();
		if (t.kind == Tok::KEYWORD) {
			TargetKind k = TargetKind::ATTRIBUTE;
			bool agg = true;
			if (t.upper == "COUNT") {
				k = TargetKind::COUNT;
			} else if (t.upper == "EXISTS") {
				k = TargetKind::EXISTS;
			} else if (t.upper == "SUM") {
				k = TargetKind::SUM;
			} else if (t.upper == "AVG") {
				k = TargetKind::AVG;
			} else if (t.upper == "MIN") {
				k = TargetKind::MIN;
			} else if (t.upper == "MAX") {
				k = TargetKind::MAX;
			} else {
				agg = false;
			}
			if (agg) {
				Next();
				e.kind = k;
				ExpectPunct("(");
				// COUNT(orders) | COUNT(*) | SUM(orders.amount)
				if (Peek().kind == Tok::PUNCT && Peek().text == "*") {
					Next();
					e.target_table = "*";
				} else {
					Ref r = ParseRef();
					if (r.qualifier.empty()) {
						e.target_table = r.name;
					} else {
						e.target_table = r.qualifier;
						e.ref.name = r.name;
					}
				}
				if (AcceptKw("WHERE")) {
					e.filter = ParseFilter();
				}
				ExpectPunct(")");
				return e;
			}
		}
		e.kind = TargetKind::ATTRIBUTE;
		e.ref = ParseRef();
		return e;
	}

	Ref ParseRef() {
		Ref r;
		r.name = ExpectIdent("column or table name");
		if (Peek().kind == Tok::PUNCT && Peek().text == ".") {
			Next();
			r.qualifier = r.name;
			if (Peek().kind == Tok::PUNCT && Peek().text == "*") {
				Next();
				r.name = "*";
			} else {
				r.name = ExpectIdent("column name");
			}
		}
		return r;
	}

	Literal ParseLiteral() {
		Literal l;
		// A sign in front of a number. Without this, `WHERE balance < -100` was a
		// syntax error, and no schema holding a negative quantity could be
		// filtered at all.
		if (Peek().kind == Tok::PUNCT && (Peek().text == "-" || Peek().text == "+") && Peek(1).kind == Tok::NUMBER) {
			const bool neg = Peek().text == "-";
			Next();
			const Token &num = Peek();
			l.kind = Literal::Kind::NUMBER;
			l.number = neg ? -num.number : num.number;
			l.text = (neg ? "-" : "") + num.text;
			Next();
			return l;
		}
		const Token &t = Peek();
		if (t.kind == Tok::NUMBER) {
			l.kind = Literal::Kind::NUMBER;
			l.number = t.number;
			l.text = t.text;
			Next();
			return l;
		}
		if (t.kind == Tok::STRING) {
			l.kind = Literal::Kind::STRING;
			l.text = t.text;
			Next();
			return l;
		}
		if (t.kind == Tok::KEYWORD && t.upper == "NULL") {
			l.kind = Literal::Kind::NUL;
			Next();
			return l;
		}
		// A bare identifier in literal position is almost always a forgotten
		// quote; say so rather than emitting "unexpected token".
		if (t.kind == Tok::IDENT) {
			throw ParseError(
			    "expected a literal but found identifier '" + t.text + "'; string literals use single quotes", t.pos);
		}
		throw ParseError("expected a literal", t.pos);
	}

	Interval ParseInterval() {
		Interval iv;
		const Token &t = Peek();
		if (t.kind != Tok::NUMBER) {
			throw ParseError("HORIZON expects a number followed by a unit, e.g. 30 DAYS", t.pos);
		}
		// Bounded before the cast: a 20-digit amount is undefined to convert and
		// came out negative, slipping past the positivity check.
		if (!(t.number >= 0.0) || t.number > 1.0e9 || t.number != std::floor(t.number)) {
			throw ParseError("interval amount must be a whole number of units, at most 1000000000", t.pos);
		}
		iv.amount = (int64_t)t.number;
		Next();
		const Token &u = Peek();
		// Units arrive as keywords when known and as identifiers otherwise; accept
		// both here so an unknown unit is named in the error rather than producing
		// a generic "needs a unit".
		if (u.kind != Tok::KEYWORD && u.kind != Tok::IDENT) {
			throw ParseError("HORIZON needs a unit (DAYS, WEEKS, MONTHS, YEARS, HOURS)", u.pos);
		}
		std::string up = u.upper;
		if (!up.empty() && up.back() == 'S') {
			up.pop_back();
		}
		if (up != "DAY" && up != "WEEK" && up != "MONTH" && up != "YEAR" && up != "HOUR") {
			throw ParseError("unknown horizon unit '" + u.text + "'; expected DAYS, WEEKS, MONTHS, YEARS or HOURS",
			                 u.pos);
		}
		iv.unit = up;
		iv.present = true;
		// Micros() multiplies in int64; keep the total inside it with a wide margin.
		if (double(iv.amount) * (up == "HOUR"    ? 3600.0e6
		                         : up == "DAY"   ? 86400.0e6
		                         : up == "WEEK"  ? 7 * 86400.0e6
		                         : up == "MONTH" ? 30.436875 * 86400.0e6
		                                         : 365.2425 * 86400.0e6) >
		    1.0e17) {
			throw ParseError("interval is too long; at most about 3000 years", t.pos);
		}
		Next();
		return iv;
	}

	// filter := or_expr ; standard precedence: NOT binds tighter than AND,
	// AND tighter than OR.
	std::unique_ptr<FilterNode> ParseFilter() {
		return ParseOr();
	}
	std::unique_ptr<FilterNode> ParseOr() {
		auto lhs = ParseAnd();
		if (!PeekKw("OR")) {
			return lhs;
		}
		auto node = std::unique_ptr<FilterNode>(new FilterNode());
		node->kind = FilterNode::Kind::OR;
		node->children.push_back(std::move(lhs));
		while (AcceptKw("OR")) {
			node->children.push_back(ParseAnd());
		}
		return node;
	}
	std::unique_ptr<FilterNode> ParseAnd() {
		auto lhs = ParseNot();
		if (!PeekKw("AND")) {
			return lhs;
		}
		auto node = std::unique_ptr<FilterNode>(new FilterNode());
		node->kind = FilterNode::Kind::AND;
		node->children.push_back(std::move(lhs));
		while (AcceptKw("AND")) {
			node->children.push_back(ParseNot());
		}
		return node;
	}
	// Every level of parentheses or NOT is a few stack frames, and a statement
	// can be as long as the caller likes; a runaway nest must fail as a parse
	// error rather than overrun the thread's stack.
	static const int kMaxFilterDepth = 200;
	void Descend(uint32_t pos) {
		if (++depth_ > kMaxFilterDepth) {
			throw ParseError("filter is nested too deeply (more than 200 levels)", pos);
		}
	}
	std::unique_ptr<FilterNode> ParseNot() {
		if (PeekKw("NOT")) {
			Descend(Peek().pos);
			Next();
			auto node = std::unique_ptr<FilterNode>(new FilterNode());
			node->kind = FilterNode::Kind::NOT;
			node->children.push_back(ParseNot());
			depth_--;
			return node;
		}
		return ParsePrimary();
	}
	std::unique_ptr<FilterNode> ParsePrimary() {
		if (PeekPunct("(")) {
			Descend(Peek().pos);
			Next();
			auto n = ParseOr();
			ExpectPunct(")");
			depth_--;
			return n;
		}
		auto node = std::unique_ptr<FilterNode>(new FilterNode());
		node->kind = FilterNode::Kind::CMP;
		if (pred_ctx_ != PredCtx::NONE && PeekKw("PREDICTION") &&
		    !(Peek(1).kind == Tok::PUNCT && Peek(1).text == ".")) {
			if (pred_ctx_ == PredCtx::REFUSED) {
				throw ParseError("WHERE picks the rows before anything is predicted, so it cannot test prediction; "
				                 "put the condition in a total instead, e.g. COUNT(*) FILTER (WHERE prediction > 0.5), "
				                 "or quote \"prediction\" to mean a column",
				                 Peek().pos);
			}
			Next();
			node->prediction = true;
			node->ref.name = "prediction";
		} else {
			node->ref = ParseRef();
		}
		const Token &t = Peek();
		if (t.kind == Tok::KEYWORD && t.upper == "IS") {
			Next();
			bool neg = AcceptKw("NOT");
			ExpectKw("NULL");
			node->op = neg ? CmpOp::IS_NOT_NULL : CmpOp::IS_NULL;
			return node;
		}
		bool neg_in = false;
		if (t.kind == Tok::KEYWORD && t.upper == "NOT") {
			Next();
			neg_in = true;
		}
		if (PeekKw("IN")) {
			Next();
			ExpectPunct("(");
			node->op = neg_in ? CmpOp::NOT_IN : CmpOp::IN;
			while (true) {
				node->values.push_back(ParseLiteral());
				if (!AcceptPunct(",")) {
					break;
				}
			}
			ExpectPunct(")");
			return node;
		}
		if (neg_in) {
			throw ParseError("expected IN after NOT", Peek().pos);
		}
		const Token &o = Peek();
		if (o.kind != Tok::PUNCT) {
			throw ParseError("expected a comparison operator after " + node->ref.ToString(), o.pos);
		}
		if (o.text == "=") {
			node->op = CmpOp::EQ;
		} else if (o.text == "!=" || o.text == "<>") {
			node->op = CmpOp::NE;
		} else if (o.text == "<") {
			node->op = CmpOp::LT;
		} else if (o.text == "<=") {
			node->op = CmpOp::LE;
		} else if (o.text == ">") {
			node->op = CmpOp::GT;
		} else if (o.text == ">=") {
			node->op = CmpOp::GE;
		} else {
			throw ParseError("unknown comparison operator '" + o.text + "'", o.pos);
		}
		Next();
		node->values.push_back(ParseLiteral());
		return node;
	}

	// -- token helpers --
	const Token &Peek(size_t ahead = 0) const {
		size_t i = idx_ + ahead;
		return i < toks_.size() ? toks_[i] : toks_.back();
	}
	const Token &Next() {
		const Token &t = Peek();
		if (idx_ < toks_.size() - 1) {
			idx_++;
		}
		return t;
	}
	bool PeekKw(const char *kw) const {
		const Token &t = Peek();
		return t.kind == Tok::KEYWORD && t.upper == kw;
	}
	bool AcceptKw(const char *kw) {
		if (PeekKw(kw)) {
			Next();
			return true;
		}
		return false;
	}
	void ExpectKw(const char *kw) {
		if (!AcceptKw(kw)) {
			throw ParseError(std::string("expected ") + kw + " but found '" + Peek().text + "'", Peek().pos);
		}
	}
	// Nothing may follow a complete statement but a semicolon. `DROP MODEL a b c`
	// used to drop `a` and discard the rest without a word.
	void EndOfStatement(const char *what) {
		AcceptPunct(";");
		if (Peek().kind != Tok::END) {
			throw ParseError("unexpected token '" + Peek().text + "' after " + what, Peek().pos);
		}
	}
	bool PeekPunct(const char *p) const {
		const Token &t = Peek();
		return t.kind == Tok::PUNCT && t.text == p;
	}
	bool AcceptPunct(const char *p) {
		if (PeekPunct(p)) {
			Next();
			return true;
		}
		return false;
	}
	void ExpectPunct(const char *p) {
		if (!AcceptPunct(p)) {
			throw ParseError(std::string("expected '") + p + "' but found '" + Peek().text + "'", Peek().pos);
		}
	}
	std::string ExpectIdent(const char *what) {
		const Token &t = Peek();
		// Keywords are accepted as identifiers where a name is required, so a
		// column called `count` or a table called `test` stays usable.
		if (t.kind == Tok::IDENT || t.kind == Tok::KEYWORD) {
			Next();
			return t.text;
		}
		throw ParseError(std::string("expected ") + what + " but found '" + t.text + "'", t.pos);
	}

	std::string src_;
	std::vector<Token> toks_;
	size_t idx_ = 0;
	int depth_ = 0;
	int expr_ops_ = 0;
	// What a bare `prediction` means in a filter: a column everywhere but
	// OPTIMIZE, the model's answer inside its totals, and an error in its WHERE,
	// which is decided before anything is predicted.
	enum class PredCtx : uint8_t { NONE, ALLOWED, REFUSED } pred_ctx_ = PredCtx::NONE;
	struct PredScope {
		PredCtx &slot;
		PredCtx saved;
		PredScope(PredCtx &s, PredCtx v) : slot(s), saved(s) {
			slot = v;
		}
		~PredScope() {
			slot = saved;
		}
	};
};

inline Statement Parse(const std::string &src) {
	Parser p(src);
	return p.ParseStatement();
}

} // namespace pql
