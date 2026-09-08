// pql.hpp — Predictive Query Language: a single-file extension of SQL with
// training and prediction statements, in the style of extras/qfp/src/qfp.hpp.
//
// The file is deliberately self-contained and DuckDB-agnostic: it never
// includes a DuckDB header. The binding in
// extension/core_functions/pql/pql_functions.cpp reads raw typed pointers out
// of DataChunk buffers (FlatVector::GetData) and fills the column arrays below.
//
// On copying, precisely: no Value is materialised per cell, and tables and
// columns that cannot influence the model are never read at all. The columns
// that ARE used are copied once into the layout the aggregation wants. That
// last copy is not avoidable for training, which makes many epochs of random
// access by entity over time-sorted indexes, while DuckDB's chunks are
// transient and offer no stable row-addressable pointer to retain. A streaming
// PREDICT that never materialises is possible and is not what this does today.
//
// ---------------------------------------------------------------------------
// GRAMMAR
// ---------------------------------------------------------------------------
//
//   train_stmt :=
//       TRAIN MODEL <ident>
//       PREDICT <predict_expr>
//       FOR <table> [AS <ident>]
//       [ WHERE <filter> ]
//       [ AT <ident> ]                     -- anchor-time column on the entity
//       [ HORIZON <interval> ]             -- future window for forecast targets
//       [ USING GRAPH ( <table> [, ...] ) ]-- explicit graph allow-list
//       [ SPLIT TEMPORAL VALIDATE FROM <literal> TEST FROM <literal> ]
//       [ OPTIONS ( <ident> = <literal> [, ...] ) ]
//
//   predict_stmt :=
//       PREDICT <predict_expr>
//       FOR <table> [AS <ident>]
//       [ WHERE <filter> ]
//       [ AT <ident> | AT <literal> ]
//       [ HORIZON <interval> ]
//       USING MODEL <ident>
//
//   predict_expr :=
//         <table>.<column>                        -- attribute imputation
//       | COUNT  ( <table> [ WHERE <filter> ] )   -- forecast: how many
//       | EXISTS ( <table> [ WHERE <filter> ] )   -- forecast: any at all
//       | SUM    ( <table>.<column> [ WHERE <filter> ] )
//       | AVG    ( <table>.<column> [ WHERE <filter> ] )
//       | MIN    ( <table>.<column> [ WHERE <filter> ] )
//       | MAX    ( <table>.<column> [ WHERE <filter> ] )
//
//   interval := <integer> ( DAY | DAYS | WEEK | WEEKS | MONTH | MONTHS |
//                           YEAR | YEARS | HOUR | HOURS )
//
//   filter   := disjunction of conjunctions of comparisons; see ParseFilter.
//              <ref> ( = | != | <> | < | <= | > | >= ) <literal>
//              <ref> IN ( <literal> [, ...] )
//              <ref> IS [NOT] NULL
//              combined with AND / OR and parenthesised freely.
//
// Two prediction shapes fall out of predict_expr, and they are genuinely
// different problems, so the parser keeps them distinct:
//
//   ATTRIBUTE  `PREDICT users.churn FOR users WHERE ...`
//              Impute a column that exists on the entity row. No horizon.
//
//   FORECAST   `PREDICT COUNT(orders WHERE orders.status='paid')
//               FOR users AT signup_ts HORIZON 30 DAYS`
//              The target is an aggregate over rows of a *related* table that
//              land strictly inside (anchor, anchor + horizon]. The label is
//              computed by the engine, never read from a column, which is what
//              makes leakage structurally impossible: every feature is drawn
//              from (-inf, anchor] and every label from (anchor, anchor+h].
//
// ---------------------------------------------------------------------------
// EXAMPLES
// ---------------------------------------------------------------------------
//
//   TRAIN MODEL churn
//     PREDICT EXISTS(orders) FOR customers AS c
//     AT c.last_seen HORIZON 90 DAYS
//     WHERE c.region = 'EU'
//     SPLIT TEMPORAL VALIDATE FROM '2023-01-01' TEST FROM '2023-07-01'
//     OPTIONS (epochs = 40, hidden = 128, layers = 2);
//
//   PREDICT EXISTS(orders) FOR customers AS c
//     AT c.last_seen HORIZON 90 DAYS
//     WHERE c.id = 42
//     USING MODEL churn;
//
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
	std::string text;   // identifiers keep original case
	std::string upper;  // uppercased, for keyword comparison
	double number = 0;
	uint32_t pos = 0;   // byte offset in the source, for error messages
};

inline std::string ToUpper(const std::string &s) {
	std::string o(s);
	for (auto &c : o) {
		c = char(std::toupper((unsigned char)c));
	}
	return o;
}

// The reserved words PQL itself dispatches on. Anything else that arrives as
// an identifier stays an identifier, so a column called `horizon` is still
// addressable as "horizon" when quoted.
inline bool IsPqlKeyword(const std::string &up) {
	static const char *kw[] = {"TRAIN",  "MODEL",  "PREDICT", "FOR",    "WHERE",  "AT",
	                           "HORIZON", "USING", "GRAPH",   "SPLIT",  "TEMPORAL", "VALIDATE",
	                           "TEST",   "FROM",   "OPTIONS", "AS",     "AND",    "OR",
	                           "NOT",    "IN",     "IS",      "NULL",   "COUNT",  "EXISTS",
	                           "SUM",    "AVG",    "MIN",     "MAX",    "DAY",    "DAYS",
	                           "WEEK",   "WEEKS",  "MONTH",   "MONTHS", "YEAR",   "YEARS",
	                           "HOUR",   "HOURS",  "DROP",    "SHOW",   "MODELS", "EVERY",
	                           "BACKTEST", "TO",
	                           nullptr};
	for (int i = 0; kw[i]; i++) {
		if (up == kw[i]) {
			return true;
		}
	}
	return false;
}

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
		Token t;
		t.pos = i;
		// single-quoted string literal, '' escapes a quote
		if (c == '\'') {
			i++;
			std::string v;
			while (i < n) {
				if (src[i] == '\'') {
					if (i + 1 < n && src[i + 1] == '\'') {
						v.push_back('\'');
						i += 2;
						continue;
					}
					i++;
					break;
				}
				v.push_back(src[i++]);
			}
			t.kind = Tok::STRING;
			t.text = v;
			t.upper = ToUpper(v);
			out.push_back(std::move(t));
			continue;
		}
		// double-quoted identifier keeps its exact spelling and never becomes a keyword
		if (c == '"') {
			i++;
			std::string v;
			while (i < n && src[i] != '"') {
				v.push_back(src[i++]);
			}
			if (i < n) {
				i++;
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
			t.number = std::strtod(t.text.c_str(), nullptr);
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

// A qualified reference: [table.]column, where table may be an alias.
struct Ref {
	std::string qualifier; // may be empty
	std::string name;

	std::string ToString() const {
		return qualifier.empty() ? name : qualifier + "." + name;
	}
};

enum class CmpOp : uint8_t { EQ, NE, LT, LE, GT, GE, IN, NOT_IN, IS_NULL, IS_NOT_NULL };

struct Literal {
	enum class Kind : uint8_t { NUMBER, STRING, NUL } kind = Kind::NUL;
	double number = 0;
	std::string text;

	std::string ToString() const {
		switch (kind) {
		case Kind::NUMBER:
			return text;
		case Kind::STRING:
			return "'" + text + "'";
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
	CmpOp op = CmpOp::EQ;
	std::vector<Literal> values;
	// AND / OR / NOT
	std::vector<std::unique_ptr<FilterNode>> children;

	std::unique_ptr<FilterNode> Copy() const {
		auto o = std::unique_ptr<FilterNode>(new FilterNode());
		o->kind = kind;
		o->ref = ref;
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
		std::string s = ref.ToString();
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
		s += ref.name.empty() ? target_table : (target_table + "." + ref.name);
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

struct Options {
	std::unordered_map<std::string, Literal> kv;

	// A silently ignored option reads as "my setting had no effect", which is the
	// most common way to waste an afternoon.
	static bool Known(const std::string &k) {
		static const char *ok[] = {"EPOCHS", "HIDDEN",    "LR",     "BATCH",
		                           "SEED",   "MEAN_COLS", "ARCH",   "LAYERS",
		                           nullptr};
		for (int i = 0; ok[i]; i++) {
			if (k == ok[i]) {
				return true;
			}
		}
		return false;
	}
	static std::string KnownList() {
		return "EPOCHS, HIDDEN, LR, BATCH, SEED, MEAN_COLS, ARCH, LAYERS";
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

enum class StmtKind : uint8_t { TRAIN, PREDICT, BACKTEST, DROP_MODEL, SHOW_MODELS };

struct Statement {
	StmtKind kind = StmtKind::PREDICT;
	std::string model;      // TRAIN MODEL <model> / USING MODEL <model>
	PredictExpr target;
	std::string entity_table;
	std::string entity_alias;
	std::unique_ptr<FilterNode> filter; // outer WHERE, restricts entities
	Ref anchor;                         // AT <column>
	Literal anchor_literal;             // AT '2024-01-01' (prediction only)
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

	Statement() = default;
	Statement(Statement &&) = default;
	Statement &operator=(Statement &&) = default;
	Statement(const Statement &o) {
		*this = o;
	}
	Statement &operator=(const Statement &o) {
		if (this == &o) {
			return *this;
		}
		kind = o.kind;
		model = o.model;
		target = o.target;
		entity_table = o.entity_table;
		entity_alias = o.entity_alias;
		filter = o.filter ? o.filter->Copy() : nullptr;
		anchor = o.anchor;
		anchor_literal = o.anchor_literal;
		anchor_is_literal = o.anchor_is_literal;
		horizon = o.horizon;
		every = o.every;
		graph_tables = o.graph_tables;
		split = o.split;
		options = o.options;
		saw_using_model = o.saw_using_model;
		backtest_from = o.backtest_from;
		backtest_to = o.backtest_to;
		has_from = o.has_from;
		has_to = o.has_to;
		return *this;
	}

	// The alias a filter may use for the entity; falls back to the table name.
	const std::string &EntityName() const {
		return entity_alias.empty() ? entity_table : entity_alias;
	}

	std::string ToString() const {
		std::string s;
		if (kind == StmtKind::DROP_MODEL) {
			return "DROP MODEL " + model;
		}
		if (kind == StmtKind::SHOW_MODELS) {
			return "SHOW MODELS";
		}
		if (kind == StmtKind::BACKTEST) {
			return "BACKTEST MODEL " + model;
		}
		s = (kind == StmtKind::TRAIN) ? ("TRAIN MODEL " + model + " PREDICT ") : "PREDICT ";
		s += target.ToString();
		s += " FOR " + entity_table;
		if (!entity_alias.empty()) {
			s += " AS " + entity_alias;
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
				s += (i ? ", " : "") + graph_tables[i];
			}
			s += ")";
		}
		if (split.present) {
			s += " SPLIT TEMPORAL VALIDATE FROM " + split.validate_from.ToString() + " TEST FROM " +
			     split.test_from.ToString();
		}
		if (kind == StmtKind::PREDICT) {
			s += " USING MODEL " + model;
		}
		return s;
	}
};

// ===========================================================================
// 4. Parser
// ===========================================================================

struct ParseError : std::runtime_error {
	uint32_t position;
	ParseError(const std::string &msg, uint32_t pos) : std::runtime_error(msg), position(pos) {
	}
};

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
		if (u == "TRAIN" || u == "PREDICT") {
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
			AcceptPunct(";");
			return st;
		}
		if (AcceptKw("DROP")) {
			ExpectKw("MODEL");
			st.kind = StmtKind::DROP_MODEL;
			st.model = ExpectIdent("model name");
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
			if (a != "MLP" && a != "SAGE") {
				throw ParseError("ARCH must be 'mlp' or 'sage'", 0);
			}
		}
		static const Bound bounds[] = {{"EPOCHS", 1, 100000},  {"HIDDEN", 1, 1024},
		                               {"LAYERS", 1, 2},
		                               {"LR", 1e-6, 1.0},      {"BATCH", 1, 65536},
		                               {"MEAN_COLS", 0, 64},   {nullptr, 0, 0}};
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
				throw ParseError(std::string(bounds[i].key) + " must be between " +
				                     tidy(bounds[i].lo) + " and " + tidy(bounds[i].hi) +
				                     " (got " + tidy(v) + ")",
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
		if (st.target.kind != TargetKind::COUNT && st.target.kind != TargetKind::EXISTS &&
		    forecast && st.target.ref.name.empty()) {
			throw ParseError(std::string(TargetKindName(st.target.kind)) +
			                     " needs a column, e.g. " + TargetKindName(st.target.kind) +
			                     "(orders.amount)",
			                 0);
		}
	}

	static bool IsClauseStart(const std::string &up) {
		return up == "WHERE" || up == "AT" || up == "HORIZON" || up == "USING" || up == "SPLIT" ||
		       up == "OPTIONS" || up == "EVERY" || up == "FROM" || up == "TO";
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
			throw ParseError("expected a literal but found identifier '" + t.text +
			                     "'; string literals use single quotes",
			                 t.pos);
		}
		throw ParseError("expected a literal", t.pos);
	}

	Interval ParseInterval() {
		Interval iv;
		const Token &t = Peek();
		if (t.kind != Tok::NUMBER) {
			throw ParseError("HORIZON expects a number followed by a unit, e.g. 30 DAYS", t.pos);
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
			throw ParseError("unknown horizon unit '" + u.text +
			                     "'; expected DAYS, WEEKS, MONTHS, YEARS or HOURS",
			                 u.pos);
		}
		iv.unit = up;
		iv.present = true;
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
	std::unique_ptr<FilterNode> ParseNot() {
		if (AcceptKw("NOT")) {
			auto node = std::unique_ptr<FilterNode>(new FilterNode());
			node->kind = FilterNode::Kind::NOT;
			node->children.push_back(ParseNot());
			return node;
		}
		return ParsePrimary();
	}
	std::unique_ptr<FilterNode> ParsePrimary() {
		if (AcceptPunct("(")) {
			auto n = ParseOr();
			ExpectPunct(")");
			return n;
		}
		auto node = std::unique_ptr<FilterNode>(new FilterNode());
		node->kind = FilterNode::Kind::CMP;
		node->ref = ParseRef();
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
			throw ParseError(std::string("expected ") + kw + " but found '" + Peek().text + "'",
			                 Peek().pos);
		}
	}
	bool AcceptPunct(const char *p) {
		const Token &t = Peek();
		if (t.kind == Tok::PUNCT && t.text == p) {
			Next();
			return true;
		}
		return false;
	}
	void ExpectPunct(const char *p) {
		if (!AcceptPunct(p)) {
			throw ParseError(std::string("expected '") + p + "' but found '" + Peek().text + "'",
			                 Peek().pos);
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
};

inline Statement Parse(const std::string &src) {
	Parser p(src);
	return p.ParseStatement();
}

// ===========================================================================
// 5. Materialized data
// ===========================================================================
//
// The binding streams DataChunks in and appends them here. Columns are kept as
// parallel arrays rather than rows: every downstream pass (aggregation,
// encoding, message passing) walks one column at a time, so this is both the
// natural layout and the cache-friendly one. Categorical values are stored as
// dictionary codes with the dictionary held alongside, so string comparison
// happens once at load rather than per row per epoch.

struct Dictionary {
	std::vector<std::string> values;                      // code -> text, 0 = unknown/null
	std::unordered_map<std::string, uint32_t> index;      // text -> code

	Dictionary() {
		values.push_back(""); // code 0 reserved for unknown
	}
	uint32_t Intern(const std::string &v) {
		auto it = index.find(v);
		if (it != index.end()) {
			return it->second;
		}
		uint32_t code = (uint32_t)values.size();
		values.push_back(v);
		index.emplace(v, code);
		return code;
	}
	// Lookup without interning: an unseen literal in a filter must not silently
	// become a new category, or `WHERE status = 'typo'` would match nothing yet
	// grow the vocabulary.
	uint32_t Lookup(const std::string &v) const {
		auto it = index.find(v);
		return it == index.end() ? 0u : it->second;
	}
	size_t size() const {
		return values.size();
	}
};

struct Column {
	std::string name;
	ColType type = ColType::INVALID;
	std::vector<double> num;    // numeric payload (also holds timestamps as micros)
	std::vector<uint32_t> code; // categorical payload
	std::vector<uint8_t> valid; // 1 = present
	Dictionary dict;

	size_t size() const {
		return valid.size();
	}
	bool IsNumeric() const {
		return type == ColType::INT64 || type == ColType::DOUBLE || type == ColType::BOOL ||
		       type == ColType::TIMESTAMP;
	}
};

struct Frame {
	std::string name;
	std::vector<Column> columns;
	size_t nrows = 0;

	int Find(const std::string &col) const {
		for (size_t i = 0; i < columns.size(); i++) {
			// Column references are matched case-insensitively, as in SQL.
			if (columns[i].name.size() == col.size() &&
			    ToUpper(columns[i].name) == ToUpper(col)) {
				return (int)i;
			}
		}
		return -1;
	}
	const Column &At(const std::string &col) const {
		int i = Find(col);
		if (i < 0) {
			throw std::runtime_error("pql: no column '" + col + "' on table '" + name + "'");
		}
		return columns[(size_t)i];
	}

};

// ===========================================================================
// 6. Schema graph
// ===========================================================================

struct ForeignKey {
	std::string child_table, child_column;
	std::string parent_table, parent_column;
};

// One directed traversal from a parent row to the child rows pointing at it,
// with the child rows pre-sorted by time. Sorting once at build time is what
// turns every windowed aggregate into two binary searches instead of a scan,
// which is the whole reason these features are affordable.
struct Link {
	size_t fk_index = 0;
	std::string child_table, parent_table;
	int child_key_col = -1, parent_key_col = -1, child_time_col = -1;
	// The child's clock, one entry per child row. A fact table often carries no
	// timestamp of its own (line_items are dated by their transaction), so when
	// the child has none this is filled by following a foreign key to a dated
	// parent. Everything downstream reads this instead of a column.
	std::vector<double> ctime;
	std::string time_source; // for reporting: "" when the child is self-dated
	double Time(uint32_t child_row) const {
		return ctime[child_row];
	}
	// CSR arena: child rows grouped by parent in one flat array, ascending in
	// time within each group. One allocation instead of one per parent, and the
	// per-parent slice is contiguous, so scans stream instead of chasing.
	std::vector<uint32_t> flat;
	std::vector<uint32_t> off; // size nparents + 1
	bool dated = false;
	bool duplicate_parent_keys = false;

	uint32_t Begin(uint32_t p) const {
		return off[p];
	}
	uint32_t End(uint32_t p) const {
		return off[p + 1];
	}
	uint32_t Count(uint32_t p) const {
		return off[p + 1] - off[p];
	}
};

// LSD radix sort of one link's buckets on the child timestamp, 11/11/10 bits
// over the order-preserving unsigned image of the float key. Linear in the
// number of children, versus the comparison sort's n log n, and every pass is a
// sequential read plus a scattered write rather than a gather per comparison.
inline void SortBucketsByTime(Link &lk, size_t nparents) {
	// Size every buffer once, to the largest bucket, so the loop below never
	// touches the allocator.
	uint32_t widest = 0;
	for (size_t p = 0; p < nparents; p++) {
		widest = std::max(widest, lk.off[p + 1] - lk.off[p]);
	}
	std::vector<uint32_t> keys(widest), rows(widest), keys2(widest), rows2(widest);
	std::vector<uint32_t> hist(2048 + 1);
	for (size_t p = 0; p < nparents; p++) {
		const uint32_t b = lk.off[p], e = lk.off[p + 1];
		const uint32_t n = e - b;
		if (n < 2) {
			continue;
		}
		if (n < 64) {
			std::sort(lk.flat.begin() + b, lk.flat.begin() + e,
			          [&](uint32_t x, uint32_t y) { return lk.Time(x) < lk.Time(y); });
			continue;
		}
		for (uint32_t i = 0; i < n; i++) {
			const float f = (float)lk.Time(lk.flat[b + i]);
			uint32_t u;
			std::memcpy(&u, &f, sizeof(u));
			// Flip so that the unsigned order matches the float order for both signs.
			u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
			keys[i] = u;
			rows[i] = lk.flat[b + i];
		}
		const int bits[3] = {11, 11, 10};
		int shift = 0;
		for (int pass = 0; pass < 3; pass++) {
			const uint32_t buckets = 1u << bits[pass];
			const uint32_t mask = buckets - 1u;
			std::fill(hist.begin(), hist.begin() + buckets + 1, 0u);
			for (uint32_t i = 0; i < n; i++) {
				hist[((keys[i] >> shift) & mask) + 1]++;
			}
			for (uint32_t i = 0; i < buckets; i++) {
				hist[i + 1] += hist[i];
			}
			for (uint32_t i = 0; i < n; i++) {
				const uint32_t d = hist[(keys[i] >> shift) & mask]++;
				keys2[d] = keys[i];
				rows2[d] = rows[i];
			}
			keys.swap(keys2);
			rows.swap(rows2);
			shift += bits[pass];
		}
		for (uint32_t i = 0; i < n; i++) {
			lk.flat[b + i] = rows[i];
		}
	}
}

struct Database {
	std::vector<Frame> tables;
	std::vector<ForeignKey> fks;
	std::vector<Link> links;

	int Find(const std::string &t) const {
		for (size_t i = 0; i < tables.size(); i++) {
			if (ToUpper(tables[i].name) == ToUpper(t)) {
				return (int)i;
			}
		}
		return -1;
	}
	Frame &At(const std::string &t) {
		int i = Find(t);
		if (i < 0) {
			throw std::runtime_error("pql: unknown table '" + t + "'");
		}
		return tables[(size_t)i];
	}
	const Frame &At(const std::string &t) const {
		int i = Find(t);
		if (i < 0) {
			throw std::runtime_error("pql: unknown table '" + t + "'");
		}
		return tables[(size_t)i];
	}

	// Guess the time column of a table: the first TIMESTAMP column, else a
	// column whose name looks temporal. Explicit AT always wins over this.
	static int GuessTimeColumn(const Frame &f) {
		for (size_t i = 0; i < f.columns.size(); i++) {
			if (f.columns[i].type == ColType::TIMESTAMP) {
				return (int)i;
			}
		}
		static const char *hints[] = {"DATE", "TIME", "TIMESTAMP", "CREATED", "_AT", nullptr};
		for (size_t i = 0; i < f.columns.size(); i++) {
			std::string up = ToUpper(f.columns[i].name);
			for (int h = 0; hints[h]; h++) {
				if (up.find(hints[h]) != std::string::npos) {
					return (int)i;
				}
			}
		}
		return -1;
	}

	// Build the parent -> children indexes for every declared foreign key,
	// restricted to `allow` when the query pinned a graph allow-list.
	void BuildLinks(const std::vector<std::string> &allow) {
		links.clear();
		for (size_t k = 0; k < fks.size(); k++) {
			const ForeignKey &fk = fks[k];
			if (!allow.empty()) {
				bool ok = false;
				for (auto &a : allow) {
					if (ToUpper(a) == ToUpper(fk.child_table)) {
						ok = true;
						break;
					}
				}
				if (!ok) {
					continue;
				}
			}
			int ci = Find(fk.child_table), pi = Find(fk.parent_table);
			if (ci < 0 || pi < 0) {
				continue;
			}
			const Frame &child = tables[(size_t)ci];
			const Frame &parent = tables[(size_t)pi];
			int ckey = child.Find(fk.child_column), pkey = parent.Find(fk.parent_column);
			if (ckey < 0 || pkey < 0) {
				continue;
			}
			Link lk;
			lk.fk_index = k;
			lk.child_table = fk.child_table;
			lk.parent_table = fk.parent_table;
			lk.child_key_col = ckey;
			lk.parent_key_col = pkey;
			lk.child_time_col = GuessTimeColumn(child);
			lk.dated = lk.child_time_col >= 0;
			if (lk.dated) {
				const Column &own = child.columns[(size_t)lk.child_time_col];
				lk.ctime.assign(child.nrows, 0.0);
				for (size_t r = 0; r < child.nrows; r++) {
					lk.ctime[r] = own.valid[r] ? own.num[r] : std::numeric_limits<double>::quiet_NaN();
				}
			} else {
				// Inherit a clock: follow one foreign key out of the child to a table
				// that is dated, and date each child row by its parent.
				for (const auto &fk2 : fks) {
					if (ToUpper(fk2.child_table) != ToUpper(fk.child_table)) {
						continue;
					}
					// Never inherit a clock from the entity this link hangs off: every
					// child would then carry the entity's own anchor, so nothing could
					// ever fall strictly after it and the horizon would be meaningless.
					if (ToUpper(fk2.parent_table) == ToUpper(fk.parent_table)) {
						continue;
					}
					const int gpi = Find(fk2.parent_table);
					if (gpi < 0) {
						continue;
					}
					const Frame &gp = tables[(size_t)gpi];
					const int gt = GuessTimeColumn(gp);
					const int ck2 = child.Find(fk2.child_column);
					const int pk2 = gp.Find(fk2.parent_column);
					if (gt < 0 || ck2 < 0 || pk2 < 0) {
						continue;
					}
					const Column &gpk = gp.columns[(size_t)pk2];
					const Column &gtc = gp.columns[(size_t)gt];
					const Column &cc2 = child.columns[(size_t)ck2];
					const bool str_key =
					    gpk.type == ColType::CATEGORY && cc2.type == ColType::CATEGORY;
					std::unordered_map<std::string, double> by_str;
					std::unordered_map<int64_t, double> by_num;
					for (size_t r = 0; r < gp.nrows; r++) {
						if (!gpk.valid[r] || !gtc.valid[r]) {
							continue;
						}
						if (str_key) {
							const uint32_t code = gpk.code[r];
							if (code < gpk.dict.values.size()) {
								by_str.emplace(gpk.dict.values[code], gtc.num[r]);
							}
						} else if ((gpk.type == ColType::CATEGORY) == (cc2.type == ColType::CATEGORY)) {
							by_num.emplace((int64_t)gpk.num[r], gtc.num[r]);
						}
					}
					if (by_str.empty() && by_num.empty()) {
						continue;
					}
					lk.ctime.assign(child.nrows, std::numeric_limits<double>::quiet_NaN());
					size_t hit = 0;
					for (size_t r = 0; r < child.nrows; r++) {
						if (!cc2.valid[r]) {
							continue;
						}
						if (str_key) {
							const uint32_t code = cc2.code[r];
							if (code >= cc2.dict.values.size()) {
								continue;
							}
							auto it = by_str.find(cc2.dict.values[code]);
							if (it != by_str.end()) {
								lk.ctime[r] = it->second;
								hit++;
							}
						} else {
							auto it = by_num.find((int64_t)cc2.num[r]);
							if (it != by_num.end()) {
								lk.ctime[r] = it->second;
								hit++;
							}
						}
					}
					// Only accept an inherited clock that actually dates the table.
					if (hit * 2 >= child.nrows) {
						lk.dated = true;
						lk.time_source = fk2.parent_table + "." + gp.columns[(size_t)gt].name;
						break;
					}
					lk.ctime.clear();
				}
			}

			// Map parent key value -> parent row. Keys are compared on their
			// numeric projection, which is why the binding must hand integer keys
			// through as INT64 rather than as strings: a float/int mismatch here
			// silently resolves nothing.
			// Keys are usually dense integers. When they are, a direct-address table
			// turns one hash lookup per child row into one array index, which is the
			// difference between chasing buckets and streaming.
			const Column &pk = parent.columns[(size_t)pkey];
			int64_t kmin = INT64_MAX, kmax = INT64_MIN;
			for (size_t r = 0; r < parent.nrows; r++) {
				if (!pk.valid[r]) {
					continue;
				}
				const int64_t key =
				    pk.type == ColType::CATEGORY ? (int64_t)pk.code[r] : (int64_t)pk.num[r];
				kmin = std::min(kmin, key);
				kmax = std::max(kmax, key);
			}
			std::vector<uint32_t> direct;
			bool use_direct = false;
			if (parent.nrows > 0 && kmin <= kmax) {
				const uint64_t span = (uint64_t)(kmax - kmin) + 1u;
				if (span <= (uint64_t)parent.nrows * 4u + 64u) {
					direct.assign((size_t)span, UINT32_MAX);
					use_direct = true;
				}
			}
			std::unordered_map<int64_t, uint32_t> key_to_row;
			if (!use_direct) {
				key_to_row.reserve(parent.nrows * 2);
			}
			for (size_t r = 0; r < parent.nrows; r++) {
				if (!pk.valid[r]) {
					continue;
				}
				const int64_t key =
				    pk.type == ColType::CATEGORY ? (int64_t)pk.code[r] : (int64_t)pk.num[r];
				if (use_direct) {
					uint32_t &slot = direct[(size_t)(key - kmin)];
					if (slot != UINT32_MAX) {
						lk.duplicate_parent_keys = true;
					}
					slot = (uint32_t)r;
				} else {
					if (!key_to_row.emplace(key, (uint32_t)r).second) {
						lk.duplicate_parent_keys = true;
					}
				}
			}
			auto lookup = [&](int64_t key, uint32_t &out) {
				if (use_direct) {
					if (key < kmin || key > kmax) {
						return false;
					}
					const uint32_t v = direct[(size_t)(key - kmin)];
					if (v == UINT32_MAX) {
						return false;
					}
					out = v;
					return true;
				}
				auto it = key_to_row.find(key);
				if (it == key_to_row.end()) {
					return false;
				}
				out = it->second;
				return true;
			};
			const Column &ck = child.columns[(size_t)ckey];
			// Text keys must be matched on the string. Every column owns its own
			// dictionary, so the same value has different codes on the two sides and
			// comparing codes silently resolves nothing.
			const bool string_key =
			    pk.type == ColType::CATEGORY && ck.type == ColType::CATEGORY;
			// One side text and the other numeric would compare a dictionary code to
			// a value and invent matches. Refuse the link instead.
			if ((pk.type == ColType::CATEGORY) != (ck.type == ColType::CATEGORY)) {
				continue;
			}
			std::unordered_map<std::string, uint32_t> smap;
			if (string_key) {
				smap.reserve(parent.nrows * 2);
				for (size_t r = 0; r < parent.nrows; r++) {
					if (!pk.valid[r]) {
						continue;
					}
					const uint32_t code = pk.code[r];
					if (code < pk.dict.values.size()) {
						if (!smap.emplace(pk.dict.values[code], (uint32_t)r).second) {
							lk.duplicate_parent_keys = true;
						}
					}
				}
			}
			std::vector<uint32_t> parent_of(child.nrows, UINT32_MAX);
			lk.off.assign(parent.nrows + 1, 0u);
			for (size_t r = 0; r < child.nrows; r++) {
				if (!ck.valid[r]) {
					continue;
				}
				uint32_t pr;
				if (string_key) {
					const uint32_t code = ck.code[r];
					if (code >= ck.dict.values.size()) {
						continue;
					}
					auto it = smap.find(ck.dict.values[code]);
					if (it == smap.end()) {
						continue;
					}
					pr = it->second;
				} else {
					const int64_t key =
					    ck.type == ColType::CATEGORY ? (int64_t)ck.code[r] : (int64_t)ck.num[r];
					if (!lookup(key, pr)) {
						continue;
					}
				}
				parent_of[r] = pr;
				lk.off[pr + 1]++;
			}
			for (size_t p = 0; p < parent.nrows; p++) {
				lk.off[p + 1] += lk.off[p];
			}
			lk.flat.assign(lk.off[parent.nrows], 0u);
			{
				std::vector<uint32_t> cursor(lk.off.begin(), lk.off.end() - 1);
				for (size_t r = 0; r < child.nrows; r++) {
					const uint32_t p = parent_of[r];
					if (p != UINT32_MAX) {
						lk.flat[cursor[p]++] = (uint32_t)r;
					}
				}
			}
			if (lk.dated) {
				// Fact tables are normally appended in time order, and the scatter
				// above preserves row order within a bucket, so the buckets are very
				// often already sorted. An O(n) check skips the O(n log n) work
				// entirely in that case.
				bool sorted = true;
				for (size_t p = 0; p < parent.nrows && sorted; p++) {
					const uint32_t b = lk.off[p], e = lk.off[p + 1];
					for (uint32_t i = b + 1; i < e; i++) {
						if (lk.Time(lk.flat[i]) < lk.Time(lk.flat[i - 1])) {
							sorted = false;
							break;
						}
					}
				}
				if (!sorted) {
					SortBucketsByTime(lk, parent.nrows);
				}
			}
			links.push_back(std::move(lk));
		}
	}

	// How many of a link's children for parent `p` fall at or before `cutoff`.
	// Binary search over the pre-sorted bucket.
	static size_t VisiblePrefix(const Link &lk, const Frame &, uint32_t p, double cutoff) {
		const uint32_t b = lk.Begin(p), e = lk.End(p);
		if (!lk.dated) {
			return e - b;
		}
		size_t lo = 0, hi = e - b;
		while (lo < hi) {
			size_t mid = (lo + hi) / 2;
			if (lk.Time(lk.flat[b + mid]) <= cutoff) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		return lo;
	}
};

// ===========================================================================
// 7. Filter evaluation
// ===========================================================================

// Check every qualifier in a filter tree against the relation it will actually
// be evaluated on. A qualifier naming something else was previously accepted and
// silently applied to whichever frame the walk happened to receive.
inline void ValidateFilterScope(const FilterNode *f, const Frame &frame, const std::string &alias,
                                const char *where) {
	if (!f) {
		return;
	}
	if (f->kind != FilterNode::Kind::CMP) {
		for (const auto &c : f->children) {
			ValidateFilterScope(c.get(), frame, alias, where);
		}
		return;
	}
	if (!f->ref.qualifier.empty()) {
		const std::string q = ToUpper(f->ref.qualifier);
		if (q != ToUpper(frame.name) && (alias.empty() || q != ToUpper(alias))) {
			throw std::runtime_error("pql: the " + std::string(where) + " filter refers to '" +
			                         f->ref.ToString() + "', but it is evaluated on '" + frame.name +
			                         "'" + (alias.empty() ? "" : " (aliased " + alias + ")"));
		}
	}
	if (frame.Find(f->ref.name) < 0) {
		throw std::runtime_error("pql: the " + std::string(where) + " filter refers to column '" +
		                         f->ref.name + "', which does not exist on '" + frame.name + "'");
	}
}

// Filters are evaluated against a single row of one frame. Comparisons against
// a category resolve the literal through that column's dictionary once per
// call site; an unseen literal resolves to code 0 and matches nothing, which is
// the SQL-correct outcome for a value not present in the data.
inline bool EvalFilter(const FilterNode *f, const Frame &frame, size_t row) {
	if (!f) {
		return true;
	}
	switch (f->kind) {
	case FilterNode::Kind::AND:
		for (auto &c : f->children) {
			if (!EvalFilter(c.get(), frame, row)) {
				return false;
			}
		}
		return true;
	case FilterNode::Kind::OR:
		for (auto &c : f->children) {
			if (EvalFilter(c.get(), frame, row)) {
				return true;
			}
		}
		return false;
	case FilterNode::Kind::NOT:
		return f->children.empty() ? true : !EvalFilter(f->children[0].get(), frame, row);
	default:
		break;
	}
	int ci = frame.Find(f->ref.name);
	if (ci < 0) {
		throw std::runtime_error("pql: filter references unknown column '" + f->ref.ToString() + "'");
	}
	const Column &col = frame.columns[(size_t)ci];
	const bool present = col.valid[row] != 0;
	if (f->op == CmpOp::IS_NULL) {
		return !present;
	}
	if (f->op == CmpOp::IS_NOT_NULL) {
		return present;
	}
	if (!present) {
		return false; // NULL compares false, as in SQL three-valued logic collapsed to a filter
	}
	auto matches_literal = [&](const Literal &lit) -> bool {
		if (col.type == ColType::CATEGORY) {
			if (lit.kind != Literal::Kind::STRING) {
				return false;
			}
			return col.code[row] == col.dict.Lookup(lit.text);
		}
		if (lit.kind != Literal::Kind::NUMBER) {
			return false;
		}
		return col.num[row] == lit.number;
	};
	switch (f->op) {
	case CmpOp::IN:
	case CmpOp::NOT_IN: {
		bool any = false;
		for (auto &lit : f->values) {
			if (matches_literal(lit)) {
				any = true;
				break;
			}
		}
		return f->op == CmpOp::IN ? any : !any;
	}
	case CmpOp::EQ:
		return !f->values.empty() && matches_literal(f->values[0]);
	case CmpOp::NE:
		return !f->values.empty() && !matches_literal(f->values[0]);
	default:
		break;
	}
	if (f->values.empty() || f->values[0].kind != Literal::Kind::NUMBER) {
		throw std::runtime_error("pql: ordered comparison needs a numeric literal");
	}
	const double a = col.num[row], b = f->values[0].number;
	switch (f->op) {
	case CmpOp::LT:
		return a < b;
	case CmpOp::LE:
		return a <= b;
	case CmpOp::GT:
		return a > b;
	default:
		return a >= b;
	}
}

// ===========================================================================
// 8. Label construction
// ===========================================================================
//
// For a forecast target the label is computed, never read. Given an entity row
// and its anchor t, the engine aggregates the target table's rows that (a)
// point at this entity, (b) satisfy the inner filter, and (c) carry a time
// strictly inside (t, t + horizon]. Features later use only (-inf, t]. The two
// intervals are disjoint by construction, so a leak would require a bug in one
// of these two comparisons rather than a modelling oversight.

struct Example {
	uint32_t entity_row = 0;
	double anchor = 0;     // micros
	double label = 0;
	// The same aggregate over the horizon immediately BEFORE the anchor. This is
	// the persistence prediction, and the model learns the residual over it: a
	// net cannot cheaply rediscover the identity function from log1p-compressed
	// counts, so handing it the level as an offset is worth more than any feature.
	double base = 0;
	bool has_label = false;
};

inline double AggregateWindow(const Database &, const Link &lk, const Frame &child,
                              uint32_t entity_row, double lo_exclusive, double hi_inclusive,
                              TargetKind kind, int value_col, const FilterNode *filter,
                              bool &any) {
	const uint32_t bslice = lk.Begin(entity_row), eslice = lk.End(entity_row);
	const bool timed = lk.dated;
	double acc = 0;
	double best = 0;
	size_t n = 0;
	size_t n_values = 0;
	bool have_best = false;
	any = false;
	// Bound the walk to the horizon window. Scanning the whole bucket per entity
	// made label construction O(all children) when only a handful can qualify.
	uint32_t wbeg = bslice, wend = eslice;
	if (timed) {
		auto lower = [&](double bound) {
			uint32_t lo = bslice, hi = eslice;
			while (lo < hi) {
				const uint32_t mid = lo + (hi - lo) / 2;
				if (lk.Time(lk.flat[mid]) <= bound) {
					lo = mid + 1;
				} else {
					hi = mid;
				}
			}
			return lo;
		};
		wbeg = lower(lo_exclusive);
		wend = lower(hi_inclusive);
	}
	for (uint32_t idx = wbeg; idx < wend; idx++) {
		const uint32_t r = lk.flat[idx];
		if (filter && !EvalFilter(filter, child, r)) {
			continue;
		}
		any = true;
		n++;
		if (value_col >= 0) {
			const Column &vc = child.columns[(size_t)value_col];
			if (!vc.valid[r]) {
				continue;
			}
			const double v = vc.num[r];
			acc += v;
			n_values++;
			if (!have_best || (kind == TargetKind::MIN && v < best) ||
			    (kind == TargetKind::MAX && v > best)) {
				best = v;
				have_best = true;
			}
		}
	}
	switch (kind) {
	case TargetKind::COUNT:
		return double(n);
	case TargetKind::EXISTS:
		return any ? 1.0 : 0.0;
	case TargetKind::SUM:
		return acc;
	case TargetKind::AVG:
		return n_values ? acc / double(n_values) : 0.0;
	case TargetKind::MIN:
	case TargetKind::MAX:
		return have_best ? best : 0.0;
	default:
		return 0.0;
	}
}

// ===========================================================================
// 9. Feature specification
// ===========================================================================

// Windows used for compiled temporal aggregates, in days. Fixed rather than
// learned: each is one binary search, and a wide set costs almost nothing.
static const double kWindowDays[] = {7.0, 30.0, 90.0, 365.0, 1.0e18};
static const int kNumWindows = 5;
static const double kMicrosPerDay = 86400.0 * 1000000.0;

// Which raw columns feed the model, fixed at training time and replayed at
// prediction time so the two vectors always line up.
struct FeatureSpec {
	struct SelfCol {
		int index = -1;
		std::string name;
		double mean = 0, sd = 1;
	};
	struct LinkAgg {
		int link = -1;
		std::string child_table, child_key;
		std::vector<int> mean_cols; // child numeric columns averaged in-window
		std::vector<std::string> mean_names;
		std::vector<double> mean_mu, mean_sd;
	};
	std::vector<SelfCol> self_cols;
	std::vector<LinkAgg> link_aggs;
	int width = 0;

	// count + recency per window, plus one mean per selected child column per window
	// Column positions are not stable across statements: the loader prunes
	// columns per query, so a filter on a text column changes the frame layout.
	// Re-resolve by name before any use, and fail loudly if something is gone.
	void Rebind(const Database &db, const Frame &entity) {
		for (auto &sc : self_cols) {
			const int i = entity.Find(sc.name);
			if (i < 0) {
				throw std::runtime_error("pql: column '" + sc.name + "' used by this model is not "
				                         "present on '" + entity.name + "' any more");
			}
			sc.index = i;
		}
		for (auto &la : link_aggs) {
			int found = -1;
			for (size_t l = 0; l < db.links.size(); l++) {
				if (ToUpper(db.links[l].child_table) == ToUpper(la.child_table) &&
				    ToUpper(db.links[l].parent_table) == ToUpper(entity.name)) {
					found = (int)l;
					break;
				}
			}
			if (found < 0) {
				throw std::runtime_error("pql: the link from '" + la.child_table + "' to '" +
				                         entity.name + "' used by this model is not available");
			}
			la.link = found;
			const Frame &child = db.At(la.child_table);
			for (size_t m = 0; m < la.mean_names.size(); m++) {
				const int c = child.Find(la.mean_names[m]);
				if (c < 0) {
					throw std::runtime_error("pql: column '" + la.child_table + "." +
					                         la.mean_names[m] + "' used by this model is missing");
				}
				la.mean_cols[m] = c;
			}
		}
	}

	// Per window: a count, a within-window recency, and one mean per selected
	// child column. Plus one never-happened flag for the link as a whole.
	int PerLinkWidth(const LinkAgg &la) const {
		return kNumWindows * (2 + (int)la.mean_cols.size()) + 1;
	}
	void ComputeWidth() {
		width = (int)self_cols.size();
		for (auto &la : link_aggs) {
			width += PerLinkWidth(la);
		}
	}
};

// Prefix-sum arena over each link's time-sorted children.
//
// The scan this replaces read vc.num[rows[i]] -- an indirect gather through a
// row-index array, which cannot vectorize and which we had to cap at 512 rows
// to bound its cost. Laying the values out once in link order, with a running
// sum per parent, turns every windowed mean into two lookups. It is exact
// (the cap is gone) and it is O(1) per window instead of O(window).
struct AggCache {
	struct LinkCache {
		std::vector<uint32_t> off;               // per parent: base into the prefix arrays
		std::vector<std::vector<float>> psum;    // [mean col][off + i] running value sum
		std::vector<std::vector<uint32_t>> pcnt; // [mean col][off + i] running valid count
	};
	std::vector<LinkCache> links; // parallel to FeatureSpec::link_aggs
};

inline AggCache BuildAggCache(const Database &db, const FeatureSpec &spec) {
	AggCache cache;
	cache.links.resize(spec.link_aggs.size());
	for (size_t li = 0; li < spec.link_aggs.size(); li++) {
		const FeatureSpec::LinkAgg &la = spec.link_aggs[li];
		const Link &lk = db.links[(size_t)la.link];
		const Frame &child = db.At(lk.child_table);
		AggCache::LinkCache &lc = cache.links[li];
		const size_t nparents = lk.off.empty() ? 0 : lk.off.size() - 1;
		lc.off.assign(nparents + 1, 0);
		uint32_t total = 0;
		for (size_t p = 0; p < nparents; p++) {
			lc.off[p] = total;
			total += lk.Count((uint32_t)p) + 1u; // +1 for the leading zero
		}
		lc.off[nparents] = total;
		lc.psum.assign(la.mean_cols.size(), {});
		lc.pcnt.assign(la.mean_cols.size(), {});
		for (size_t c = 0; c < la.mean_cols.size(); c++) {
			const Column &vc = child.columns[(size_t)la.mean_cols[c]];
			std::vector<float> &ps = lc.psum[c];
			std::vector<uint32_t> &pc = lc.pcnt[c];
			ps.assign(total, 0.0f);
			pc.assign(total, 0u);
			for (size_t p = 0; p < nparents; p++) {
				const uint32_t b = lk.Begin((uint32_t)p), e = lk.End((uint32_t)p);
				const uint32_t base = lc.off[p];
				float run = 0.0f;
				uint32_t cnt = 0;
				for (uint32_t i = b; i < e; i++) {
					const uint32_t r = lk.flat[i];
					// Branchless: a null contributes zero and does not advance the
					// count, so the loop carries no data-dependent branch.
					const float v = vc.valid[r] ? (float)vc.num[r] : 0.0f;
					const uint32_t ok = vc.valid[r] ? 1u : 0u;
					run += v;
					cnt += ok;
					ps[base + (i - b) + 1] = run;
					pc[base + (i - b) + 1] = cnt;
				}
			}
		}
	}
	return cache;
}

// Build the feature vector for one entity row at a given anchor time. Every
// value is drawn from (-inf, anchor], which is what keeps forecasts honest.
inline void BuildFeatures(const Database &db, const Frame &entity, const FeatureSpec &spec,
                          const AggCache &cache, uint32_t row, double anchor,
                          std::vector<float> &out) {
	out.assign((size_t)spec.width, 0.0f);
	size_t k = 0;
	for (const auto &sc : spec.self_cols) {
		const Column &c = entity.columns[(size_t)sc.index];
		out[k++] = c.valid[row] ? float((c.num[row] - sc.mean) / sc.sd) : 0.0f;
	}
	for (size_t li = 0; li < spec.link_aggs.size(); li++) {
		const FeatureSpec::LinkAgg &la = spec.link_aggs[li];
		const AggCache::LinkCache &lc = cache.links[li];
		const Link &lk = db.links[(size_t)la.link];
		const Frame &child = db.At(lk.child_table);
		const uint32_t bslice = lk.Begin(row);
		const size_t visible = Database::VisiblePrefix(lk, child, row, anchor);
		const bool timed = lk.dated;

		// Recency is a min over child timestamps, so it cannot be expressed as a
		// count; it earns its own slot because it dominated every count feature
		// we measured on engagement-style tasks.
		double since = -1.0;
		if (timed && visible > 0) {
			since = (anchor - lk.Time(lk.flat[bslice + visible - 1])) / kMicrosPerDay;
			if (since < 0) {
				since = 0;
			}
		}
		out[k++] = since < 0 ? 1.0f : 0.0f; // never happened at all
		for (int w = 0; w < kNumWindows; w++) {
			const double lo = anchor - kWindowDays[w] * kMicrosPerDay;
			size_t start = 0;
			if (timed && w < kNumWindows - 1) {
				size_t a = 0, b = visible;
				while (a < b) {
					size_t mid = (a + b) / 2;
					if (lk.Time(lk.flat[bslice + mid]) <= lo) {
						a = mid + 1;
					} else {
						b = mid;
					}
				}
				start = a;
			}
			const size_t n = visible > start ? visible - start : 0;
			out[k++] = float(std::log1p(double(n)));
			// Recency *within this window*: the same instant when the window holds
			// anything, and absent when it does not. Emitting one global value made
			// all five slots identical; this makes "nothing in 7 days, but something
			// in 90" expressible.
			out[k++] = (n > 0 && since >= 0) ? float(std::log1p(since) * 0.25) : 0.0f;
			const uint32_t base = lc.off[row];
			for (size_t mi = 0; mi < la.mean_cols.size(); mi++) {
				const float sum = lc.psum[mi][base + visible] - lc.psum[mi][base + start];
				const uint32_t m = lc.pcnt[mi][base + visible] - lc.pcnt[mi][base + start];
				const double mu = mi < la.mean_mu.size() ? la.mean_mu[mi] : 0.0;
				const double sd = mi < la.mean_sd.size() ? la.mean_sd[mi] : 1.0;
				out[k++] = m ? float(((double(sum) / double(m)) - mu) / sd) : 0.0f;
			}
		}
	}
}

inline void MatMulNT(const float *A, const float *W, const float *bias, float *C, int B, int K,
                     int N, bool relu);

// ===========================================================================
// 9b. Heterogeneous GraphSAGE
// ===========================================================================
//
// Transposed from RelML's HeteroGraphSAGE: a W_self per node type, a W_neigh
// per edge type, mean aggregation over neighbours, ReLU between layers.
//
//     h_v = ReLU( W_self . x_v  +  SUM_r W_r . mean_{u in N_r(v)} x_u )
//
// Two things are done differently here, both because the CSR arena is sorted by
// time.
//
// 1. Aggregation is EXACT AND TEMPORAL. RelML has no temporal neighbour
//    sampling, which is why its own benchmark has to keep fact tables out of the
//    graph or a 2010 row message-passes into a 2015 prediction. Here the
//    neighbours visible at an anchor are a prefix of a time-sorted slice, so a
//    running sum over that slice gives the mean of any prefix by subtracting two
//    points. Nothing after the anchor can enter, by construction.
//
// 2. That also makes it O(C) per node instead of O(degree * C): the running sums
//    are built once per epoch and every entity reads two rows out of them.
//
// Layout is chosen for the cache: node features and running sums are row-major
// [N x C] in one arena block, so an aggregation is a contiguous subtract and a
// batch of them is a GEMM through the same kernels the dense layers use.

struct SageSpec {
	struct NodeType {
		int table = -1;
		std::vector<int> cols;      // numeric columns encoding this node type
		std::vector<double> mu, sd; // standardisation, fitted on the train fold
		int dim = 0;
	};
	struct EdgeType {
		int link = -1;   // index into Database::links
		int src_node = -1; // index into nodes: the child
		int dst_node = -1; // index into nodes: the parent (the entity)
	};
	std::vector<NodeType> nodes;
	std::vector<EdgeType> edges;
	int entity_node = -1;

	int NodeDim(int n) const {
		return nodes[(size_t)n].dim;
	}
};

// Raw node features, [N x dim] row-major, one block per node type.
struct SageFeatures {
	std::vector<std::vector<float>> x; // per node type
};

// Running sums of a source node type's features along one link's CSR order.
// psum[e] holds the sum of the first (e - base) children of that parent, so the
// mean over [lo, hi) is (psum[base+hi] - psum[base+lo]) / (hi - lo).
struct SagePrefix {
	std::vector<std::vector<float>> psum; // per edge type, [(E + P) x src_dim]
	std::vector<std::vector<uint32_t>> off;
};

inline SageFeatures BuildSageFeatures(const Database &db, const SageSpec &spec) {
	SageFeatures f;
	f.x.resize(spec.nodes.size());
	for (size_t n = 0; n < spec.nodes.size(); n++) {
		const SageSpec::NodeType &nt = spec.nodes[n];
		const Frame &fr = db.tables[(size_t)nt.table];
		f.x[n].assign(fr.nrows * (size_t)nt.dim, 0.0f);
		for (size_t c = 0; c < nt.cols.size(); c++) {
			const Column &col = fr.columns[(size_t)nt.cols[c]];
			const double mu = nt.mu[c], sd = nt.sd[c];
			float *dst = f.x[n].data() + c;
			const int stride = nt.dim;
			for (size_t r = 0; r < fr.nrows; r++) {
				dst[r * (size_t)stride] = col.valid[r] ? float((col.num[r] - mu) / sd) : 0.0f;
			}
		}
	}
	return f;
}

inline SagePrefix BuildSagePrefix(const Database &db, const SageSpec &spec,
                                  const SageFeatures &feat) {
	SagePrefix pre;
	pre.psum.resize(spec.edges.size());
	pre.off.resize(spec.edges.size());
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const SageSpec::EdgeType &et = spec.edges[e];
		const Link &lk = db.links[(size_t)et.link];
		const int d = spec.NodeDim(et.src_node);
		const size_t nparents = lk.off.empty() ? 0 : lk.off.size() - 1;
		pre.off[e].assign(nparents + 1, 0u);
		uint32_t total = 0;
		for (size_t p = 0; p < nparents; p++) {
			pre.off[e][p] = total;
			total += lk.Count((uint32_t)p) + 1u;
		}
		pre.off[e][nparents] = total;
		pre.psum[e].assign((size_t)total * (size_t)d, 0.0f);
		const float *src = feat.x[(size_t)et.src_node].data();
		for (size_t p = 0; p < nparents; p++) {
			const uint32_t b = lk.Begin((uint32_t)p), en = lk.End((uint32_t)p);
			float *run = pre.psum[e].data() + (size_t)pre.off[e][p] * (size_t)d;
			// run[0..d) is the zero row; each step adds one child, contiguous in d.
			for (uint32_t i = b; i < en; i++) {
				const float *xu = src + (size_t)lk.flat[i] * (size_t)d;
				float *cur = run + (size_t)(i - b + 1) * (size_t)d;
				const float *prev = run + (size_t)(i - b) * (size_t)d;
				for (int k = 0; k < d; k++) {
					cur[k] = prev[k] + xu[k];
				}
			}
		}
	}
	return pre;
}

// ---------------------------------------------------------------------------
// Second hop.
//
// A child's embedding is computed at the child's OWN timestamp, aggregating its
// own children up to that instant. Since every visible child satisfies
// t_child <= anchor, this cannot see past the anchor, and because it does not
// depend on the anchor it is computed once per epoch rather than per example.
// That is what makes an exact two-hop affordable; RelML sidesteps the question
// by not sampling temporally at all.
// ---------------------------------------------------------------------------

struct SageHop2 {
	// Per entity-edge: the transformed embedding of each child row, [N_child x C].
	std::vector<std::vector<float>> h_child;
	// Prefix sums of those embeddings along the entity link's CSR order.
	std::vector<std::vector<float>> psum;
	std::vector<std::vector<uint32_t>> off;
	bool active = false;
};

// Parameters of the child layer: one W_self per child node type, one W_neigh per
// grandchild edge. Mirrors SageParams but is indexed per entity-edge.
struct SageChildParams {
	int channels = 0;
	std::vector<int> self_dim;                            // per entity-edge
	std::vector<std::vector<float>> w_self;               // [C x self_dim]
	std::vector<std::vector<float>> bias;                 // [C]
	std::vector<std::vector<int>> gc_link;                // grandchild link ids
	std::vector<std::vector<int>> gc_dim;                 // + degree slot
	std::vector<std::vector<std::vector<float>>> w_gc;    // [C x gc_dim]

	size_t ParamCount() const {
		size_t n = 0;
		for (const auto &w : w_self) {
			n += w.size();
		}
		for (const auto &b : bias) {
			n += b.size();
		}
		for (const auto &e : w_gc) {
			for (const auto &w : e) {
				n += w.size();
			}
		}
		return n;
	}
};

// Mean of a neighbourhood visible at `anchor`, written into `out` (length d).
inline void SageMeanAt(const Database &db, const SageSpec &spec, const SagePrefix &pre, size_t e,
                       uint32_t parent_row, double anchor, float *out, int d) {
	const SageSpec::EdgeType &et = spec.edges[e];
	const Link &lk = db.links[(size_t)et.link];
	const Frame &child = db.At(lk.child_table);
	const size_t visible = Database::VisiblePrefix(lk, child, parent_row, anchor);
	const uint32_t base = pre.off[e][parent_row];
	const float *hi = pre.psum[e].data() + (size_t)(base + visible) * (size_t)d;
	const float *lo = pre.psum[e].data() + (size_t)base * (size_t)d;
	if (visible == 0) {
		for (int k = 0; k <= d; k++) {
			out[k] = 0.0f;
		}
		return;
	}
	const float inv = 1.0f / float(visible);
	for (int k = 0; k < d; k++) {
		out[k] = (hi[k] - lo[k]) * inv;
	}
	// Degree rides along with the mean. A mean is scale-stable but says nothing
	// about how many neighbours produced it, and on relational data the count is
	// usually the signal. RelML aggregates by mean alone; RelBench's RDL sums,
	// which keeps degree implicitly. Carrying both keeps mean's conditioning and
	// makes the count learnable.
	out[d] = float(std::log1p(double(visible)));
}

// ---------------------------------------------------------------------------
// Step 1: build the node/edge schema the layers will run over.
// ---------------------------------------------------------------------------

// Numeric columns of a table that can encode a node, excluding keys and clocks
// (an id is identity, not signal; a timestamp is the axis, not a feature).
inline void SageNodeColumns(const Database &db, int table, const Link *incoming,
                            std::vector<int> &out) {
	const Frame &f = db.tables[(size_t)table];
	for (size_t c = 0; c < f.columns.size(); c++) {
		if (!f.columns[c].IsNumeric()) {
			continue;
		}
		if (incoming && ((int)c == incoming->child_key_col || (int)c == incoming->child_time_col)) {
			continue;
		}
		bool is_key = false;
		for (const auto &fk : db.fks) {
			const bool child_side = ToUpper(fk.child_table) == ToUpper(f.name) &&
			                        ToUpper(fk.child_column) == ToUpper(f.columns[c].name);
			const bool parent_side = ToUpper(fk.parent_table) == ToUpper(f.name) &&
			                         ToUpper(fk.parent_column) == ToUpper(f.columns[c].name);
			if (child_side || parent_side) {
				is_key = true;
				break;
			}
		}
		if (!is_key) {
			out.push_back((int)c);
		}
	}
}

inline SageSpec BuildSageSpec(const Database &db, const Frame &entity, int anchor_col,
                              int target_col, double train_cutoff) {
	SageSpec spec;
	auto add_node = [&](int table, const Link *incoming) {
		for (size_t n = 0; n < spec.nodes.size(); n++) {
			if (spec.nodes[n].table == table) {
				return (int)n;
			}
		}
		SageSpec::NodeType nt;
		nt.table = table;
		SageNodeColumns(db, table, incoming, nt.cols);
		if (table == db.Find(entity.name)) {
			// the anchor and the target are not inputs
			std::vector<int> keep;
			for (int c : nt.cols) {
				if (c != anchor_col && c != target_col) {
					keep.push_back(c);
				}
			}
			nt.cols.swap(keep);
		}
		const Frame &f = db.tables[(size_t)table];
		const Column *anch = (table == db.Find(entity.name) && anchor_col >= 0)
		                         ? &f.columns[(size_t)anchor_col]
		                         : nullptr;
		for (int c : nt.cols) {
			const Column &col = f.columns[(size_t)c];
			double sum = 0, sq = 0;
			uint64_t n = 0;
			for (size_t r = 0; r < f.nrows; r++) {
				if (!col.valid[r]) {
					continue;
				}
				// Statistics come from the training era only.
				if (anch && std::isfinite(train_cutoff) &&
				    (!anch->valid[r] || anch->num[r] > train_cutoff)) {
					continue;
				}
				sum += col.num[r];
				sq += col.num[r] * col.num[r];
				n++;
			}
			const double mu = n ? sum / double(n) : 0.0;
			const double var = n > 1 ? (sq - double(n) * mu * mu) / double(n - 1) : 0.0;
			nt.mu.push_back(mu);
			nt.sd.push_back(var > 1e-18 ? std::sqrt(var) : 1.0);
		}
		nt.dim = (int)nt.cols.size();
		spec.nodes.push_back(nt);
		return (int)spec.nodes.size() - 1;
	};

	const int entity_table = db.Find(entity.name);
	spec.entity_node = add_node(entity_table, nullptr);
	for (size_t l = 0; l < db.links.size(); l++) {
		const Link &lk = db.links[l];
		if (ToUpper(lk.parent_table) != ToUpper(entity.name)) {
			continue;
		}
		const int ct = db.Find(lk.child_table);
		if (ct < 0) {
			continue;
		}
		SageSpec::EdgeType et;
		et.link = (int)l;
		et.src_node = add_node(ct, &lk);
		et.dst_node = spec.entity_node;
		if (spec.NodeDim(et.src_node) > 0) {
			spec.edges.push_back(et);
		}
	}
	return spec;
}

// ---------------------------------------------------------------------------
// Step 2: parameters, carved from one arena block.
// ---------------------------------------------------------------------------

struct SageParams {
	int channels = 0;
	int self_dim = 0;
	std::vector<int> src_dim;                // per edge type
	std::vector<float> w_self;               // [C x self_dim]
	std::vector<std::vector<float>> w_neigh; // per edge: [C x src_dim]
	std::vector<float> bias;                 // [C]
	std::vector<float> w_out;                // [C]
	float b_out = 0.0f;

	size_t ParamCount() const {
		size_t n = w_self.size() + bias.size() + w_out.size() + 1;
		for (const auto &w : w_neigh) {
			n += w.size();
		}
		return n;
	}

	void Init(const SageSpec &spec, int C, uint64_t seed) {
		channels = C;
		self_dim = spec.NodeDim(spec.entity_node);
		src_dim.clear();
		w_neigh.assign(spec.edges.size(), {});
		for (size_t e = 0; e < spec.edges.size(); e++) {
			src_dim.push_back(spec.NodeDim(spec.edges[e].src_node) + 1); // + degree
		}
		uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
		auto rnd = [&]() {
			st ^= st << 13;
			st ^= st >> 7;
			st ^= st << 17;
			return double(int64_t(st >> 11)) / double(1ULL << 53);
		};
		auto fill = [&](std::vector<float> &w, size_t n, int fan_in) {
			w.assign(n, 0.0f);
			const double s = std::sqrt(2.0 / double(std::max(1, fan_in)));
			for (auto &v : w) {
				v = float((rnd() * 2 - 1) * s);
			}
		};
		fill(w_self, (size_t)C * (size_t)std::max(1, self_dim), std::max(1, self_dim));
		for (size_t e = 0; e < w_neigh.size(); e++) {
			fill(w_neigh[e], (size_t)C * (size_t)std::max(1, src_dim[e]), std::max(1, src_dim[e]));
		}
		bias.assign((size_t)C, 0.0f);
		fill(w_out, (size_t)C, C);
		b_out = 0.0f;
	}
};

// ===========================================================================
// 10. Model
// ===========================================================================

// A two-layer network over [self features | per-link neighbour means], which is
// one round of GraphSAGE-style aggregation with the neighbourhood restricted to
// rows visible at the anchor. Kept dense and small: the compiled aggregates
// carry the history, so depth buys much less here than width of the feature
// vocabulary does.
struct Model {
	std::string name;
	Statement spec_stmt;      // the training statement, replayed at predict time
	FeatureSpec features;
	int hidden = 64;
	bool classification = true;
	bool sage = false;        // hetero GraphSAGE instead of the dense head
	int sage_layers = 1;
	SageSpec sage_spec;
	SageParams sage_params;
	SageChildParams sage_child;
	bool residual = false;    // output is base + f(x)
	bool nonnegative = false; // counts and sums of non-negative columns
	double label_mean = 0, label_sd = 1; // regression targets are standardized

	std::vector<float> w1, b1; // width x hidden
	std::vector<float> w2, b2; // hidden x hidden
	std::vector<float> w3, b3; // hidden x 1

	void Init(int width, int hid, uint64_t seed) {
		hidden = hid;
		w1.assign((size_t)width * hid, 0.0f);
		b1.assign((size_t)hid, 0.0f);
		w2.assign((size_t)hid * hid, 0.0f);
		b2.assign((size_t)hid, 0.0f);
		w3.assign((size_t)hid, 0.0f);
		b3.assign(1, 0.0f);
		uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
		auto rnd = [&]() {
			st ^= st << 13;
			st ^= st >> 7;
			st ^= st << 17;
			return double(int64_t(st >> 11)) / double(1ULL << 53);
		};
		const double s1 = std::sqrt(2.0 / std::max(1, width));
		for (auto &v : w1) {
			v = float((rnd() * 2 - 1) * s1);
		}
		const double s2 = std::sqrt(2.0 / std::max(1, hid));
		for (auto &v : w2) {
			v = float((rnd() * 2 - 1) * s2);
		}
		for (auto &v : w3) {
			v = float((rnd() * 2 - 1) * s2);
		}
	}

	double Forward(const std::vector<float> &x, std::vector<float> &h1v,
	               std::vector<float> &h2v) const {
		const int width = (int)(w1.size() / (size_t)hidden);
		h1v.assign((size_t)hidden, 0.0f);
		for (int j = 0; j < hidden; j++) {
			double acc = b1[(size_t)j];
			const float *col = &w1[(size_t)j * (size_t)width];
			for (int i = 0; i < width; i++) {
				acc += double(col[i]) * double(x[(size_t)i]);
			}
			h1v[(size_t)j] = float(acc > 0 ? acc : 0);
		}
		h2v.assign((size_t)hidden, 0.0f);
		for (int j = 0; j < hidden; j++) {
			double acc = b2[(size_t)j];
			const float *col = &w2[(size_t)j * (size_t)hidden];
			for (int i = 0; i < hidden; i++) {
				acc += double(col[i]) * double(h1v[(size_t)i]);
			}
			h2v[(size_t)j] = float(acc > 0 ? acc : 0);
		}
		double o = b3[0];
		for (int i = 0; i < hidden; i++) {
			o += double(w3[(size_t)i]) * double(h2v[(size_t)i]);
		}
		return o;
	}

	// Single-row scoring for the prediction path. Scratch is a member so a call
	// allocates nothing; batch work should use ScoreRows instead.
	double Predict(const float *x, int width) const {
		scratch_h1.resize((size_t)hidden);
		scratch_h2.resize((size_t)hidden);
		MatMulNT(x, w1.data(), b1.data(), scratch_h1.data(), 1, width, hidden, true);
		MatMulNT(scratch_h1.data(), w2.data(), b2.data(), scratch_h2.data(), 1, hidden, hidden, true);
		double o = b3[0];
		for (int j = 0; j < hidden; j++) {
			o += double(w3[(size_t)j]) * double(scratch_h2[(size_t)j]);
		}
		if (!classification) {
			return o * label_sd + label_mean;
		}
		return 1.0 / (1.0 + std::exp(-o));
	}

	mutable std::vector<float> scratch_h1, scratch_h2;
};

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------
//
// Every buffer a training run touches is carved from one block sized up front.
// The working set is a known function of (rows, width, hidden), so there is no
// reason to return to the allocator once the shape is known; the hot loops then
// run against memory that is contiguous and already resident.
class Arena {
public:
	void Reserve(size_t bytes) {
		if (buf_.size() < bytes) {
			buf_.resize(bytes);
		}
		used_ = 0;
	}
	template <typename T>
	T *Alloc(size_t n) {
		const size_t align = alignof(T) > 16 ? alignof(T) : 16;
		used_ = (used_ + align - 1) & ~(align - 1);
		const size_t bytes = n * sizeof(T);
		if (used_ + bytes > buf_.size()) {
			// Growing would reallocate and dangle every pointer already returned.
			throw std::runtime_error("pql: arena exhausted; Reserve() was undersized");
		}
		T *p = reinterpret_cast<T *>(buf_.data() + used_);
		used_ += bytes;
		return p;
	}
	template <typename T>
	T *Zeroed(size_t n) {
		T *p = Alloc<T>(n);
		std::memset(p, 0, n * sizeof(T));
		return p;
	}
	void Reset() {
		used_ = 0;
	}
	size_t Used() const {
		return used_;
	}
	size_t Capacity() const {
		return buf_.size();
	}

private:
	std::vector<uint8_t> buf_;
	size_t used_ = 0;
};

// ---------------------------------------------------------------------------
// Batched kernels. Each takes contiguous float buffers with the reduction on
// the innermost, unit-stride axis of both operands, which is the shape clang
// auto-vectorizes into FMA lanes. Working a minibatch at a time turns B
// matrix-vector products into one matrix-matrix product and is worth several
// times the scalar form.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shape-specialised kernels
// ---------------------------------------------------------------------------
//
// A network is a program, and its shape (channel count, per-relation widths) is
// fixed the moment the model is built. QFP makes a program a type so the whole
// dataflow specialises at compile time; the same trick applies here without a
// JIT: instantiate the hot kernels for the channel counts people actually use,
// and dispatch once. With N a compile-time constant the trip count disappears,
// the loop unrolls fully, and the accumulators stay in registers.
//
// Anything else falls back to the runtime-width kernel below, so no shape is
// unsupported, only unspecialised.

template <int N>
inline void MatMulNT_N(const float *A, const float *W, const float *bias, float *C_out, int B,
                       int K, bool relu) {
	for (int b = 0; b < B; b++) {
		const float *arow = A + (size_t)b * (size_t)K;
		float *crow = C_out + (size_t)b * (size_t)N;
		for (int j = 0; j < N; j++) {
			const float *wrow = W + (size_t)j * (size_t)K;
			float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
			int k = 0;
			for (; k + 3 < K; k += 4) {
				a0 += arow[k] * wrow[k];
				a1 += arow[k + 1] * wrow[k + 1];
				a2 += arow[k + 2] * wrow[k + 2];
				a3 += arow[k + 3] * wrow[k + 3];
			}
			float acc = (a0 + a1) + (a2 + a3);
			for (; k < K; k++) {
				acc += arow[k] * wrow[k];
			}
			if (bias) {
				acc += bias[j];
			}
			crow[j] = relu ? (acc > 0.0f ? acc : 0.0f) : acc;
		}
	}
}

// C[B x N] = A[B x K] * W[N x K]^T + bias, optionally ReLU'd.
inline void MatMulNT(const float *A, const float *W, const float *bias, float *C, int B, int K,
                     int N, bool relu) {
	for (int b = 0; b < B; b++) {
		const float *arow = A + (size_t)b * (size_t)K;
		float *crow = C + (size_t)b * (size_t)N;
		for (int j = 0; j < N; j++) {
			const float *wrow = W + (size_t)j * (size_t)K;
			// Four independent partial sums. Float addition is not associative, so
			// a single accumulator forces clang to keep the reduction serial;
			// splitting it here is what lets the loop issue FMA lanes, and it does
			// not require -ffast-math.
			float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
			int k = 0;
			for (; k + 3 < K; k += 4) {
				a0 += arow[k] * wrow[k];
				a1 += arow[k + 1] * wrow[k + 1];
				a2 += arow[k + 2] * wrow[k + 2];
				a3 += arow[k + 3] * wrow[k + 3];
			}
			float acc = (a0 + a1) + (a2 + a3);
			for (; k < K; k++) {
				acc += arow[k] * wrow[k];
			}
			if (bias) {
				acc += bias[j];
			}
			crow[j] = relu ? (acc > 0 ? acc : 0.0f) : acc;
		}
	}
}

// G[N x K] += D[B x N]^T * A[B x K]
// Both dimensions compile-time. K is the relation width, typically 3 to 6, so a
// runtime trip count there is nearly all overhead: fully unrolled it becomes a
// handful of FMAs with the accumulator in a register.
template <int N, int K>
inline void MatMulNT_NK(const float *A, const float *W, const float *bias, float *C_out, int B,
                        bool relu) {
	// Only the K loop is fully unrolled. Unrolling the N loop as well blows up the
	// body, and when B is large (the child layer runs over every child row) that
	// costs more in instruction cache than it saves in loop overhead.
	for (int b = 0; b < B; b++) {
		const float *arow = A + (size_t)b * (size_t)K;
		float *crow = C_out + (size_t)b * (size_t)N;
		for (int j = 0; j < N; j++) {
			const float *wrow = W + (size_t)j * (size_t)K;
			float acc = bias ? bias[j] : 0.0f;
#pragma clang loop unroll(full)
			for (int k = 0; k < K; k++) {
				acc += arow[k] * wrow[k];
			}
			crow[j] = relu ? (acc > 0.0f ? acc : 0.0f) : acc;
		}
	}
}

// One dispatch point: the specialised kernel when the width is a known one,
// the general kernel otherwise.
inline void MatMulNT(const float *A, const float *W, const float *bias, float *C_out, int B, int K,
                     int N, bool relu);

inline void MatMulDispatch(const float *A, const float *W, const float *bias, float *C_out, int B,
                           int K, int N, bool relu) {
	// Both dimensions known: the fully unrolled form.
#define PQL_NK(NN, KK)                                                                             \
	if (N == (NN) && K == (KK)) {                                                                  \
		MatMulNT_NK<NN, KK>(A, W, bias, C_out, B, relu);                                           \
		return;                                                                                    \
	}
	PQL_NK(32, 3) PQL_NK(32, 4) PQL_NK(32, 5) PQL_NK(32, 6)
	PQL_NK(48, 3) PQL_NK(48, 4) PQL_NK(48, 5) PQL_NK(48, 6)
	PQL_NK(64, 3) PQL_NK(64, 4) PQL_NK(64, 5) PQL_NK(64, 6)
	PQL_NK(128, 3) PQL_NK(128, 4) PQL_NK(128, 5) PQL_NK(128, 6)
#undef PQL_NK
	switch (N) {
	case 16:
		MatMulNT_N<16>(A, W, bias, C_out, B, K, relu);
		return;
	case 32:
		MatMulNT_N<32>(A, W, bias, C_out, B, K, relu);
		return;
	case 48:
		MatMulNT_N<48>(A, W, bias, C_out, B, K, relu);
		return;
	case 64:
		MatMulNT_N<64>(A, W, bias, C_out, B, K, relu);
		return;
	case 128:
		MatMulNT_N<128>(A, W, bias, C_out, B, K, relu);
		return;
	default:
		MatMulNT(A, W, bias, C_out, B, K, N, relu);
		return;
	}
}

inline void AccumOuter(const float *D, const float *A, float *G, int B, int K, int N) {
	for (int b = 0; b < B; b++) {
		const float *drow = D + (size_t)b * (size_t)N;
		const float *arow = A + (size_t)b * (size_t)K;
		for (int j = 0; j < N; j++) {
			// Keep the zero-test: ReLU makes roughly half these gradients exactly
			// zero, so this skips a whole K-length inner loop rather than a single
			// operation. Measured 19% faster than the branchless form.
			const float d = drow[j];
			if (d == 0.0f) {
				continue;
			}
			float *grow = G + (size_t)j * (size_t)K;
			for (int k = 0; k < K; k++) {
				grow[k] += d * arow[k];
			}
		}
	}
}

// G[N x K] += D[B x N]^T * A[B x K], both dimensions compile-time.
template <int N, int K>
inline void AccumOuter_NK(const float *D, const float *A, float *G, int B) {
	for (int b = 0; b < B; b++) {
		const float *drow = D + (size_t)b * (size_t)N;
		const float *arow = A + (size_t)b * (size_t)K;
#pragma clang loop unroll(full)
		for (int j = 0; j < N; j++) {
			const float d = drow[j];
			if (d == 0.0f) {
				continue; // ReLU sparsity: skips a whole row of work
			}
			float *grow = G + (size_t)j * (size_t)K;
#pragma clang loop unroll(full)
			for (int k = 0; k < K; k++) {
				grow[k] += d * arow[k];
			}
		}

	}
}

inline void AccumOuterDispatch(const float *D, const float *A, float *G, int B, int K, int N) {
#define PQL_AO(NN, KK)                                                                             \
	if (N == (NN) && K == (KK)) {                                                                  \
		AccumOuter_NK<NN, KK>(D, A, G, B);                                                         \
		return;                                                                                    \
	}
	PQL_AO(32, 3) PQL_AO(32, 4) PQL_AO(32, 5) PQL_AO(32, 6)
	PQL_AO(48, 3) PQL_AO(48, 4) PQL_AO(48, 5) PQL_AO(48, 6)
	PQL_AO(64, 3) PQL_AO(64, 4) PQL_AO(64, 5) PQL_AO(64, 6)
	PQL_AO(128, 3) PQL_AO(128, 4) PQL_AO(128, 5) PQL_AO(128, 6)
#undef PQL_AO
	AccumOuter(D, A, G, B, K, N);
}

// dA[B x K] = D[B x N] * W[N x K], masked by the pre-activation sign of H.
inline void BackThroughRelu(const float *D, const float *W, const float *H, float *dA, int B,
                            int K, int N) {
	for (int b = 0; b < B; b++) {
		const float *drow = D + (size_t)b * (size_t)N;
		float *arow = dA + (size_t)b * (size_t)K;
		for (int k = 0; k < K; k++) {
			arow[k] = 0.0f;
		}
		for (int j = 0; j < N; j++) {
			const float d = drow[j];
			if (d == 0.0f) {
				continue; // same sparsity skip as AccumOuter
			}
			const float *wrow = W + (size_t)j * (size_t)K;
			for (int k = 0; k < K; k++) {
				arow[k] += d * wrow[k];
			}
		}
		// Mask by multiply rather than branch: the ReLU gate is data dependent and
		// splits roughly evenly, which is the worst case for a predictor.
		const float *hrow = H + (size_t)b * (size_t)K;
		for (int k = 0; k < K; k++) {
			arow[k] *= (hrow[k] > 0.0f) ? 1.0f : 0.0f;
		}
	}
}

// Score a batch through the same kernels training uses. Scratch is supplied by
// the caller so the evaluation path allocates nothing per epoch.
struct EvalScratch {
	std::vector<float> h1, h2;
	void Ensure(size_t rows, int hidden) {
		if (h1.size() < rows * (size_t)hidden) {
			h1.assign(rows * (size_t)hidden, 0.0f);
			h2.assign(rows * (size_t)hidden, 0.0f);
		}
	}
};

// Adam over a flat parameter block; one instance per tensor.
struct Adam {
	std::vector<float> m, v;
	int t = 0;
	void Step(std::vector<float> &w, const float *g, double lr) {
		if (m.size() != w.size()) {
			m.assign(w.size(), 0.0f);
			v.assign(w.size(), 0.0f);
		}
		t++;
		const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
		const double c1 = 1.0 - std::pow(b1, t), c2 = 1.0 - std::pow(b2, t);
		for (size_t i = 0; i < w.size(); i++) {
			m[i] = float(b1 * m[i] + (1 - b1) * g[i]);
			v[i] = float(b2 * v[i] + (1 - b2) * double(g[i]) * double(g[i]));
			const double mh = m[i] / c1, vh = v[i] / c2;
			w[i] = float(w[i] - lr * mh / (std::sqrt(vh) + eps));
		}
	}
};

// ===========================================================================
// 11. Metrics
// ===========================================================================

inline double AUROC(const std::vector<std::pair<double, int>> &scored) {
	std::vector<std::pair<double, int>> v(scored);
	std::sort(v.begin(), v.end());
	double rank_sum = 0;
	int64_t npos = 0, nneg = 0;
	size_t i = 0;
	while (i < v.size()) {
		size_t j = i;
		while (j < v.size() && v[j].first == v[i].first) {
			j++;
		}
		const double avg_rank = (double(i) + double(j - 1)) / 2.0 + 1.0;
		for (size_t k = i; k < j; k++) {
			if (v[k].second) {
				rank_sum += avg_rank;
				npos++;
			} else {
				nneg++;
			}
		}
		i = j;
	}
	if (!npos || !nneg) {
		// Undefined, not "chance". Returning 0.5 made every epoch tie, so strict
		// improvement never fired and training silently kept epoch 1.
		return std::numeric_limits<double>::quiet_NaN();
	}
	return (rank_sum - double(npos) * double(npos + 1) / 2.0) / (double(npos) * double(nneg));
}

inline double MAE(const std::vector<std::pair<double, double>> &pred_true) {
	if (pred_true.empty()) {
		return 0.0;
	}
	double acc = 0;
	for (auto &p : pred_true) {
		acc += std::fabs(p.first - p.second);
	}
	return acc / double(pred_true.size());
}

// ===========================================================================
// 12. Example collection
// ===========================================================================

struct Dataset {
	std::vector<Example> examples;
	// One flat buffer, reserved up front: a vector per example meant one heap
	// allocation per entity and scattered rows. Here a row is a contiguous span
	// and the whole set streams.
	std::vector<float> x;
	int width = 0;
	size_t censored = 0;
	size_t no_anchor = 0;

	const float *Row(size_t i) const {
		return &x[i * (size_t)width];
	}
	size_t Count() const {
		return examples.size();
	}
};

// Resolve the anchor column: explicit AT wins, otherwise the entity's own time
// column. Attribute prediction with no time anywhere is legitimate and yields a
// single non-temporal snapshot.
inline int ResolveAnchorColumn(const Frame &entity, const Statement &stmt) {
	if (stmt.every.present) {
		return -1; // anchors are generated, not read from a column
	}
	if (!stmt.anchor.name.empty()) {
		const int i = entity.Find(stmt.anchor.name);
		if (i < 0) {
			throw std::runtime_error("pql: AT column '" + stmt.anchor.name + "' not found on '" +
			                         entity.name + "'");
		}
		return i;
	}
	return Database::GuessTimeColumn(entity);
}

// Choose the columns and links the model reads. Deliberately conservative: the
// target, the anchor and anything that is a declared key are excluded, since a
// key's numeric value carries identity rather than signal.
// `train_cutoff` bounds which entity rows contribute to the normalisation
// statistics. Fitting mean/sd over every row, including the folds the model will
// be scored on, is transductive: harmless-looking, but it leaks the shape of the
// test distribution into training.
inline FeatureSpec FitFeatureSpec(const Database &db, const Frame &entity, const Statement &,
                                  int anchor_col, int target_col, int max_mean_cols,
                                  double train_cutoff = std::numeric_limits<double>::infinity()) {
	FeatureSpec spec;
	std::vector<bool> excluded(entity.columns.size(), false);
	if (anchor_col >= 0) {
		excluded[(size_t)anchor_col] = true;
	}
	if (target_col >= 0) {
		excluded[(size_t)target_col] = true;
	}
	for (const auto &fk : db.fks) {
		if (ToUpper(fk.parent_table) == ToUpper(entity.name)) {
			const int i = entity.Find(fk.parent_column);
			if (i >= 0) {
				excluded[(size_t)i] = true;
			}
		}
		if (ToUpper(fk.child_table) == ToUpper(entity.name)) {
			const int i = entity.Find(fk.child_column);
			if (i >= 0) {
				excluded[(size_t)i] = true;
			}
		}
	}
	for (size_t i = 0; i < entity.columns.size(); i++) {
		if (excluded[i] || !entity.columns[i].IsNumeric()) {
			continue;
		}
		// An entity row is read as it stands now, not as it stood at the anchor, so
		// any other timestamp on it (last_seen_at, closed_at) may record something
		// that happened after the anchor. Those are perfect leaks, so they are not
		// features. The child-link path is unaffected: it is cut at the anchor.
		if (entity.columns[i].type == ColType::TIMESTAMP) {
			continue;
		}
		const Column &c = entity.columns[i];
		// One branchless pass with four partial sums: a null contributes zero, so
		// the reduction vectorizes and the column is read once instead of twice.
		// Raw pointers plus a multiply-by-mask instead of a select: the vector<>
		// indirection and the branch both blocked vectorization here.
		const double *num = c.num.data();
		const uint8_t *val = c.valid.data();
		const size_t rows_n = entity.nrows;
		const Column *anchor_c = anchor_col >= 0 ? &entity.columns[(size_t)anchor_col] : nullptr;
		if (anchor_c && std::isfinite(train_cutoff)) {
			// Bounded pass: only rows at or before the cutoff inform mean and sd.
			double bs = 0, bq = 0;
			uint64_t bn = 0;
			for (size_t r2 = 0; r2 < rows_n; r2++) {
				if (!val[r2] || !anchor_c->valid[r2] || anchor_c->num[r2] > train_cutoff) {
					continue;
				}
				bs += num[r2];
				bq += num[r2] * num[r2];
				bn++;
			}
			if (bn > 1) {
				const double mean_b = bs / double(bn);
				const double var_b = (bq - double(bn) * mean_b * mean_b) / double(bn - 1);
				FeatureSpec::SelfCol sc_b;
				sc_b.index = (int)i;
				sc_b.name = c.name;
				sc_b.mean = mean_b;
				sc_b.sd = var_b > 1e-18 ? std::sqrt(var_b) : 1.0;
				spec.self_cols.push_back(sc_b);
				continue;
			}
		}
		double s0 = 0, s1 = 0, q0 = 0, q1 = 0;
		uint64_t n0 = 0, n1 = 0;
		size_t r = 0;
		for (; r + 1 < rows_n; r += 2) {
			const double m0 = double(val[r]), m1 = double(val[r + 1]);
			const double v0 = num[r] * m0, v1 = num[r + 1] * m1;
			s0 += v0;
			s1 += v1;
			q0 += v0 * num[r];
			q1 += v1 * num[r + 1];
			n0 += val[r];
			n1 += val[r + 1];
		}
		for (; r < rows_n; r++) {
			const double m = double(val[r]);
			const double v = num[r] * m;
			s0 += v;
			q0 += v * num[r];
			n0 += val[r];
		}
		const size_t n = (size_t)(n0 + n1);
		const double sum = s0 + s1, sumsq = q0 + q1;
		const double mean = n ? sum / double(n) : 0.0;
		const double var = n > 1 ? (sumsq - double(n) * mean * mean) / double(n - 1) : 0.0;
		const double sd = var > 0 ? std::sqrt(var) : 1.0;
		FeatureSpec::SelfCol sc;
		sc.index = (int)i;
		sc.name = entity.columns[i].name;
		sc.mean = mean;
		sc.sd = sd > 1e-9 ? sd : 1.0;
		spec.self_cols.push_back(sc);
	}
	for (size_t l = 0; l < db.links.size(); l++) {
		const Link &lk = db.links[l];
		if (ToUpper(lk.parent_table) != ToUpper(entity.name)) {
			continue;
		}
		FeatureSpec::LinkAgg la;
		la.link = (int)l;
		la.child_table = lk.child_table;
		const Frame &child = db.At(lk.child_table);
		la.child_key = lk.child_key_col >= 0
		                   ? child.columns[(size_t)lk.child_key_col].name
		                   : std::string();
		for (size_t c = 0; c < child.columns.size() && (int)la.mean_cols.size() < max_mean_cols; c++) {
			if ((int)c == lk.child_key_col || (int)c == lk.child_time_col) {
				continue;
			}
			if (!child.columns[c].IsNumeric()) {
				continue;
			}
			// Identifier columns average to a meaningless magnitude; averaging a
			// key says nothing about behaviour.
			const std::string up = ToUpper(child.columns[c].name);
			if (up == "ID" || (up.size() > 3 && up.compare(up.size() - 3, 3, "_ID") == 0)) {
				continue;
			}
			const Column &cc = child.columns[c];
			const double *cnum = cc.num.data();
			const uint8_t *cval = cc.valid.data();
			const size_t crows = child.nrows;
			double cs0 = 0, cs1 = 0, cq0 = 0, cq1 = 0;
			uint64_t k0 = 0, k1 = 0;
			size_t cr = 0;
			for (; cr + 1 < crows; cr += 2) {
				const double v0 = cnum[cr] * double(cval[cr]);
				const double v1 = cnum[cr + 1] * double(cval[cr + 1]);
				cs0 += v0;
				cs1 += v1;
				cq0 += v0 * cnum[cr];
				cq1 += v1 * cnum[cr + 1];
				k0 += cval[cr];
				k1 += cval[cr + 1];
			}
			for (; cr < crows; cr++) {
				const double v = cnum[cr] * double(cval[cr]);
				cs0 += v;
				cq0 += v * cnum[cr];
				k0 += cval[cr];
			}
			const size_t cn = (size_t)(k0 + k1);
			const double mu = cn ? (cs0 + cs1) / double(cn) : 0.0;
			const double cvar =
			    cn > 1 ? ((cq0 + cq1) - double(cn) * mu * mu) / double(cn - 1) : 0.0;
			const double sd = cvar > 0 ? std::sqrt(cvar) : 1.0;
			la.mean_cols.push_back((int)c);
			la.mean_names.push_back(cc.name);
			la.mean_mu.push_back(mu);
			la.mean_sd.push_back(sd > 1e-9 ? sd : 1.0);
		}
		spec.link_aggs.push_back(la);
	}
	spec.ComputeWidth();
	return spec;
}

// Assemble labelled examples. For a forecast the label is aggregated from the
// target table over (anchor, anchor+horizon]; for an attribute it is read from
// the entity row and rows without one are skipped.
inline Dataset CollectExamples(const Database &db, const Statement &stmt, const FeatureSpec &spec,
                               const AggCache &cache, int anchor_col, bool need_labels) {
	const Frame &entity = db.At(stmt.entity_table);
	Dataset ds;
	ds.width = spec.width;
	// Bound the buffer up front: at most one row per entity.
	ds.examples.reserve(entity.nrows);
	ds.x.reserve(entity.nrows * (size_t)spec.width);

	int target_link = -1, value_col = -1;
	const bool forecast = IsForecast(stmt.target.kind);
	if (forecast && stmt.target.target_table != "*") {
		for (size_t l = 0; l < db.links.size(); l++) {
			if (ToUpper(db.links[l].parent_table) == ToUpper(entity.name) &&
			    ToUpper(db.links[l].child_table) == ToUpper(stmt.target.target_table)) {
				target_link = (int)l;
				break;
			}
		}
		if (target_link < 0) {
			throw std::runtime_error("pql: no foreign key connects '" + stmt.target.target_table +
			                         "' to '" + entity.name +
			                         "', so its rows cannot be attributed to an entity");
		}
		// Without a time column on the child, every row is inside every window: the
		// horizon silently becomes a no-op and the label collapses to the lifetime
		// total, which is also fed back in as a feature. That is a leak, so refuse.
		if (!db.links[(size_t)target_link].dated) {
			throw std::runtime_error(
			    "pql: '" + stmt.target.target_table +
			    "' has no time column, so HORIZON cannot place its rows in the future. "
			    "Join a timestamp onto it (e.g. from its parent transaction) and predict "
			    "over that table instead.");
		}
		if (!stmt.target.ref.name.empty()) {
			value_col = db.At(stmt.target.target_table).Find(stmt.target.ref.name);
			if (value_col < 0) {
				throw std::runtime_error("pql: no column '" + stmt.target.ref.name + "' on '" +
				                         stmt.target.target_table + "'");
			}
		}
	}
	const int attr_col = forecast ? -1 : entity.Find(stmt.target.ref.name);
	if (!forecast && attr_col < 0) {
		throw std::runtime_error("pql: no column '" + stmt.target.ref.ToString() + "' on '" +
		                         entity.name + "'");
	}

	const double horizon = stmt.horizon.present ? double(stmt.horizon.Micros()) : 0.0;
	// A label window that runs past the end of the child data is silently
	// truncated, so those examples are not labelled, they are censored.
	double child_tmax = std::numeric_limits<double>::infinity();
	if (forecast && target_link >= 0) {
		const Link &tl = db.links[(size_t)target_link];
		if (tl.dated) {
			const Frame &tf = db.At(stmt.target.target_table);
			double m = -std::numeric_limits<double>::infinity();
			for (size_t r = 0; r < tf.nrows; r++) {
				const double t = tl.Time((uint32_t)r);
				if (std::isfinite(t) && t > m) {
					m = t;
				}
			}
			child_tmax = m;
		}
	}
	// EVERY: derive the anchor grid from the data. Start one horizon in (so the
	// first example has history) and stop one horizon short of the end (so no
	// label is censored).
	std::vector<double> generated;
	if (stmt.every.present && forecast && target_link >= 0) {
		const Link &tl = db.links[(size_t)target_link];
		if (tl.dated) {
			const Frame &tf = db.At(stmt.target.target_table);
			double lo = std::numeric_limits<double>::infinity();
			double hi = -std::numeric_limits<double>::infinity();
			for (size_t r = 0; r < tf.nrows; r++) {
				const double t = tl.Time((uint32_t)r);
				if (!std::isfinite(t)) {
					continue;
				}
				lo = std::min(lo, t);
				hi = std::max(hi, t);
			}
			const double step = double(stmt.every.Micros());
			const double first = lo + horizon;
			const double last = hi - horizon;
			if (step > 0 && std::isfinite(first) && last > first) {
				for (double t = first; t <= last; t += step) {
					generated.push_back(t);
				}
			}
			if (generated.empty()) {
				throw std::runtime_error(
				    "pql: EVERY produced no anchors; '" + stmt.target.target_table +
				    "' does not span enough time for one horizon before and after");
			}
		}
	}
	size_t censored = 0;
	size_t no_anchor = 0;
	std::vector<float> feat;
	for (size_t r = 0; r < entity.nrows; r++) {
		if (!EvalFilter(stmt.filter.get(), entity, r)) {
			continue;
		}
		if (!generated.empty()) {
			for (double gen_anchor : generated) {
				Example gex;
				gex.entity_row = (uint32_t)r;
				gex.anchor = gen_anchor;
				bool any_prev = false;
				gex.base = AggregateWindow(db, db.links[(size_t)target_link],
				                           db.At(stmt.target.target_table), (uint32_t)r,
				                           gex.anchor - horizon, gex.anchor, stmt.target.kind,
				                           value_col, stmt.target.filter.get(), any_prev);
				if (need_labels) {
					bool any = false;
					gex.label = AggregateWindow(db, db.links[(size_t)target_link],
					                            db.At(stmt.target.target_table), (uint32_t)r,
					                            gex.anchor, gex.anchor + horizon, stmt.target.kind,
					                            value_col, stmt.target.filter.get(), any);
					gex.has_label = true;
				}
				BuildFeatures(db, entity, spec, cache, (uint32_t)r, gex.anchor, feat);
				ds.examples.push_back(gex);
				ds.x.insert(ds.x.end(), feat.begin(), feat.end());
			}
			continue;
		}
		Example ex;
		ex.entity_row = (uint32_t)r;
		if (stmt.anchor_is_literal) {
			ex.anchor = stmt.anchor_literal.kind == Literal::Kind::NUMBER
			                ? stmt.anchor_literal.number
			                : 0.0;
		} else if (anchor_col >= 0) {
			const Column &ac = entity.columns[(size_t)anchor_col];
			if (!ac.valid[r]) {
				no_anchor++; // no anchor, no honest cutoff
				continue;
			}
			ex.anchor = ac.num[r];
		} else {
			ex.anchor = 1.0e18; // no time dimension: everything is visible
		}

		if (forecast && target_link >= 0) {
			bool any_prev = false;
			ex.base = AggregateWindow(db, db.links[(size_t)target_link],
			                          db.At(stmt.target.target_table), (uint32_t)r,
			                          ex.anchor - horizon, ex.anchor, stmt.target.kind, value_col,
			                          stmt.target.filter.get(), any_prev);
		}
		if (need_labels) {
			if (forecast) {
				if (target_link < 0) {
					continue;
				}
				if (ex.anchor + horizon > child_tmax) {
					censored++;
					continue;
				}
				bool any = false;
				ex.label = AggregateWindow(db, db.links[(size_t)target_link],
				                           db.At(stmt.target.target_table), (uint32_t)r, ex.anchor,
				                           ex.anchor + horizon, stmt.target.kind, value_col,
				                           stmt.target.filter.get(), any);
				ex.has_label = true;
			} else {
				const Column &tc = entity.columns[(size_t)attr_col];
				if (!tc.valid[r]) {
					continue;
				}
				ex.label = tc.type == ColType::CATEGORY ? double(tc.code[r]) : tc.num[r];
				ex.has_label = true;
			}
		}
		BuildFeatures(db, entity, spec, cache, (uint32_t)r, ex.anchor, feat);
		ds.examples.push_back(ex);
		ds.x.insert(ds.x.end(), feat.begin(), feat.end());
	}
	ds.censored = censored;
	ds.no_anchor = no_anchor;
	return ds;
}

// ===========================================================================
// 13. Training
// ===========================================================================

// Score an index set in batches through MatMulNT. Reuses the training scratch,
// so an epoch's validation pass performs no allocation at all.
inline void ScoreRows(const Model &model, const Dataset &ds, const std::vector<size_t> &idx,
                      float *xbuf, size_t xcap, EvalScratch &scratch, std::vector<double> &out) {
	const int width = ds.width, hidden = model.hidden;
	const size_t cap = std::max<size_t>(1, xcap / (size_t)std::max(1, width));
	out.assign(idx.size(), 0.0);
	scratch.Ensure(cap, hidden);
	for (size_t base = 0; base < idx.size(); base += cap) {
		const int B = (int)std::min(cap, idx.size() - base);
		for (int i = 0; i < B; i++) {
			const float *row = ds.Row(idx[base + (size_t)i]);
			std::memcpy(xbuf + (size_t)i * (size_t)width, row, sizeof(float) * (size_t)width);
		}
		MatMulNT(xbuf, model.w1.data(), model.b1.data(), scratch.h1.data(), B, width, hidden,
		         true);
		MatMulNT(scratch.h1.data(), model.w2.data(), model.b2.data(), scratch.h2.data(), B, hidden,
		         hidden, true);
		for (int i = 0; i < B; i++) {
			const float *h2 = &scratch.h2[(size_t)i * (size_t)hidden];
			double o = model.b3[0];
			for (int j = 0; j < hidden; j++) {
				o += double(model.w3[(size_t)j]) * double(h2[j]);
			}
			double v = model.classification ? 1.0 / (1.0 + std::exp(-o))
			                                : o * model.label_sd + model.label_mean;
			if (!model.classification) {
				if (model.residual) {
					v += ds.examples[idx[base + (size_t)i]].base;
				}
				if (model.nonnegative && v < 0.0) {
					v = 0.0; // a negative count is not a number anyone can use
				}
			}
			out[base + (size_t)i] = v;
		}
	}
}

// ---------------------------------------------------------------------------
// Step 3/4: batched forward and backward.
//
// Scratch is supplied by the caller from the arena, so a training step performs
// no allocation. Everything is row-major [B x k] so each GEMM reduces along the
// unit-stride axis of both operands, which is what lets MatMulNT vectorize.
// ---------------------------------------------------------------------------

// All per-step buffers come from one block. The working set is an exact function
// of (batch, channels, per-relation widths), all known once the model is built,
// so there is no reason to return to the allocator afterwards; the pointers below
// are carved once and stay valid for the run.
struct SageScratch {
	Arena arena;
	float *xself = nullptr;      // [B x self_dim]
	std::vector<float *> means;  // per edge: [B x src_dim]
	float *z = nullptr;          // [B x C] pre-activation
	float *h = nullptr;          // [B x C] post-ReLU
	float *dz = nullptr;         // [B x C]
	float *acc = nullptr;        // [B x C] accumulator for edge terms
	float *dmean = nullptr;      // [C] message gradient, hoisted out of the child loop
	size_t self_stride = 0;
	std::vector<size_t> mean_stride;

	size_t Bytes(size_t B, const SageParams &p) const {
		size_t f = B * (size_t)std::max(1, p.self_dim);
		for (size_t e = 0; e < p.w_neigh.size(); e++) {
			f += B * (size_t)std::max(1, p.src_dim[e]);
		}
		f += 4 * B * (size_t)p.channels + (size_t)p.channels;
		// one alignment pad per allocation, generously
		return f * sizeof(float) + 64 * (8 + p.w_neigh.size());
	}

	void Ensure(size_t B, const SageParams &p) {
		arena.Reserve(Bytes(B, p));
		self_stride = (size_t)std::max(1, p.self_dim);
		xself = arena.Zeroed<float>(B * self_stride);
		means.assign(p.w_neigh.size(), nullptr);
		mean_stride.assign(p.w_neigh.size(), 0);
		for (size_t e = 0; e < p.w_neigh.size(); e++) {
			mean_stride[e] = (size_t)std::max(1, p.src_dim[e]);
			means[e] = arena.Zeroed<float>(B * mean_stride[e]);
		}
		z = arena.Zeroed<float>(B * (size_t)p.channels);
		h = arena.Zeroed<float>(B * (size_t)p.channels);
		dz = arena.Zeroed<float>(B * (size_t)p.channels);
		acc = arena.Zeroed<float>(B * (size_t)p.channels);
		dmean = arena.Zeroed<float>((size_t)p.channels);
	}
};

// Gather the batch's own features and its neighbourhood means. This is the only
// part that touches the graph; everything after it is dense linear algebra.
inline void SageGather(const Database &db, const SageSpec &spec, const SageFeatures &feat,
                       const SagePrefix &pre, const Dataset &ds, const std::vector<size_t> &idx,
                       size_t from, int B, const SageParams &params, SageScratch &sc) {
	const Frame &entity = db.tables[(size_t)spec.nodes[(size_t)spec.entity_node].table];
	(void)entity;
	const int sd = std::max(1, params.self_dim);
	const float *xsrc = feat.x[(size_t)spec.entity_node].data();
	for (int i = 0; i < B; i++) {
		const Example &ex = ds.examples[idx[from + (size_t)i]];
		if (params.self_dim > 0) {
			std::memcpy(sc.xself + (size_t)i * (size_t)sd,
			            xsrc + (size_t)ex.entity_row * (size_t)params.self_dim,
			            sizeof(float) * (size_t)params.self_dim);
		}
		for (size_t e = 0; e < spec.edges.size(); e++) {
			const int d = params.src_dim[e];
			// src_dim already includes the degree slot, so pass the feature width.
			SageMeanAt(db, spec, pre, e, ex.entity_row, ex.anchor,
			           sc.means[e] + (size_t)i * (size_t)std::max(1, d), d - 1);
		}
	}
}

// z = W_self . x_self + SUM_r W_r . mean_r ; h = ReLU(z + b) ; out = w_out . h
inline void SageForward(const SageParams &p, int B, SageScratch &sc, std::vector<double> &out) {
	const int C = p.channels;
	if (p.self_dim > 0) {
		MatMulDispatch(sc.xself, p.w_self.data(), nullptr, sc.z, B, p.self_dim, C, false);
	} else {
		std::memset(sc.z, 0, sizeof(float) * (size_t)B * (size_t)C);
	}
	for (size_t e = 0; e < p.w_neigh.size(); e++) {
		const int d = p.src_dim[e];
		if (d <= 0) {
			continue;
		}
		MatMulDispatch(sc.means[e], p.w_neigh[e].data(), nullptr, sc.acc, B, d, C, false);
		float *z = sc.z;
		const float *a = sc.acc;
		const size_t n = (size_t)B * (size_t)C;
		for (size_t k = 0; k < n; k++) {
			z[k] += a[k];
		}
	}
	out.assign((size_t)B, 0.0);
	for (int i = 0; i < B; i++) {
		float *zr = sc.z + (size_t)i * (size_t)C;
		float *hr = sc.h + (size_t)i * (size_t)C;
		double o = p.b_out;
		for (int k = 0; k < C; k++) {
			const float v = zr[k] + p.bias[(size_t)k];
			zr[k] = v;
			hr[k] = v > 0.0f ? v : 0.0f;
			o += double(p.w_out[(size_t)k]) * double(hr[k]);
		}
		out[(size_t)i] = o;
	}
}

struct SageGrads {
	std::vector<float> gw_self, gbias, gw_out;
	std::vector<std::vector<float>> gw_neigh;
	float gb_out = 0.0f;

	void Ensure(const SageParams &p) {
		gw_self.assign(p.w_self.size(), 0.0f);
		gbias.assign(p.bias.size(), 0.0f);
		gw_out.assign(p.w_out.size(), 0.0f);
		gw_neigh.resize(p.w_neigh.size());
		for (size_t e = 0; e < gw_neigh.size(); e++) {
			gw_neigh[e].assign(p.w_neigh[e].size(), 0.0f);
		}
		gb_out = 0.0f;
	}
	void Zero() {
		std::fill(gw_self.begin(), gw_self.end(), 0.0f);
		std::fill(gbias.begin(), gbias.end(), 0.0f);
		std::fill(gw_out.begin(), gw_out.end(), 0.0f);
		for (auto &g : gw_neigh) {
			std::fill(g.begin(), g.end(), 0.0f);
		}
		gb_out = 0.0f;
	}
};

// `dout` holds dLoss/dout per row. Layer-1 inputs are raw features, so no
// gradient flows past the aggregation; that is exact for a single layer.
inline void SageBackward(const SageParams &p, int B, const std::vector<float> &dout,
                         SageScratch &sc, SageGrads &g) {
	const int C = p.channels;
	for (int i = 0; i < B; i++) {
		const float d = dout[(size_t)i];
		const float *hr = sc.h + (size_t)i * (size_t)C;
		const float *zr = sc.z + (size_t)i * (size_t)C;
		float *dzr = sc.dz + (size_t)i * (size_t)C;
		for (int k = 0; k < C; k++) {
			g.gw_out[(size_t)k] += d * hr[k];
			dzr[k] = (zr[k] > 0.0f) ? d * p.w_out[(size_t)k] : 0.0f;
			g.gbias[(size_t)k] += dzr[k];
		}
		g.gb_out += d;
	}
	if (p.self_dim > 0) {
		AccumOuterDispatch(sc.dz, sc.xself, g.gw_self.data(), B, p.self_dim, C);
	}
	for (size_t e = 0; e < p.w_neigh.size(); e++) {
		const int d = p.src_dim[e];
		if (d > 0) {
			AccumOuterDispatch(sc.dz, sc.means[e], g.gw_neigh[e].data(), B, d, C);
		}
	}
}

// Build the child layer: for every entity-edge, find the child's own children,
// and size a W_self plus one W_neigh per grandchild relation.
inline SageChildParams BuildChildLayer(const Database &db, const SageSpec &spec, int C,
                                       uint64_t seed) {
	SageChildParams cp;
	cp.channels = C;
	uint64_t st = seed ? seed ^ 0xA5A5A5A5u : 0x1234567u;
	auto rnd = [&]() {
		st ^= st << 13;
		st ^= st >> 7;
		st ^= st << 17;
		return double(int64_t(st >> 11)) / double(1ULL << 53);
	};
	auto fill = [&](std::vector<float> &w, size_t n, int fan) {
		w.assign(n, 0.0f);
		const double sc = std::sqrt(2.0 / double(std::max(1, fan)));
		for (auto &v : w) {
			v = float((rnd() * 2 - 1) * sc);
		}
	};
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const int sd = spec.NodeDim(spec.edges[e].src_node);
		cp.self_dim.push_back(sd);
		cp.w_self.push_back({});
		fill(cp.w_self.back(), (size_t)C * (size_t)std::max(1, sd), std::max(1, sd));
		cp.bias.push_back(std::vector<float>((size_t)C, 0.0f));
		const std::string child_tbl =
		    db.tables[(size_t)spec.nodes[(size_t)spec.edges[e].src_node].table].name;
		std::vector<int> links;
		std::vector<int> dims;
		std::vector<std::vector<float>> ws;
		for (size_t l = 0; l < db.links.size(); l++) {
			if (ToUpper(db.links[l].parent_table) != ToUpper(child_tbl)) {
				continue;
			}
			const int gt = db.Find(db.links[l].child_table);
			if (gt < 0) {
				continue;
			}
			std::vector<int> cols;
			SageNodeColumns(db, gt, &db.links[l], cols);
			if (cols.empty()) {
				continue;
			}
			links.push_back((int)l);
			dims.push_back((int)cols.size() + 1); // + degree
			ws.push_back({});
			fill(ws.back(), (size_t)C * (size_t)dims.back(), dims.back());
		}
		cp.gc_link.push_back(links);
		cp.gc_dim.push_back(dims);
		cp.w_gc.push_back(ws);
	}
	return cp;
}

// Everything the child layer needs to run and to be differentiated.
// Data-only part of the child layer: grandchild features, their running sums,
// and the message each child sees at its own timestamp. None of this depends on
// the weights, so recomputing it per epoch was pure waste; it is built once.
struct ChildStatic {
	// per entity-edge, per grandchild relation: [N_child x gc_dim] messages
	std::vector<std::vector<std::vector<float>>> gc_mean;
	bool built = false;
};

struct ChildEmbed {
	std::vector<std::vector<float>> h; // per entity-edge: [N_child x C] post-ReLU
	std::vector<std::vector<float>> z; // pre-activation, kept for backward
	// per entity-edge, per grandchild relation: [N_child x gc_dim] messages
	std::vector<std::vector<float>> psum; // prefix sums of h along the entity link
	std::vector<std::vector<uint32_t>> off;
};

// Forward the child layer for every child row, each at its own timestamp, then
// build the running sums the entity will read.
// Build the data-only part once: for every grandchild relation, the message each
// child row sees at its OWN timestamp. Weight-independent, so it never needs
// recomputing.
inline void BuildChildStatic(const Database &db, const SageSpec &spec, const SageChildParams &cp,
                             ChildStatic &cs) {
	cs.gc_mean.resize(spec.edges.size());
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &elk = db.links[(size_t)spec.edges[e].link];
		const Frame &child = db.At(elk.child_table);
		const size_t N = child.nrows;
		cs.gc_mean[e].resize(cp.gc_link[e].size());
		for (size_t g = 0; g < cp.gc_link[e].size(); g++) {
			const Link &glk = db.links[(size_t)cp.gc_link[e][g]];
			const Frame &gf = db.At(glk.child_table);
			const int gd = cp.gc_dim[e][g];
			const int gfeat = gd - 1;
			std::vector<int> gcols;
			SageNodeColumns(db, db.Find(glk.child_table), &glk, gcols);
			std::vector<float> gx(gf.nrows * (size_t)std::max(1, gfeat), 0.0f);
			for (int c = 0; c < gfeat; c++) {
				const Column &col = gf.columns[(size_t)gcols[(size_t)c]];
				double sum = 0;
				uint64_t n = 0;
				for (size_t r = 0; r < gf.nrows; r++) {
					if (col.valid[r]) {
						sum += col.num[r];
						n++;
					}
				}
				const double mu = n ? sum / double(n) : 0.0;
				double sq = 0;
				for (size_t r = 0; r < gf.nrows; r++) {
					if (col.valid[r]) {
						sq += (col.num[r] - mu) * (col.num[r] - mu);
					}
				}
				const double sd2 = n > 1 ? std::sqrt(sq / double(n - 1)) : 1.0;
				const double inv = sd2 > 1e-12 ? 1.0 / sd2 : 1.0;
				for (size_t r = 0; r < gf.nrows; r++) {
					gx[r * (size_t)std::max(1, gfeat) + (size_t)c] =
					    col.valid[r] ? float((col.num[r] - mu) * inv) : 0.0f;
				}
			}
			const size_t gparents = glk.off.empty() ? 0 : glk.off.size() - 1;
			std::vector<uint32_t> goff(gparents + 1, 0u);
			uint32_t tot = 0;
			for (size_t p = 0; p < gparents; p++) {
				goff[p] = tot;
				tot += glk.Count((uint32_t)p) + 1u;
			}
			goff[gparents] = tot;
			std::vector<float> gps((size_t)tot * (size_t)std::max(1, gfeat), 0.0f);
			for (size_t p = 0; p < gparents; p++) {
				const uint32_t b = glk.Begin((uint32_t)p), en = glk.End((uint32_t)p);
				float *run = gps.data() + (size_t)goff[p] * (size_t)std::max(1, gfeat);
				for (uint32_t i = b; i < en; i++) {
					const float *xw = gx.data() + (size_t)glk.flat[i] * (size_t)std::max(1, gfeat);
					float *cur = run + (size_t)(i - b + 1) * (size_t)std::max(1, gfeat);
					const float *prev = run + (size_t)(i - b) * (size_t)std::max(1, gfeat);
					for (int k = 0; k < gfeat; k++) {
						cur[k] = prev[k] + xw[k];
					}
				}
			}
			std::vector<float> &M = cs.gc_mean[e][g];
			M.assign(N * (size_t)gd, 0.0f);
			for (size_t u = 0; u < N && u < gparents; u++) {
				const double tu = elk.dated ? elk.Time((uint32_t)u)
				                            : std::numeric_limits<double>::infinity();
				const size_t vis = Database::VisiblePrefix(glk, gf, (uint32_t)u, tu);
				if (vis == 0) {
					continue;
				}
				float *out = &M[u * (size_t)gd];
				const float *hi = gps.data() + (size_t)(goff[u] + vis) * (size_t)std::max(1, gfeat);
				const float *lo = gps.data() + (size_t)goff[u] * (size_t)std::max(1, gfeat);
				const float inv = 1.0f / float(vis);
				for (int k = 0; k < gfeat; k++) {
					out[k] = (hi[k] - lo[k]) * inv;
				}
				out[gfeat] = float(std::log1p(double(vis)));
			}
		}
	}
	cs.built = true;
}

// Per epoch, only the weight-dependent part: the GEMMs, the ReLU, and the
// running sums the entity reads. Buffers are sized on the first call and reused.
inline void ComputeChildEmbeddings(const Database &db, const SageSpec &spec,
                                   const SageFeatures &feat, const SageChildParams &cp,
                                   const ChildStatic &cs, ChildEmbed &ce) {
	const int C = cp.channels;
	ce.h.resize(spec.edges.size());
	ce.z.resize(spec.edges.size());
	ce.psum.resize(spec.edges.size());
	ce.off.resize(spec.edges.size());
	static thread_local std::vector<float> tmp;

	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &elk = db.links[(size_t)spec.edges[e].link];
		const Frame &child = db.At(elk.child_table);
		const size_t N = child.nrows;
		const int sd = cp.self_dim[e];
		if (ce.h[e].size() != N * (size_t)C) {
			ce.h[e].assign(N * (size_t)C, 0.0f);
			ce.z[e].assign(N * (size_t)C, 0.0f);
		}
		const float *xc = feat.x[(size_t)spec.edges[e].src_node].data();
		if (sd > 0) {
			MatMulDispatch(xc, cp.w_self[e].data(), nullptr, ce.z[e].data(), (int)N, sd, C, false);
		} else {
			std::fill(ce.z[e].begin(), ce.z[e].end(), 0.0f);
		}
		for (size_t g = 0; g < cp.gc_link[e].size(); g++) {
			const int gd = cp.gc_dim[e][g];
			if (tmp.size() < N * (size_t)C) {
				tmp.assign(N * (size_t)C, 0.0f);
			}
			MatMulDispatch(cs.gc_mean[e][g].data(), cp.w_gc[e][g].data(), nullptr, tmp.data(),
			               (int)N, gd, C, false);
			float *z = ce.z[e].data();
			const float *t = tmp.data();
			const size_t n = N * (size_t)C;
			for (size_t k = 0; k < n; k++) {
				z[k] += t[k];
			}
		}
		for (size_t u = 0; u < N; u++) {
			float *zr = &ce.z[e][u * (size_t)C];
			float *hr = &ce.h[e][u * (size_t)C];
			const float *bs = cp.bias[e].data();
			for (int k = 0; k < C; k++) {
				zr[k] += bs[k];
				hr[k] = zr[k] > 0.0f ? zr[k] : 0.0f;
			}
		}
		const size_t nparents = elk.off.empty() ? 0 : elk.off.size() - 1;
		if (ce.off[e].size() != nparents + 1) {
			ce.off[e].assign(nparents + 1, 0u);
			uint32_t total = 0;
			for (size_t p = 0; p < nparents; p++) {
				ce.off[e][p] = total;
				total += elk.Count((uint32_t)p) + 1u;
			}
			ce.off[e][nparents] = total;
			ce.psum[e].assign((size_t)total * (size_t)C, 0.0f);
		}
		for (size_t p = 0; p < nparents; p++) {
			const uint32_t b = elk.Begin((uint32_t)p), en = elk.End((uint32_t)p);
			float *run = ce.psum[e].data() + (size_t)ce.off[e][p] * (size_t)C;
			for (int k = 0; k < C; k++) {
				run[k] = 0.0f;
			}
			for (uint32_t i = b; i < en; i++) {
				const float *hu = &ce.h[e][(size_t)elk.flat[i] * (size_t)C];
				float *cur = run + (size_t)(i - b + 1) * (size_t)C;
				const float *prev = run + (size_t)(i - b) * (size_t)C;
				for (int k = 0; k < C; k++) {
					cur[k] = prev[k] + hu[k];
				}
			}
		}
	}
}

// Gather for the two-hop case: the entity's message is the mean of its
// children's EMBEDDINGS (plus degree), not of their raw columns.
inline void SageGather2(const Database &db, const SageSpec &spec, const ChildEmbed &ce,
                        const Dataset &ds, const std::vector<size_t> &idx, size_t from, int B,
                        int C, SageScratch &sc) {
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &lk = db.links[(size_t)spec.edges[e].link];
		const Frame &child = db.At(lk.child_table);
		const int d = C + 1;
		for (int i = 0; i < B; i++) {
			const Example &ex = ds.examples[idx[from + (size_t)i]];
			float *out = sc.means[e] + (size_t)i * (size_t)d;
			const size_t vis = Database::VisiblePrefix(lk, child, ex.entity_row, ex.anchor);
			if (vis == 0) {
				for (int k = 0; k <= C; k++) {
					out[k] = 0.0f;
				}
				continue;
			}
			const uint32_t base = ce.off[e][ex.entity_row];
			const float *hi = ce.psum[e].data() + (size_t)(base + vis) * (size_t)C;
			const float *lo = ce.psum[e].data() + (size_t)base * (size_t)C;
			const float inv = 1.0f / float(vis);
			for (int k = 0; k < C; k++) {
				out[k] = (hi[k] - lo[k]) * inv;
			}
			out[C] = float(std::log1p(double(vis)));
		}
	}
}

// Gradients of the child layer.
struct ChildGrads {
	std::vector<std::vector<float>> gw_self, gbias;
	std::vector<std::vector<std::vector<float>>> gw_gc;
	void Ensure(const SageChildParams &cp) {
		gw_self.resize(cp.w_self.size());
		gbias.resize(cp.bias.size());
		gw_gc.resize(cp.w_gc.size());
		for (size_t e = 0; e < cp.w_self.size(); e++) {
			gw_self[e].assign(cp.w_self[e].size(), 0.0f);
			gbias[e].assign(cp.bias[e].size(), 0.0f);
			gw_gc[e].resize(cp.w_gc[e].size());
			for (size_t g = 0; g < cp.w_gc[e].size(); g++) {
				gw_gc[e][g].assign(cp.w_gc[e][g].size(), 0.0f);
			}
		}
	}
	void Zero() {
		for (auto &v : gw_self) {
			std::fill(v.begin(), v.end(), 0.0f);
		}
		for (auto &v : gbias) {
			std::fill(v.begin(), v.end(), 0.0f);
		}
		for (auto &e : gw_gc) {
			for (auto &v : e) {
				std::fill(v.begin(), v.end(), 0.0f);
			}
		}
	}
};

// Push the entity's gradient back through the aggregation into each visible
// child, then through the child layer. Only rows actually touched by this batch
// are cleared and updated, so the cost is O(batch * degree * C) rather than
// O(all children * C).
inline void SageBackward2(const Database &db, const SageSpec &spec, const SageParams &p,
                          const SageChildParams &cp, const SageFeatures &feat,
                          const ChildStatic &cs, const ChildEmbed &ce, const Dataset &ds,
                          const std::vector<size_t> &idx, size_t from, int B, SageScratch &sc,
                          ChildGrads &cg, std::vector<float> &dh_scratch,
                          std::vector<uint32_t> &touched, float *dmean) {
	const int C = p.channels;
	for (size_t e = 0; e < spec.edges.size(); e++) {
		const Link &lk = db.links[(size_t)spec.edges[e].link];
		const Frame &child = db.At(lk.child_table);
		const size_t N = child.nrows;
		if (dh_scratch.size() < N * (size_t)C) {
			dh_scratch.assign(N * (size_t)C, 0.0f);
		}
		touched.clear();

		// dmean = W_e^T . dz, then split (1/n) to each visible child
		for (int i = 0; i < B; i++) {
			const Example &ex = ds.examples[idx[from + (size_t)i]];
			const size_t vis = Database::VisiblePrefix(lk, child, ex.entity_row, ex.anchor);
			if (vis == 0) {
				continue;
			}
			const float *dzr = sc.dz + (size_t)i * (size_t)C;
			const float *W = p.w_neigh[e].data(); // [C x (C+1)]
			const int d = C + 1;
			const float inv = 1.0f / float(vis);
			const uint32_t b = lk.Begin(ex.entity_row);

			// The message gradient is the same for every child of this entity, so
			// it is formed once and then distributed. Computing it inside the child
			// loop cost O(degree * C^2) and strided through W; this is O(C^2) once
			// plus O(degree * C), and both loops now run along contiguous memory.
			std::memset(dmean, 0, sizeof(float) * (size_t)C);
			for (int c = 0; c < C; c++) {
				const float g = dzr[c];
				if (g == 0.0f) {
					continue;
				}
				const float *wrow = W + (size_t)c * (size_t)d;
				for (int k = 0; k < C; k++) {
					dmean[k] += g * wrow[k];
				}
			}
			for (int k = 0; k < C; k++) {
				dmean[k] *= inv;
			}
			for (size_t j = 0; j < vis; j++) {
				const uint32_t u = lk.flat[b + j];
				float *dh = &dh_scratch[(size_t)u * (size_t)C];
				if (dh[0] == 0.0f && dh[C - 1] == 0.0f) {
					touched.push_back(u);
				}
				for (int k = 0; k < C; k++) {
					dh[k] += dmean[k];
				}
			}
		}
		// through the child ReLU into the child layer's weights
		for (uint32_t u : touched) {
			const float *zr = &ce.z[e][(size_t)u * (size_t)C];
			float *dh = &dh_scratch[(size_t)u * (size_t)C];
			for (int k = 0; k < C; k++) {
				const float g = (zr[k] > 0.0f) ? dh[k] : 0.0f;
				dh[k] = g;
				cg.gbias[e][(size_t)k] += g;
			}
			const int sd = cp.self_dim[e];
			if (sd > 0) {
				const float *xu = feat.x[(size_t)spec.edges[e].src_node].data() +
				                  (size_t)u * (size_t)sd;
				for (int c = 0; c < C; c++) {
					const float g = dh[c];
					if (g == 0.0f) {
						continue;
					}
					float *row = &cg.gw_self[e][(size_t)c * (size_t)sd];
					for (int k = 0; k < sd; k++) {
						row[k] += g * xu[k];
					}
				}
			}
			for (size_t gidx = 0; gidx < cp.gc_link[e].size(); gidx++) {
				const int gd = cp.gc_dim[e][gidx];
				const float *m = &cs.gc_mean[e][gidx][(size_t)u * (size_t)gd];
				for (int c = 0; c < C; c++) {
					const float g = dh[c];
					if (g == 0.0f) {
						continue;
					}
					float *row = &cg.gw_gc[e][gidx][(size_t)c * (size_t)gd];
					for (int k = 0; k < gd; k++) {
						row[k] += g * m[k];
					}
				}
			}
		}
		for (uint32_t u : touched) {
			std::fill(dh_scratch.begin() + (size_t)u * (size_t)C,
			          dh_scratch.begin() + (size_t)(u + 1) * (size_t)C, 0.0f);
		}
	}
}

struct TrainReport {
	size_t n_train = 0, n_val = 0, n_test = 0;
	size_t n_positives = 0; // classification only; 0 here explains a NaN metric
	size_t n_censored = 0;  // dropped: label window ran past the data
	size_t n_no_anchor = 0; // dropped: the anchor column was NULL
	double baseline_metric = 0; // persistence, so a user can see what to beat
	double val_metric = 0, test_metric = 0;
	std::string metric_name = "auroc";
	int epochs_run = 0;
	int width = 0;
};

// Called once per epoch. The host passes something that throws when the user
// has interrupted; without it a long fit ignored Ctrl-C entirely.
using CancelCheck = std::function<void()>;

inline TrainReport TrainModel(const Database &db, const Statement &stmt, Model &model,
                              const CancelCheck &cancelled = CancelCheck()) {
	TrainReport rep;
	const Frame &entity = db.At(stmt.entity_table);
	const int anchor_col = ResolveAnchorColumn(entity, stmt);
	const bool forecast = IsForecast(stmt.target.kind);
	// Qualifiers are checked once, up front, against the relation each filter is
	// actually evaluated on.
	ValidateFilterScope(stmt.filter.get(), entity, stmt.entity_alias, "WHERE");
	if (forecast && stmt.target.filter && stmt.target.target_table != "*" &&
	    db.Find(stmt.target.target_table) >= 0) {
		ValidateFilterScope(stmt.target.filter.get(), db.At(stmt.target.target_table), "",
		                    "inner WHERE");
	}
	const int target_col = forecast ? -1 : entity.Find(stmt.target.ref.name);

	// A text target would be regressed as an arbitrary dictionary code.
	if (!forecast && target_col >= 0 &&
	    entity.columns[(size_t)target_col].type == ColType::CATEGORY) {
		throw std::runtime_error("pql: '" + stmt.target.ref.ToString() +
		                         "' is text; PQL predicts numeric and boolean attributes only");
	}
	const int max_mean_cols = (int)stmt.options.Num("MEAN_COLS", 3);
	// Work out where training ends before fitting normalisation, so the statistics
	// never see the validation or test folds. Derived from the anchors directly,
	// which is cheap and needs no features.
	double train_cutoff = std::numeric_limits<double>::infinity();
	if (stmt.every.present && forecast) {
		// The grid is uniform, so the 60% point of the child table's span is the
		// same boundary the split will land on.
		int tl = -1;
		for (size_t l = 0; l < db.links.size(); l++) {
			if (ToUpper(db.links[l].parent_table) == ToUpper(entity.name) &&
			    ToUpper(db.links[l].child_table) == ToUpper(stmt.target.target_table)) {
				tl = (int)l;
				break;
			}
		}
		if (tl >= 0 && db.links[(size_t)tl].dated) {
			const Frame &tf = db.At(stmt.target.target_table);
			const Link &tlk = db.links[(size_t)tl];
			double lo = std::numeric_limits<double>::infinity(), hi = -lo;
			for (size_t r = 0; r < tf.nrows; r++) {
				const double t = tlk.Time((uint32_t)r);
				if (std::isfinite(t)) {
					lo = std::min(lo, t);
					hi = std::max(hi, t);
				}
			}
			if (std::isfinite(lo) && hi > lo) {
				train_cutoff = lo + (hi - lo) * 0.6;
			}
		}
	} else if (anchor_col >= 0) {
		if (stmt.split.present) {
			train_cutoff = stmt.split.validate_from.number;
		} else {
			std::vector<double> anchors;
			anchors.reserve(entity.nrows);
			const Column &ac = entity.columns[(size_t)anchor_col];
			for (size_t r = 0; r < entity.nrows; r++) {
				if (ac.valid[r] && EvalFilter(stmt.filter.get(), entity, r)) {
					anchors.push_back(ac.num[r]);
				}
			}
			if (anchors.size() > 10) {
				std::sort(anchors.begin(), anchors.end());
				train_cutoff = anchors[(anchors.size() * 6) / 10]; // matches the 60/20/20 default
			}
		}
	}
	model.features =
	    FitFeatureSpec(db, entity, stmt, anchor_col, target_col, max_mean_cols, train_cutoff);
	const AggCache cache = BuildAggCache(db, model.features);
	Dataset ds = CollectExamples(db, stmt, model.features, cache, anchor_col, true);
	if (ds.examples.empty()) {
		if (ds.censored > 0) {
			throw std::runtime_error(
			    "pql: all " + std::to_string(ds.censored) +
			    " candidate rows were dropped because their label window ends after the last "
			    "row of '" + stmt.target.target_table +
			    "'. Use a shorter HORIZON or anchors further from the end of the data.");
		}
		throw std::runtime_error("pql: no training rows matched; check WHERE and AT");
	}
	if (model.features.width == 0) {
		throw std::runtime_error("pql: no usable features for '" + entity.name +
		                         "'; it has no numeric columns and no linked child tables");
	}
	for (const auto &la : model.features.link_aggs) {
		if (db.links[(size_t)la.link].duplicate_parent_keys) {
			throw std::runtime_error(
			    "pql: '" + entity.name +
			    "' has duplicate key values, so child rows cannot be attributed to one entity row. "
			    "Give the entity a unique key (a panel needs a synthetic id).");
		}
	}

	// Deliberately NOT decided here: the fold split happens first, so the task
	// type cannot be chosen by test-fold values, and a COUNT whose labels merely
	// happen to be 0/1 must not silently become a probability capped at 1.
	bool binary = false;
	// Residual only helps a real forecast; an attribute has no "previous window".
	model.spec_stmt = stmt;

	// Temporal split. Without an explicit SPLIT, hold out the latest 20% by
	// anchor so validation is always forward in time, never a random shuffle.
	std::vector<size_t> order(ds.examples.size());
	for (size_t i = 0; i < order.size(); i++) {
		order[i] = i;
	}
	std::sort(order.begin(), order.end(),
	          [&](size_t a, size_t b) { return ds.examples[a].anchor < ds.examples[b].anchor; });
	std::vector<size_t> tr, va, te;
	if (stmt.split.present) {
		const double v = stmt.split.validate_from.number, t = stmt.split.test_from.number;
		for (size_t i : order) {
			const double a = ds.examples[i].anchor;
			if (a < v) {
				tr.push_back(i);
			} else if (a < t) {
				va.push_back(i);
			} else {
				te.push_back(i);
			}
		}
	} else {
		const size_t n = order.size();
		const size_t c1 = n > 10 ? (n * 6) / 10 : n;
		const size_t c2 = n > 10 ? (n * 8) / 10 : n;
		for (size_t i = 0; i < n; i++) {
			if (i < c1) {
				tr.push_back(order[i]);
			} else if (i < c2) {
				va.push_back(order[i]);
			} else {
				te.push_back(order[i]);
			}
		}
	}
	// Now that the folds exist, decide the task type from training labels alone.
	{
		const bool countish = stmt.target.kind == TargetKind::COUNT ||
		                      stmt.target.kind == TargetKind::SUM ||
		                      stmt.target.kind == TargetKind::AVG ||
		                      stmt.target.kind == TargetKind::MIN ||
		                      stmt.target.kind == TargetKind::MAX;
		binary = !countish;
		if (binary) {
			for (size_t i : tr) {
				const double v = ds.examples[i].label;
				if (v != 0.0 && v != 1.0) {
					binary = false;
					break;
				}
			}
		}
	}
	model.classification = binary;
	rep.metric_name = binary ? "auroc" : "mae";
	// Residual only helps a real forecast; an attribute has no "previous window".
	model.residual = forecast && !binary;
	model.nonnegative = stmt.target.kind == TargetKind::COUNT ||
	                    stmt.target.kind == TargetKind::EXISTS;
	if (model.residual) {
		for (auto &e : ds.examples) {
			e.label -= e.base;
		}
	}
	if (tr.empty()) {
		// Falling back to "train on everything" here would report a test metric
		// measured on the training rows, which reads as a great result.
		throw std::runtime_error(
		    "pql: SPLIT leaves no training rows (every anchor is at or after VALIDATE FROM). "
		    "Check the split boundaries against the anchor column's range.");
	}
	if (stmt.split.present && (!va.empty() || !te.empty())) {
		for (size_t i : tr) {
			(void)i;
		}
	}

	if (!binary) {
		double sum = 0;
		for (size_t i : tr) {
			sum += ds.examples[i].label;
		}
		model.label_mean = tr.empty() ? 0 : sum / double(tr.size());
		double var = 0;
		for (size_t i : tr) {
			const double d = ds.examples[i].label - model.label_mean;
			var += d * d;
		}
		const double sd = tr.size() > 1 ? std::sqrt(var / double(tr.size() - 1)) : 1.0;
		model.label_sd = sd > 1e-9 ? sd : 1.0;
	}

	const int hidden = (int)stmt.options.Num("HIDDEN", 64);
	const int epochs = (int)stmt.options.Num("EPOCHS", 60);
	const double lr = stmt.options.Num("LR", 0.01);
	const uint64_t seed = (uint64_t)stmt.options.Num("SEED", 42);
	const int width = model.features.width;
	model.Init(width, hidden, seed);

	// ---- hetero GraphSAGE path -------------------------------------------
	if (ToUpper(stmt.options.Str("ARCH", "mlp")) == "SAGE") {
		model.sage = true;
		model.sage_spec = BuildSageSpec(db, entity, anchor_col, target_col, train_cutoff);
		if (model.sage_spec.edges.empty()) {
			throw std::runtime_error("pql: arch='sage' needs at least one linked child table with "
			                         "numeric columns");
		}
		const int C = (int)stmt.options.Num("HIDDEN", 64);
		const int n_layers = (int)stmt.options.Num("LAYERS", 1);
		model.sage_layers = n_layers;
		model.sage_params.Init(model.sage_spec, C, seed);
		const SageFeatures feats = BuildSageFeatures(db, model.sage_spec);
		const SagePrefix prefix = BuildSagePrefix(db, model.sage_spec, feats);
		ChildEmbed ce;
		ChildStatic cstat;
		ChildGrads cg;
		std::vector<float> dh_scratch;
		std::vector<uint32_t> touched;
		if (n_layers >= 2) {
			// The entity now aggregates child EMBEDDINGS, so each message is C wide
			// plus the degree slot.
			model.sage_child = BuildChildLayer(db, model.sage_spec, C, seed);
			for (size_t e = 0; e < model.sage_params.src_dim.size(); e++) {
				model.sage_params.src_dim[e] = C + 1;
				model.sage_params.w_neigh[e].assign((size_t)C * (size_t)(C + 1), 0.0f);
				uint64_t st2 = seed * 2654435761u + (uint64_t)e + 1;
				const double sc2 = std::sqrt(2.0 / double(C + 1));
				for (auto &v : model.sage_params.w_neigh[e]) {
					st2 ^= st2 << 13;
					st2 ^= st2 >> 7;
					st2 ^= st2 << 17;
					v = float(((double(int64_t(st2 >> 11)) / double(1ULL << 53)) * 2 - 1) * sc2);
				}
			}
			cg.Ensure(model.sage_child);
			BuildChildStatic(db, model.sage_spec, model.sage_child, cstat);
		}

		uint64_t srng = seed | 1ULL;
		auto sshuffle = [&](std::vector<size_t> &v) {
			for (size_t i = v.size(); i > 1; i--) {
				srng ^= srng << 13;
				srng ^= srng >> 7;
				srng ^= srng << 17;
				std::swap(v[i - 1], v[srng % i]);
			}
		};
		SageScratch sc;
		SageGrads gr;
		gr.Ensure(model.sage_params);
		std::vector<double> scores;
		std::vector<double> outs;
		std::vector<float> douts;
		SageParams best_p = model.sage_params;
		double best_m = -std::numeric_limits<double>::infinity();
		bool have_m = false;
		const size_t sbatch = (size_t)std::max(1.0, stmt.options.Num("BATCH", 64));
		sc.Ensure(sbatch, model.sage_params);

		auto score = [&](const std::vector<size_t> &fold, std::vector<double> &into) {
			into.assign(fold.size(), 0.0);
			for (size_t b = 0; b < fold.size(); b += sbatch) {
				const int B = (int)std::min(sbatch, fold.size() - b);
				if (n_layers >= 2) {
					SageGather2(db, model.sage_spec, ce, ds, fold, b, B, C, sc);
				} else {
					SageGather(db, model.sage_spec, feats, prefix, ds, fold, b, B, model.sage_params,
					           sc);
				}
				SageForward(model.sage_params, B, sc, outs);
				for (int i = 0; i < B; i++) {
					double v = binary ? 1.0 / (1.0 + std::exp(-outs[(size_t)i]))
					                  : outs[(size_t)i] * model.label_sd + model.label_mean;
					if (!binary) {
						if (model.residual) {
							v += ds.examples[fold[b + (size_t)i]].base;
						}
						if (model.nonnegative && v < 0.0) {
							v = 0.0;
						}
					}
					into[b + (size_t)i] = v;
				}
			}
		};

		for (int ep = 0; ep < epochs; ep++) {
			sshuffle(tr);
			if (n_layers >= 2) {
				// Child embeddings depend on the child weights, so they are refreshed
				// once per epoch rather than per batch.
				ComputeChildEmbeddings(db, model.sage_spec, feats, model.sage_child, cstat, ce);
			}
			for (size_t b = 0; b < tr.size(); b += sbatch) {
				const int B = (int)std::min(sbatch, tr.size() - b);
				if (n_layers >= 2) {
					SageGather2(db, model.sage_spec, ce, ds, tr, b, B, C, sc);
				} else {
					SageGather(db, model.sage_spec, feats, prefix, ds, tr, b, B, model.sage_params,
					           sc);
				}
				SageForward(model.sage_params, B, sc, outs);
				douts.assign((size_t)B, 0.0f);
				for (int i = 0; i < B; i++) {
					double y = ds.examples[tr[b + (size_t)i]].label;
					if (binary) {
						douts[(size_t)i] = float(1.0 / (1.0 + std::exp(-outs[(size_t)i])) - y);
					} else {
						y = (y - model.label_mean) / model.label_sd;
						const double d = outs[(size_t)i] - y;
						douts[(size_t)i] = float(d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d));
					}
				}
				gr.Zero();
				SageBackward(model.sage_params, B, douts, sc, gr);
				if (n_layers >= 2) {
					cg.Zero();
					SageBackward2(db, model.sage_spec, model.sage_params, model.sage_child, feats, cstat, ce,
					              ds, tr, b, B, sc, cg, dh_scratch, touched, sc.dmean);
				}
				const float inv = 1.0f / float(std::max(1, B));
				auto step = [&](std::vector<float> &w, std::vector<float> &g, Adam &opt) {
					for (auto &v : g) {
						v *= inv;
					}
					opt.Step(w, g.data(), lr);
				};
				static thread_local std::vector<Adam> opts;
				if (opts.size() < 3 + model.sage_params.w_neigh.size()) {
					opts.assign(3 + model.sage_params.w_neigh.size(), Adam());
				}
				step(model.sage_params.w_self, gr.gw_self, opts[0]);
				step(model.sage_params.bias, gr.gbias, opts[1]);
				step(model.sage_params.w_out, gr.gw_out, opts[2]);
				for (size_t e = 0; e < gr.gw_neigh.size(); e++) {
					step(model.sage_params.w_neigh[e], gr.gw_neigh[e], opts[3 + e]);
				}
				if (n_layers >= 2) {
					static thread_local std::vector<Adam> copts;
					size_t need = 0;
					for (size_t e = 0; e < cg.gw_self.size(); e++) {
						need += 2 + cg.gw_gc[e].size();
					}
					if (copts.size() < need) {
						copts.assign(need, Adam());
					}
					size_t oi = 0;
					for (size_t e = 0; e < cg.gw_self.size(); e++) {
						step(model.sage_child.w_self[e], cg.gw_self[e], copts[oi++]);
						step(model.sage_child.bias[e], cg.gbias[e], copts[oi++]);
						for (size_t g = 0; g < cg.gw_gc[e].size(); g++) {
							step(model.sage_child.w_gc[e][g], cg.gw_gc[e][g], copts[oi++]);
						}
					}
				}
				model.sage_params.b_out -= float(lr * double(gr.gb_out) * double(inv));
			}
			rep.epochs_run = ep + 1;
			if (cancelled) {
				cancelled();
			}
			if (!va.empty()) {
				score(va, scores);
				double m;
				if (binary) {
					std::vector<std::pair<double, int>> sc2;
					for (size_t i = 0; i < va.size(); i++) {
						sc2.emplace_back(scores[i], ds.examples[va[i]].label > 0.5 ? 1 : 0);
					}
					m = AUROC(sc2);
				} else {
					std::vector<std::pair<double, double>> pt;
					for (size_t i = 0; i < va.size(); i++) {
						const Example &ex = ds.examples[va[i]];
						pt.emplace_back(scores[i], model.residual ? ex.label + ex.base : ex.label);
					}
					m = -MAE(pt);
				}
				if (!(m < best_m)) {
					best_m = std::isnan(m) ? best_m : m;
					best_p = model.sage_params;
					have_m = true;
				}
			}
		}
		if (have_m) {
			model.sage_params = best_p;
			rep.val_metric = std::isfinite(best_m) ? (binary ? best_m : -best_m)
			                                       : std::numeric_limits<double>::quiet_NaN();
		}
		rep.n_train = tr.size();
		rep.n_val = va.size();
		rep.n_test = te.size();
		rep.n_censored = ds.censored;
		rep.n_no_anchor = ds.no_anchor;
		rep.width = (int)(model.sage_params.ParamCount() +
		                  (n_layers >= 2 ? model.sage_child.ParamCount() : 0));
		if (binary) {
			for (const auto &e : ds.examples) {
				rep.n_positives += (e.label > 0.5) ? 1 : 0;
			}
		}
		if (!te.empty()) {
			score(te, scores);
			if (binary) {
				std::vector<std::pair<double, int>> sc2;
				for (size_t i = 0; i < te.size(); i++) {
					sc2.emplace_back(scores[i], ds.examples[te[i]].label > 0.5 ? 1 : 0);
				}
				rep.test_metric = AUROC(sc2);
			} else {
				std::vector<std::pair<double, double>> pt;
				double be = 0;
				for (size_t i = 0; i < te.size(); i++) {
					const Example &ex = ds.examples[te[i]];
					const double truth = model.residual ? ex.label + ex.base : ex.label;
					pt.emplace_back(scores[i], truth);
					be += std::fabs(ex.base - truth);
				}
				rep.test_metric = MAE(pt);
				rep.baseline_metric = be / double(te.size());
			}
		}
		return rep;
	}

	const size_t batch = (size_t)std::max(1.0, stmt.options.Num("BATCH", 64));
	// One block for the whole run. The working set is a known function of
	// (batch, width, hidden), so it is sized once here and never grown.
	const size_t bw = batch * (size_t)width, bh = batch * (size_t)hidden;
	const size_t n_w1 = (size_t)width * (size_t)hidden, n_w2 = (size_t)hidden * (size_t)hidden;
	// Activations AND gradients. Sizing only the activations meant the gradient
	// allocations grew the buffer, which reallocates and dangles every pointer
	// already handed out.
	const size_t need =
	    sizeof(float) * (bw + 4 * bh + 2 * batch + n_w1 + n_w2 + 3 * (size_t)hidden + 1) + 64 * 16;
	Arena arena;
	arena.Reserve(need);
	float *X = arena.Zeroed<float>(bw);
	float *H1 = arena.Zeroed<float>(bh);
	float *H2 = arena.Zeroed<float>(bh);
	float *DH2 = arena.Zeroed<float>(bh);
	float *DH1 = arena.Zeroed<float>(bh);
	float *DO = arena.Zeroed<float>(batch);
	float *YB = arena.Zeroed<float>(batch);
	EvalScratch escratch;
	std::vector<double> scores;

	Adam a1, a2, a3, ab1, ab2, ab3;
	float *gw1 = arena.Zeroed<float>(n_w1);
	float *gb1 = arena.Zeroed<float>((size_t)hidden);
	float *gw2 = arena.Zeroed<float>(n_w2);
	float *gb2 = arena.Zeroed<float>((size_t)hidden);
	float *gw3 = arena.Zeroed<float>((size_t)hidden);
	float *gb3 = arena.Zeroed<float>(1);
	const size_t n_gw1 = n_w1, n_gw2 = n_w2;
	std::vector<float> h1v, h2v, dh1((size_t)hidden), dz1((size_t)hidden), dh2((size_t)hidden),
	    dz2((size_t)hidden);

	Model best = model;
	// Regression scores as -MAE, which is negative, so the sentinel must be
	// below every attainable value or selection silently never fires.
	double best_val = -std::numeric_limits<double>::infinity();
	bool have_val = false;
	rep.width = width;

	uint64_t rng = seed | 1ULL;
	auto shuffle = [&](std::vector<size_t> &v) {
		for (size_t i = v.size(); i > 1; i--) {
			rng ^= rng << 13;
			rng ^= rng >> 7;
			rng ^= rng << 17;
			std::swap(v[i - 1], v[rng % i]);
		}
	};

	// Batch-major scratch, allocated once and reused every step.

	for (int ep = 0; ep < epochs; ep++) {
		shuffle(tr);
		for (size_t bstart = 0; bstart < tr.size(); bstart += batch) {
			const size_t bend = std::min(bstart + batch, tr.size());
			const int B = (int)(bend - bstart);
			for (int i = 0; i < B; i++) {
				const float *xv = ds.Row(tr[bstart + (size_t)i]);
				std::memcpy(X + (size_t)i * (size_t)width, xv, sizeof(float) * (size_t)width);
				YB[(size_t)i] = (float)ds.examples[tr[bstart + (size_t)i]].label;
			}
			MatMulNT(X, model.w1.data(), model.b1.data(), H1, B, width, hidden, true);
			MatMulNT(H1, model.w2.data(), model.b2.data(), H2, B, hidden, hidden, true);

			for (int i = 0; i < B; i++) {
				const float *h2 = &H2[(size_t)i * (size_t)hidden];
				double o = model.b3[0];
				for (int j = 0; j < hidden; j++) {
					o += double(model.w3[(size_t)j]) * double(h2[j]);
				}
				double y = YB[(size_t)i];
				if (binary) {
					DO[(size_t)i] = float(1.0 / (1.0 + std::exp(-o)) - y);
				} else {
					y = (y - model.label_mean) / model.label_sd;
					// Huber, not squared: the reported metric is MAE, which is the
					// conditional median, and squared loss fits the mean. On skewed
					// counts those differ a lot.
					const double d = o - y;
					DO[(size_t)i] = float(d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d));
				}
			}

			std::memset(gw1, 0, sizeof(float) * n_gw1);
			std::memset(gb1, 0, sizeof(float) * (size_t)hidden);
			std::memset(gw2, 0, sizeof(float) * n_gw2);
			std::memset(gb2, 0, sizeof(float) * (size_t)hidden);
			std::memset(gw3, 0, sizeof(float) * (size_t)hidden);
			gb3[0] = 0.0f;

			for (int i = 0; i < B; i++) {
				const float d = DO[(size_t)i];
				const float *h2 = &H2[(size_t)i * (size_t)hidden];
				float *dh2 = &DH2[(size_t)i * (size_t)hidden];
				for (int j = 0; j < hidden; j++) {
					gw3[(size_t)j] += d * h2[j];
					dh2[j] = (h2[j] > 0.0f) ? d * model.w3[(size_t)j] : 0.0f;
				}
				gb3[0] += d;
			}
			AccumOuter(DH2, H1, gw2, B, hidden, hidden);
			for (int i = 0; i < B; i++) {
				const float *dh2 = &DH2[(size_t)i * (size_t)hidden];
				for (int j = 0; j < hidden; j++) {
					gb2[(size_t)j] += dh2[j];
				}
			}
			BackThroughRelu(DH2, model.w2.data(), H1, DH1, B, hidden, hidden);
			AccumOuter(DH1, X, gw1, B, width, hidden);
			for (int i = 0; i < B; i++) {
				const float *dh1 = &DH1[(size_t)i * (size_t)hidden];
				for (int j = 0; j < hidden; j++) {
					gb1[(size_t)j] += dh1[j];
				}
			}

			const float scale = 1.0f / float(std::max(1, B));
			for (size_t i = 0; i < n_gw1; i++) {
				gw1[i] *= scale;
			}
			for (int i = 0; i < hidden; i++) {
				gb1[i] *= scale;
				gb2[i] *= scale;
				gw3[i] *= scale;
			}
			for (size_t i = 0; i < n_gw2; i++) {
				gw2[i] *= scale;
			}
			gb3[0] *= scale;

			a1.Step(model.w1, gw1, lr);
			ab1.Step(model.b1, gb1, lr);
			a2.Step(model.w2, gw2, lr);
			ab2.Step(model.b2, gb2, lr);
			a3.Step(model.w3, gw3, lr);
			ab3.Step(model.b3, gb3, lr);
		}
		rep.epochs_run = ep + 1;
		if (cancelled) {
			cancelled();
		}

		if (!va.empty()) {
			ScoreRows(model, ds, va, X, bw, escratch, scores);
			double m;
			if (binary) {
				std::vector<std::pair<double, int>> sc;
				sc.reserve(va.size());
				for (size_t i = 0; i < va.size(); i++) {
					sc.emplace_back(scores[i], ds.examples[va[i]].label > 0.5 ? 1 : 0);
				}
				m = AUROC(sc);
			} else {
				std::vector<std::pair<double, double>> pt;
				pt.reserve(va.size());
				for (size_t i = 0; i < va.size(); i++) {
					const Example &e = ds.examples[va[i]];
					pt.emplace_back(scores[i], model.residual ? e.label + e.base : e.label);
				}
				m = -MAE(pt);
			}
			// >= so a later epoch wins ties; NaN (undefined metric) falls back to the
			// most recent weights rather than freezing at epoch 1.
			if (!(m < best_val)) {
				best_val = std::isnan(m) ? best_val : m;
				best = model;
				have_val = true;
			}
		}
	}
	if (have_val) {
		model = best;
		// -inf means no epoch ever produced a defined metric (a single-class fold),
		// which must read as "undefined", not as a very bad score.
		rep.val_metric = std::isfinite(best_val) ? (binary ? best_val : -best_val)
		                                        : std::numeric_limits<double>::quiet_NaN();
	}
	rep.n_train = tr.size();
	rep.n_val = va.size();
	rep.n_test = te.size();
	rep.n_censored = ds.censored;
	rep.n_no_anchor = ds.no_anchor;
	if (binary) {
		for (const auto &e : ds.examples) {
			rep.n_positives += (e.label > 0.5) ? 1 : 0;
		}
	}
	if (!te.empty()) {
		ScoreRows(model, ds, te, X, bw, escratch, scores);
		if (binary) {
			std::vector<std::pair<double, int>> sc;
			for (size_t i = 0; i < te.size(); i++) {
				sc.emplace_back(scores[i], ds.examples[te[i]].label > 0.5 ? 1 : 0);
			}
			rep.test_metric = AUROC(sc);
		} else {
			std::vector<std::pair<double, double>> pt;
			double base_err = 0;
			for (size_t i = 0; i < te.size(); i++) {
				const Example &e = ds.examples[te[i]];
				const double truth = model.residual ? e.label + e.base : e.label;
				pt.emplace_back(scores[i], truth);
				base_err += std::fabs(e.base - truth);
			}
			rep.test_metric = MAE(pt);
			rep.baseline_metric = te.empty() ? 0.0 : base_err / double(te.size());
		}
	}
	return rep;
}

// ===========================================================================
// 14. Prediction and registry
// ===========================================================================

struct Prediction {
	uint32_t entity_row = 0;
	double anchor = 0; // the time the prediction is made from
	double value = 0;
};

// Predict with a trained model. The stored training statement supplies the
// anchor and horizon so a PREDICT that omits them still lines up; anything the
// caller does restate must match, or the features would be built differently
// from how they were learned.
inline std::vector<Prediction> RunPredict(const Database &db, const Model &model,
                                          const Statement &stmt) {
	const Frame &entity = db.At(stmt.entity_table);
	Statement eff = stmt;
	if (eff.anchor.name.empty() && !eff.anchor_is_literal) {
		eff.anchor = model.spec_stmt.anchor;
	}
	if (!eff.horizon.present) {
		eff.horizon = model.spec_stmt.horizon;
	}
	// A model trained on a generated grid has no anchor column to read. Unless
	// the caller names a moment, predict from the most recent point in the data:
	// "what happens in the next horizon, starting now".
	if (model.spec_stmt.every.present && eff.anchor.name.empty() && !eff.anchor_is_literal) {
		double latest = -std::numeric_limits<double>::infinity();
		for (size_t l = 0; l < db.links.size(); l++) {
			const Link &lk = db.links[l];
			if (ToUpper(lk.parent_table) != ToUpper(entity.name) || !lk.dated) {
				continue;
			}
			const Frame &cf = db.At(lk.child_table);
			for (size_t r = 0; r < cf.nrows; r++) {
				const double t = lk.Time((uint32_t)r);
				if (std::isfinite(t) && t > latest) {
					latest = t;
				}
			}
		}
		if (std::isfinite(latest)) {
			eff.anchor_literal.kind = Literal::Kind::NUMBER;
			eff.anchor_literal.number = latest;
			eff.anchor_is_literal = true;
		}
	}
	// Restating the target or horizon was previously discarded, so a PREDICT could
	// ask for EXISTS and be answered with a count.
	if (stmt.target.kind != TargetKind::ATTRIBUTE || !stmt.target.ref.name.empty()) {
		const std::string asked = stmt.target.ToString();
		const std::string trained = model.spec_stmt.target.ToString();
		if (ToUpper(asked) != ToUpper(trained)) {
			throw std::runtime_error("pql: model '" + model.name + "' predicts " + trained +
			                         ", not " + asked + "; drop the target or retrain");
		}
	}
	if (stmt.horizon.present && model.spec_stmt.horizon.present &&
	    stmt.horizon.Micros() != model.spec_stmt.horizon.Micros()) {
		throw std::runtime_error("pql: model '" + model.name + "' was trained with HORIZON " +
		                         model.spec_stmt.horizon.ToString() + ", not " +
		                         stmt.horizon.ToString());
	}
	eff.target = model.spec_stmt.target;
	const int anchor_col = eff.anchor_is_literal ? -1 : ResolveAnchorColumn(entity, eff);

	ValidateFilterScope(eff.filter.get(), entity, eff.entity_alias, "WHERE");
	FeatureSpec spec = model.features;
	spec.Rebind(db, entity);
	const AggCache cache = BuildAggCache(db, spec);
	Dataset ds = CollectExamples(db, eff, spec, cache, anchor_col, false);
	std::vector<Prediction> out;
	out.reserve(ds.examples.size());
	if (model.sage) {
		SageSpec spec_s = model.sage_spec;
		// Re-resolve the links: indices are per-statement, as with the dense path.
		for (auto &et : spec_s.edges) {
			const std::string ct = db.tables[(size_t)spec_s.nodes[(size_t)et.src_node].table].name;
			int found = -1;
			for (size_t l = 0; l < db.links.size(); l++) {
				if (ToUpper(db.links[l].child_table) == ToUpper(ct) &&
				    ToUpper(db.links[l].parent_table) == ToUpper(entity.name)) {
					found = (int)l;
					break;
				}
			}
			if (found < 0) {
				throw std::runtime_error("pql: the link from '" + ct + "' used by this model is "
				                         "not available");
			}
			et.link = found;
		}
		const SageFeatures feats = BuildSageFeatures(db, spec_s);
		const SagePrefix prefix = BuildSagePrefix(db, spec_s, feats);
		SageScratch sc;
		const size_t B0 = 256;
		sc.Ensure(B0, model.sage_params);
		std::vector<size_t> all(ds.examples.size());
		for (size_t i = 0; i < all.size(); i++) {
			all[i] = i;
		}
		std::vector<double> outs;
		for (size_t b = 0; b < all.size(); b += B0) {
			const int B = (int)std::min(B0, all.size() - b);
			SageGather(db, spec_s, feats, prefix, ds, all, b, B, model.sage_params, sc);
			SageForward(model.sage_params, B, sc, outs);
			for (int i = 0; i < B; i++) {
				Prediction p;
				p.entity_row = ds.examples[b + (size_t)i].entity_row;
				p.anchor = ds.examples[b + (size_t)i].anchor;
				double v = model.classification
				               ? 1.0 / (1.0 + std::exp(-outs[(size_t)i]))
				               : outs[(size_t)i] * model.label_sd + model.label_mean;
				if (!model.classification) {
					if (model.residual) {
						v += ds.examples[b + (size_t)i].base;
					}
					if (model.nonnegative && v < 0.0) {
						v = 0.0;
					}
				}
				p.value = v;
				out.push_back(p);
			}
		}
		return out;
	}
	for (size_t i = 0; i < ds.examples.size(); i++) {
		Prediction p;
		p.entity_row = ds.examples[i].entity_row;
		p.anchor = ds.examples[i].anchor;
		double v = model.Predict(ds.Row(i), spec.width);
		// Same post-processing as the scoring path, or PREDICT would disagree with
		// the metric the model was judged by.
		if (!model.classification) {
			if (model.residual) {
				v += ds.examples[i].base;
			}
			if (model.nonnegative && v < 0.0) {
				v = 0.0;
			}
		}
		p.value = v;
		out.push_back(p);
	}
	return out;
}

// One replayed prediction with the outcome that actually followed.
struct BacktestRow {
	uint32_t entity_row = 0;
	double anchor = 0;
	double predicted = 0;
	double actual = 0;
	double baseline = 0; // persistence, so every row can be judged against it
};

// Replay a trained model across its anchors and pair each prediction with the
// label that actually materialised. This is the same machinery training uses,
// with labels kept rather than discarded, so a backtest cannot silently diverge
// from how the model was scored.
inline std::vector<BacktestRow> RunBacktest(const Database &db, const Model &model,
                                            const Statement &stmt) {
	Statement eff = model.spec_stmt;
	eff.kind = StmtKind::BACKTEST;
	if (!stmt.entity_table.empty()) {
		eff.entity_table = stmt.entity_table;
	}
	if (stmt.filter) {
		eff.filter = stmt.filter->Copy();
	}
	const Frame &entity = db.At(eff.entity_table);
	ValidateFilterScope(eff.filter.get(), entity, eff.entity_alias, "WHERE");
	const int anchor_col = ResolveAnchorColumn(entity, eff);

	FeatureSpec spec = model.features;
	spec.Rebind(db, entity);
	const AggCache cache = BuildAggCache(db, spec);
	Dataset ds = CollectExamples(db, eff, spec, cache, anchor_col, /*need_labels=*/true);

	std::vector<double> preds;
	if (model.sage) {
		SageSpec ss = model.sage_spec;
		for (auto &et : ss.edges) {
			const std::string ct = db.tables[(size_t)ss.nodes[(size_t)et.src_node].table].name;
			for (size_t l = 0; l < db.links.size(); l++) {
				if (ToUpper(db.links[l].child_table) == ToUpper(ct) &&
				    ToUpper(db.links[l].parent_table) == ToUpper(entity.name)) {
					et.link = (int)l;
					break;
				}
			}
		}
		const SageFeatures feats = BuildSageFeatures(db, ss);
		const SagePrefix pre = BuildSagePrefix(db, ss, feats);
		SageScratch sc;
		const size_t B0 = 256;
		sc.Ensure(B0, model.sage_params);
		std::vector<size_t> all(ds.examples.size());
		for (size_t i = 0; i < all.size(); i++) {
			all[i] = i;
		}
		preds.assign(all.size(), 0.0);
		std::vector<double> outs;
		for (size_t b = 0; b < all.size(); b += B0) {
			const int B = (int)std::min(B0, all.size() - b);
			SageGather(db, ss, feats, pre, ds, all, b, B, model.sage_params, sc);
			SageForward(model.sage_params, B, sc, outs);
			for (int i = 0; i < B; i++) {
				double v = model.classification
				               ? 1.0 / (1.0 + std::exp(-outs[(size_t)i]))
				               : outs[(size_t)i] * model.label_sd + model.label_mean;
				if (!model.classification) {
					if (model.residual) {
						v += ds.examples[b + (size_t)i].base;
					}
					if (model.nonnegative && v < 0.0) {
						v = 0.0;
					}
				}
				preds[b + (size_t)i] = v;
			}
		}
	} else {
		preds.assign(ds.examples.size(), 0.0);
		for (size_t i = 0; i < ds.examples.size(); i++) {
			double v = model.Predict(ds.Row(i), spec.width);
			if (!model.classification) {
				if (model.residual) {
					v += ds.examples[i].base;
				}
				if (model.nonnegative && v < 0.0) {
					v = 0.0;
				}
			}
			preds[i] = v;
		}
	}

	const double lo = stmt.has_from ? stmt.backtest_from.number
	                                : -std::numeric_limits<double>::infinity();
	const double hi =
	    stmt.has_to ? stmt.backtest_to.number : std::numeric_limits<double>::infinity();
	std::vector<BacktestRow> out;
	out.reserve(ds.examples.size());
	for (size_t i = 0; i < ds.examples.size(); i++) {
		const Example &ex = ds.examples[i];
		if (ex.anchor < lo || ex.anchor > hi) {
			continue;
		}
		BacktestRow r;
		r.entity_row = ex.entity_row;
		r.anchor = ex.anchor;
		r.predicted = preds[i];
		// CollectExamples returns the raw label here. Training mutates its own copy
		// (label -= base) for the residual head, so truth there is label + base;
		// adding base again on a fresh collection would double-count it.
		r.actual = ex.label;
		r.baseline = ex.base;
		out.push_back(r);
	}
	return out;
}

class Registry {
public:
	// shared_ptr, not a bare pointer: PREDICT reads the model for the whole of a
	// load (hundreds of ms), and a concurrent DROP or re-TRAIN would otherwise
	// destroy it underneath the reader.
	void Put(const std::string &name, Model m) {
		models_[ToUpper(name)] = std::make_shared<Model>(std::move(m));
	}
	std::shared_ptr<const Model> Get(const std::string &name) const {
		auto it = models_.find(ToUpper(name));
		return it == models_.end() ? nullptr : it->second;
	}
	bool Drop(const std::string &name) {
		return models_.erase(ToUpper(name)) > 0;
	}
	std::vector<std::shared_ptr<const Model>> All() const {
		std::vector<std::shared_ptr<const Model>> v;
		v.reserve(models_.size());
		for (auto &kv : models_) {
			v.push_back(kv.second);
		}
		return v;
	}

private:
	std::unordered_map<std::string, std::shared_ptr<Model>> models_;
};

} // namespace pql
