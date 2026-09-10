#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/storage/object_cache.hpp"

// pql.hpp is DuckDB-agnostic and declares its own namespace; keep this TU isolated.
#include "../../../extras/pql/src/pql.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>

namespace duckdb {

TableFunction PqlExecFunction();

//===--------------------------------------------------------------------===//
// Model registry
//===--------------------------------------------------------------------===//

// Models live in the database's own ObjectCache, so they are scoped to the
// instance and die with it. A previous version kept a process-global map keyed
// by the DatabaseInstance address, which both leaked (nothing ever erased an
// entry) and, once an address was reused, handed one database's models to
// another.
struct PqlRegistryEntry : public ObjectCacheEntry {
	static string ObjectType() {
		return "pql_model_registry";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		// Under the lock: DuckDB asks for this from its own thread, and walking the
		// registry while a TRAIN inserts into it can rehash the map underneath the
		// iterator. The count also used to miss every graph parameter and the
		// category labels, so a sage model reported as very nearly nothing.
		std::lock_guard<std::mutex> guard(lock);
		idx_t bytes = 0;
		for (const auto &m : registry.All()) {
			bytes += (idx_t)m->ApproxBytes();
		}
		return bytes;
	}
	mutable std::mutex lock;
	pql::Registry registry;
};

static shared_ptr<PqlRegistryEntry> GetRegistryEntry(ClientContext &context) {
	auto &cache = ObjectCache::GetObjectCache(context);
	return cache.GetOrCreate<PqlRegistryEntry>("pql_models");
}

//===--------------------------------------------------------------------===//
// Loading DuckDB tables into pql frames
//===--------------------------------------------------------------------===//

static pql::ColType MapType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return pql::ColType::BOOL;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return pql::ColType::INT64;
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return pql::ColType::DOUBLE;
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
		return pql::ColType::TIMESTAMP;
	case LogicalTypeId::VARCHAR:
		return pql::ColType::CATEGORY;
	default:
		return pql::ColType::INVALID;
	}
}

// Pull one table into a frame by reading the DataChunk buffers directly.
// Columns are cast to canonical types inside DuckDB's vectorized engine, so the
// reader below handles exactly four physical layouts and never materializes a
// per-cell Value. Training needs the data resident (many epochs, random access
// by entity, time-sorted child indexes), so the bulk transfer is a typed
// memcpy out of the chunk rather than a per-value construction.
static pql::Frame LoadFrame(Connection &con, const std::string &table) {
	auto probe = con.Query("SELECT * FROM \"" + table + "\" LIMIT 0");
	if (probe->HasError()) {
		throw InvalidInputException("pql: cannot read table '%s': %s", table, probe->GetError());
	}
	pql::Frame frame;
	frame.name = table;
	const idx_t ncol = probe->ColumnCount();
	frame.columns.resize(ncol);
	std::string select;
	for (idx_t c = 0; c < ncol; c++) {
		const std::string cname = probe->ColumnName(c).GetIdentifierName();
		frame.columns[c].name = cname;
		frame.columns[c].type = MapType(probe->GetTypes()[c]);
		if (!select.empty()) {
			select += ", ";
		}
		const std::string q = "\"" + cname + "\"";
		switch (frame.columns[c].type) {
		case pql::ColType::CATEGORY:
			select += "CAST(" + q + " AS VARCHAR)";
			break;
		case pql::ColType::TIMESTAMP:
			select += "CAST(" + q + " AS TIMESTAMP)";
			break;
		case pql::ColType::DOUBLE:
			select += "CAST(" + q + " AS DOUBLE)";
			break;
		case pql::ColType::INT64:
		case pql::ColType::BOOL:
			select += "CAST(" + q + " AS BIGINT)";
			break;
		default:
			select += "NULL";
			break;
		}
	}
	auto result = con.Query("SELECT " + select + " FROM \"" + table + "\"");
	if (result->HasError()) {
		throw InvalidInputException("pql: cannot read table '%s': %s", table, result->GetError());
	}
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		const idx_t n = chunk->size();
		for (idx_t c = 0; c < ncol; c++) {
			pql::Column &dst = frame.columns[c];
			Vector &vec = chunk->data[c];
			vec.Flatten(n);
			auto &validity = FlatVector::Validity(vec);
			const size_t base = dst.valid.size();
			dst.valid.resize(base + n);
			dst.num.resize(base + n, 0.0);
			dst.code.resize(base + n, 0u);
			switch (dst.type) {
			case pql::ColType::CATEGORY: {
				const string_t *data = FlatVector::GetData<string_t>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					if (ok) {
						dst.code[base + r] = dst.dict.Intern(data[r].GetString());
					}
				}
				break;
			}
			case pql::ColType::TIMESTAMP:
			case pql::ColType::INT64:
			case pql::ColType::BOOL: {
				const int64_t *data = FlatVector::GetData<int64_t>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					dst.num[base + r] = ok ? double(data[r]) : 0.0;
				}
				break;
			}
			case pql::ColType::DOUBLE: {
				const double *data = FlatVector::GetData<double>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					dst.num[base + r] = ok ? data[r] : 0.0;
				}
				break;
			}
			default:
				for (idx_t r = 0; r < n; r++) {
					dst.valid[base + r] = 0;
				}
				break;
			}
		}
		frame.nrows += n;
	}
	return frame;
}

static std::vector<std::string> ListTables(Connection &con) {
	std::vector<std::string> out;
	auto r = con.Query("SELECT table_name FROM duckdb_tables() WHERE NOT internal");
	if (r->HasError()) {
		return out;
	}
	while (true) {
		auto chunk = r->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		for (idx_t i = 0; i < chunk->size(); i++) {
			out.push_back(chunk->GetValue(0, i).ToString());
		}
	}
	return out;
}

// Foreign keys come from declared constraints when present, and otherwise from
// the <table>_id / <table>id naming convention. Declared keys win: an explicit
// schema should never be second-guessed.
static std::vector<pql::ForeignKey> DiscoverForeignKeys(Connection &con, const std::vector<std::string> &tables,
                                                        const pql::Database &db) {
	std::vector<pql::ForeignKey> fks;
	auto r = con.Query("SELECT table_name, constraint_column_names, referenced_table, "
	                   "referenced_column_names FROM duckdb_constraints() "
	                   "WHERE constraint_type = 'FOREIGN KEY'");
	if (!r->HasError()) {
		while (true) {
			auto chunk = r->Fetch();
			if (!chunk || chunk->size() == 0) {
				break;
			}
			for (idx_t i = 0; i < chunk->size(); i++) {
				const Value cols = chunk->GetValue(1, i);
				const Value refs = chunk->GetValue(3, i);
				auto cl = ListValue::GetChildren(cols);
				auto rl = ListValue::GetChildren(refs);
				if (cl.size() != 1 || rl.size() != 1) {
					continue; // composite keys are not modelled yet
				}
				pql::ForeignKey fk;
				fk.child_table = chunk->GetValue(0, i).ToString();
				fk.child_column = cl[0].ToString();
				fk.parent_table = chunk->GetValue(2, i).ToString();
				fk.parent_column = rl[0].ToString();
				fks.push_back(fk);
			}
		}
	}
	auto declared = [&](const std::string &t, const std::string &c) {
		for (auto &fk : fks) {
			if (pql::ToUpper(fk.child_table) == pql::ToUpper(t) && pql::ToUpper(fk.child_column) == pql::ToUpper(c)) {
				return true;
			}
		}
		return false;
	};
	for (const auto &t : tables) {
		const int ti = db.Find(t);
		if (ti < 0) {
			continue;
		}
		for (const auto &col : db.tables[(size_t)ti].columns) {
			const std::string up = pql::ToUpper(col.name);
			if (up.size() < 3 || up == "ID" || declared(t, col.name)) {
				continue;
			}
			std::string stem;
			if (up.size() > 3 && up.compare(up.size() - 3, 3, "_ID") == 0) {
				stem = up.substr(0, up.size() - 3);
			} else if (up.compare(up.size() - 2, 2, "ID") == 0) {
				stem = up.substr(0, up.size() - 2);
			} else {
				continue;
			}
			for (const auto &other : tables) {
				if (pql::ToUpper(other) == pql::ToUpper(t)) {
					continue;
				}
				std::string ot = pql::ToUpper(other);
				std::string singular = ot;
				if (singular.size() > 1 && singular.back() == 'S') {
					singular.pop_back();
				}
				if (ot != stem && singular != stem) {
					continue;
				}
				const int oi = db.Find(other);
				if (oi < 0) {
					continue;
				}
				// Point at the other table's own "id" column when it has one.
				const int idc = db.tables[(size_t)oi].Find("id");
				if (idc < 0) {
					continue;
				}
				pql::ForeignKey fk;
				fk.child_table = t;
				fk.child_column = col.name;
				fk.parent_table = other;
				fk.parent_column = db.tables[(size_t)oi].columns[(size_t)idc].name;
				fks.push_back(fk);
				break;
			}
		}
	}
	return fks;
}

// Column names a filter tree mentions, so pruning never drops one.
static void CollectFilterColumns(const pql::FilterNode *f, std::set<std::string> &out) {
	if (!f) {
		return;
	}
	if (f->kind == pql::FilterNode::Kind::CMP) {
		out.insert(pql::ToUpper(f->ref.name));
		return;
	}
	for (const auto &c : f->children) {
		CollectFilterColumns(c.get(), out);
	}
}

// A date written as a string parsed to 0 (1970-01-01), which silently made
// SPLIT boundaries meaningless, anchors epoch-zero, and timestamp filters match
// nothing. Resolve them through DuckDB's own parser before anything reads them.
static bool ParseTimestampLiteral(const std::string &text, double &micros_out) {
	Value v(text);
	string err;
	auto converted = v.DefaultTryCastAs(LogicalType::TIMESTAMP, &err);
	if (!converted.has_value() || converted->IsNull()) {
		return false;
	}
	micros_out = double(Timestamp::GetEpochMicroSeconds(converted->GetValue<timestamp_t>()));
	return true;
}

static void ResolveTemporalLiteral(pql::Literal &lit, const char *what) {
	if (lit.kind != pql::Literal::Kind::STRING) {
		return;
	}
	double micros = 0;
	if (!ParseTimestampLiteral(lit.text, micros)) {
		throw InvalidInputException("pql: %s value '%s' is not a timestamp", what, lit.text);
	}
	lit.kind = pql::Literal::Kind::NUMBER;
	lit.number = micros;
}

struct TableSchema {
	std::string schema; // the SQL schema it lives in, "main" unless said otherwise
	std::string name;
	std::vector<std::string> columns;
	std::vector<pql::ColType> types;

	// How to name it in a query. PQL itself refers to tables by their bare name,
	// which is why two schemas holding the same one has to be resolved before
	// anything is read.
	std::string Qualified() const {
		return "\"" + schema + "\".\"" + name + "\"";
	}
};

// One catalog query: names and types for everything, no data touched.
static std::vector<TableSchema> LoadSchema(Connection &con) {
	std::vector<TableSchema> out;
	// Grouped by schema AND name. Grouping by name alone merged two same-named
	// tables from different schemas into one entry carrying both column lists,
	// and the SELECT built from it named columns that were not there.
	auto r = con.Query("SELECT t.schema_name, t.table_name, c.column_name, c.data_type "
	                   "FROM duckdb_tables() t JOIN duckdb_columns() c "
	                   "  ON c.table_oid = t.table_oid "
	                   "WHERE NOT t.internal "
	                   "ORDER BY t.schema_name, t.table_name, c.column_index");
	if (r->HasError()) {
		return out;
	}
	while (true) {
		auto chunk = r->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		for (idx_t i = 0; i < chunk->size(); i++) {
			const std::string sn = chunk->GetValue(0, i).ToString();
			const std::string tn = chunk->GetValue(1, i).ToString();
			const std::string cn = chunk->GetValue(2, i).ToString();
			const std::string ty = pql::ToUpper(chunk->GetValue(3, i).ToString());
			if (out.empty() || out.back().name != tn || out.back().schema != sn) {
				out.push_back(TableSchema {sn, tn, {}, {}});
			}
			pql::ColType t = pql::ColType::INVALID;
			if (ty.find("TIMESTAMP") != std::string::npos || ty.find("DATE") != std::string::npos) {
				t = pql::ColType::TIMESTAMP;
			} else if (ty.find("BOOLEAN") != std::string::npos) {
				t = pql::ColType::BOOL;
			} else if (ty.find("INT") != std::string::npos) {
				t = pql::ColType::INT64;
			} else if (ty.find("DOUBLE") != std::string::npos || ty.find("FLOAT") != std::string::npos ||
			           ty.find("DECIMAL") != std::string::npos || ty.find("REAL") != std::string::npos) {
				t = pql::ColType::DOUBLE;
			} else if (ty.find("VARCHAR") != std::string::npos || ty.find("CHAR") != std::string::npos ||
			           ty.find("TEXT") != std::string::npos) {
				t = pql::ColType::CATEGORY;
			}
			out.back().columns.push_back(cn);
			out.back().types.push_back(t);
		}
	}
	return out;
}

// Load one table, restricted to the columns that can matter. Values are read
// straight out of the DataChunk buffers; the SELECT list is the pruning.
static pql::Frame LoadFrameColumns(Connection &con, const TableSchema &schema, const std::vector<size_t> &keep) {
	pql::Frame frame;
	frame.name = schema.name;
	frame.columns.resize(keep.size());
	std::string select;
	for (size_t k = 0; k < keep.size(); k++) {
		const size_t c = keep[k];
		frame.columns[k].name = schema.columns[c];
		frame.columns[k].type = schema.types[c];
		if (!select.empty()) {
			select += ", ";
		}
		const std::string q = "\"" + schema.columns[c] + "\"";
		switch (schema.types[c]) {
		case pql::ColType::CATEGORY:
			select += "CAST(" + q + " AS VARCHAR)";
			break;
		case pql::ColType::TIMESTAMP:
			select += "CAST(" + q + " AS TIMESTAMP)";
			break;
		case pql::ColType::DOUBLE:
			select += "CAST(" + q + " AS DOUBLE)";
			break;
		default:
			select += "CAST(" + q + " AS BIGINT)";
			break;
		}
	}
	if (select.empty()) {
		return frame;
	}
	auto result = con.Query("SELECT " + select + " FROM " + schema.Qualified());
	if (result->HasError()) {
		throw InvalidInputException("pql: cannot read table '%s': %s", schema.name, result->GetError());
	}
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		const idx_t n = chunk->size();
		for (idx_t c = 0; c < frame.columns.size(); c++) {
			pql::Column &dst = frame.columns[c];
			Vector &vec = chunk->data[c];
			vec.Flatten(n);
			auto &validity = FlatVector::Validity(vec);
			const size_t base = dst.valid.size();
			dst.valid.resize(base + n);
			dst.num.resize(base + n, 0.0);
			dst.code.resize(base + n, 0u);
			if (dst.type == pql::ColType::CATEGORY) {
				const string_t *data = FlatVector::GetData<string_t>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					if (ok) {
						dst.code[base + r] = dst.dict.Intern(data[r].GetString());
					}
				}
			} else if (dst.type == pql::ColType::DOUBLE) {
				const double *data = FlatVector::GetData<double>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					dst.num[base + r] = ok ? data[r] : 0.0;
				}
			} else {
				const int64_t *data = FlatVector::GetData<int64_t>(vec);
				for (idx_t r = 0; r < n; r++) {
					const bool ok = validity.RowIsValid(r);
					dst.valid[base + r] = ok ? 1 : 0;
					dst.num[base + r] = ok ? double(data[r]) : 0.0;
				}
			}
		}
		frame.nrows += n;
	}
	return frame;
}

// Foreign keys straight from the catalog plus the <table>_id convention,
// computed on names alone so nothing has to be read to decide what to read.
static std::vector<pql::ForeignKey> ForeignKeysFromSchema(Connection &con, const std::vector<TableSchema> &schema) {
	std::vector<pql::ForeignKey> fks;
	auto r = con.Query("SELECT table_name, constraint_column_names, referenced_table, "
	                   "referenced_column_names FROM duckdb_constraints() "
	                   "WHERE constraint_type = 'FOREIGN KEY'");
	if (!r->HasError()) {
		while (true) {
			auto chunk = r->Fetch();
			if (!chunk || chunk->size() == 0) {
				break;
			}
			for (idx_t i = 0; i < chunk->size(); i++) {
				auto cl = ListValue::GetChildren(chunk->GetValue(1, i));
				auto rl = ListValue::GetChildren(chunk->GetValue(3, i));
				if (cl.size() != 1 || rl.size() != 1) {
					continue;
				}
				pql::ForeignKey fk;
				fk.child_table = chunk->GetValue(0, i).ToString();
				fk.child_column = cl[0].ToString();
				fk.parent_table = chunk->GetValue(2, i).ToString();
				fk.parent_column = rl[0].ToString();
				fks.push_back(fk);
			}
		}
	}
	auto find_col = [&](const TableSchema &t, const std::string &up) -> int {
		for (size_t i = 0; i < t.columns.size(); i++) {
			if (pql::ToUpper(t.columns[i]) == up) {
				return (int)i;
			}
		}
		return -1;
	};
	for (const auto &t : schema) {
		for (const auto &col : t.columns) {
			const std::string up = pql::ToUpper(col);
			if (up.size() < 3 || up == "ID") {
				continue;
			}
			bool already = false;
			for (const auto &fk : fks) {
				if (pql::ToUpper(fk.child_table) == pql::ToUpper(t.name) && pql::ToUpper(fk.child_column) == up) {
					already = true;
					break;
				}
			}
			if (already) {
				continue;
			}
			std::string stem;
			if (up.size() > 3 && up.compare(up.size() - 3, 3, "_ID") == 0) {
				stem = up.substr(0, up.size() - 3);
			} else if (up.compare(up.size() - 2, 2, "ID") == 0) {
				stem = up.substr(0, up.size() - 2);
			} else {
				continue;
			}
			for (const auto &other : schema) {
				if (pql::ToUpper(other.name) == pql::ToUpper(t.name)) {
					continue;
				}
				const std::string ot = pql::ToUpper(other.name);
				// Every spelling the table's name might have been singularised to,
				// tried together rather than picking one rule. categories -> category
				// needs -ies -> -y; boxes -> box and addresses -> address need -es
				// dropped whole, which the single trailing -s rule turned into "boxe"
				// and "addresse" and so linked nothing.
				std::vector<std::string> cands;
				cands.push_back(ot);
				if (ot.size() > 3 && ot.compare(ot.size() - 3, 3, "IES") == 0) {
					cands.push_back(ot.substr(0, ot.size() - 3) + "Y");
				}
				if (ot.size() > 2 && ot.compare(ot.size() - 2, 2, "ES") == 0) {
					cands.push_back(ot.substr(0, ot.size() - 2));
				}
				if (ot.size() > 1 && ot.back() == 'S') {
					cands.push_back(ot.substr(0, ot.size() - 1));
				}
				bool matches = false;
				for (const auto &c : cands) {
					if (c == stem) {
						matches = true;
						break;
					}
				}
				if (!matches) {
					continue;
				}
				// The parent key is usually spelled the same as the referencing
				// column (products.product_id), and only sometimes plain "id".
				// Checking only for "id" missed every key in a <table>_id schema.
				int idc = find_col(other, up);
				if (idc < 0) {
					idc = find_col(other, "ID");
				}
				if (idc < 0) {
					continue;
				}
				pql::ForeignKey fk;
				fk.child_table = t.name;
				fk.child_column = col;
				fk.parent_table = other.name;
				fk.parent_column = other.columns[(size_t)idc];
				fks.push_back(fk);
				break;
			}
		}
	}
	return fks;
}

// Walk a filter tree and convert string literals compared against a TIMESTAMP
// column into microseconds, so `WHERE as_of = '2026-04-01'` means what it reads.
static void ResolveFilterTimestamps(pql::FilterNode *f, const TableSchema *schema) {
	if (!f || !schema) {
		return;
	}
	if (f->kind != pql::FilterNode::Kind::CMP) {
		for (auto &c : f->children) {
			ResolveFilterTimestamps(c.get(), schema);
		}
		return;
	}
	const std::string up = pql::ToUpper(f->ref.name);
	for (size_t i = 0; i < schema->columns.size(); i++) {
		if (pql::ToUpper(schema->columns[i]) != up) {
			continue;
		}
		if (schema->types[i] == pql::ColType::TIMESTAMP) {
			for (auto &lit : f->values) {
				ResolveTemporalLiteral(lit, "timestamp comparison");
			}
		}
		return;
	}
}

static const TableSchema *FindSchema(const std::vector<TableSchema> &schema, const std::string &name) {
	for (const auto &t : schema) {
		if (pql::ToUpper(t.name) == pql::ToUpper(name)) {
			return &t;
		}
	}
	return nullptr;
}

static pql::Database LoadDatabase(ClientContext &context, const pql::Statement &stmt) {
	Connection con(DatabaseInstance::GetDatabase(context));
	std::vector<TableSchema> schema = LoadSchema(con);
	{
		// PQL names tables without a schema, so a bare name has to mean one table.
		// Prefer the session's own schema, the way an unqualified name resolves in
		// SQL; take a unique match elsewhere; refuse when it is genuinely ambiguous
		// rather than picking whichever the catalog listed first.
		std::string current = "main";
		auto cs = con.Query("SELECT current_schema()");
		if (cs && !cs->HasError()) {
			auto ch = cs->Fetch();
			if (ch && ch->size() > 0) {
				current = ch->GetValue(0, 0).ToString();
			}
		}
		std::map<std::string, std::vector<size_t>> by_name;
		for (size_t i = 0; i < schema.size(); i++) {
			by_name[pql::ToUpper(schema[i].name)].push_back(i);
		}
		std::vector<TableSchema> resolved;
		for (auto &kv : by_name) {
			if (kv.second.size() == 1) {
				resolved.push_back(schema[kv.second[0]]);
				continue;
			}
			int pick = -1;
			for (size_t idx : kv.second) {
				if (pql::ToUpper(schema[idx].schema) == pql::ToUpper(current)) {
					pick = (int)idx;
					break;
				}
			}
			if (pick >= 0) {
				resolved.push_back(schema[(size_t)pick]);
				continue;
			}
			// Only a problem if the statement actually wants this table; recorded
			// here and reported below, once we know what it asked for.
			TableSchema amb = schema[kv.second[0]];
			amb.columns.clear();
			amb.types.clear();
			amb.schema.clear(); // empty schema marks it unresolvable
			resolved.push_back(amb);
		}
		schema.swap(resolved);
	}
	const std::vector<pql::ForeignKey> all_fks = ForeignKeysFromSchema(con, schema);

	auto same = [](const std::string &a, const std::string &b) {
		return pql::ToUpper(a) == pql::ToUpper(b);
	};
	// Only the entity, its children, and the target can influence the model, so
	// nothing else is read at all.
	std::set<std::string> want;
	want.insert(pql::ToUpper(stmt.entity_table));
	if (!stmt.target.target_table.empty() && stmt.target.target_table != "*") {
		want.insert(pql::ToUpper(stmt.target.target_table));
	}
	for (const auto &fk : all_fks) {
		if (same(fk.parent_table, stmt.entity_table)) {
			want.insert(pql::ToUpper(fk.child_table));
		}
	}
	// A child with no timestamp of its own is dated through a foreign key
	// (line_items are dated by their transaction), so the table holding that
	// clock has to be loaded even though it is not itself a child of the entity.
	auto has_time = [&](const TableSchema &t) {
		for (size_t i = 0; i < t.types.size(); i++) {
			if (t.types[i] == pql::ColType::TIMESTAMP) {
				return true;
			}
		}
		for (const auto &c : t.columns) {
			const std::string up = pql::ToUpper(c);
			if (up.find("DATE") != std::string::npos || up.find("TIME") != std::string::npos) {
				return true;
			}
		}
		return false;
	};
	{
		std::set<std::string> clock_tables;
		for (const auto &w : want) {
			const TableSchema *ct = FindSchema(schema, w);
			if (!ct || has_time(*ct)) {
				continue;
			}
			for (const auto &fk : all_fks) {
				if (pql::ToUpper(fk.child_table) != w) {
					continue;
				}
				const TableSchema *pt = FindSchema(schema, fk.parent_table);
				if (pt && has_time(*pt)) {
					clock_tables.insert(pql::ToUpper(fk.parent_table));
				}
			}
		}
		want.insert(clock_tables.begin(), clock_tables.end());
	}
	if (!stmt.graph_tables.empty()) {
		std::set<std::string> allow;
		allow.insert(pql::ToUpper(stmt.entity_table));
		if (!stmt.target.target_table.empty() && stmt.target.target_table != "*") {
			allow.insert(pql::ToUpper(stmt.target.target_table));
		}
		for (const auto &g : stmt.graph_tables) {
			allow.insert(pql::ToUpper(g));
		}
		std::set<std::string> pruned;
		for (const auto &w : want) {
			if (allow.count(w)) {
				pruned.insert(w);
			}
		}
		want.swap(pruned);
	}

	std::set<std::string> filter_cols;
	CollectFilterColumns(stmt.filter.get(), filter_cols);
	CollectFilterColumns(stmt.target.filter.get(), filter_cols);
	if (!stmt.anchor.name.empty()) {
		filter_cols.insert(pql::ToUpper(stmt.anchor.name));
	}
	if (!stmt.target.ref.name.empty()) {
		filter_cols.insert(pql::ToUpper(stmt.target.ref.name));
	}

	const std::string entity_up = pql::ToUpper(stmt.entity_table);

	pql::Database db;
	for (const auto &t : schema) {
		if (t.schema.empty() || !want.count(pql::ToUpper(t.name))) {
			continue;
		}
		// The entity's own text columns are features, so they are read. Elsewhere a
		// text column is still only read when a filter or a key names it.
		const bool is_entity = pql::ToUpper(t.name) == entity_up;
		std::vector<size_t> keep;
		for (size_t c = 0; c < t.columns.size(); c++) {
			const pql::ColType ty = t.types[c];
			if (ty == pql::ColType::INVALID) {
				continue;
			}
			const std::string up = pql::ToUpper(t.columns[c]);
			bool needed = ty != pql::ColType::CATEGORY || is_entity;
			if (!needed && filter_cols.count(up)) {
				needed = true;
			}
			if (!needed) {
				for (const auto &fk : all_fks) {
					if ((same(fk.child_table, t.name) && pql::ToUpper(fk.child_column) == up) ||
					    (same(fk.parent_table, t.name) && pql::ToUpper(fk.parent_column) == up)) {
						needed = true;
						break;
					}
				}
			}
			if (needed) {
				keep.push_back(c);
			}
		}
		db.tables.push_back(LoadFrameColumns(con, t, keep));
	}
	for (const auto &t : schema) {
		if (!t.schema.empty() || !want.count(pql::ToUpper(t.name))) {
			continue;
		}
		throw InvalidInputException(
		    "pql: '%s' exists in more than one schema and none of them is the current one. "
		    "SET schema to the one you mean, or rename so the name is unique",
		    t.name);
	}
	if (db.Find(stmt.entity_table) < 0) {
		throw InvalidInputException("pql: entity table '%s' not found", stmt.entity_table);
	}
	for (const auto &fk : all_fks) {
		if (db.Find(fk.child_table) >= 0 && db.Find(fk.parent_table) >= 0) {
			db.fks.push_back(fk);
		}
	}
	db.BuildLinks(stmt.graph_tables);
	return db;
}

//===--------------------------------------------------------------------===//
// pql_exec table function
//===--------------------------------------------------------------------===//

struct PqlBindData : public TableFunctionData {
	vector<Identifier> names;
	vector<LogicalType> types;
	// Result rows are produced during bind and replayed here.
	std::vector<std::vector<Value>> rows;
};

struct PqlGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static void RunTrain(ClientContext &context, pql::Statement &stmt, PqlBindData &bind) {
	// Checked before any work, not after: refusing at the end would mean the user
	// waited for a model that is then thrown away.
	if (!stmt.or_replace) {
		auto entry = GetRegistryEntry(context);
		std::lock_guard<std::mutex> guard(entry->lock);
		if (entry->registry.Get(stmt.model)) {
			throw InvalidInputException(
			    "pql: a model named '%s' already exists. Use TRAIN OR REPLACE MODEL to replace it, "
			    "or DROP MODEL %s first",
			    stmt.model, stmt.model);
		}
	}
	pql::Database db = LoadDatabase(context, stmt);
	pql::Model model;
	model.name = stmt.model;
	auto rep = pql::TrainModel(db, stmt, model, [&context]() { context.InterruptCheck(); });
	{
		auto entry = GetRegistryEntry(context);
		std::lock_guard<std::mutex> guard(entry->lock);
		entry->registry.Put(stmt.model, std::move(model));
	}
	bind.names = {Identifier("model"),     Identifier("target"),    Identifier("train_rows"), Identifier("val_rows"),
	              Identifier("test_rows"), Identifier("positives"), Identifier("metric"),     Identifier("val"),
	              Identifier("test"),      Identifier("pr_auc"),    Identifier("baseline"),   Identifier("censored"),
	              Identifier("no_anchor"), Identifier("features"),  Identifier("epochs")};
	bind.types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT,  LogicalType::BIGINT,
	              LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::DOUBLE,
	              LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::BIGINT,
	              LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BIGINT};
	// An undefined metric (single-class fold) surfaces as NULL, not as a number.
	auto metric_value = [](double v) {
		return std::isnan(v) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(v);
	};
	bind.rows.push_back({Value(stmt.model), Value(stmt.target.ToString()), Value::BIGINT((int64_t)rep.n_train),
	                     Value::BIGINT((int64_t)rep.n_val), Value::BIGINT((int64_t)rep.n_test),
	                     Value::BIGINT((int64_t)rep.n_positives), Value(rep.metric_name), metric_value(rep.val_metric),
	                     metric_value(rep.test_metric), metric_value(rep.pr_auc),
	                     rep.baseline_metric > 0 ? Value::DOUBLE(rep.baseline_metric) : Value(LogicalType::DOUBLE),
	                     Value::BIGINT((int64_t)rep.n_censored), Value::BIGINT((int64_t)rep.n_no_anchor),
	                     Value::BIGINT(rep.width), Value::BIGINT(rep.epochs_run)});
}

static void RunPredictStmt(ClientContext &context, pql::Statement &stmt, PqlBindData &bind) {
	std::shared_ptr<const pql::Model> model;
	{
		auto entry = GetRegistryEntry(context);
		std::lock_guard<std::mutex> guard(entry->lock);
		model = entry->registry.Get(stmt.model);
	}
	if (!model) {
		throw InvalidInputException("pql: no model named '%s'; TRAIN it first", stmt.model);
	}
	pql::Statement effective = stmt;
	if (effective.entity_table.empty()) {
		effective.entity_table = model->spec_stmt.entity_table;
	}
	pql::Database db = LoadDatabase(context, effective);
	auto preds = pql::RunPredict(db, *model, effective);

	// Echo an identifying column alongside the prediction so a result row can be
	// joined back. The entity's first declared key is the natural choice.
	const pql::Frame &entity = db.At(effective.entity_table);
	// Prefer the entity's own key: "id", else "<singular>_id", else any declared
	// parent key on this table.
	int key_col = entity.Find("id");
	if (key_col < 0) {
		std::string sing = entity.name;
		if (sing.size() > 1 && (sing.back() == 's' || sing.back() == 'S')) {
			sing.pop_back();
		}
		key_col = entity.Find(sing + "_id");
	}
	if (key_col < 0) {
		for (const auto &fk : db.fks) {
			if (pql::ToUpper(fk.parent_table) == pql::ToUpper(entity.name)) {
				key_col = entity.Find(fk.parent_column);
				break;
			}
		}
	}
	// Emit the key in its own type. Forcing it through a DOUBLE turned every
	// string key into 0.0 and lost precision past 2^53, so the documented
	// join-back silently produced nothing.
	const pql::ColType key_type = key_col >= 0 ? entity.columns[(size_t)key_col].type : pql::ColType::INT64;
	LogicalType key_logical = LogicalType::BIGINT;
	if (key_col >= 0) {
		switch (key_type) {
		case pql::ColType::CATEGORY:
			key_logical = LogicalType::VARCHAR;
			break;
		case pql::ColType::DOUBLE:
			key_logical = LogicalType::DOUBLE;
			break;
		case pql::ColType::TIMESTAMP:
			key_logical = LogicalType::TIMESTAMP;
			break;
		default:
			key_logical = LogicalType::BIGINT;
			break;
		}
	}
	// Echo the anchor beside the key: a forecast is "for this entity, at this
	// time", and a panel returns one row per as-of date. Without it the result is
	// several numbers with nothing to tell them apart.
	// A model trained on a generated grid has no anchor column, and the entity's
	// own timestamps (first_seen_at and friends) are not the forecast moment.
	const bool generated_anchor = model->spec_stmt.every.present;
	const int anchor_out =
	    (effective.anchor_is_literal || generated_anchor) ? -1 : pql::ResolveAnchorColumn(entity, effective);
	// A generated or explicit anchor has no column to name it after, but it is
	// still the moment the forecast is made from, so it is still reported.
	const bool literal_anchor = (effective.anchor_is_literal || generated_anchor) && !preds.empty();
	const bool with_anchor = anchor_out >= 0 || literal_anchor;
	bind.names = {Identifier("entity"), Identifier("prediction")};
	bind.types = {key_logical, LogicalType::DOUBLE};
	if (with_anchor) {
		bind.names = {Identifier("entity"), Identifier("anchor"), Identifier("prediction")};
		bind.types = {key_logical, LogicalType::TIMESTAMP, LogicalType::DOUBLE};
		if (anchor_out >= 0) {
			bind.names[1] = Identifier(entity.columns[(size_t)anchor_out].name);
		}
	}
	if (key_col >= 0) {
		bind.names[0] = Identifier(entity.columns[(size_t)key_col].name);
	}
	for (const auto &p : preds) {
		Value key_value;
		if (key_col < 0) {
			key_value = Value::BIGINT((int64_t)p.entity_row);
		} else {
			const pql::Column &kc = entity.columns[(size_t)key_col];
			if (key_type == pql::ColType::CATEGORY) {
				const uint32_t code = kc.code[p.entity_row];
				key_value = Value(code < kc.dict.values.size() ? kc.dict.values[code] : std::string());
			} else if (key_type == pql::ColType::DOUBLE) {
				key_value = Value::DOUBLE(kc.num[p.entity_row]);
			} else if (key_type == pql::ColType::TIMESTAMP) {
				key_value = Value::TIMESTAMP(timestamp_t((int64_t)kc.num[p.entity_row]));
			} else {
				key_value = Value::BIGINT((int64_t)kc.num[p.entity_row]);
			}
		}
		if (with_anchor) {
			bind.rows.push_back({key_value, Value::TIMESTAMP(timestamp_t((int64_t)p.anchor)), Value::DOUBLE(p.value)});
		} else {
			bind.rows.push_back({key_value, Value::DOUBLE(p.value)});
		}
	}
}

// Replay a model and show, per row, what it predicted and what happened.
static void RunExplainStmt(ClientContext &context, pql::Statement &stmt, PqlBindData &bind) {
	std::shared_ptr<const pql::Model> model;
	{
		auto entry = GetRegistryEntry(context);
		std::lock_guard<std::mutex> guard(entry->lock);
		model = entry->registry.Get(stmt.model);
	}
	if (!model) {
		throw InvalidInputException("pql: no model named '%s'; TRAIN it first", stmt.model);
	}
	pql::Statement effective = stmt;
	if (effective.entity_table.empty()) {
		effective.entity_table = model->spec_stmt.entity_table;
	}
	pql::Database db = LoadDatabase(context, effective);
	auto imp = pql::RunExplain(db, *model, effective);
	bind.names = {Identifier("feature"), Identifier("slots"), Identifier("importance")};
	bind.types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::DOUBLE};
	for (const auto &f : imp) {
		bind.rows.push_back({Value(f.feature), Value::BIGINT((int64_t)f.slots),
		                     std::isnan(f.drop) ? Value(LogicalType::DOUBLE)
		                                        : Value::DOUBLE(f.drop)});
	}
}

static void RunBacktestStmt(ClientContext &context, pql::Statement &stmt, PqlBindData &bind) {
	std::shared_ptr<const pql::Model> model;
	{
		auto entry = GetRegistryEntry(context);
		std::lock_guard<std::mutex> guard(entry->lock);
		model = entry->registry.Get(stmt.model);
	}
	if (!model) {
		throw InvalidInputException("pql: no model named '%s'; TRAIN it first", stmt.model);
	}
	pql::Statement effective = stmt;
	if (effective.entity_table.empty()) {
		effective.entity_table = model->spec_stmt.entity_table;
	}
	pql::Database db = LoadDatabase(context, effective);
	auto rows = pql::RunBacktest(db, *model, effective);

	const pql::Frame &entity = db.At(effective.entity_table);
	int key_col = entity.Find("id");
	if (key_col < 0) {
		std::string sing = entity.name;
		if (sing.size() > 1 && (sing.back() == 's' || sing.back() == 'S')) {
			sing.pop_back();
		}
		key_col = entity.Find(sing + "_id");
	}
	if (key_col < 0) {
		for (const auto &fk : db.fks) {
			if (pql::ToUpper(fk.parent_table) == pql::ToUpper(entity.name)) {
				key_col = entity.Find(fk.parent_column);
				break;
			}
		}
	}
	LogicalType key_logical = LogicalType::BIGINT;
	if (key_col >= 0 && entity.columns[(size_t)key_col].type == pql::ColType::CATEGORY) {
		key_logical = LogicalType::VARCHAR;
	} else if (key_col >= 0 && entity.columns[(size_t)key_col].type == pql::ColType::DOUBLE) {
		key_logical = LogicalType::DOUBLE;
	}
	bind.names = {Identifier("entity"), Identifier("anchor"), Identifier("predicted"),
	              Identifier("actual"), Identifier("error"),  Identifier("baseline")};
	if (key_col >= 0) {
		bind.names[0] = Identifier(entity.columns[(size_t)key_col].name);
	}
	bind.types = {key_logical,         LogicalType::TIMESTAMP, LogicalType::DOUBLE,
	              LogicalType::DOUBLE, LogicalType::DOUBLE,    LogicalType::DOUBLE};
	for (const auto &r : rows) {
		Value key_value = Value::BIGINT((int64_t)r.entity_row);
		if (key_col >= 0) {
			const pql::Column &kc = entity.columns[(size_t)key_col];
			if (kc.type == pql::ColType::CATEGORY) {
				const uint32_t code = kc.code[r.entity_row];
				key_value = Value(code < kc.dict.values.size() ? kc.dict.values[code] : std::string());
			} else if (kc.type == pql::ColType::DOUBLE) {
				key_value = Value::DOUBLE(kc.num[r.entity_row]);
			} else {
				key_value = Value::BIGINT((int64_t)kc.num[r.entity_row]);
			}
		}
		bind.rows.push_back({key_value, Value::TIMESTAMP(timestamp_t((int64_t)r.anchor)), Value::DOUBLE(r.predicted),
		                     Value::DOUBLE(r.actual), Value::DOUBLE(r.predicted - r.actual),
		                     Value::DOUBLE(r.baseline)});
	}
}

static void RunShowModels(ClientContext &context, PqlBindData &bind) {
	bind.names = {Identifier("model"), Identifier("target"), Identifier("entity"), Identifier("kind"),
	              Identifier("features"), Identifier("statement")};
	bind.types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	              LogicalType::BIGINT,   LogicalType::VARCHAR};
	auto entry = GetRegistryEntry(context);
	std::lock_guard<std::mutex> guard(entry->lock);
	for (const auto &m : entry->registry.All()) {
		// The defining statement, echoed back in full. Copy it to retrain, or read
		// it to see exactly what a model was given.
		bind.rows.push_back({Value(m->name), Value(m->spec_stmt.target.ToString()), Value(m->spec_stmt.entity_table),
		                     Value(m->classification ? "classification" : "regression"),
		                     Value::BIGINT(m->features.width), Value(m->spec_stmt.ToString())});
	}
}

static unique_ptr<FunctionData> PqlBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto bind = make_uniq<PqlBindData>();
	const std::string query = input.inputs[0].ToString();
	pql::Statement stmt;
	try {
		stmt = pql::Parse(query);
	} catch (const pql::ParseError &e) {
		throw ParserException("PQL: %s", e.what());
	}
	// Temporal literals are resolved before anything reads them, so a quoted date
	// is a real timestamp rather than epoch zero.
	if (stmt.split.present) {
		ResolveTemporalLiteral(stmt.split.validate_from, "SPLIT VALIDATE FROM");
		ResolveTemporalLiteral(stmt.split.test_from, "SPLIT TEST FROM");
	}
	if (stmt.anchor_is_literal) {
		ResolveTemporalLiteral(stmt.anchor_literal, "AT");
	}
	if (stmt.has_from) {
		ResolveTemporalLiteral(stmt.backtest_from, "BACKTEST FROM");
	}
	if (stmt.has_to) {
		ResolveTemporalLiteral(stmt.backtest_to, "BACKTEST TO");
	}
	if (stmt.kind == pql::StmtKind::TRAIN || stmt.kind == pql::StmtKind::PREDICT) {
		Connection probe(DatabaseInstance::GetDatabase(context));
		const std::vector<TableSchema> sch = LoadSchema(probe);
		ResolveFilterTimestamps(stmt.filter.get(), FindSchema(sch, stmt.entity_table));
		if (!stmt.target.target_table.empty() && stmt.target.target_table != "*") {
			ResolveFilterTimestamps(stmt.target.filter.get(), FindSchema(sch, stmt.target.target_table));
		}
	}
	switch (stmt.kind) {
	case pql::StmtKind::TRAIN:
		RunTrain(context, stmt, *bind);
		break;
	case pql::StmtKind::PREDICT:
		RunPredictStmt(context, stmt, *bind);
		break;
	case pql::StmtKind::BACKTEST:
		RunBacktestStmt(context, stmt, *bind);
		break;
	case pql::StmtKind::EXPLAIN:
		RunExplainStmt(context, stmt, *bind);
		break;
	case pql::StmtKind::SHOW_MODELS:
		RunShowModels(context, *bind);
		break;
	case pql::StmtKind::DROP_MODEL: {
		bool dropped;
		{
			auto entry = GetRegistryEntry(context);
			std::lock_guard<std::mutex> guard(entry->lock);
			dropped = entry->registry.Drop(stmt.model);
		}
		if (!dropped && !stmt.if_exists) {
			throw InvalidInputException("pql: no model named '%s'; use DROP MODEL IF EXISTS to "
			                            "ignore that",
			                            stmt.model);
		}
		bind->names = {Identifier("dropped")};
		bind->types = {LogicalType::BOOLEAN};
		bind->rows.push_back({Value::BOOLEAN(dropped)});
		break;
	}
	}
	return_types = bind->types;
	names = bind->names;
	return std::move(bind);
}

// SHOW MODELS cannot reach the parser extension: DuckDB's own SHOW grammar
// accepts it and fails later at catalog lookup. pql_models() is the usable form.
static unique_ptr<FunctionData> PqlModelsBind(ClientContext &context, TableFunctionBindInput &,
                                              vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto bind = make_uniq<PqlBindData>();
	RunShowModels(context, *bind);
	return_types = bind->types;
	names = bind->names;
	return std::move(bind);
}

static unique_ptr<GlobalTableFunctionState> PqlInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<PqlGlobalState>();
}

static void PqlScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<PqlBindData>();
	auto &state = data.global_state->Cast<PqlGlobalState>();
	idx_t count = 0;
	while (state.offset < bind.rows.size() && count < STANDARD_VECTOR_SIZE) {
		const auto &row = bind.rows[state.offset++];
		for (idx_t c = 0; c < row.size() && c < output.ColumnCount(); c++) {
			output.SetValue(c, count, row[c]);
		}
		count++;
	}
	output.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// Parser extension
//===--------------------------------------------------------------------===//

struct PqlParseData : public ParserExtensionParseData {
	explicit PqlParseData(std::string query_p) : query(std::move(query_p)) {
	}
	std::string query;

	unique_ptr<ParserExtensionParseData> Copy() const override {
		return make_uniq<PqlParseData>(query);
	}
	string ToString() const override {
		return query;
	}
};

// Rebuild statement text from the token view. String literals keep their quotes
// when the tokenizer preserved them and are re-quoted otherwise, so the PQL
// parser sees the same literal the user wrote.
static std::string Reassemble(const vector<SimpleToken> &tokens, idx_t upto) {
	std::string out;
	for (idx_t i = 0; i < upto; i++) {
		const auto &t = tokens[i];
		if (!out.empty()) {
			out += ' ';
		}
		if (t.type == TokenType::STRING_LITERAL && (t.text.empty() || t.text[0] != '\'')) {
			out += "'" + t.text + "'";
		} else {
			out += t.text;
		}
	}
	return out;
}

static ParserExtensionParseResult PqlParseFunction(ParserExtensionInfo *, const vector<SimpleToken> &tokens) {
	if (tokens.empty()) {
		return ParserExtensionParseResult();
	}
	const std::string first = pql::ToUpper(tokens[0].text);
	const std::string second = tokens.size() > 1 ? pql::ToUpper(tokens[1].text) : "";
	// EXPLAIN is matched only with MODEL after it, so DuckDB's own EXPLAIN keeps
	// working on every other statement.
	const bool ours = first == "TRAIN" || first == "PREDICT" || (first == "BACKTEST" && second == "MODEL") ||
	                  (first == "EXPLAIN" && second == "MODEL") ||
	                  (first == "DROP" && second == "MODEL") || (first == "SHOW" && second == "MODELS");
	if (!ours) {
		return ParserExtensionParseResult();
	}
	// Claim through the statement terminator, or the whole tail if there is none.
	idx_t upto = tokens.size();
	for (idx_t i = 0; i < tokens.size(); i++) {
		if (tokens[i].type == TokenType::TERMINATOR || tokens[i].text == ";") {
			upto = i;
			break;
		}
	}
	const std::string query = Reassemble(tokens, upto);
	try {
		pql::Parse(query); // validate now so errors surface at parse time
	} catch (const pql::ParseError &e) {
		ParserExtensionParseResult err(std::string("PQL: ") + e.what());
		err.consumed_tokens = -1;
		return err;
	}
	ParserExtensionParseResult result(make_uniq<PqlParseData>(query));
	result.consumed_tokens = (int64_t)upto;
	return result;
}

static ParserExtensionPlanResult PqlPlanFunction(ParserExtensionInfo *, ClientContext &context,
                                                 unique_ptr<ParserExtensionParseData> parse_data) {
	auto &data = parse_data->Cast<PqlParseData>();
	ParserExtensionPlanResult result;
	result.function = PqlExecFunction();
	result.parameters.push_back(Value(data.query));
	result.requires_valid_transaction = true;
	result.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

TableFunction PqlExecFunction() {
	vector<LogicalType> args;
	args.push_back(LogicalType::VARCHAR);
	TableFunction fn("pql_exec", args, PqlScan, PqlBind, PqlInit);
	return fn;
}

void RegisterPqlExtension(ExtensionLoader &loader) {
	loader.RegisterFunction(PqlExecFunction());
	TableFunction models("pql_models", vector<LogicalType>(), PqlScan, PqlModelsBind, PqlInit);
	loader.RegisterFunction(models);
	auto &db = loader.GetDatabaseInstance();
	ParserExtension pql_ext;
	pql_ext.parse_function = PqlParseFunction;
	pql_ext.plan_function = PqlPlanFunction;
	ParserExtension::Register(DBConfig::GetConfig(db), pql_ext);
}

} // namespace duckdb
