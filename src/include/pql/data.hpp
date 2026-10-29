// data.hpp — Materialized data, the schema graph of links, filter evaluation and label construction.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/lang.hpp"

namespace pql {

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
	std::vector<std::string> values;                 // code -> text, 0 = unknown/null
	std::unordered_map<std::string, uint32_t> index; // text -> code

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
		return type == ColType::INT64 || type == ColType::DOUBLE || type == ColType::BOOL || type == ColType::TIMESTAMP;
	}
};

struct Frame {
	std::string name;
	std::vector<Column> columns;
	size_t nrows = 0;

	int Find(const std::string &col) const {
		for (size_t i = 0; i < columns.size(); i++) {
			// Column references are matched case-insensitively, as in SQL.
			if (columns[i].name.size() == col.size() && ToUpper(columns[i].name) == ToUpper(col)) {
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

// Project a numeric key onto int64 for matching. An integral value inside the
// int64 range maps to itself, so integer keys stored as doubles round-trip
// exactly and the dense direct-address table below still applies. Anything else
// (a fraction, a magnitude at or past 2^63) maps to its bit pattern: a plain
// cast there is undefined behaviour, and truncation matched 12.5 onto 12.
inline int64_t NumericKey(double v) {
	if (v >= -9223372036854775808.0 && v < 9223372036854775808.0 && v == std::floor(v)) {
		return (int64_t)v;
	}
	int64_t bits = 0;
	std::memcpy(&bits, &v, sizeof(bits));
	return bits;
}

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
		// The float key is coarse: at 1e15 micros one float step is ~134 s, so
		// children closer than that share a key and came out in input order. Each
		// run of equal keys is the only span that can still be unsorted, so sort
		// those on the double; runs are short, so the pass stays linear.
		for (uint32_t i = 0; i < n;) {
			uint32_t j = i + 1;
			while (j < n && keys[j] == keys[i]) {
				j++;
			}
			if (j - i >= 2) {
				std::sort(lk.flat.begin() + b + i, lk.flat.begin() + b + j,
				          [&](uint32_t x, uint32_t y) { return lk.Time(x) < lk.Time(y); });
			}
			i = j;
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
	// A NaN or an infinity is not a value, so it is treated as one that is
	// missing. Left alone it poisons whatever it touches: a column holding one
	// NaN gives a NaN mean, every feature built from it is NaN, every weight
	// becomes NaN within a step, and the run still reports a metric. That is the
	// worst shape a failure can take, so it is stopped at the door.
	void MarkNonFiniteMissing() {
		for (auto &t : tables) {
			for (auto &c : t.columns) {
				if (!c.IsNumeric()) {
					continue;
				}
				for (size_t r = 0; r < c.num.size() && r < c.valid.size(); r++) {
					if (c.valid[r] && !std::isfinite(c.num[r])) {
						c.valid[r] = 0;
						c.num[r] = 0.0;
					}
				}
			}
		}
	}

	void BuildLinks(const std::vector<std::string> &allow) {
		MarkNonFiniteMissing();
		links.clear();
		// Rows are addressed with uint32 throughout the link index.
		for (const auto &t : tables) {
			if (t.nrows >= (size_t)UINT32_MAX) {
				throw std::runtime_error("pql: table '" + t.name + "' has " + std::to_string(t.nrows) +
				                         " rows; at most 4294967294 are supported");
			}
		}
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
					const bool str_key = gpk.type == ColType::CATEGORY && cc2.type == ColType::CATEGORY;
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
							by_num.emplace(NumericKey(gpk.num[r]), gtc.num[r]);
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
							auto it = by_num.find(NumericKey(cc2.num[r]));
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
				const int64_t key = pk.type == ColType::CATEGORY ? (int64_t)pk.code[r] : NumericKey(pk.num[r]);
				kmin = std::min(kmin, key);
				kmax = std::max(kmax, key);
			}
			std::vector<uint32_t> direct;
			bool use_direct = false;
			if (parent.nrows > 0 && kmin <= kmax) {
				// Unsigned: kmax - kmin overflows when a column holds both int64 ends.
				const uint64_t span = (uint64_t)kmax - (uint64_t)kmin;
				if (span < (uint64_t)parent.nrows * 4u + 64u) {
					direct.assign((size_t)span + 1u, UINT32_MAX);
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
				const int64_t key = pk.type == ColType::CATEGORY ? (int64_t)pk.code[r] : NumericKey(pk.num[r]);
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
			const bool string_key = pk.type == ColType::CATEGORY && ck.type == ColType::CATEGORY;
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
				// A dated child with no time can fall inside no window, and kept in
				// the bucket it breaks the sort the binary searches rely on.
				if (lk.dated && !std::isfinite(lk.ctime[r])) {
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
					const int64_t key = ck.type == ColType::CATEGORY ? (int64_t)ck.code[r] : NumericKey(ck.num[r]);
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
inline void ValidateFilterScope(const FilterNode *f, const Frame &frame, const std::string &alias, const char *where) {
	if (!f) {
		return;
	}
	if (f->kind != FilterNode::Kind::CMP) {
		for (const auto &c : f->children) {
			ValidateFilterScope(c.get(), frame, alias, where);
		}
		return;
	}
	if (f->prediction) {
		return; // the model's answer, not a column
	}
	if (!f->ref.qualifier.empty()) {
		const std::string q = ToUpper(f->ref.qualifier);
		if (q != ToUpper(frame.name) && (alias.empty() || q != ToUpper(alias))) {
			throw std::runtime_error("pql: the " + std::string(where) + " filter refers to '" + f->ref.ToString() +
			                         "', but it is evaluated on '" + frame.name + "'" +
			                         (alias.empty() ? "" : " (aliased " + alias + ")"));
		}
	}
	if (frame.Find(f->ref.name) < 0) {
		throw std::runtime_error("pql: the " + std::string(where) + " filter refers to column '" + f->ref.name +
		                         "', which does not exist on '" + frame.name + "'");
	}
}

// Filters are evaluated against a single row of one frame. Comparisons against
// a category resolve the literal through that column's dictionary once per
// call site; an unseen literal resolves to code 0 and matches nothing, which is
// the SQL-correct outcome for a value not present in the data.
// `prediction` is the model's answer for the row, for the comparisons inside
// OPTIMIZE's totals that test it; everywhere else there is none.
inline bool EvalFilter(const FilterNode *f, const Frame &frame, size_t row,
                       double prediction = std::numeric_limits<double>::quiet_NaN()) {
	if (!f) {
		return true;
	}
	switch (f->kind) {
	case FilterNode::Kind::AND:
		for (auto &c : f->children) {
			if (!EvalFilter(c.get(), frame, row, prediction)) {
				return false;
			}
		}
		return true;
	case FilterNode::Kind::OR:
		for (auto &c : f->children) {
			if (EvalFilter(c.get(), frame, row, prediction)) {
				return true;
			}
		}
		return false;
	case FilterNode::Kind::NOT:
		return f->children.empty() ? true : !EvalFilter(f->children[0].get(), frame, row, prediction);
	default:
		break;
	}
	if (f->prediction) {
		const bool known = std::isfinite(prediction);
		if (f->op == CmpOp::IS_NULL || f->op == CmpOp::IS_NOT_NULL) {
			return (f->op == CmpOp::IS_NULL) != known;
		}
		if (!known) {
			return false;
		}
		auto eq = [&](const Literal &lit) {
			return lit.kind == Literal::Kind::NUMBER && prediction == lit.number;
		};
		switch (f->op) {
		case CmpOp::IN:
		case CmpOp::NOT_IN: {
			bool any = false;
			for (const auto &lit : f->values) {
				any = any || eq(lit);
			}
			return f->op == CmpOp::IN ? any : !any;
		}
		case CmpOp::EQ:
			return !f->values.empty() && eq(f->values[0]);
		case CmpOp::NE:
			return !f->values.empty() && !eq(f->values[0]);
		default:
			break;
		}
		if (f->values.empty() || f->values[0].kind != Literal::Kind::NUMBER) {
			throw std::runtime_error("pql: prediction is a number, so it compares with numbers");
		}
		const double b = f->values[0].number;
		switch (f->op) {
		case CmpOp::LT:
			return prediction < b;
		case CmpOp::LE:
			return prediction <= b;
		case CmpOp::GT:
			return prediction > b;
		default:
			return prediction >= b;
		}
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
	double anchor = 0; // micros
	double label = 0;
	// The same aggregate over the horizon immediately BEFORE the anchor. This is
	// the persistence prediction, and the model learns the residual over it: a
	// net cannot cheaply rediscover the identity function from log1p-compressed
	// counts, so handing it the level as an offset is worth more than any feature.
	double base = 0;
	bool has_label = false;
};

inline double AggregateWindow(const Database &, const Link &lk, const Frame &child, uint32_t entity_row,
                              double lo_exclusive, double hi_inclusive, TargetKind kind, int value_col,
                              const FilterNode *filter, bool &any) {
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
			if (!have_best || (kind == TargetKind::MIN && v < best) || (kind == TargetKind::MAX && v > best)) {
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

} // namespace pql
