// features.hpp — Feature specification, windowed aggregates over links, example collection.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/data.hpp"

namespace pql {

// ===========================================================================
// 9. Feature specification
// ===========================================================================

// Windows used for compiled temporal aggregates, in days. Fixed rather than
// learned: each is one binary search, and a wide set costs almost nothing.
static const double kWindowDays[] = {7.0, 30.0, 90.0, 365.0, 1.0e18};
static const int kNumWindows = 5;
static const double kMicrosPerDay = 86400.0 * 1000000.0;

// Child row -> parent row for one link, UINT32_MAX where the row has none.
inline std::vector<uint32_t> InvertLink(const Link &lk, size_t nchild) {
	std::vector<uint32_t> out(nchild, UINT32_MAX);
	const size_t nparents = lk.off.empty() ? 0 : lk.off.size() - 1;
	for (size_t p = 0; p < nparents; p++) {
		for (uint32_t i = lk.off[p]; i < lk.off[p + 1]; i++) {
			out[lk.flat[i]] = (uint32_t)p;
		}
	}
	return out;
}

// Entity row -> row of the table its `via_key` column points at.
inline std::vector<uint32_t> ViaRows(const Database &db, const Frame &entity, const std::string &via_table,
                                     const std::string &via_key) {
	for (const Link &lk : db.links) {
		if (ToUpper(lk.child_table) != ToUpper(entity.name) || ToUpper(lk.parent_table) != ToUpper(via_table)) {
			continue;
		}
		const std::string key = lk.child_key_col >= 0 ? entity.columns[(size_t)lk.child_key_col].name : std::string();
		if (ToUpper(key) == ToUpper(via_key)) {
			return InvertLink(lk, entity.nrows);
		}
	}
	throw std::runtime_error("pql: the link from '" + entity.name + "' to '" + via_table +
	                         "' used by this model is not available");
}

// Which raw columns feed the model, fixed at training time and replayed at
// prediction time so the two vectors always line up.
struct FeatureSpec {
	struct SelfCol {
		int index = -1;
		std::string name;
		double mean = 0, sd = 1;
	};
	// A text column, encoded so the model can use it.
	//
	// Categories are stored as dictionary codes, and a code is an arbitrary
	// integer: feeding it in as a number would tell the model that 'basic' sits
	// between 'premium' and 'trial'. So a low-cardinality column becomes one slot
	// per value, and a high-cardinality one becomes how often its value occurs.
	// Neither looks at the label, so neither can leak.
	//
	// Slots are keyed by the text, not the code. Dictionary codes are assigned in
	// load order and the loader prunes columns per statement, so a code learned at
	// TRAIN need not mean the same thing at PREDICT.
	struct CatCol {
		int index = -1;
		std::string name;
		bool one_hot = true;
		// Whether the training fold held any row this encoding could not name: a
		// NULL, or a value dropped from the kept set. Only then is a spare slot
		// worth carrying.
		bool has_other = false;
		std::vector<std::string> labels; // one slot each, most frequent first
		std::vector<double> freqs;       // training-fold share of each label
		double freq_other = 0;           // share held by everything unlisted
		double freq_mean = 0, freq_sd = 1;
		// Resolved by Rebind: code -> slot, and code -> frequency. Both are direct
		// lookups so a row costs no hashing.
		std::vector<int> slot_by_code;
		std::vector<float> freq_by_code;

		// One slot per kept label, plus one for everything else when training saw
		// such rows. When it did not, an unrecognised value sets no slot at all.
		//
		// A spare slot that never fires during training keeps whatever weights it
		// was initialised with, because a zero input gets zero gradient. Firing it
		// at prediction time would then inject an untrained random number: an
		// unseen category would get an arbitrary answer rather than no answer.
		int Width() const {
			return one_hot ? (int)labels.size() + (has_other ? 1 : 0) : 1;
		}

		// Turn the labels into direct lookups against this frame's dictionary.
		// Must run before any row is built, at training time as well as at
		// prediction time: without it every row falls to the other slot and the
		// column looks like a constant.
		void Resolve(const Column &col) {
			const size_t ncodes = col.dict.values.size();
			slot_by_code.assign(ncodes, -1);
			freq_by_code.assign(ncodes, (float)freq_other);
			for (size_t sl = 0; sl < labels.size(); sl++) {
				const uint32_t code = col.dict.Lookup(labels[sl]);
				if (code == 0 || (size_t)code >= ncodes) {
					continue; // code 0 is the reserved unknown, never a real label
				}
				slot_by_code[code] = (int)sl;
				freq_by_code[code] = (float)freqs[sl];
			}
		}
	};
	struct LinkAgg {
		int link = -1;
		std::string child_table, child_key;
		// Set when the link hangs off a table the entity points at rather than
		// off the entity itself (panel row -> customer -> transactions): the
		// entity column holding that key, the table it reaches, and per entity
		// row the matching row there, or UINT32_MAX when it has none. One history
		// then serves every row that refers to the same customer, so a panel
		// with many anchors per customer carries the history once, not per row.
		std::string via_table, via_key;
		std::vector<uint32_t> via;
		std::vector<int> mean_cols; // child numeric columns averaged in-window
		std::vector<std::string> mean_names;
		std::vector<double> mean_mu, mean_sd;
	};
	std::vector<SelfCol> self_cols;
	std::vector<CatCol> cat_cols;
	std::vector<LinkAgg> link_aggs;
	// Where the anchor falls in the week and in the year, as sine and cosine
	// pairs. The anchor's raw value is deliberately not a feature: it is a
	// straight line through the training era and a model that leans on it
	// extrapolates nonsense. Its position within a cycle is different, it is known
	// at prediction time by definition, and demand data is full of it.
	bool anchor_cycles = false;
	static const int kCycleSlots = 4;
	int width = 0;

	// count + recency per window, plus one mean per selected child column per window
	// Column positions are not stable across statements: the loader prunes
	// columns per query, so a filter on a text column changes the frame layout.
	// Re-resolve by name before any use, and fail loudly if something is gone.
	void Rebind(const Database &db, const Frame &entity) {
		for (auto &sc : self_cols) {
			const int i = entity.Find(sc.name);
			if (i < 0) {
				throw std::runtime_error("pql: column '" + sc.name +
				                         "' used by this model is not "
				                         "present on '" +
				                         entity.name + "' any more");
			}
			sc.index = i;
		}
		for (auto &cc : cat_cols) {
			const int i = entity.Find(cc.name);
			if (i < 0) {
				throw std::runtime_error("pql: column '" + cc.name +
				                         "' used by this model is not "
				                         "present on '" +
				                         entity.name + "' any more");
			}
			cc.index = i;
			const Column &col = entity.columns[(size_t)i];
			if (col.type != ColType::CATEGORY) {
				throw std::runtime_error("pql: column '" + cc.name +
				                         "' was text when this model "
				                         "was trained and is not any more");
			}
			// Against this frame's own dictionary: a label it has never seen has no
			// code, and those rows fall to the other slot, which is the honest answer.
			cc.Resolve(col);
		}
		for (auto &la : link_aggs) {
			// Two foreign keys between the same pair of tables are two links, so the
			// child key column has to take part or both aggregates land on the first.
			const std::string owner = la.via_table.empty() ? entity.name : la.via_table;
			int found = -1;
			for (size_t l = 0; l < db.links.size(); l++) {
				const Link &lk = db.links[l];
				if (ToUpper(lk.child_table) != ToUpper(la.child_table) || ToUpper(lk.parent_table) != ToUpper(owner)) {
					continue;
				}
				const Frame &child = db.At(lk.child_table);
				const std::string key =
				    lk.child_key_col >= 0 ? child.columns[(size_t)lk.child_key_col].name : std::string();
				if (la.child_key.empty() || ToUpper(key) == ToUpper(la.child_key)) {
					found = (int)l;
					break;
				}
			}
			if (found < 0) {
				throw std::runtime_error("pql: the link from '" + la.child_table + "' to '" + owner +
				                         "' used by this model is not available");
			}
			la.link = found;
			if (!la.via_table.empty()) {
				la.via = ViaRows(db, entity, la.via_table, la.via_key);
			}
			const Frame &child = db.At(la.child_table);
			for (size_t m = 0; m < la.mean_names.size(); m++) {
				const int c = child.Find(la.mean_names[m]);
				if (c < 0) {
					throw std::runtime_error("pql: column '" + la.child_table + "." + la.mean_names[m] +
					                         "' used by this model is missing");
				}
				la.mean_cols[m] = c;
			}
		}
	}

	// One label per feature group, in the order BuildFeatures emits them, with the
	// number of slots each occupies. A one-hot block is one group: permuting its
	// slots separately would split the column's contribution across them and make
	// every piece look unimportant.
	std::vector<std::pair<std::string, int>> Describe() const {
		static const char *wname[] = {"7d", "30d", "90d", "365d", "all"};
		std::vector<std::pair<std::string, int>> out;
		for (const auto &sc : self_cols) {
			out.emplace_back(sc.name, 1);
		}
		if (anchor_cycles) {
			out.emplace_back("anchor: phase of week", 2);
			out.emplace_back("anchor: phase of year", 2);
		}
		for (const auto &cc : cat_cols) {
			out.emplace_back(cc.name + (cc.one_hot ? " (category)" : " (how common)"), cc.Width());
		}
		for (const auto &la : link_aggs) {
			const std::string name = la.via_table.empty() ? la.child_table : la.via_table + "." + la.child_table;
			out.emplace_back(name + ": never happened", 1);
			for (int w = 0; w < kNumWindows; w++) {
				const std::string suffix = std::string(" ") + wname[w];
				out.emplace_back(name + ": count" + suffix, 1);
				out.emplace_back(name + ": recency" + suffix, 1);
				out.emplace_back(name + ": spacing" + suffix, 1);
				for (const auto &mn : la.mean_names) {
					out.emplace_back(name + "." + mn + ": mean" + suffix, 1);
				}
			}
		}
		return out;
	}

	// Per window: a count, a within-window recency, the spacing of the events, and
	// one mean per selected child column. Plus one never-happened flag per link.
	int PerLinkWidth(const LinkAgg &la) const {
		return kNumWindows * (3 + (int)la.mean_cols.size()) + 1;
	}
	void ComputeWidth() {
		width = (int)self_cols.size();
		if (anchor_cycles) {
			width += kCycleSlots;
		}
		for (const auto &cc : cat_cols) {
			width += cc.Width();
		}
		for (auto &la : link_aggs) {
			width += PerLinkWidth(la);
		}
	}
};

// Fit an encoding for one text column, counting only rows from the training era.
// `anch` is the column that decides which era a row belongs to, or null when
// every row counts. Returns false when the column carries nothing usable.
//
// Shared by the dense path and by GraphSAGE so the two cannot drift apart.
inline bool FitCatCol(const Frame &f, size_t col_index, const Column *anch, double train_cutoff, int max_categories,
                      FeatureSpec::CatCol &out) {
	const Column &c = f.columns[col_index];
	if (c.type != ColType::CATEGORY) {
		return false;
	}
	const bool bounded = anch && std::isfinite(train_cutoff);
	auto in_fold = [&](size_t r) {
		return !bounded || (anch->valid[r] && anch->num[r] <= train_cutoff);
	};
	std::vector<uint64_t> counts(c.dict.values.size(), 0ull);
	uint64_t seen = 0, rows_in_fold = 0;
	for (size_t r = 0; r < f.nrows; r++) {
		if (!in_fold(r)) {
			continue;
		}
		rows_in_fold++;
		if (!c.valid[r]) {
			continue;
		}
		const uint32_t code = c.code[r];
		if (code < counts.size()) {
			counts[code]++;
			seen++;
		}
	}
	if (seen == 0) {
		return false;
	}
	std::vector<uint32_t> order;
	for (uint32_t code = 1; code < (uint32_t)counts.size(); code++) {
		if (counts[code] > 0) {
			order.push_back(code);
		}
	}
	if (order.size() < 2) {
		return false; // one value everywhere is a constant, not a feature
	}
	// Nearly one value per row is an identifier or free text, not a category: an
	// email column would otherwise become a frequency feature holding the same
	// number in every row.
	if ((int)order.size() > max_categories && order.size() * 4 > (size_t)seen) {
		return false;
	}
	// Most frequent first, ties broken by label so the layout is deterministic.
	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		if (counts[a] != counts[b]) {
			return counts[a] > counts[b];
		}
		return c.dict.values[a] < c.dict.values[b];
	});
	out = FeatureSpec::CatCol();
	out.index = (int)col_index;
	out.name = c.name;
	const double inv_seen = 1.0 / double(seen);
	uint64_t named = 0;
	for (uint32_t code : order) {
		out.labels.push_back(c.dict.values[code]);
		out.freqs.push_back(double(counts[code]) * inv_seen);
		named += counts[code];
	}
	out.one_hot = (int)order.size() <= max_categories;
	// Did any training row fall outside the kept labels? With one-hot every seen
	// value gets a slot, so this is exactly "were there NULLs".
	out.has_other = named < rows_in_fold;
	if (!out.one_hot) {
		// Too many values to give each a slot. How often a value occurs is a weaker
		// feature than the value itself, but it is a real one and it costs one slot
		// instead of thousands.
		double fs = 0, fq = 0;
		uint64_t fn = 0;
		for (size_t r = 0; r < f.nrows; r++) {
			if (!c.valid[r] || !in_fold(r)) {
				continue;
			}
			const uint32_t code = c.code[r];
			const double freq = code < counts.size() ? double(counts[code]) * inv_seen : 0.0;
			fs += freq;
			fq += freq * freq;
			fn++;
		}
		out.freq_mean = fn ? fs / double(fn) : 0.0;
		const double var = fn > 1 ? (fq - double(fn) * out.freq_mean * out.freq_mean) / double(fn - 1) : 0.0;
		out.freq_sd = var > 1e-18 ? std::sqrt(var) : 1.0;
	}
	out.Resolve(c);
	return true;
}

// Prefix-sum arena over each link's time-sorted children.
//
// The scan this replaces read vc.num[rows[i]] -- an indirect gather through a
// row-index array, which cannot vectorize and which we had to cap at 512 rows
// to bound its cost. Laying the values out once in link order, with a running
// sum per parent, turns every windowed mean into two lookups. It is exact
// (the cap is gone) and it is O(1) per window instead of O(window).
// Length of a link's prefix-sum arena: one running row per child plus a zero
// row per parent. Those arenas are indexed with uint32, so a link the index
// cannot hold is refused here, before any of them is sized, rather than wrapped.
inline uint32_t PrefixTotal(const Link &lk) {
	const uint64_t nparents = lk.off.empty() ? 0 : (uint64_t)lk.off.size() - 1;
	const uint64_t total = (uint64_t)lk.flat.size() + nparents;
	if (total > (uint64_t)UINT32_MAX) {
		throw std::runtime_error("pql: link " + lk.child_table + " -> " + lk.parent_table + " has " +
		                         std::to_string(total) +
		                         " rows plus parents, more than the 4294967295 a model can index");
	}
	return (uint32_t)total;
}

struct AggCache {
	struct LinkCache {
		std::vector<uint32_t> off;               // per parent: base into the prefix arrays
		std::vector<std::vector<float>> psum;    // [mean col][off + i] running value sum
		std::vector<std::vector<uint32_t>> pcnt; // [mean col][off + i] running valid count
		// Running sum of squared gaps between consecutive children, in days, so the
		// spread of the gaps inside any window is two lookups like everything else.
		std::vector<float> pgap2;
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
		PrefixTotal(lk); // refuses what the uint32 running index below cannot hold
		uint32_t total = 0;
		for (size_t p = 0; p < nparents; p++) {
			lc.off[p] = total;
			total += lk.Count((uint32_t)p) + 1u; // +1 for the leading zero
		}
		lc.off[nparents] = total;
		lc.pgap2.assign(total, 0.0f);
		if (lk.dated) {
			for (size_t p = 0; p < nparents; p++) {
				const uint32_t b = lk.Begin((uint32_t)p), e = lk.End((uint32_t)p);
				const uint32_t base = lc.off[p];
				double run = 0.0;
				for (uint32_t i = b; i < e; i++) {
					const uint32_t j = i - b;
					if (j >= 1) {
						const double g = (lk.Time(lk.flat[i]) - lk.Time(lk.flat[i - 1])) / kMicrosPerDay;
						run += g * g;
					}
					lc.pgap2[base + j + 1] = (float)run;
				}
			}
		}
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
inline void BuildFeatures(const Database &db, const Frame &entity, const FeatureSpec &spec, const AggCache &cache,
                          uint32_t row, double anchor, std::vector<float> &out) {
	out.assign((size_t)spec.width, 0.0f);
	size_t k = 0;
	for (const auto &sc : spec.self_cols) {
		const Column &c = entity.columns[(size_t)sc.index];
		out[k++] = c.valid[row] ? float((c.num[row] - sc.mean) / sc.sd) : 0.0f;
	}
	if (spec.anchor_cycles) {
		// Days since the epoch, which began on a Thursday. Which weekday maps to
		// which angle does not matter, only that it is the same every time.
		if (std::isfinite(anchor)) {
			const double days = std::floor(anchor / kMicrosPerDay);
			const double dow = std::fmod(std::fmod(days, 7.0) + 7.0, 7.0);
			const double doy = std::fmod(std::fmod(days, 365.25) + 365.25, 365.25);
			const double tw = 6.283185307179586;
			out[k + 0] = float(std::sin(tw * dow / 7.0));
			out[k + 1] = float(std::cos(tw * dow / 7.0));
			out[k + 2] = float(std::sin(tw * doy / 365.25));
			out[k + 3] = float(std::cos(tw * doy / 365.25));
		}
		k += (size_t)FeatureSpec::kCycleSlots;
	}
	for (const auto &cc : spec.cat_cols) {
		const Column &c = entity.columns[(size_t)cc.index];
		const uint32_t code = c.valid[row] ? c.code[row] : 0u;
		const bool known = code < cc.slot_by_code.size();
		if (cc.one_hot) {
			const int slot = (known && c.valid[row]) ? cc.slot_by_code[code] : -1;
			if (slot >= 0) {
				out[k + (size_t)slot] = 1.0f;
			} else if (cc.has_other) {
				out[k + cc.labels.size()] = 1.0f;
			}
			// Otherwise the block stays zero: the encoding has nothing to say.
			k += (size_t)cc.Width();
		} else {
			const double f = (known && c.valid[row]) ? (double)cc.freq_by_code[code] : cc.freq_other;
			out[k++] = float((f - cc.freq_mean) / cc.freq_sd);
		}
	}
	for (size_t li = 0; li < spec.link_aggs.size(); li++) {
		const FeatureSpec::LinkAgg &la = spec.link_aggs[li];
		const AggCache::LinkCache &lc = cache.links[li];
		const Link &lk = db.links[(size_t)la.link];
		const Frame &child = db.At(lk.child_table);
		// The row the link is read from: the entity row itself, or the row it
		// points at when the history belongs to a parent.
		const uint32_t prow = la.via.empty() ? row : la.via[row];
		if (prow == UINT32_MAX) {
			out[k] = 1.0f; // points at nothing: never happened, every window empty
			k += (size_t)spec.PerLinkWidth(la);
			continue;
		}
		const uint32_t bslice = lk.Begin(prow);
		const size_t visible = Database::VisiblePrefix(lk, child, prow, anchor);
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
			const uint32_t base = lc.off[prow];
			// How evenly the events are spaced, as the coefficient of variation of
			// the gaps between them. Divided by the mean gap, so it says nothing
			// about the rate, which the count already carries: a user who returns
			// every nine days and one who arrives in two bursts have the same count,
			// the same recency and the same window totals, and only this separates
			// them.
			float cv = 0.0f;
			if (timed && n >= 2) {
				const double span =
				    (lk.Time(lk.flat[bslice + visible - 1]) - lk.Time(lk.flat[bslice + start])) / kMicrosPerDay;
				const double ng = double(n - 1);
				const double mean_gap = span / ng; // the gaps telescope
				const double sumsq = double(lc.pgap2[base + visible]) - double(lc.pgap2[base + start + 1]);
				if (mean_gap > 1e-9) {
					const double var = sumsq / ng - mean_gap * mean_gap;
					const double c = std::sqrt(var > 0 ? var : 0.0) / mean_gap;
					cv = float(c > 4.0 ? 4.0 : c);
				}
			}
			out[k++] = cv;
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

// Does EXCLUDE name this column? A bare name means the entity's own column;
// qualify it to reach a child table's.
inline bool IsExcluded(const Statement &stmt, const Frame &entity, const std::string &table,
                       const std::string &column) {
	for (const auto &r : stmt.excluded) {
		if (ToUpper(r.name) != ToUpper(column)) {
			continue;
		}
		if (r.qualifier.empty()) {
			if (ToUpper(table) == ToUpper(entity.name)) {
				return true;
			}
			continue;
		}
		if (ToUpper(r.qualifier) == ToUpper(table) || ToUpper(r.qualifier) == ToUpper(stmt.entity_alias)) {
			return true;
		}
	}
	return false;
}

// A name that matches nothing is a mistake, not a no-op. Silently ignoring a
// misspelt EXCLUDE would leave the column in the model and say nothing.
inline void ValidateExcluded(const Database &db, const Frame &entity, const Statement &stmt) {
	for (const auto &r : stmt.excluded) {
		bool found = false;
		for (const auto &t : db.tables) {
			if (!r.qualifier.empty() && ToUpper(r.qualifier) != ToUpper(t.name) &&
			    ToUpper(r.qualifier) != ToUpper(stmt.entity_alias)) {
				continue;
			}
			if (r.qualifier.empty() && ToUpper(t.name) != ToUpper(entity.name)) {
				continue;
			}
			if (t.Find(r.name) >= 0) {
				found = true;
				break;
			}
		}
		if (!found) {
			throw std::runtime_error("pql: EXCLUDE names '" + r.ToString() + "', which is not a column of " +
			                         (r.qualifier.empty() ? ("'" + entity.name + "'") : ("'" + r.qualifier + "'")));
		}
	}
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

// The 60/20/20 split by anchor, or the boundaries an explicit SPLIT names.
// Always forward in time, never a shuffle, so validation and test sit after
// everything the model was fitted on.
//
// One definition, because anything that wants to reproduce the folds later (to
// explain a model on rows it did not learn from, say) has to agree with training
// exactly or it is measuring something else.
inline void SplitByAnchor(const Dataset &ds, const Statement &stmt, std::vector<size_t> &tr, std::vector<size_t> &va,
                          std::vector<size_t> &te) {
	tr.clear();
	va.clear();
	te.clear();
	std::vector<size_t> order(ds.examples.size());
	for (size_t i = 0; i < order.size(); i++) {
		order[i] = i;
	}
	std::sort(order.begin(), order.end(),
	          [&](size_t a, size_t b) { return ds.examples[a].anchor < ds.examples[b].anchor; });
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
		return;
	}
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
			throw std::runtime_error("pql: AT column '" + stmt.anchor.name + "' not found on '" + entity.name + "'");
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
inline FeatureSpec FitFeatureSpec(const Database &db, const Frame &entity, const Statement &stmt, int anchor_col,
                                  int target_col, int max_mean_cols, int max_categories = 16,
                                  double train_cutoff = std::numeric_limits<double>::infinity()) {
	FeatureSpec spec;
	// Only when the anchor is a real instant. An attribute model with no clock
	// anywhere has nothing to take a phase of.
	spec.anchor_cycles =
	    stmt.every.present || (anchor_col >= 0 && entity.columns[(size_t)anchor_col].type == ColType::TIMESTAMP);
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
		if (IsExcluded(stmt, entity, entity.name, entity.columns[i].name)) {
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
				// Values can be finite and still sum past the end of double. A column
				// whose own mean does not exist cannot be standardised, so it is not a
				// feature.
				if (!std::isfinite(mean_b) || !std::isfinite(var_b)) {
					continue;
				}
				FeatureSpec::SelfCol sc_b;
				sc_b.index = (int)i;
				sc_b.name = c.name;
				sc_b.mean = mean_b;
				sc_b.sd = var_b > 1e-18 ? std::sqrt(var_b) : 1.0;
				spec.self_cols.push_back(sc_b);
			}
			// Fewer than two training-era values is not a feature; falling through
			// to the pass over every row would fit it on the folds being scored.
			continue;
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
		if (!std::isfinite(mean) || !std::isfinite(sd)) {
			continue;
		}
		FeatureSpec::SelfCol sc;
		sc.index = (int)i;
		sc.name = entity.columns[i].name;
		sc.mean = mean;
		sc.sd = sd > 1e-9 ? sd : 1.0;
		spec.self_cols.push_back(sc);
	}
	// Text columns, counted over the training fold only like every other
	// statistic here, so the encoding never sees a validation or test row.
	for (size_t i = 0; i < entity.columns.size(); i++) {
		if (excluded[i] || entity.columns[i].type != ColType::CATEGORY) {
			continue;
		}
		if (IsExcluded(stmt, entity, entity.name, entity.columns[i].name)) {
			continue;
		}
		const Column *anch = anchor_col >= 0 ? &entity.columns[(size_t)anchor_col] : nullptr;
		FeatureSpec::CatCol cc;
		if (FitCatCol(entity, i, anch, train_cutoff, max_categories, cc)) {
			spec.cat_cols.push_back(cc);
		}
	}
	auto add_link = [&](size_t l, const std::string &via_table, const std::string &via_key,
	                    const std::vector<uint32_t> *via) {
		const Link &lk = db.links[l];
		FeatureSpec::LinkAgg la;
		la.link = (int)l;
		la.child_table = lk.child_table;
		la.via_table = via_table;
		la.via_key = via_key;
		if (via) {
			la.via = *via;
		}
		const Frame &child = db.At(lk.child_table);
		la.child_key = lk.child_key_col >= 0 ? child.columns[(size_t)lk.child_key_col].name : std::string();
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
			if (IsExcluded(stmt, entity, lk.child_table, child.columns[c].name)) {
				continue;
			}
			const Column &cc = child.columns[c];
			const double *cnum = cc.num.data();
			const uint8_t *cval = cc.valid.data();
			const size_t crows = child.nrows;
			// The child's clock bounds these statistics to the training era, the
			// way the entity's anchor bounds its own columns above. Same mask
			// multiply as before, so the loop stays branchless.
			const bool bounded = lk.dated && std::isfinite(train_cutoff) && lk.ctime.size() == crows;
			const double *ct = bounded ? lk.ctime.data() : nullptr;
			double cs0 = 0, cs1 = 0, cq0 = 0, cq1 = 0;
			uint64_t k0 = 0, k1 = 0;
			size_t cr = 0;
			for (; cr + 1 < crows; cr += 2) {
				const uint8_t m0 = cval[cr] & (uint8_t)(!bounded || ct[cr] <= train_cutoff);
				const uint8_t m1 = cval[cr + 1] & (uint8_t)(!bounded || ct[cr + 1] <= train_cutoff);
				const double v0 = cnum[cr] * double(m0);
				const double v1 = cnum[cr + 1] * double(m1);
				cs0 += v0;
				cs1 += v1;
				cq0 += v0 * cnum[cr];
				cq1 += v1 * cnum[cr + 1];
				k0 += m0;
				k1 += m1;
			}
			for (; cr < crows; cr++) {
				const uint8_t m = cval[cr] & (uint8_t)(!bounded || ct[cr] <= train_cutoff);
				const double v = cnum[cr] * double(m);
				cs0 += v;
				cq0 += v * cnum[cr];
				k0 += m;
			}
			const size_t cn = (size_t)(k0 + k1);
			const double mu = cn ? (cs0 + cs1) / double(cn) : 0.0;
			const double cvar = cn > 1 ? ((cq0 + cq1) - double(cn) * mu * mu) / double(cn - 1) : 0.0;
			const double sd = cvar > 0 ? std::sqrt(cvar) : 1.0;
			if (!std::isfinite(mu) || !std::isfinite(sd)) {
				continue;
			}
			la.mean_cols.push_back((int)c);
			la.mean_names.push_back(cc.name);
			la.mean_mu.push_back(mu);
			la.mean_sd.push_back(sd > 1e-9 ? sd : 1.0);
		}
		spec.link_aggs.push_back(la);
	};
	for (size_t l = 0; l < db.links.size(); l++) {
		if (ToUpper(db.links[l].parent_table) == ToUpper(entity.name)) {
			add_link(l, "", "", nullptr);
		}
	}
	// History through a parent. A panel row that points at a customer reads the
	// customer's children as its own, censored at the row's anchor; the entity
	// table itself is not one of those children, or every row would see the
	// other rows about the same customer.
	for (size_t u = 0; u < db.links.size(); u++) {
		const Link &up = db.links[u];
		if (ToUpper(up.child_table) != ToUpper(entity.name) || ToUpper(up.parent_table) == ToUpper(entity.name) ||
		    up.child_key_col < 0) {
			continue;
		}
		const std::string via_key = entity.columns[(size_t)up.child_key_col].name;
		std::vector<uint32_t> via;
		for (size_t l = 0; l < db.links.size(); l++) {
			const Link &lk = db.links[l];
			if (ToUpper(lk.parent_table) != ToUpper(up.parent_table) ||
			    ToUpper(lk.child_table) == ToUpper(entity.name)) {
				continue;
			}
			if (via.empty()) {
				via = InvertLink(up, entity.nrows);
			}
			add_link(l, up.parent_table, via_key, &via);
		}
	}
	spec.ComputeWidth();
	return spec;
}

// Assemble labelled examples. For a forecast the label is aggregated from the
// target table over (anchor, anchor+horizon]; for an attribute it is read from
// the entity row and rows without one are skipped.
// `only_rows`, when given, restricts the entity rows to those it marks, on top
// of the statement's own filter.
inline Dataset CollectExamples(const Database &db, const Statement &stmt, const FeatureSpec &spec,
                               const AggCache &cache, int anchor_col, bool need_labels,
                               const std::vector<uint8_t> *only_rows = nullptr) {
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
			// Say which of the two things went wrong. A typo in the table name used
			// to be reported as a missing relationship, sending people to look at
			// their foreign keys for a table that was never there.
			if (db.Find(stmt.target.target_table) < 0) {
				throw std::runtime_error("pql: there is no table '" + stmt.target.target_table + "' to aggregate over");
			}
			throw std::runtime_error("pql: no foreign key connects '" + stmt.target.target_table + "' to '" +
			                         entity.name + "', so its rows cannot be attributed to an entity");
		}
		// Without a time column on the child, every row is inside every window: the
		// horizon silently becomes a no-op and the label collapses to the lifetime
		// total, which is also fed back in as a feature. That is a leak, so refuse.
		if (!db.links[(size_t)target_link].dated) {
			throw std::runtime_error("pql: '" + stmt.target.target_table +
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
		throw std::runtime_error("pql: no column '" + stmt.target.ref.ToString() + "' on '" + entity.name + "'");
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
				throw std::runtime_error("pql: EVERY produced no anchors; '" + stmt.target.target_table +
				                         "' does not span enough time for one horizon before and after");
			}
		}
	}
	size_t censored = 0;
	size_t no_anchor = 0;
	std::vector<float> feat;
	for (size_t r = 0; r < entity.nrows; r++) {
		if (only_rows && (r >= only_rows->size() || !(*only_rows)[r])) {
			continue;
		}
		if (!EvalFilter(stmt.filter.get(), entity, r)) {
			continue;
		}
		if (!generated.empty()) {
			for (double gen_anchor : generated) {
				Example gex;
				gex.entity_row = (uint32_t)r;
				gex.anchor = gen_anchor;
				bool any_prev = false;
				gex.base = AggregateWindow(db, db.links[(size_t)target_link], db.At(stmt.target.target_table),
				                           (uint32_t)r, gex.anchor - horizon, gex.anchor, stmt.target.kind, value_col,
				                           stmt.target.filter.get(), any_prev);
				if (need_labels) {
					bool any = false;
					gex.label = AggregateWindow(db, db.links[(size_t)target_link], db.At(stmt.target.target_table),
					                            (uint32_t)r, gex.anchor, gex.anchor + horizon, stmt.target.kind,
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
			ex.anchor = stmt.anchor_literal.kind == Literal::Kind::NUMBER ? stmt.anchor_literal.number : 0.0;
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
			ex.base = AggregateWindow(db, db.links[(size_t)target_link], db.At(stmt.target.target_table), (uint32_t)r,
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
				ex.label = AggregateWindow(db, db.links[(size_t)target_link], db.At(stmt.target.target_table),
				                           (uint32_t)r, ex.anchor, ex.anchor + horizon, stmt.target.kind, value_col,
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

} // namespace pql
