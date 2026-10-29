// common.hpp — Standard includes and the two helpers every part uses.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pql {

inline std::string ToUpper(const std::string &s) {
	std::string o(s);
	for (auto &c : o) {
		c = char(std::toupper((unsigned char)c));
	}
	return o;
}

// Called between units of work (a minibatch, a scoring block); it throws to
// abort. The DuckDB binding uses it for Ctrl-C and max_execution_time.
using CancelCheck = std::function<void()>; // empty means never cancelled

} // namespace pql
