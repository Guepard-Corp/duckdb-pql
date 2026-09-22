# Building and testing

The extension has no dependency beyond DuckDB and the C++17 standard library:
no packages to fetch, no toolchain beyond a C++ compiler, CMake and (optionally)
Ninja.

```bash
git clone --recurse-submodules https://github.com/Guepard-Corp/duckdb-pql
cd duckdb-pql
GEN=ninja make            # build/release/duckdb has PQL linked in
make test                 # the SQL tests under test/sql
```

The build also produces `build/release/extension/pql/pql.duckdb_extension`,
which any DuckDB of the same version can load:

```sql
LOAD 'build/release/extension/pql/pql.duckdb_extension';
```

The language itself is tested without DuckDB. Each file under
[`test/cpp`](../test/cpp) is a standalone program against
[`pql.hpp`](../src/include/pql/pql.hpp):

```bash
make test_cpp             # parse, model, categorical, links, degenerate, fuzz
make test_cpp_sanitize    # the fuzzer and the degenerate harness under ASan/UBSan
```

| file | what it covers |
|---|---|
| `test_parse.cpp` | grammar: every clause, every error, statements round-trip |
| `test_model.cpp` | end to end on synthetic data, including the leakage canary |
| `test_categorical.cpp` | text columns: vocabularies, unseen categories at prediction time |
| `test_links.cpp` | link tables and windows against a brute-force oracle over random schemas |
| `test_degenerate.cpp` | databases that should not exist: empty, one row, duplicate keys, NaN, infinity, cancellation |
| `test_fuzz.cpp` | mutated statements and garbage, checking that anything accepted round-trips |
| `test_memory.cpp` | allocation discipline (see below) |

PQL was developed inside a fork of DuckDB, where the same header is compiled
in-tree as part of `core_functions`. That fork, with its own tests and the
history of the language, is the `duckdb-fork` branch of this repository.

`test_memory.cpp` is not part of `make test_cpp`. It counts heap allocations
per training step and currently fails its section A: training still allocates
a handful of vectors per step. That is a performance goal, not a correctness
one, and it is tracked as work in progress.

---
