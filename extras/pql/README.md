# PQL

The PQL documentation lives at the repository root: [../../README.md](../../README.md).

The implementation is in-tree, in
[`extension/core_functions/pql/`](../../extension/core_functions/pql/):

| path | what it is |
|---|---|
| [`pql.hpp`](../../extension/core_functions/pql/pql.hpp) | the whole language: grammar, parser, features, model, training. One self-contained header that includes no DuckDB header. |
| [`pql_functions.cpp`](../../extension/core_functions/pql/pql_functions.cpp) | the DuckDB binding: the `pql_exec` and `pql_models` table functions and the parser extension |

The SQL-level tests are in [`test/sql/pql/`](../../test/sql/pql/) and run with
the rest of the suite: `build/reldebug/test/unittest "test/sql/pql/*"`.

This directory holds what is not part of the DuckDB build:

| path | what it is |
|---|---|
| [`tests/`](tests/) | standalone programs against `pql.hpp`; `make test` builds and runs them, `make sanitize` runs the fuzzer, the degenerate-data harness and the link oracle under ASan/UBSan |
| [`examples/tutorial.sql`](examples/tutorial.sql) | the dataset Part 1 of the root README walks through |
| [`examples/shop_demo.sql`](examples/shop_demo.sql) | self-contained demo; creates its own tables and trains three models |
| [`examples/sunnyside_demo.sql`](examples/sunnyside_demo.sql) | demo against a real point-of-sale database, brought over with [`sunnyside_export.sh`](examples/sunnyside_export.sh) and [`sunnyside_load.sql`](examples/sunnyside_load.sql) |
| [`console/server.py`](console/server.py) | a small local browser console |

| test | what it covers |
|---|---|
| `test_parse.cpp` | grammar: every clause, every error, statements round-trip |
| `test_model.cpp` | end to end on synthetic data, including the leakage canary |
| `test_categorical.cpp` | text columns: vocabularies, unseen categories at prediction time |
| `test_links.cpp` | link tables and windows against a brute-force oracle over random schemas |
| `test_degenerate.cpp` | databases that should not exist: empty, one row, duplicate keys, NaN, infinity, cancellation |
| `test_fuzz.cpp` | mutated statements and garbage, checking that anything accepted round-trips |
| `test_memory.cpp` | allocation discipline; not in `make test`, its section A is a known failure (training still allocates per step) |

The same language ships on its own as a DuckDB community extension from this
repository's `main` branch; this branch is the in-tree fork it was developed in.
