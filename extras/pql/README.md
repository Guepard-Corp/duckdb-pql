# PQL

The PQL documentation lives at the repository root: [../../README.md](../../README.md).

This directory holds the implementation and its examples.

| path | what it is |
|---|---|
| [`src/pql.hpp`](src/pql.hpp) | the whole language: grammar, parser, features, model, training. One self-contained header that includes no DuckDB header. |
| [`examples/tutorial.sql`](examples/tutorial.sql) | the dataset Part 1 of the root README walks through |
| [`examples/shop_demo.sql`](examples/shop_demo.sql) | self-contained demo; creates its own tables and trains three models |
| [`examples/sunnyside_demo.sql`](examples/sunnyside_demo.sql) | demo against a real point-of-sale database |
| [`examples/sunnyside_export.sh`](examples/sunnyside_export.sh) | copies that schema out of Postgres |
| [`examples/sunnyside_load.sql`](examples/sunnyside_load.sql) | loads the export into DuckDB |
| [`test_parse.cpp`](test_parse.cpp) | grammar tests, compile standalone with `clang++ -std=c++17` |
| [`test_model.cpp`](test_model.cpp) | end-to-end model tests including the leakage canary |
| [`test_fuzz.cpp`](test_fuzz.cpp) | parser fuzzer: mutated statements and garbage, checking that anything accepted round-trips |
| [`test_degenerate.cpp`](test_degenerate.cpp) | databases that should not exist: empty, one row, duplicate keys, NaN, infinity |
| [`test_memory.cpp`](test_memory.cpp) | memory discipline: zero heap allocations per training step, the arena sized exactly by its own carve, prediction allocating per call not per row, a trained model keeping none of its data |
| [`console/server.py`](console/server.py) | a small local browser console |

The DuckDB binding is in
[`../../extension/core_functions/pql/pql_functions.cpp`](../../extension/core_functions/pql/pql_functions.cpp),
and the SQL-level tests are in
[`../../test/sql/pql/`](../../test/sql/pql/).

Running the standalone tests needs no DuckDB build:

```bash
clang++ -std=c++17 -O2 -o /tmp/pt test_parse.cpp && /tmp/pt
clang++ -std=c++17 -O2 -o /tmp/pm test_model.cpp && /tmp/pm
```

The fuzzer and the degenerate-data harness are worth running under sanitizers,
which is how most of what they now guard was found:

```bash
clang++ -std=c++17 -O1 -g -I. -fsanitize=address,undefined -o /tmp/tf test_fuzz.cpp && /tmp/tf
clang++ -std=c++17 -O1 -g -I. -fsanitize=address,undefined -o /tmp/td test_degenerate.cpp && /tmp/td
```
