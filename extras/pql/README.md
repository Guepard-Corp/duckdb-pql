# PQL

The PQL documentation lives at the repository root: [../../README.md](../../README.md).

This directory holds the implementation and its examples.

| path | what it is |
|---|---|
| [`src/pql.hpp`](src/pql.hpp) | the whole language: grammar, parser, features, model, training. One self-contained header that includes no DuckDB header. |
| [`examples/shop_demo.sql`](examples/shop_demo.sql) | self-contained demo; creates its own tables and trains three models |
| [`examples/sunnyside_demo.sql`](examples/sunnyside_demo.sql) | demo against a real point-of-sale database |
| [`examples/sunnyside_export.sh`](examples/sunnyside_export.sh) | copies that schema out of Postgres |
| [`examples/sunnyside_load.sql`](examples/sunnyside_load.sql) | loads the export into DuckDB |
| [`test_parse.cpp`](test_parse.cpp) | grammar tests, compile standalone with `clang++ -std=c++17` |
| [`test_model.cpp`](test_model.cpp) | end-to-end model tests including the leakage canary |
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
