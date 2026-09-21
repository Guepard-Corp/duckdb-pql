# Tests

`sql/` holds [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html)
that exercise PQL through DuckDB: the parser extension, `pql_exec`,
`pql_models`, error paths, quoting, schema changes and keys. Run them with
`make test` after `make`.

`cpp/` holds standalone programs against `src/include/pql/pql.hpp` that need
no DuckDB build. Run them with `make test_cpp`, and the fuzzer and the
degenerate-data harness under sanitizers with `make test_cpp_sanitize`.
