# DuckDB + PQL

This is a fork of [DuckDB](README_DUCKDB.md) that adds **PQL**, a predictive
query language: two SQL statements that train a model from the tables you
already have and ask it questions. Upstream DuckDB's own README is preserved at
[README_DUCKDB.md](README_DUCKDB.md).

## Build and run

```bash
make reldebug                                  # builds build/reldebug/duckdb
build/reldebug/duckdb                          # PQL is compiled in, nothing to load

# try it on a self-contained demo
build/reldebug/duckdb -c ".read extras/pql/examples/shop_demo.sql"

# tests
build/reldebug/test/unittest test/sql/pql/pql_basic.test
build/reldebug/test/unittest test/sql/pql/pql_guards.test
```

Models live for the life of a session, so keep a `TRAIN` and the `PREDICT` that
uses it in the same connection.

---

PQL extends DuckDB's SQL with two statements: one that trains a model from the
tables you already have, and one that asks it a question. There is no feature
engineering step, no export, and no separate training service. The database is
the input.

```sql
TRAIN MODEL churn
  PREDICT EXISTS(orders) FOR customers AS c
  AT c.last_seen HORIZON 90 DAYS;

PREDICT EXISTS(orders) FOR customers WHERE region = 'EU' USING MODEL churn;
```

The implementation is one self-contained header, [`extras/pql/src/pql.hpp`](extras/pql/src/pql.hpp),
in the style of `extras/qfp/src/qfp.hpp` in the sibling CJIT fork: it includes no DuckDB header and reads
`DataChunk` buffers directly. The binding that adapts DuckDB onto it lives in
[`extension/core_functions/pql/pql_functions.cpp`](extension/core_functions/pql/pql_functions.cpp).

---

## The two questions PQL can answer

The distinction matters more than any other part of the syntax, because the two
are different problems with different failure modes.

**Attribute prediction** fills in a column that already exists on the entity
row. It is imputation, and it has no time dimension.

```sql
TRAIN MODEL grade PREDICT students.final_score FOR students;
```

**Forecasting** asks what will happen next. The target is an aggregate over rows
of a *related* table that fall strictly after an anchor time, so the label is
computed by the engine rather than read from a column.

Every forecast needs an anchor: a moment to stand at, with the past visible and
the future not. Where that moment comes from is the most important decision you
make, so it has its own section below.

---

## Anchors: the part that trips everyone up

A model learns from **examples**, and every example needs two halves: what was
known at a moment, and what actually happened after it. Concretely, for one
product:

| standing on | sales so far (the input) | sales in next 30d (the answer) |
|---|---|---|
| 2025-12-01 | 199 | 99 |
| 2026-01-01 | 304 | 149 |
| 2026-02-01 | 464 | 161 |
| 2026-03-01 | 612 | 224 |
| 2026-04-01 | 839 | 253 |

Five examples. The model trains on the early rows and is scored on the later
ones, which is what makes the score honest: standing on 1 April it has genuinely
never seen what followed.

**Some tables carry a usable moment and some do not.** A `products` row is a
catalog entry: a name, a category, a sku. There is no moment in it, so it cannot
produce a single example. Its `first_seen_at` and `last_seen_at` do not help:
`last_seen_at` is *defined* as the last sale, so the 30 days after it are empty
by construction, and `first_seen_at` leaves exactly one prior transaction to
learn from.

PQL gives you two ways to supply the missing moment.

### `EVERY`: let PQL generate the grid

```sql
TRAIN MODEL demand PREDICT COUNT(line_items) FOR products
  EVERY 1 MONTH HORIZON 30 DAYS;
```

Nothing to build. PQL derives the grid from the data's own span, starting one
horizon in so the first example has history and stopping one horizon short so no
label is censored, then makes one example per (product, date). It also dates
`line_items` automatically by following the foreign key to `transactions`, since
line items carry no clock of their own.

Use this when you want an answer without preparing anything.

### A panel: choose the moments yourself

```sql
CREATE TABLE snapshots AS               -- the questions you want to ask
SELECT row_number() OVER () AS id, p.product_id, d.as_of
FROM products p
CROSS JOIN (SELECT unnest([TIMESTAMP '2025-12-01', TIMESTAMP '2026-01-01',
                           TIMESTAMP '2026-02-01', TIMESTAMP '2026-03-01',
                           TIMESTAMP '2026-04-01']) AS as_of) d;

CREATE TABLE snap_sales AS              -- that product's sales, per snapshot
SELECT row_number() OVER () AS id, s.id AS snapshot_id, t.occurred_at, li.qty
FROM snapshots s
JOIN line_items li ON li.product_id = s.product_id
JOIN transactions t ON t.transaction_id = li.transaction_id;

TRAIN MODEL demand PREDICT COUNT(snap_sales) FOR snapshots
  AT as_of HORIZON 30 DAYS;
```

`snapshots` is **not data**. It is a list of questions built from your data:
"product P, on date D". The answers are computed from the fact table each time
and never stored. `snap_sales` re-attaches each product's sales to each of its
snapshots, which is why it is larger than the table it came from.

Use this when the moments matter: month ends, a promotion calendar, the dates
your business actually reports on.

### Which to use

Measured on the same point-of-sale database, same model settings:

| approach | examples | test MAE | its persistence baseline |
|---|---|---|---|
| panel, five winter dates | 475 | **9.30** | 9.82 |
| `EVERY 1 MONTH` over the full span | 456 | 20.34 | 18.55 |

The panel wins here, but read the baselines before concluding anything: 9.82
against 18.55 says these are *different questions*, and the automatic one is
much harder. Five winter dates is a narrower, more predictable slice than the
whole 2025-2026 range. Compare a model only against the baseline on its own row.

Start with `EVERY` because it costs nothing. Move to a panel when you know which
moments you care about, or when `EVERY` cannot beat its baseline.

---

Features come from `(-inf, anchor]` and labels from `(anchor, anchor + horizon]`.
The intervals are disjoint by construction, so a forecast cannot see its own
answer.

**Read this before trusting that sentence.** The guarantee covers the *child*
tables PQL aggregates. Two things sit outside it, and PQL now defends both:

- A child table with no time column would be visible in every window, which
  would make `HORIZON` a no-op and collapse the label into a lifetime total.
  PQL refuses such a forecast rather than answering it.
- The entity's own row is read as it stands *now*, not as it stood at the
  anchor. A column like `last_seen_at` or `closed_at` records the future.
  PQL therefore excludes every timestamp column on the entity except the
  anchor itself. Non-temporal columns that are updated in place (a `status`
  flag, a running `lifetime_value`) are still your responsibility.

**Attribute prediction has no leakage guarantee at all.** Every other numeric
column on the same row is a feature, including ones that are consequences of the
target. Predicting `line_items.voided` scores a perfect 1.0 because
`net_sales = 0` exactly when `voided` is true. A near-perfect attribute score
usually means a same-row tautology, not a good model.

---

## Syntax

```
TRAIN MODEL <name>
  PREDICT <target>
  FOR <table> [AS <alias>]
  [ WHERE <filter> ]                        -- which entities to train on
  [ AT <column> | EVERY <n> <unit> ]        -- anchor time: read it, or generate it
  [ HORIZON <n> <unit> ]                    -- how far ahead to look
  [ USING GRAPH (<table>, ...) ]            -- restrict the graph
  [ SPLIT TEMPORAL VALIDATE FROM <ts> TEST FROM <ts> ]
  [ OPTIONS (<key> = <value>, ...) ]

PREDICT <target>
  FOR <table> [AS <alias>]
  [ WHERE <filter> ]
  USING MODEL <name>

BACKTEST MODEL <name>
  [ FOR <table> ]                           -- defaults to the model's entity
  [ WHERE <filter> ]
  [ FROM <ts> ] [ TO <ts> ]

DROP MODEL <name>
```

`PREDICT` and `FOR` come first and in that order. The optional clauses after
`FOR` may appear in any order; repeating one is an error rather than a silent
overwrite.

`PREDICT` inherits `AT` and `HORIZON` from the model, so you normally omit
them. Restating them is allowed but must match what the model was trained with;
a mismatch is an error, because a model that was fitted to a 30-day horizon
cannot answer a 365-day question.

### Targets

| form | meaning | task |
|---|---|---|
| `<table>.<column>` | impute this column | classification if the column is 0/1, else regression |
| `COUNT(<table>)` | how many related rows land in the horizon | regression |
| `EXISTS(<table>)` | will there be any at all | classification |
| `SUM(<table>.<col>)` | total over the horizon | regression |
| `AVG` / `MIN` / `MAX` | the corresponding aggregate | regression |

`COUNT(*)` counts the linked table's rows. Text columns cannot be targets: PQL
predicts numeric and boolean attributes only, and says so rather than regressing
dictionary codes.

### Horizon units

`DAY`, `WEEK`, `MONTH`, `YEAR`, `HOUR`, singular or plural, and strictly
positive. Months and years use calendar-average lengths (30.436875 and 365.2425
days), so `HORIZON 1 MONTH` is not a calendar month; say it in days if that
matters.

### Filters

Filters appear in two positions and they mean different things.

The **outer** `WHERE` chooses which entities take part:

```sql
PREDICT EXISTS(orders) FOR customers WHERE region = 'EU' AND tier >= 2
  USING MODEL churn;
```

The **inner** `WHERE`, inside the aggregate, narrows what counts toward the
label:

```sql
TRAIN MODEL refunds
  PREDICT EXISTS(orders WHERE orders.status = 'refunded')
  FOR customers AT last_seen HORIZON 60 DAYS;
```

Same customers, different question. Operators are `=`, `!=`, `<>`, `<`, `<=`,
`>`, `>=`, `IN (...)`, `NOT IN (...)` and `IS [NOT] NULL`, combined with `AND`,
`OR`, `NOT` and parentheses. Comparisons are column-against-literal.

A qualifier must name the relation the filter is evaluated on: the entity (or
its alias) for the outer `WHERE`, the aggregated table for the inner one.
Naming anything else is an error, not a silent match against whatever was to
hand.

Quoted dates work wherever a timestamp is expected, in `SPLIT`, in `AT` and in
a comparison against a timestamp column:

```sql
PREDICT COUNT(sales) FOR snapshots WHERE as_of = '2026-04-01' USING MODEL demand;
```

### Options

| option | default | range |
|---|---|---|
| `EPOCHS` | 60 | 1 to 100000 |
| `HIDDEN` | 64 | 1 to 1024 |
| `LR` | 0.01 | 1e-6 to 1 |
| `BATCH` | 64 | 1 to 65536 |
| `MEAN_COLS` | 3 | 0 to 64 |
| `SEED` | 42 | any |
| `ARCH` | `mlp` | `mlp` or `sage` |
| `LAYERS` | 1 | 1 or 2 (`sage` only) |

An unknown option key is an error. A silently ignored `epocs = 500` is the most
expensive typo in machine learning.

---

## Reading the result

`TRAIN` returns one row:

```
┌────────┬───────────────────┬────────────┬──────────┬───────────┬───────────┬────────┬───────┬───────┬──────────┬──────────┬────────┐
│ model  │      target       │ train_rows │ val_rows │ test_rows │ positives │ metric │  val  │ test  │ baseline │ censored │ epochs │
├────────┼───────────────────┼────────────┼──────────┼───────────┼───────────┼────────┼───────┼───────┼──────────┼──────────┼────────┤
│ demand │ COUNT(snap_sales) │        285 │       95 │        95 │         0 │ mae    │ 10.24 │ 9.394 │    9.821 │        0 │    120 │
└────────┴───────────────────┴────────────┴──────────┴───────────┴───────────┴────────┴───────┴───────┴──────────┴──────────┴────────┘
```

- **`test`** is the honest number: scored once, on a later split that model
  selection never saw.
- **`val`** is the *best* epoch by that metric, so it is optimistic by
  construction, typically by 15-25%. It is a selection statistic, not an
  estimate. Do not quote it.
- **`baseline`** is persistence: repeat the previous window's value. For a
  forecast this is the number to beat, and it is usually much harder to beat
  than predicting the mean. Above, 9.394 versus 9.821 is a real but modest win.
- **`positives`** explains a NULL metric. AUROC is undefined when a fold has one
  class, and PQL reports NULL rather than a flattering 0.5.
- **`censored`** counts examples dropped because their label window ran past the
  end of the data. A long horizon on a short table silently scores against
  labels that do not exist yet; those rows are removed and counted instead.

Splits are temporal. Without `SPLIT`, the latest 40% by anchor is held out as
validation and test. Run several seeds before believing a difference: on the
demand example the seed-to-seed spread is about 1 MAE.

`PREDICT` returns the entity key in its own type, the anchor it predicted from,
and the prediction: a probability for classification, an expected value
otherwise. Counts are never negative.

```
┌───────┬─────────────────────┬────────────┐
│  id   │        as_of        │ prediction │
├───────┼─────────────────────┼────────────┤
│   321 │ 2025-12-01 00:00:00 │     112.29 │
│   325 │ 2026-04-01 00:00:00 │     234.27 │
└───────┴─────────────────────┴────────────┘
```

The anchor is part of the answer, not decoration: a panel holds one row per
(entity, date), so a filter like `WHERE product_id = 133` returns one prediction
per as-of date and the column is what tells them apart. Attribute prediction has
no anchor, so the column is omitted there.

### Backtesting

`PREDICT` answers "what happens next". `BACKTEST` replays the model over its
anchors and pairs every prediction with the outcome that actually followed:

```sql
BACKTEST MODEL daily WHERE product_id = 133 FROM '2026-04-20';
```

```
┌────────────┬─────────────────────┬───────────┬────────┬────────┬──────────┐
│ product_id │       anchor        │ predicted │ actual │ error  │ baseline │
├────────────┼─────────────────────┼───────────┼────────┼────────┼──────────┤
│        133 │ 2026-05-17 15:45:25 │      9.58 │   10.0 │  -0.42 │     10.0 │
│        133 │ 2026-05-13 15:45:25 │      2.95 │   13.0 │ -10.05 │      3.0 │
└────────────┴─────────────────────┴───────────┴────────┴────────┴──────────┘
```

`baseline` is persistence on that same row, so you can see day by day whether the
model actually beat "repeat last time". Aggregate over the window to compare:

```sql
SELECT count(*) AS n,
       round(avg(abs(error)), 4) AS model_mae,
       round(avg(abs(baseline - actual)), 4) AS persistence_mae
FROM pql_exec('BACKTEST MODEL daily FROM ''2026-04-20''');
```

It reuses the machinery training uses, keeping the labels instead of discarding
them, so a backtest over the test window reproduces the `test` column of `TRAIN`
exactly. That agreement is the point: two independent paths to the same number
is what catches a mistake in either.

### Composing with SQL

```sql
SELECT s.product_id, round(p.prediction, 1) AS next_30d
FROM pql_exec('PREDICT COUNT(snap_sales) FOR snapshots USING MODEL demand') p
JOIN snapshots s ON s.id = p.id
ORDER BY p.prediction DESC LIMIT 5;

SELECT * FROM pql_models();
```

`SHOW MODELS` cannot be a bare statement, because DuckDB's own `SHOW` grammar
accepts it before the extension is consulted. Use `pql_models()`.

---

## Worked example

See [`extras/pql/examples/sunnyside_demo.sql`](extras/pql/examples/sunnyside_demo.sql), which runs
end to end against a real point-of-sale database, and
[`extras/pql/examples/shop_demo.sql`](extras/pql/examples/shop_demo.sql) for a self-contained one.

A product has no time to anchor on, so demand forecasting needs a panel: one row
per (product, as-of date). That is the same construction RelBench uses.

```sql
CREATE TABLE snapshots AS
SELECT row_number() OVER () AS id, p.product_id, d.as_of
FROM products p CROSS JOIN (SELECT unnest([...]) AS as_of) d;

CREATE TABLE snap_sales AS
SELECT row_number() OVER () AS id, s.id AS snapshot_id, t.occurred_at, li.qty
FROM snapshots s JOIN line_items li ON li.product_id = s.product_id
JOIN transactions t ON t.transaction_id = li.transaction_id;

TRAIN MODEL demand PREDICT COUNT(snap_sales) FOR snapshots
  AT as_of HORIZON 30 DAYS OPTIONS (epochs = 120, hidden = 48);
```

Note the join in `snap_sales`: `line_items` has no timestamp of its own, so
`transactions.occurred_at` is brought along. Without it the table is undated and
PQL refuses the forecast.

---

## How it works

**Foreign keys** come from declared constraints when present, and otherwise from
the `<table>_id` convention, matching the parent's own key column
(`products.product_id`) or a plain `id`. Plurals are handled, including
`categories` to `category`. Keys are matched on their values: text keys compare
as text, and a text-to-numeric key pair is refused rather than compared as
dictionary codes.

**Only what matters is read.** The catalog is consulted first, so the loader
knows the schema before touching data, then reads only the entity, its children
and the target, and within those only the columns that can influence the model.
Adding 2.5M rows of unrelated tables leaves training time unchanged.

**Features** are compiled temporal aggregates: per foreign-key link and per
window of 7/30/90/365/all days, a count, a recency, and means of the most
informative child columns, computed exactly by prefix sums over time-sorted
children.

**Two architectures** are available, chosen with `OPTIONS (arch = ...)`.

`arch = 'mlp'` (the default) runs a two-layer network over the entity's own
columns plus the compiled aggregates above. Small and fast: 23 parameters on the
worked example.

`arch = 'sage'` is a heterogeneous GraphSAGE, transposed from RelML's
`HeteroGraphSAGE`: one `W_self` per node type, one `W_neigh` per edge type, mean
aggregation, ReLU, linear head.

```
h_v = ReLU( W_self . x_v  +  SUM_r W_r . [ mean_{u in N_r(v)} x_u ; log1p(deg_r(v)) ] )
```

Two things differ from the original, both deliberate.

*Aggregation is exact and temporal.* RelML has no temporal neighbour sampling,
which is why its own benchmark has to keep fact tables out of the graph or a
2010 row message-passes into a 2015 prediction. Here the CSR arena is sorted by
time, so the neighbours visible at an anchor are a prefix of a slice: a running
sum along that slice makes the mean of any prefix a subtraction of two rows.
Nothing after the anchor can enter, and the same property makes it O(C) per node
instead of O(degree * C).

*Degree rides along with the mean.* A mean is scale-stable but says nothing
about how many neighbours produced it, and on relational data the count is
usually the signal. Measured: mean alone scored 0.541 on a task whose ceiling is
0.833; mean plus degree scored 0.755. RelBench's RDL sums instead, which keeps
degree implicitly; carrying both keeps mean's conditioning as well.

`OPTIONS (layers = 2)` adds a second hop. A child's embedding is computed at the
child's *own* timestamp, which is <= the anchor by construction, so it stays
leak-free and, being anchor-independent, is computed once per epoch rather than
per example. On a task where the signal sits two hops out and one hop provably
cannot reach it, one hop scores 0.470 and two score 0.860.

**Which to use.** On the worked example, `sage` beat `mlp` on all five seeds
(mean 8.99 vs 9.47 MAE, persistence 9.82) and with a much tighter spread, at the
cost of 337 parameters against 23. `mlp` remains the default because it is
cheaper and its worst case is easier to reason about. For a forecast it predicts the *residual* over persistence rather
than the level, because a network cannot cheaply rediscover the identity
function from log-compressed counts; handing it the level as an offset was worth
more than any feature (15.55 to 9.39 MAE on the demand example). Regression uses
a Huber objective to match the reported MAE, since squared loss fits the mean
and MAE rewards the median.

**Performance.** Children live in a CSR arena, one flat array per link with
contiguous per-parent slices. Buckets are checked for existing time order before
sorting, which is free for append-ordered fact tables, and radix-sorted
otherwise. Training scratch is carved from a single arena block sized up front.
The dense layers run as batched GEMMs whose reductions vectorize through
independent accumulators, without `-ffast-math`.

**Models** live in the database's `ObjectCache`, so they are scoped to the
database and released with it. A long `TRAIN` honours `max_execution_time` and
Ctrl-C.

---

## What PQL refuses

Each of these was once accepted and silently produced a meaningless model:

- a forecast over a child table with no time column;
- a text column as a prediction target;
- an entity with no usable features;
- an entity whose key is not unique (a panel needs a synthetic id);
- `HORIZON 0`, an unknown `OPTIONS` key, an out-of-range option;
- `USING MODEL` in a `TRAIN` statement, or `SPLIT` / `OPTIONS` / `USING GRAPH`
  in a `PREDICT`;
- a filter qualifier naming a relation the filter is not evaluated on;
- a `PREDICT` whose restated target or horizon disagrees with the model;
- a `SPLIT` whose boundaries leave no training rows.

---

## Limits

- Filters compare a column to a literal; no column-to-column comparisons,
  arithmetic, functions or subqueries, and no `BETWEEN` or `LIKE`.
- `FOR` takes a bare table name, not a subquery or a schema-qualified name, so
  a panel must be materialized first.
- Composite foreign keys are not modelled.
- Models do not survive the process and are not transactional: a `ROLLBACK` will
  not remove one.
- `TRAIN` runs in the bind phase on its own connection, so it does not see
  uncommitted data from the calling transaction, and `EXPLAIN` on a `TRAIN`
  still trains.
- Work is single-threaded and outside the buffer manager, so `memory_limit` does
  not apply and a table larger than RAM has no spill path.
- Each statement reloads its tables; a `TRAIN` followed by five `PREDICT`s pays
  the load six times.
- `TIMESTAMP WITH TIME ZONE` is read through the session time zone, so a model
  trained in one session and used in another with a different zone has shifted
  anchors.
