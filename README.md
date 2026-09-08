# DuckDB + PQL

A fork of [DuckDB](README_DUCKDB.md) that adds **PQL**: predictive queries as
first-class SQL. You train models on the tables you already have, and ask them
questions the same way you ask anything else.

```sql
TRAIN MODEL churn PREDICT EXISTS(orders) FOR customers
  EVERY 1 WEEK HORIZON 30 DAYS;

PREDICT EXISTS(orders) FOR customers WHERE region = 'EU' USING MODEL churn;
```

No feature table to build, no export, no training service, no separate
vocabulary. The database is the input, SQL is the interface, and the model is an
object you can name, use and drop.

```bash
make reldebug
build/reldebug/duckdb -c ".read extras/pql/examples/shop_demo.sql"
```

---

# Part 1 — A tutorial

Work through this and you will know PQL. Every query below is real and runs.

## 1.1 Your first model

Suppose you have customers, and some of them churned. You want to predict which
others will.

```sql
TRAIN MODEL churn PREDICT customers.churned FOR customers;
```

That is the whole statement. PQL reads the catalog, finds that `orders` points
at `customers` through a foreign key, and builds features from both: the
customer's own columns, plus counts, averages and recency of their orders. Then
it trains, holds out a slice, and reports.

```
┌────────────┬───────────┬────────┬───────┬──────────┐
│ train_rows │ test_rows │ metric │ test  │ features │
├────────────┼───────────┼────────┼───────┼──────────┤
│        540 │       180 │ auroc  │ 0.615 │       17 │
└────────────┴───────────┴────────┴───────┴──────────┘
```

Seventeen features you did not have to write, from a statement that names no
features at all. 0.615 AUROC is a modest model, which is the honest result on
data where churn is only partly explained by order history — and it is the sort
of number you should expect to see, not 0.99. Ask it about anyone:

```sql
PREDICT customers.churned FOR customers WHERE tier >= 5 USING MODEL churn;
```

**This needs no dates.** Predicting a column that already exists is imputation,
and imputation has no time dimension. If your schema has no timestamps at all,
this still works.

## 1.2 The idea that makes forecasting different

Now a harder question: *will this customer order again in the next 30 days?*

That is not a column anyone filled in. It is a fact about the future, and to
learn it you need examples of the shape **"here is what was known at a moment,
and here is what happened after it."**

Take one product and look at what that means concretely:

| standing on | sales so far (input) | sales in next 30d (answer) |
|---|---|---|
| 2025-12-01 | 199 | 99 |
| 2026-01-01 | 304 | 149 |
| 2026-02-01 | 464 | 161 |
| 2026-03-01 | 612 | 224 |
| 2026-04-01 | 839 | 253 |

Five training examples from one product. Each is a question with a known answer.
The moment you stand at is the **anchor**, and everything in PQL's forecasting
follows from it: features come from at-or-before the anchor, the label comes
from strictly after it.

## 1.3 Forecasting

Say it directly and PQL builds those examples for you:

```sql
TRAIN MODEL demand PREDICT COUNT(line_items) FOR products
  EVERY 1 MONTH HORIZON 30 DAYS;
```

`EVERY 1 MONTH` generates the anchors from the data's own span. `HORIZON 30 DAYS`
says how far ahead to look. 95 products become 570 training examples.

The target can be any aggregate over a related table:

| you write | you get |
|---|---|
| `COUNT(orders)` | how many orders in the window |
| `EXISTS(orders)` | will there be any at all (a classifier) |
| `SUM(orders.amount)` | how much they will spend |
| `AVG` / `MIN` / `MAX` | the corresponding statistic |

And you can narrow what counts:

```sql
TRAIN MODEL refunds
  PREDICT EXISTS(orders WHERE orders.status = 'refunded')
  FOR customers EVERY 1 WEEK HORIZON 60 DAYS;
```

Same customers, different question. That inner `WHERE` is the label definition;
the outer one picks which customers take part.

## 1.4 Reading the answer honestly

Every `TRAIN` prints the number you have to beat:

```
│ metric │  test  │ baseline │
│  mae   │ 1.0616 │  1.1387  │
```

**`baseline` is persistence**: just repeat what happened last window. For
forecasting it is a genuinely hard opponent and a much better yardstick than
"predict the average". Above, the model wins by 7%.

`test` is scored once, on a later slice model selection never saw. `val` is the
best epoch by that metric and is optimistic by construction — a selection
statistic, not an estimate. Do not quote it.

## 1.5 Backtesting

Aggregate metrics hide things. Replay the model and see it day by day:

```sql
BACKTEST MODEL demand WHERE product_id = 133 FROM '2026-04-20';
```

```
┌────────────┬─────────────────────┬───────────┬────────┬────────┬──────────┐
│ product_id │       anchor        │ predicted │ actual │ error  │ baseline │
├────────────┼─────────────────────┼───────────┼────────┼────────┼──────────┤
│        133 │ 2026-05-17 15:45:25 │      9.58 │   10.0 │  -0.42 │     10.0 │
│        133 │ 2026-05-13 15:45:25 │      2.95 │   13.0 │ -10.05 │      3.0 │
└────────────┴─────────────────────┴───────────┴────────┴────────┴──────────┘
```

Now you can see the character of the model, not just its average: it tracks
quiet days closely and underestimates spikes. Aggregate it to compare properly:

```sql
SELECT count(*) AS n,
       round(avg(abs(error)), 4)              AS model_mae,
       round(avg(abs(baseline - actual)), 4)  AS persistence_mae
FROM pql_exec('BACKTEST MODEL demand FROM ''2026-04-20''');
```

A backtest over the test window reproduces `TRAIN`'s `test` column exactly. Two
independent paths agreeing to four decimals is what tells you both are right.

## 1.6 It is all still SQL

```sql
SELECT p.product_id, round(p.prediction, 1) AS next_30d
FROM pql_exec('PREDICT COUNT(line_items) FOR products USING MODEL demand') p
JOIN products p2 ON p2.product_id = p.product_id
ORDER BY p.prediction DESC LIMIT 10;

SELECT * FROM pql_models();
DROP MODEL demand;
```

---

# Part 2 — Reference

## Syntax

```
TRAIN MODEL <name>
  PREDICT <target>
  FOR <table> [AS <alias>]
  [ WHERE <filter> ]                        -- which entities take part
  [ AT <column> | EVERY <n> <unit> ]        -- anchor: read one, or generate them
  [ HORIZON <n> <unit> ]                    -- how far ahead
  [ USING GRAPH (<table>, ...) ]            -- restrict which tables are used
  [ SPLIT TEMPORAL VALIDATE FROM <ts> TEST FROM <ts> ]
  [ OPTIONS (<key> = <value>, ...) ]

PREDICT <target> FOR <table> [AS <alias>] [ WHERE <filter> ] USING MODEL <name>

BACKTEST MODEL <name> [ FOR <table> ] [ WHERE <filter> ] [ FROM <ts> ] [ TO <ts> ]

DROP MODEL <name>
```

`PREDICT` and `FOR` come first; the optional clauses may follow in any order.
`PREDICT` inherits `AT` and `HORIZON` from the model.

### Anchors: `AT` or `EVERY`

Use `AT <column>` when your rows already carry the moment they are about — a
snapshot table, an events table, anything with a meaningful timestamp per row.

Use `EVERY <n> <unit>` when they do not. A `products` row is a catalog entry
with no moment in it, so PQL generates a grid across the data's span: starting
one horizon in, so the first example has history, and stopping one horizon
short, so no label is cut off by the end of the data.

### Filters

`=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`, `IN (...)`, `NOT IN (...)`,
`IS [NOT] NULL`, combined with `AND`, `OR`, `NOT` and parentheses. Quoted dates
work wherever a timestamp belongs:

```sql
PREDICT COUNT(sales) FOR products WHERE as_of = '2026-04-01' USING MODEL demand;
```

### Options

| option | default | range |
|---|---|---|
| `EPOCHS` | 60 | 1 to 100000 |
| `HIDDEN` | 64 | 1 to 1024 |
| `LR` | 0.01 | 1e-6 to 1 |
| `BATCH` | 64 | 1 to 65536 |
| `SEED` | 42 | any |
| `MEAN_COLS` | 3 | 0 to 64 |
| `ARCH` | `mlp` | `mlp` or `sage` |
| `LAYERS` | 1 | 1 or 2 (`sage` only) |

### Result columns

| column | meaning |
|---|---|
| `test` | the honest number: scored once, on a later split |
| `baseline` | persistence — the number to beat |
| `val` | best epoch by the metric; optimistic, for selection only |
| `positives` | explains a NULL metric (AUROC is undefined on one class) |
| `censored` | examples dropped because their label window ran past the data |
| `no_anchor` | examples dropped for a NULL anchor |

---

# Part 3 — How it works

## Leakage is structural

Features are drawn from `(-inf, anchor]` and labels from
`(anchor, anchor + horizon]`. Disjoint by construction, so a forecast cannot see
its own answer.

PQL enforces the corners rather than trusting them. A forecast over a child
table with no clock is refused, because every row would fall inside every window
and the label would collapse into a lifetime total that is also a feature.
Timestamps on the entity other than the anchor are excluded, because a column
like `last_seen_at` records the future. Labels whose window runs past the end of
the data are dropped and counted in `censored`, rather than scored against
outcomes that have not happened yet.

The same discipline shows up in the errors. Each of these was once accepted and
produced a plausible-looking, meaningless model:

- a text column as a target (it was being regressed as a dictionary code);
- an entity with no usable features (a constant model);
- non-unique entity keys (children attributed to one arbitrary row);
- `HORIZON 0`, unknown or out-of-range `OPTIONS`;
- a filter qualifier naming a relation it is not evaluated on;
- a `SPLIT` whose boundaries leave no training rows.

## Features

Per foreign-key link and per window of 7/30/90/365/all days: a count, a recency,
and means of the most informative child columns — computed exactly by prefix
sums over time-sorted children, so a windowed statistic is two array lookups
rather than a scan.

Foreign keys come from declared constraints, and otherwise from the `<table>_id`
convention (including `categories` → `category`). Text keys match as text. If a
child table has no clock, PQL follows a foreign key to find one — `line_items`
dated by its `transaction`, for instance — so you rarely need to denormalize by
hand.

## Two architectures

`arch = 'mlp'` (default) runs a small dense network over the entity's columns
plus those aggregates. Cheap: 23 parameters on the worked example.

`arch = 'sage'` is a heterogeneous GraphSAGE, one `W_self` per node type and one
`W_neigh` per edge type:

```
h_v = ReLU( W_self . x_v  +  SUM_r W_r . [ mean_{u in N_r(v)} x_u ; log1p(deg_r(v)) ] )
```

Because the neighbour arena is sorted by time, the neighbours visible at an
anchor are a *prefix* of a slice, so a running sum makes the mean of any prefix
a subtraction of two rows. That is both the exact-temporal guarantee and an O(C)
aggregation instead of O(degree × C).

Degree travels with the mean because a mean alone cannot tell how many
neighbours produced it, and on relational data the count is usually the signal:
mean alone scored 0.541 on a task whose ceiling is 0.833, mean plus degree
scored 0.755.

`OPTIONS (layers = 2)` adds a second hop, with each child embedded at its own
timestamp. On a task where the signal sits two hops out and one hop provably
cannot reach it, one hop scores 0.470 and two score 0.860.

Measured on the worked example, `sage` beat `mlp` on all five seeds (8.99 vs
9.47 MAE, persistence 9.82) at 337 parameters against 23.

## Performance

The language lives in one self-contained header,
[`extras/pql/src/pql.hpp`](extras/pql/src/pql.hpp), which never includes a
DuckDB header.

Children live in a CSR arena — one flat array per link, contiguous per-parent
slices. Buckets already in time order skip sorting entirely; the rest are radix
sorted. Kernels are specialised on their shape and dispatched once, so trip
counts vanish and bodies unroll. Per-step buffers come from a single arena sized
up front from `(batch, channels, relation widths)`.

The loader consults the catalog first and reads only the tables and columns that
can influence the model, straight from `DataChunk` buffers with no per-cell
`Value`. Adding 2.5M rows of unrelated tables leaves training time unchanged.

---

# Part 4 — Worked examples

[`examples/shop_demo.sql`](extras/pql/examples/shop_demo.sql) is
self-contained: it creates its own tables and trains three models.

[`examples/sunnyside_demo.sql`](extras/pql/examples/sunnyside_demo.sql) runs
against a real point-of-sale database, with
[`sunnyside_export.sh`](extras/pql/examples/sunnyside_export.sh) and
[`sunnyside_load.sql`](extras/pql/examples/sunnyside_load.sql) to bring it over
from Postgres.

A daily forecast with the last 30 days held back:

```sql
TRAIN MODEL daily
  PREDICT COUNT(line_items) FOR products
  EVERY 1 DAY HORIZON 1 DAY
  SPLIT TEMPORAL VALIDATE FROM '2026-03-21' TEST FROM '2026-04-20'
  OPTIONS (epochs = 60, hidden = 48);

BACKTEST MODEL daily FROM '2026-04-20';
```

15,010 training rows, 2,755 held out, no tables built by hand.

---

# Part 5 — Where the edges are

Honest about what is not there yet, so you can plan around it.

**Language.** Filters compare a column to a literal — no column-to-column
comparisons, arithmetic, `BETWEEN` or `LIKE`. `FOR` takes a bare table name, not
a subquery or a schema-qualified name. Composite foreign keys are not modelled.
Text targets are refused rather than trained as multi-class.

**Attribute prediction has no leakage guarantee.** Every other numeric column on
the row is a feature, including ones that are consequences of the target.
Predicting `line_items.voided` scores a perfect 1.0 because `net_sales = 0`
exactly when voided. A near-perfect attribute score usually means a same-row
tautology; forecasting is the path with the structural guarantee.

**Execution.** `TRAIN` runs during binding on its own connection, so it does not
see uncommitted data and `EXPLAIN` on a `TRAIN` still trains. Work is
single-threaded and outside the buffer manager, so `memory_limit` does not apply.
Each statement reloads its tables. All four dissolve together by moving the work
into a physical operator with a sink, which is the main thing left to build.

**Models** live for the session: not persisted, not transactional.

**Time zones.** `TIMESTAMP WITH TIME ZONE` reads through the session zone, so a
model trained in one session and used in another with a different zone has
shifted anchors.
