# PQL: predictive queries for DuckDB

A DuckDB extension that adds **PQL**: predictive queries as first-class SQL. You train models on the tables you already have, and ask them
questions the same way you ask anything else.

```sql
TRAIN MODEL churn PREDICT EXISTS(orders) FOR customers
  EVERY 1 WEEK HORIZON 30 DAYS;

PREDICT EXISTS(orders) FOR customers WHERE region = 'EU' USING MODEL churn;
```

No feature table to build, no export, no training service, no separate
vocabulary. The database is the input, SQL is the interface, and the model is an
object you can name, use and drop.

```sql
INSTALL pql FROM community;
LOAD pql;
```

Or build it yourself and run the demo (see [Building](#building-and-testing)):

```bash
make
build/release/duckdb -c ".read examples/shop_demo.sql"
```

---

# Part 1 — A tutorial

Work through this and you will know PQL. Every query below is real and runs.

## 1.1 Your first model

Everything in Part 1 runs against one small dataset. Load it and follow along:

```bash
build/release/duckdb -c ".read examples/tutorial.sql"
```

It makes 1,500 `customers` and their `orders`. Some customers churned. You want
to predict which others will.

```sql
TRAIN MODEL churn PREDICT customers.churned FOR customers;
```

That is the whole statement. PQL reads the catalog, finds that `orders` points
at `customers` through a foreign key, and builds features from both: the
customer's own columns, plus counts, averages, recency and spacing of their
orders. Then it trains, holds out a slice, and reports.

```
┌────────────┬───────────┬────────┬───────┬──────────┐
│ train_rows │ test_rows │ metric │ test  │ features │
├────────────┼───────────┼────────┼───────┼──────────┤
│        900 │       300 │ auroc  │ 0.718 │       29 │
└────────────┴───────────┴────────┴───────┴──────────┘
```

Twenty-nine features you did not have to write, from a statement that names no
features at all. 0.718 AUROC is a modest model, which is the honest result on
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
TRAIN MODEL reorder PREDICT EXISTS(orders) FOR customers
  AT last_seen HORIZON 30 DAYS OPTIONS (epochs = 60);
```

`AT last_seen` says where each customer's anchor is. `HORIZON 30 DAYS` says how
far ahead to look. When your rows carry no such moment, `EVERY 1 MONTH` builds a
grid of anchors from the data's own span instead.

```
┌────────────┬───────────┬────────┬────────┬────────┬──────────┬──────────┐
│ train_rows │ test_rows │ metric │  test  │ pr_auc │ baseline │ features │
├────────────┼───────────┼────────┼────────┼────────┼──────────┼──────────┤
│        900 │       300 │ auroc  │ 0.8861 │ 0.8377 │    0.400 │       30 │
└────────────┴───────────┴────────┴────────┴────────┴──────────┴──────────┘
```

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
  FOR customers AT last_seen HORIZON 60 DAYS;
```

Same customers, different question. That inner `WHERE` is the label definition;
the outer one picks which customers take part.

## 1.4 Reading the answer honestly

Every `TRAIN` prints the number you have to beat. For a forecast of a quantity:

```sql
TRAIN MODEL spend PREDICT SUM(orders.amount) FOR customers
  AT last_seen HORIZON 60 DAYS OPTIONS (epochs = 60);
```
```
│ metric │  test   │ baseline │
│  mae   │ 56.1322 │  94.9800 │
```

**`baseline` is what a model with no information gets.** For a forecast that is
persistence: just repeat what happened last window. It is a genuinely hard
opponent and a much better yardstick than "predict the average". Above, the
model wins by 41%.

For classification it is the positive rate, which is what a coin scores on
`pr_auc`. That column matters when positives are rare, because AUROC does not
degrade there in a way anyone notices. The `reorder` model above reaches 0.8377
average precision against a 0.400 base rate. On a rarer target the two numbers
part company sharply: one tested here scored 0.974 AUROC and 0.513 average
precision on a 4.6% positive rate, which is eleven times chance rather than the
near-perfect classifier the first number suggests on its own.

`test` is scored once, on a later slice model selection never saw. `val` is the
best epoch by that metric and is optimistic by construction — a selection
statistic, not an estimate. Do not quote it.

## 1.5 What is it using?

A number tells you how well. This tells you why:

```sql
EXPLAIN MODEL reorder;
```
```
┌──────────────────────────┬───────┬────────────┐
│         feature          │ slots │ importance │
├──────────────────────────┼───────┼────────────┤
│ churned                  │     1 │     0.1435 │
│ region (category)        │     3 │     0.0781 │
│ orders.amount: mean 90d  │     1 │     0.0738 │
│ orders: count 90d        │     1 │     0.0688 │
│ orders.amount: mean 365d │     1 │     0.0598 │
└──────────────────────────┴───────┴────────────┘
```

Each group is shuffled across rows in turn, and `importance` is what the model
loses. A feature it does not use costs nothing to destroy. If something you did
not expect sits at the top, that is usually a leak, and `EXCLUDE` is how you
answer it.

## 1.6 Backtesting

Aggregate metrics hide things. Replay the model and see it row by row:

```sql
TRAIN OR REPLACE MODEL spend PREDICT SUM(orders.amount) FOR customers
  AT last_seen HORIZON 60 DAYS
  SPLIT TEMPORAL VALIDATE FROM '2025-08-01' TEST FROM '2025-09-15'
  OPTIONS (epochs = 60);

BACKTEST MODEL spend FROM '2025-09-15';
```

`OR REPLACE` because `spend` already exists from 1.4. Training over a name that
is taken is an error without it: rerunning a statement and quietly discarding
the model that was there is not something anyone asks for.

```
┌───────┬─────────────────────┬───────────┬────────┬─────────┬──────────┐
│  id   │       anchor        │ predicted │ actual │  error  │ baseline │
├───────┼─────────────────────┼───────────┼────────┼─────────┼──────────┤
│    20 │ 2025-09-18 00:00:00 │     44.14 │   43.0 │    1.14 │     43.0 │
│    21 │ 2025-10-01 00:00:00 │     16.79 │    0.0 │   16.79 │     69.0 │
│    22 │ 2025-10-14 00:00:00 │      0.78 │    0.0 │    0.78 │     98.0 │
│    23 │ 2025-10-27 00:00:00 │     44.33 │  130.0 │  -85.67 │    130.0 │
└───────┴─────────────────────┴───────────┴────────┴─────────┴──────────┘
```

Now you can see the character of the model, not just its average: close on quiet
customers, badly under on the spikes. Aggregate it to compare properly:

```sql
SELECT count(*) AS n,
       round(avg(abs(error)), 4)              AS model_mae,
       round(avg(abs(baseline - actual)), 4)  AS persistence_mae
FROM pql_exec('BACKTEST MODEL spend FROM ''2025-09-15''');
```
```
│   n   │ model_mae │ persistence_mae │
│   215 │   58.1319 │         93.5860 │
```

Those are `TRAIN`'s own `test` and `baseline` columns, reproduced to four
decimals by a separate path over the same rows. Two independent routes agreeing
is what tells you both are right.

## 1.7 It is all still SQL

```sql
SELECT c.region, count(*) AS n, round(avg(p.prediction), 3) AS avg_chance
FROM pql_exec('PREDICT EXISTS(orders) FOR customers USING MODEL reorder') p
JOIN customers c ON c.id = p.id
GROUP BY c.region ORDER BY 3 DESC;
```
```
│ region  │   n   │ avg_chance │
│ US      │   500 │      0.445 │
│ APAC    │   500 │      0.417 │
│ EU      │   500 │      0.330 │
```

```sql
SELECT * FROM pql_models();
DROP MODEL IF EXISTS spend;
```

---

# Part 2 — Reference

## Syntax

```
TRAIN [OR REPLACE] MODEL <name>
  PREDICT <target>
  FOR <table> [AS <alias>]
  [ WHERE <filter> ]                        -- which entities take part
  [ AT <column> | EVERY <n> <unit> ]        -- anchor: read one, or generate them
  [ HORIZON <n> <unit> ]                    -- how far ahead
  [ USING GRAPH (<table>, ...) ]            -- restrict which tables are used
  [ EXCLUDE (<column>, ...) ]               -- columns the model must not see
  [ SPLIT TEMPORAL VALIDATE FROM <ts> TEST FROM <ts> ]
  [ OPTIONS (<key> = <value>, ...) ]

PREDICT <target> FOR <table> [AS <alias>] [ WHERE <filter> ] USING MODEL <name>

BACKTEST MODEL <name> [ FOR <table> ] [ WHERE <filter> ] [ FROM <ts> ] [ TO <ts> ]

EXPLAIN MODEL <name> [ FOR <table> ] [ WHERE <filter> ]

DROP MODEL [IF EXISTS] <name>

SELECT * FROM pql_models()               -- the models in this session
```

Every statement is also reachable as a table function, `pql_exec('<statement>')`,
which is how you join a prediction to other tables or wrap a `TRAIN` in a
`SELECT`. `pql_exec('SHOW MODELS')` is the same listing as `pql_models()`.
(`SHOW MODELS` on its own is not available: DuckDB parses `SHOW <name>` as
`DESCRIBE <name>` before an extension sees it.)

Tables are named without a schema, so a bare name has to mean one table. PQL
resolves it the way an unqualified name resolves in SQL: the session's own
schema first, then a unique match anywhere else. When the same name lives in
several schemas and none of them is the current one, it refuses rather than
picking whichever the catalog listed first.

`PREDICT` and `FOR` come first; the optional clauses may follow in any order.
`PREDICT` inherits `AT` and `HORIZON` from the model.

### Training over a name that is taken

`TRAIN MODEL churn` twice is an error the second time:

```
pql: a model named 'churn' already exists. Use TRAIN OR REPLACE MODEL to
replace it, or DROP MODEL churn first
```

Rerunning a statement and silently discarding the model that was there is not
something anyone asks for, and the failure is quiet: the name still resolves, so
nothing looks wrong. `TRAIN OR REPLACE MODEL` says you meant it. The check
happens before any work, so you are not made to wait for a model that is then
thrown away.

`pql_models()` carries a `statement` column holding the full defining statement,
including `OPTIONS` and `EXCLUDE`. Copy it to retrain, or read it to see exactly
what a model was given.

### `EXCLUDE`: columns the model must not see

`EXPLAIN` tells you what a model leans on. `EXCLUDE` is how you answer it.

```sql
TRAIN MODEL demand PREDICT COUNT(visits) FOR members AT joined HORIZON 30 DAYS
  EXCLUDE (internal_note, visits.staff_flag);
```

A bare name is a column of the entity; qualify it to reach a child table's. It
applies before anything is fitted, so an excluded column contributes no
standardisation statistics and no category counts either, and it means the same
thing whichever `arch` is asked for.

A name that matches no column is an error rather than a no-op. A misspelt
`EXCLUDE` that quietly left the column in the model would be the worst of both:
you would believe it was gone.

Reach for it when a column is a leak you cannot remove upstream, when it is free
text that is really an identifier, or when `EXPLAIN` says it is worth nothing and
you would rather not pay for the width.

### `EXPLAIN MODEL`: what it is actually using

```sql
EXPLAIN MODEL demand;
```
```
│ feature                │ slots │ importance │
│ plan (category)        │     3 │     7.7444 │
│ anchor: phase of week  │     2 │     0.0018 │
│ visits: count 30d      │     1 │     0.0000 │
```

Each feature group is shuffled across rows in turn, leaving every other feature
and the labels where they were, and `importance` is how much worse the model
then scores. A feature it does not use costs nothing to destroy. This is
measured rather than read off the weights, so it reflects what the network does
with a feature and not how large its first-layer weights happen to be.

Scored on the same fold training held out. Shuffling on rows the model was
fitted on rewards memorised noise: a column of random numbers looked as
important as a real driver until this used the test fold instead.

A one-hot block counts as one group. Splitting it would divide the column's
contribution across its slots and make every piece look unimportant.

`arch = 'sage'` models are refused: their inputs are gathered from the graph per
batch rather than laid out as a vector, so there is nothing to permute.

### Anchors: `AT` or `EVERY`

Use `AT <column>` when your rows already carry the moment they are about — a
snapshot table, an events table, anything with a meaningful timestamp per row.

Use `EVERY <n> <unit>` when they do not. A `products` row is a catalog entry
with no moment in it, so PQL generates a grid across the data's span: starting
one horizon in, so the first example has history, and stopping one horizon
short, so no label is cut off by the end of the data.

### Filters

`=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`, `IN (...)`, `NOT IN (...)`,
`IS [NOT] NULL`, combined with `AND`, `OR`, `NOT` and parentheses. Numbers may
carry a sign. Quoted dates work wherever a timestamp belongs:

```sql
PREDICT COUNT(sales) FOR products WHERE as_of = '2026-04-01' USING MODEL demand;
PREDICT COUNT(txns) FOR accounts WHERE balance < -100 USING MODEL churn;
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
| `MAX_CATEGORIES` | 16 | 0 to 4096 — above it a text column is frequency-encoded rather than given a slot per value |
| `L2` | 0 | 0 to 1 — decoupled weight decay, on weights and not biases |

### Result columns

| column | meaning |
|---|---|
| `test` | the honest number: scored once, on a later split |
| `pr_auc` | average precision, for classification; NULL for a forecast |
| `baseline` | the number to beat: persistence for a forecast, the positive rate for a classifier |
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
how evenly the events are spaced, and means of the most informative child
columns — computed exactly by prefix sums over time-sorted children, so a
windowed statistic is two array lookups rather than a scan.

Spacing is the coefficient of variation of the gaps between consecutive events,
divided by the mean gap so it says nothing about the rate the count already
carries. A member who returns every nine days and one who arrives in two bursts
have the same count, the same recency and the same window totals, and nothing
else separates them.

The anchor contributes where it falls in the week and in the year, as sine and
cosine pairs. Its raw value is deliberately not a feature: that is a straight
line through the training era, and a model that leans on it extrapolates
nonsense past the end of the data. Its phase within a cycle is a different thing,
and it is known at prediction time by definition.

Text columns are features too. A category is stored as a dictionary code, and a
code is an arbitrary integer, so feeding it in as a number would tell the model
that `basic` sits between `premium` and `trial`. Instead a column with few enough
values gets one slot per value, and a column with too many gets how often its
value occurs. Neither looks at the label, so neither can leak, and both are
counted over the training fold only. `MAX_CATEGORIES` sets the boundary, 16 by
default.

Two cases are left alone: a column holding one value everywhere carries nothing,
and a column with nearly one value per row is an identifier or free text rather
than a category. A value that appears only after training lands in the spare slot
when training had one, and otherwise sets no slot at all, so an unfamiliar
category gets no answer rather than an arbitrary one.

Foreign keys come from declared constraints, and otherwise from the `<table>_id`
convention (including `categories` → `category`). Text keys match as text. If a
child table has no clock, PQL follows a foreign key to find one — `line_items`
dated by its `transaction`, for instance — so you rarely need to denormalize by
hand.

## Two architectures

`arch = 'mlp'` (default) runs a small dense network over the entity's columns
plus those aggregates. Cheap: 30 features on the tutorial dataset.

`L2` adds decoupled weight decay, applied to weights and not to biases. It
defaults to 0, and on the tasks measured here it made no difference either way;
it is there for wide schemas, where the feature count grows with every linked
table.

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
cannot reach it, one hop scores 0.495 and two score 0.850.

Which one wins depends on the shape of the data, and it is worth trying both.
On the tutorial's spend forecast the dense model is ahead (56.13 MAE
against 61.80, persistence 94.98) while carrying 30 features to `sage`'s 641
parameters. `sage` earns its keep where structure reaches further than one hop,
which is what the two-hop number above measures.

## Performance

The language lives in one self-contained header,
[`src/include/pql/pql.hpp`](src/include/pql/pql.hpp), which never includes a
DuckDB header.

Children live in a CSR arena: one flat array per link, contiguous per-parent
slices, and the child layer's own features and gate masks are permuted into that
same order so a batch reads them in one sweep. Buckets already in time order skip
sorting entirely; the rest are radix sorted. Per-step buffers come from a single
arena sized up front from `(batch, channels, relation widths)`.

The linear algebra is three kernels, each holding a block of the *output* in
vector registers and reducing on an outer loop. That is the opposite of writing
a dot product per output element, which ends in a horizontal reduction and
reloads the weight row once per row of the batch; the difference is roughly ten
times. Block shapes are compile-time constants, chosen per K by measurement
rather than by rule, and a weight matrix is transposed when the reduction axis
would otherwise be the wrong one — K*N writes against B*K*N multiply-adds.

The loader consults the catalog first and reads only the tables and columns that
can influence the model, straight from `DataChunk` buffers with no per-cell
`Value`. Adding 2.5M rows of unrelated tables leaves training time unchanged.

---

# Part 4 — Worked examples

[`examples/tutorial.sql`](examples/tutorial.sql) is the dataset Part
1 walks through: load it and every statement above runs as written.

[`examples/shop_demo.sql`](examples/shop_demo.sql) is self-contained:
it creates its own tables and trains three models end to end.

[`examples/sunnyside_demo.sql`](examples/sunnyside_demo.sql) runs
against a real point-of-sale database, with
[`sunnyside_export.sh`](examples/sunnyside_export.sh) and
[`sunnyside_load.sql`](examples/sunnyside_load.sql) to bring it over
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

On that database: 15,010 training rows, 2,755 held out, no tables built by hand.
It needs the Postgres export above, so unlike Part 1 it is not reproducible from
this repository alone.

---

# Building and testing

The extension has no dependency beyond DuckDB and the C++17 standard library:
no vcpkg manifest, no toolchain beyond a C++ compiler, CMake and (optionally)
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
[`test/cpp`](test/cpp) is a standalone program against
[`pql.hpp`](src/include/pql/pql.hpp):

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

# Part 5 — Where the edges are

Honest about what is not there yet, so you can plan around it.

**Language.** Filters compare a column to a literal — no column-to-column
comparisons, arithmetic, `BETWEEN` or `LIKE`. `FOR` takes a bare table name, not
a subquery and not an explicitly schema-qualified one: a bare name resolves the
way SQL resolves it, and genuine ambiguity across schemas is refused, but there
is no syntax for saying which one you meant. Composite foreign keys are not
modelled. Text targets are refused rather than trained as multi-class.

**Attribute prediction has no leakage guarantee.** Every other numeric column on
the row is a feature, including ones that are consequences of the target.
Predicting `line_items.voided` scores a perfect 1.0 because `net_sales = 0`
exactly when voided. A near-perfect attribute score usually means a same-row
tautology; forecasting is the path with the structural guarantee.

**Execution.** `TRAIN` runs during binding on its own connection, so it does not
see uncommitted data. Work is single-threaded and outside the buffer manager, so
`memory_limit` does not apply, and each statement reloads its tables. All three
dissolve together by moving the work into a physical operator with a sink, which
is the main thing left to build. (DuckDB's own `EXPLAIN` cannot be put in front
of a PQL statement at all — its parser rejects it before the extension is
reached. `EXPLAIN MODEL` is a separate thing and is described above.)

**Models** live for the session: not persisted, not transactional.

**Reproducibility is per build.** The same statement on the same data gives the
same model every time within one binary, which is tested. It is not bit-identical
across compilers or optimisation levels: the kernels let the compiler reassociate
float addition, and `-O0` and `-O2` differ around the ninth significant digit.
Nothing is fitted to that precision, but do not diff metrics across builds and
expect zeros.

**Time zones.** `TIMESTAMP WITH TIME ZONE` reads through the session zone, so a
model trained in one session and used in another with a different zone has
shifted anchors.
