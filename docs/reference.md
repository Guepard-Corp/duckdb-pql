# Reference

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

Use `AT <column>` when your rows already carry the moment they are about, and a
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
| `MAX_CATEGORIES` | 16 | 0 to 4096; above it a text column is frequency-encoded rather than given a slot per value |
| `L2` | 0 | 0 to 1; decoupled weight decay, on weights and not biases |

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

---

## Limitations

Honest about what is not there yet, so you can plan around it.

**Language.** Filters compare a column to a literal: no column-to-column
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
of a PQL statement at all; its parser rejects it before the extension is
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
