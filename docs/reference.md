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

[EXPLAIN] OPTIMIZE <table> [AS <alias>]
  [ WHERE <filter> ]                        -- the rows a plan may act on
  [ AT <column> | AT <timestamp> ]
  [ CHOOSE <column> IN (<value>, ...) ]     -- a value per row, instead of which rows
  MAXIMIZE | MINIMIZE <total>
  [ SUBJECT TO <total> (<= | >=) <number> [ AND ... ] ]
  USING MODEL <name>

  <total> := SUM(<expression>) [ FILTER (WHERE <filter>) ]
           | COUNT(*)          [ FILTER (WHERE <filter>) ]

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

### `OPTIMIZE`: from a prediction to a decision

A prediction says what will happen. `OPTIMIZE` says what to do about it, given
what each action is worth and what you can afford. It asks one of two things.

**Which rows to act on.** Five service bays this week, a breakdown costs 4200,
a service costs what the truck's row says:

```sql
OPTIMIZE trucks WHERE status = 'active'
  MAXIMIZE SUM(prediction * 4200 - service_cost)
  SUBJECT TO COUNT(*) <= 5
  USING MODEL breakdown;
```

The result is the rows to act on, the most valuable first, keyed and anchored
the way `PREDICT`'s rows are: the key, the anchor when the model has one,
`prediction`, and `objective`, which is that row's share of the total. Without
`SUBJECT TO`, it is every row whose share is worth having.

**A value per row.** `CHOOSE` names a column and the values it may take, and
`OPTIMIZE` picks one for every row:

```sql
OPTIMIZE customers
  CHOOSE offer IN ('none', '10%', '20%')
  MAXIMIZE SUM(prediction - CASE offer WHEN '10%' THEN 10 WHEN '20%' THEN 20 ELSE 0 END)
  SUBJECT TO SUM(CASE offer WHEN '10%' THEN 10 WHEN '20%' THEN 20 ELSE 0 END) <= 5000
  USING MODEL spend;
```

To learn what an offer does, PQL sets each row's `offer` to each value in turn
and predicts again. Inside the totals, `offer` reads the value being tried, so
costs can depend on it. The result has a row for every candidate with the value
chosen, in a column named after the lever. A numeric column works the same way:
`CHOOSE price IN (9.99, 12.99, 14.99)` with `MAXIMIZE SUM(prediction * price)`.
`WHERE` reads the lever as it stands today, so `WHERE offer = 'none'` plans for
the customers on no offer now.

**Expressions** are numbers, the entity's numeric columns, `prediction`, `+ - *
/`, and `CASE`, in both its forms. Text only appears in comparisons, inside
`CASE WHEN` or `FILTER (WHERE ...)`. The bare word `prediction` is the model's
answer; a column of that name is still reachable quoted or qualified.

A row whose total has no value is refused: a `NULL` in a column it reads, or a
division by zero. SQL's `SUM` would skip that row, which in a plan means treating
an unknown cost as free, and the plan would then fill up with exactly those rows.
Leave them out with `WHERE ... IS NOT NULL` instead. One departure from SQL: a
`CASE` with no `ELSE` and no matching `WHEN` is 0, not `NULL`. Inside a `SUM` they
mean the same thing.

**Limits** are `<=` or `>=`. An exact `=` is rarely reachable with whole
choices, so bound a total from both sides instead. `FILTER (WHERE ...)` narrows a
total to some rows: `COUNT(*) FILTER (WHERE offer != 'none') <= 300` is at most
300 offers.

**`EXPLAIN OPTIMIZE`** solves the same plan and reports its totals:

| measure | value | bound |
|---|---|---|
| `objective` | the plan's total | the best any plan meeting the limits could reach |
| `status quo` | the total with every lever as it is today (`CHOOSE` only) | |
| each limit, as written | the plan's total for it | the limit |
| `rows` | the rows the plan was chosen from | |
| `<lever> = <value>`, or `selected` | how many rows got it | |

`status quo` is what today's levers are predicted to score, for comparison;
nothing makes it meet the limits, and it often does not.

When `bound` equals `value` on the objective, the plan is proven to be the best
one. That is the case for every limit written in whole units at a size the
solver can tabulate (a count, a budget in whole currency units), for problems
small enough to enumerate, and whenever its search finishes. Otherwise the gap
between the two is how much better a plan could possibly be, which on thousands
of rows is usually a rounding error.

A plan that cannot meet its limits is an error, not an empty result. The message
says which limit no plan can meet on its own, or that the limits contradict each
other. When the solver could not prove either, it says it found no plan and
that one may still exist.

The limits are meant to be a handful: a budget, a capacity, a cap or a minimum
per segment. With a few dozen the plan is still usually proven best or close to
it. With hundreds, say a minimum for each of hundreds of small groups, the
search can stop short: the gap to the bound widens, and it can fail to find a
plan that exists.

**What `CHOOSE` refuses.** The question "what if this row's offer were X" only
has an answer when the model reads `offer` as an input, and only for values it
saw while training. So `CHOOSE` refuses:

- a column the model does not read: a key, a constant, an `EXCLUDE`d column, or
  one with nearly one value per row;
- a text value that never appeared in training, which would set no input at all;
- a text column with more values than `MAX_CATEGORIES`, which the model only
  knows by how common each value is (retrain with a higher `max_categories`);
- a number outside every value the column has in the table;
- the model's own target, its anchor, and timestamps.

**How far to trust a plan.** The model's answer to "what if" is only as good as
the history behind it. If offers went mostly to customers who were about to buy
anyway, the model learns that offers come before purchases, not that they cause
them, and the plan will hand out offers where they change nothing. History
where the lever was set at random, or by a rule that only uses columns the model
can see, is what makes the answer causal. And a plan is chosen because the model
scores it highly, so its `objective` leans optimistic: it is the model's
estimate, not a measurement. Try the plan on part of the rows before rolling it
out to all of them.

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

**`OPTIMIZE`** plans with one model per statement, and its levers are columns of
the entity it plans over. It estimates a plan's value with the same model that
chose it, so there is no `BACKTEST` for a plan yet: replaying a policy against
history needs outcomes for the actions that were not taken, which takes
off-policy evaluation or an experiment. Its solver is exact for whole-number
limits it can tabulate and for small problems; otherwise it reports the bound
alongside the plan rather than claiming the plan is the best.

**Reproducibility is per build.** The same statement on the same data gives the
same model every time within one binary, which is tested. It is not bit-identical
across compilers or optimisation levels: the kernels let the compiler reassociate
float addition, and `-O0` and `-O2` differ around the ninth significant digit.
Nothing is fitted to that precision, but do not diff metrics across builds and
expect zeros.

**Time zones.** `TIMESTAMP WITH TIME ZONE` reads through the session zone, so a
model trained in one session and used in another with a different zone has
shifted anchors.
