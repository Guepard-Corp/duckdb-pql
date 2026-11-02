# PQL: predictive queries for DuckDB

PQL lets you train a model and ask it questions in SQL, on the tables you
already have. No feature engineering, no export, no separate service.

```sql
INSTALL pql FROM community;
LOAD pql;

TRAIN MODEL churn PREDICT EXISTS(orders) FOR customers AT last_seen HORIZON 30 DAYS;

PREDICT EXISTS(orders) FOR customers WHERE region = 'EU' USING MODEL churn;
```

The first statement reads `customers`, follows the foreign key to `orders`,
builds features from both (counts, averages, how recent, how regular), trains,
and reports how well it did on data it held back. The second asks the model,
for every EU customer, "will they order again in the next 30 days?"

## Try it in two minutes

Paste this into any DuckDB shell with the extension loaded:

```sql
CREATE TABLE customers AS
  SELECT i AS id, (i % 7) + 1 AS tier, ['EU','US','APAC'][1 + i % 3] AS region,
         (i % 5) = 0 AS churned
  FROM range(300) t(i);

TRAIN MODEL churn PREDICT customers.churned FOR customers;
```
```
┌────────────┬───────────┬────────┬───────┬──────────┐
│ train_rows │ test_rows │ metric │ test  │ features │
├────────────┼───────────┼────────┼───────┼──────────┤
│        180 │        60 │ auroc  │ 0.439 │        5 │
└────────────┴───────────┴────────┴───────┴──────────┘
```

```sql
PREDICT customers.churned FOR customers WHERE tier >= 5 USING MODEL churn;
```
```
┌───────┬────────────┐
│  id   │ prediction │
├───────┼────────────┤
│     4 │      0.200 │
│     5 │      0.167 │
│   ... │        ... │
```

That model is useless (0.439 is a coin flip) because the toy data has nothing
to learn from: `churned` is `i % 5`, which no column explains. That is the point: PQL always tells you how
good the model actually is, on rows it never trained on.

## The six statements

| statement | what it does |
|---|---|
| `TRAIN MODEL name PREDICT target FOR table ...` | build a model and report its score |
| `PREDICT target FOR table [WHERE ...] USING MODEL name` | one prediction per row |
| `OPTIMIZE table MAXIMIZE SUM(...) SUBJECT TO ... USING MODEL name` | turn the predictions into a plan: which rows to act on, a value per row (`CHOOSE`), or each row placed on a resource (`ASSIGN`) |
| `BACKTEST MODEL name` | replay the model row by row against what really happened |
| `EXPLAIN MODEL name` | which features it relies on |
| `DROP MODEL name` | forget it |

`SELECT * FROM pql_models()` lists the models in the session. Every statement
also works as a table function, `pql_exec('...')`, so you can join predictions
to anything:

```sql
SELECT c.region, avg(p.prediction)
FROM pql_exec('PREDICT EXISTS(orders) FOR customers USING MODEL churn') p
JOIN customers c ON c.id = p.id
GROUP BY c.region;
```

## Two kinds of question

**"What is this row's value?"** A column that exists but you want to fill in
or verify. No dates needed.

```sql
TRAIN MODEL vip PREDICT customers.is_vip FOR customers;
```

**"What will happen next?"** Something in the future, counted over a window.
You say where "now" is for each row (`AT` a timestamp column) and how far to
look (`HORIZON`).

```sql
TRAIN MODEL spend PREDICT SUM(orders.amount) FOR customers AT last_seen HORIZON 60 DAYS;
TRAIN MODEL reorder PREDICT EXISTS(orders) FOR customers AT last_seen HORIZON 30 DAYS;
TRAIN MODEL refunds PREDICT EXISTS(orders WHERE orders.status = 'refunded')
  FOR customers AT last_seen HORIZON 60 DAYS;
```

For forecasts, PQL only ever shows the model what was known before each row's
anchor time, so the answer cannot leak into the features. `TRAIN` reports a
`baseline` next to the score: what you would get by just repeating the last
window. If the model does not beat it, you know.

## From a prediction to a decision: `OPTIMIZE`

`OPTIMIZE` is one statement. It turns a model's predictions into a plan that
maximises (or minimises) a total under limits, and it comes in three forms,
depending on what is being decided:

| form | decides | limits | rows interact |
|---|---|---|---|
| `OPTIMIZE t ... SUBJECT TO ...` | which rows to act on | totals over the whole plan | only through the totals |
| `OPTIMIZE t CHOOSE col IN (...) ...` | one value per row, from a list | totals over the whole plan | only through the totals |
| `OPTIMIZE t ASSIGN col OVER r.key ...` | which row of another table each row goes to, or none | a limit for each of those rows (`PER col`), and with `SEQUENCE` their order and timing | through the resource they share |

**Which rows to act on:**

```sql
OPTIMIZE customers
  MAXIMIZE SUM((1 - prediction) * tier * 100)
  SUBJECT TO COUNT(*) <= 50
  USING MODEL reorder;
```

Fifty calls this week: the customers worth most who would not order again on
their own.

**A value per row:** `CHOOSE offer IN ('none', '10%', '20%')` picks one offer for
each customer, by asking the model what each value would change.

**Rows placed on resources:** `ASSIGN` is the form for loads on trucks, jobs on
crews and orders on machines. With `CHOOSE`, one row's choice never changes what
another row is worth. With `ASSIGN`, it does: a job uses up its crew's hours, and
with `SEQUENCE` its travel time depends on the job before it. That is why `ASSIGN`
has its own solver and its own plan columns (`sequence`, `begin`, `setup`,
`duration`), shown in the next section.

In every form, `EXPLAIN OPTIMIZE` reports the plan's total beside a bound no plan
can beat, so you know how good it is.

### `OPTIMIZE ... ASSIGN`: placing work on resources

`ASSIGN` puts every row (a task) on a row of another table (a resource), or on
nothing, under a limit per resource. With `SEQUENCE` the tasks on each resource
are also put in order, with travel or changeover time between them and a time
window each must start in. The same words cover loads on trucks, jobs on crews,
orders on machines and shifts on staff: nothing in the language names a domain.

<details>
<summary>Toy data for the crew examples: four crews, thirty past jobs, six open ones</summary>

```sql
CREATE TABLE crews AS SELECT * FROM (VALUES
  ('A', 'van',    40.0, -75.0, 20.0, TIMESTAMP '2026-03-02 06:00'),
  ('B', 'van',    41.0, -74.0, 16.0, TIMESTAMP '2026-03-02 08:00'),
  ('C', 'reefer', 40.5, -76.0, 24.0, TIMESTAMP '2026-03-02 06:00'),
  ('D', 'van',    39.5, -75.5, 12.0, TIMESTAMP '2026-03-02 10:00')) t(crew, kind, lat, lon, hours, free_from);

CREATE TABLE jobs AS
SELECT i AS job_id,
       TIMESTAMP '2026-01-05' + INTERVAL (i * 2) DAY AS received_at,
       39.5 + ((i * 7) % 20) / 10.0 AS from_lat, -76.5 + ((i * 11) % 30) / 10.0 AS from_lon,
       39.5 + ((i * 13) % 20) / 10.0 AS to_lat,  -76.5 + ((i * 3) % 30) / 10.0 AS to_lon,
       60 + (i * 37) % 180 AS miles,
       400 + (i * 53) % 900 AS rate,
       CASE WHEN i % 5 = 0 THEN 'reefer' ELSE 'van' END AS kind,
       TIMESTAMP '2026-03-02 06:00' + INTERVAL ((i % 3) * 8) HOUR AS window_from,
       TIMESTAMP '2026-03-02 06:00' + INTERVAL ((i % 3) * 8 + 10) HOUR AS window_to,
       CASE WHEN i < 30 THEN (ARRAY['A','B','C','D','none'])[1 + i % 5] ELSE 'none' END AS crew,
       CASE WHEN i < 30 THEN
            CASE WHEN i % 5 = 4 THEN (400 + (i * 53) % 900) * 0.10
                 ELSE (400 + (i * 53) % 900) * (0.35 + 0.02 * (i % 5)) - (60 + (i * 37) % 180) * 0.8 END
       END AS margin,
       (i >= 30)::INT AS open
FROM range(36) t(i);

TRAIN MODEL job_margin PREDICT jobs.margin FOR jobs AT received_at WHERE open = 0
  EXCLUDE (job_id, open) OPTIONS (epochs = 40, hidden = 16, seed = 1);
```
</details>

**One job per crew, each crew on its own kind of job.** The model is asked what
each job is worth on each crew, and on no crew at all (`ELSE`):

```sql
OPTIMIZE jobs WHERE open = 1 AT received_at
  ASSIGN crew OVER crews.crew ELSE 'none'
    ONLY IF resource.kind = kind
  MAXIMIZE SUM(prediction)
  SUBJECT TO COUNT(*) PER crew <= 1
  USING MODEL job_margin;
```
```
┌────────┬──────┬────────────┐
│ job_id │ crew │ prediction │
├────────┼──────┼────────────┤
│     30 │ C    │      385.1 │
│     31 │ none │      258.2 │
│     32 │ B    │      295.8 │
│     33 │ D    │      302.3 │
│     34 │ none │        9.6 │
│     35 │ none │      173.0 │
```

**In order, with travel, time windows and each crew's hours.** `START` is the
hours from a crew to its first job, `SETUP` the hours between two jobs
(`prev.` is the job before), `DURATION` the hours a job takes, and each job
must begin inside its window. `plan.setup` is the empty driving the plan
chose, so the total can charge for it:

```sql
OPTIMIZE jobs WHERE open = 1 AT received_at
  ASSIGN crew OVER crews.crew ELSE 'none'
    ONLY IF resource.kind = kind OR (resource.kind = 'reefer' AND kind = 'van')
    SEQUENCE FROM resource.free_from
      START    DISTANCE(resource.lat, resource.lon, from_lat, from_lon) / 40
      SETUP    DISTANCE(prev.to_lat, prev.to_lon, from_lat, from_lon) / 40
      DURATION miles / 40 + 1
      WINDOW BETWEEN window_from AND window_to
  MAXIMIZE SUM(prediction - plan.setup * 60)
  SUBJECT TO SUM(plan.setup + plan.duration) PER crew <= resource.hours
  USING MODEL job_margin;
```
```
┌────────┬──────┬──────────┬──────────────────┬───────┬──────────┐
│ job_id │ crew │ sequence │      begin       │ setup │ duration │
├────────┼──────┼──────────┼──────────────────┼───────┼──────────┤
│     30 │ C    │        0 │ 2026-03-02 06:39 │  0.66 │     3.25 │
│     31 │ none │     NULL │ NULL             │  NULL │     NULL │
│     32 │ none │     NULL │ NULL             │  NULL │     NULL │
│     33 │ C    │        1 │ 2026-03-02 10:20 │  0.43 │    6.025 │
│     34 │ none │     NULL │ NULL             │  NULL │     NULL │
│     35 │ none │     NULL │ NULL             │  NULL │     NULL │
```

Crew C takes jobs 30 and 33, in that order and inside both windows. The other
jobs are worth more left to `none` than the driving it would take to reach them.

**Orders on two presses, where changing colour costs 45 minutes.** The same
clause on a different problem: no map, no clock, just a changeover between
consecutive orders.

```sql
CREATE TABLE presses AS SELECT * FROM (VALUES ('P1', 8.0), ('P2', 8.0)) t(press, hours);
CREATE TABLE orders AS
SELECT i AS order_id, (ARRAY['red','blue','green'])[1 + i % 3] AS colour, 1 + (i % 4) * 0.5 AS run_hours,
       CASE WHEN i < 60 THEN (ARRAY['P1','P2','idle'])[1 + i % 3] ELSE 'idle' END AS press,
       CASE WHEN i < 60 THEN CASE WHEN i % 3 = 2 THEN 0 ELSE 100 + (i * 17) % 60 END END AS value,
       (i >= 60)::INT AS open
FROM range(72) t(i);
TRAIN MODEL order_value PREDICT orders.value FOR orders WHERE open = 0 EXCLUDE (order_id, open) OPTIONS (seed = 1);

OPTIMIZE orders WHERE open = 1
  ASSIGN press OVER presses.press ELSE 'idle'
    SEQUENCE
      SETUP CASE WHEN prev.colour = colour THEN 0 ELSE 0.75 END
      DURATION run_hours
  MAXIMIZE SUM(prediction - 40 * plan.setup)
  SUBJECT TO SUM(plan.setup + plan.duration) PER press <= resource.hours
  USING MODEL order_value;
```
```
┌──────────┬────────┬───────┬──────────┬───────┬──────────┐
│ order_id │ colour │ press │ sequence │ setup │ duration │
├──────────┼────────┼───────┼──────────┼───────┼──────────┤
│       65 │ green  │ P1    │        0 │  0.00 │      1.5 │
│       68 │ green  │ P1    │        1 │  0.00 │      1.0 │
│       60 │ red    │ P1    │        2 │  0.75 │      1.0 │
│       66 │ red    │ P1    │        3 │  0.00 │      2.0 │
│       69 │ red    │ P1    │        4 │  0.00 │      1.5 │
│       61 │ blue   │ P2    │        0 │  0.00 │      1.5 │
│       70 │ blue   │ P2    │        1 │  0.00 │      2.0 │
│       64 │ blue   │ P2    │        2 │  0.00 │      1.0 │
│       67 │ blue   │ P2    │        3 │  0.00 │      2.5 │
│       62 │ green  │ idle  │     NULL │  NULL │     NULL │
│       63 │ red    │ idle  │     NULL │  NULL │     NULL │
│       71 │ green  │ idle  │     NULL │  NULL │     NULL │
```

The plan groups the colours (P2 runs only blue, P1 changes over once) and
leaves the three orders that do not fit in eight hours idle.

**How good is the plan?** `EXPLAIN` gives the total, the bound no plan can
beat, and each limit per resource:

```sql
EXPLAIN OPTIMIZE orders WHERE open = 1 ... -- the statement above
```
```
┌────────────────────────────────────────────────────────────────────────┬────────┬────────┐
│ measure                                                                │ value  │ bound  │
├────────────────────────────────────────────────────────────────────────┼────────┼────────┤
│ objective                                                              │ 1041.6 │ 1136.6 │
│ SUM(plan.setup + plan.duration) PER press <= resource."hours" for 'P1' │    7.8 │    8.0 │
│ SUM(plan.setup + plan.duration) PER press <= resource."hours" for 'P2' │    7.0 │    8.0 │
│ rows                                                                   │   12.0 │   NULL │
│ press = 'P1'                                                           │    5.0 │   NULL │
│ press = 'P2'                                                           │    4.0 │   NULL │
│ press = 'idle'                                                         │    3.0 │   NULL │
```

The bound relaxes the ordering (every order at its cheapest possible
changeover), so with `SEQUENCE` it is an upper limit rather than a target.
When `value` equals `bound`, as it does for the one-job-per-crew plan above,
the plan is proven best. The full grammar is in the
[reference](docs/reference.md#optimize--assign-tasks-on-resources).

## Reading the score

| column | meaning |
|---|---|
| `metric` | `auroc` for yes/no targets, `mae` for amounts |
| `test` | the score on rows the model never saw. This is the number to trust |
| `baseline` | the no-model answer: the positive rate, or "same as last time" |
| `pr_auc` | for rare yes/no targets, the more honest score |
| `features` | how many features PQL built for you |

## DuckDB version support

| branch | builds against | used by the registry as |
|---|---|---|
| `main` | the current stable DuckDB (v1.5.5) | `ref` |
| `next` | DuckDB `main` (v2.0-cyanoptera) | `ref_next` |

PQL is a parser extension, and the parser extension API changed between the
two: on v1.5.5 it receives the statement as a string, on v2.0 as a token
stream it reports back how much of it consumed. Result column names changed
type as well. One source file, `src/pql_extension.cpp`, differs between the
branches; the language itself is the same file on both.

Keeping both means the registry tests PQL against the next DuckDB release on
every change, so PQL keeps working the day that release ships rather than
disappearing until someone fixes it. When v2.0 is released, `next` becomes
`main`.

## Learn more

- [Tutorial](docs/tutorial.md): a 15-minute walk from first model to backtest, with real output
- [Reference](docs/reference.md): full syntax, options, result columns, and the current limitations
- [How it works](docs/how-it-works.md): features, leakage, the two model architectures
- [Building and testing](docs/building.md): build from source, run the tests

## Good to know

- Models live for the session. They are not saved to the database file.
- `TRAIN` reads committed data only.
- Table names are bare (`customers`, not `main.customers`) and resolve like SQL does.
- Filters compare a column to a value: `region = 'EU'`, `tier >= 5`, `region IN ('EU', 'US')`.

PQL is MIT licensed. Issues and questions: https://github.com/Guepard-Corp/duckdb-pql/issues
