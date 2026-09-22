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

## The five statements

| statement | what it does |
|---|---|
| `TRAIN MODEL name PREDICT target FOR table ...` | build a model and report its score |
| `PREDICT target FOR table [WHERE ...] USING MODEL name` | one prediction per row |
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
