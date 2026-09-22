# Tutorial

Work through this and you will know PQL. Every query below is real and runs.

## Your first model

Everything here runs against one small dataset. Load it and follow along:

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
data where churn is only partly explained by order history, and it is the sort
of number you should expect to see, not 0.99. Ask it about anyone:

```sql
PREDICT customers.churned FOR customers WHERE tier >= 5 USING MODEL churn;
```

**This needs no dates.** Predicting a column that already exists is imputation,
and imputation has no time dimension. If your schema has no timestamps at all,
this still works.

## The idea that makes forecasting different

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

## Forecasting

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

## Reading the answer honestly

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
best epoch by that metric and is optimistic by construction, a selection
statistic, not an estimate. Do not quote it.

## What is it using?

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

## Backtesting

Aggregate metrics hide things. Replay the model and see it row by row:

```sql
TRAIN OR REPLACE MODEL spend PREDICT SUM(orders.amount) FOR customers
  AT last_seen HORIZON 60 DAYS
  SPLIT TEMPORAL VALIDATE FROM '2025-08-01' TEST FROM '2025-09-15'
  OPTIONS (epochs = 60);

BACKTEST MODEL spend FROM '2025-09-15';
```

`OR REPLACE` because `spend` already exists from "Reading the answer honestly". Training over a name that
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

## It is all still SQL

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
