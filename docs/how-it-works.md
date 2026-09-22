# How it works

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
columns, computed exactly by prefix sums over time-sorted children, so a
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
child table has no clock, PQL follows a foreign key to find one (`line_items`
dated by its `transaction`, for instance), so you rarely need to denormalize by
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
[`src/include/pql/pql.hpp`](../src/include/pql/pql.hpp), which never includes a
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
would otherwise be the wrong one: K*N writes against B*K*N multiply-adds.

The loader consults the catalog first and reads only the tables and columns that
can influence the model, straight from `DataChunk` buffers with no per-cell
`Value`. Adding 2.5M rows of unrelated tables leaves training time unchanged.

---

---

## Worked examples

[`examples/tutorial.sql`](../examples/tutorial.sql) is the dataset the
[tutorial](tutorial.md) walks through: load it and every statement there runs
as written.

[`examples/shop_demo.sql`](../examples/shop_demo.sql) is self-contained:
it creates its own tables and trains three models end to end.

[`examples/sunnyside_demo.sql`](../examples/sunnyside_demo.sql) runs
against a real point-of-sale database, with
[`sunnyside_export.sh`](../examples/sunnyside_export.sh) and
[`sunnyside_load.sql`](../examples/sunnyside_load.sql) to bring it over
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
It needs the Postgres export above, so unlike the tutorial it is not reproducible from
this repository alone.

---
