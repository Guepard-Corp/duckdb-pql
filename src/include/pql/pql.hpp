// pql.hpp — Predictive Query Language: an extension of SQL with training and
// prediction statements. This header includes every part of it, in order.
//
// The parts are deliberately self-contained and DuckDB-agnostic: none of them
// includes a DuckDB header. The binding in
// src/pql_extension.cpp reads raw typed pointers out
// of DataChunk buffers (FlatVector::GetData) and fills the column arrays below.
//
// On copying, precisely: no Value is materialised per cell, and tables and
// columns that cannot influence the model are never read at all. The columns
// that ARE used are copied once into the layout the aggregation wants. That
// last copy is not avoidable for training, which makes many epochs of random
// access by entity over time-sorted indexes, while DuckDB's chunks are
// transient and offer no stable row-addressable pointer to retain. A streaming
// PREDICT that never materialises is possible and is not what this does today.
//
// ---------------------------------------------------------------------------
// GRAMMAR
// ---------------------------------------------------------------------------
//
//   train_stmt :=
//       TRAIN MODEL <ident>
//       PREDICT <predict_expr>
//       FOR <table> [AS <ident>]
//       [ WHERE <filter> ]
//       [ AT <ident> ]                     -- anchor-time column on the entity
//       [ HORIZON <interval> ]             -- future window for forecast targets
//       [ USING GRAPH ( <table> [, ...] ) ]-- explicit graph allow-list
//       [ SPLIT TEMPORAL VALIDATE FROM <literal> TEST FROM <literal> ]
//       [ OPTIONS ( <ident> = <literal> [, ...] ) ]
//
//   predict_stmt :=
//       PREDICT <predict_expr>
//       FOR <table> [AS <ident>]
//       [ WHERE <filter> ]
//       [ AT <ident> | AT <literal> ]
//       [ HORIZON <interval> ]
//       USING MODEL <ident>
//
//   predict_expr :=
//         <table>.<column>                        -- attribute imputation
//       | COUNT  ( <table> [ WHERE <filter> ] )   -- forecast: how many
//       | EXISTS ( <table> [ WHERE <filter> ] )   -- forecast: any at all
//       | SUM    ( <table>.<column> [ WHERE <filter> ] )
//       | AVG    ( <table>.<column> [ WHERE <filter> ] )
//       | MIN    ( <table>.<column> [ WHERE <filter> ] )
//       | MAX    ( <table>.<column> [ WHERE <filter> ] )
//
//   optimize_stmt :=
//       [EXPLAIN] OPTIMIZE <table> [AS <ident>]
//       [ WHERE <filter> ]                 -- the rows a plan may act on
//       [ AT <ident> | AT <literal> ]
//       [ CHOOSE <ref> IN ( <literal> [, ...] ) ]
//       ( MAXIMIZE | MINIMIZE ) <total>
//       [ SUBJECT TO <total> ( <= | >= ) <number> [ AND ... ] ]
//       USING MODEL <ident>
//
//   total    := SUM ( <expr> ) [ FILTER ( WHERE <filter> ) ]
//             | COUNT ( * )    [ FILTER ( WHERE <filter> ) ]
//   expr     := numbers, numeric columns, `prediction`, + - * /, parentheses,
//               CASE WHEN <filter> THEN <expr> ... [ELSE <expr>] END, and
//               CASE <ref> WHEN <literal> THEN <expr> ... [ELSE <expr>] END
//
//   interval := <integer> ( DAY | DAYS | WEEK | WEEKS | MONTH | MONTHS |
//                           YEAR | YEARS | HOUR | HOURS )
//
//   filter   := disjunction of conjunctions of comparisons; see ParseFilter.
//              <ref> ( = | != | <> | < | <= | > | >= ) <literal>
//              <ref> IN ( <literal> [, ...] )
//              <ref> IS [NOT] NULL
//              combined with AND / OR and parenthesised freely.
//
// Two prediction shapes fall out of predict_expr, and they are genuinely
// different problems, so the parser keeps them distinct:
//
//   ATTRIBUTE  `PREDICT users.churn FOR users WHERE ...`
//              Impute a column that exists on the entity row. No horizon.
//
//   FORECAST   `PREDICT COUNT(orders WHERE orders.status='paid')
//               FOR users AT signup_ts HORIZON 30 DAYS`
//              The target is an aggregate over rows of a *related* table that
//              land strictly inside (anchor, anchor + horizon]. The label is
//              computed by the engine, never read from a column, which is what
//              makes leakage structurally impossible: every feature is drawn
//              from (-inf, anchor] and every label from (anchor, anchor+h].
//
// ---------------------------------------------------------------------------
// EXAMPLES
// ---------------------------------------------------------------------------
//
//   TRAIN MODEL churn
//     PREDICT EXISTS(orders) FOR customers AS c
//     AT c.last_seen HORIZON 90 DAYS
//     WHERE c.region = 'EU'
//     SPLIT TEMPORAL VALIDATE FROM '2023-01-01' TEST FROM '2023-07-01'
//     OPTIONS (epochs = 40, hidden = 128, layers = 2);
//
//   PREDICT EXISTS(orders) FOR customers AS c
//     AT c.last_seen HORIZON 90 DAYS
//     WHERE c.id = 42
//     USING MODEL churn;
//
#pragma once

#include "pql/common.hpp"
#include "pql/lang.hpp"
#include "pql/data.hpp"
#include "pql/features.hpp"
#include "pql/metrics.hpp"
#include "pql/kernels.hpp"
#include "pql/learner.hpp"
#include "pql/mlp.hpp"
#include "pql/sage.hpp"
#include "pql/train.hpp"
#include "pql/predict.hpp"
#include "pql/solver.hpp"
#include "pql/optimize.hpp"
