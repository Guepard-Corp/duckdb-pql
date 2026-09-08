-- PQL on the sunnyside_v2 point-of-sale data.
--   sh extras/pql/examples/sunnyside_export.sh /tmp/sunnyside
--   build/reldebug/duckdb /tmp/sunny.duckdb -c ".read extras/pql/examples/sunnyside_load.sql"
--   build/reldebug/duckdb /tmp/sunny.duckdb -c ".read extras/pql/examples/sunnyside_demo.sql"

-- A product has no time to anchor on, so demand needs a panel: one row per
-- (product, as-of date). The column is as_of, not asof: ASOF is a DuckDB keyword.
CREATE OR REPLACE TABLE snapshots AS
SELECT row_number() OVER (ORDER BY p.product_id, d.as_of) AS id,
       p.product_id, d.as_of, p.category_id
FROM products p
CROSS JOIN (SELECT unnest([TIMESTAMP '2025-12-01', TIMESTAMP '2026-01-01',
                           TIMESTAMP '2026-02-01', TIMESTAMP '2026-03-01',
                           TIMESTAMP '2026-04-01']) AS as_of) d;

-- line_items carries no timestamp of its own, so occurred_at is joined in from
-- transactions. An undated child would be invisible to HORIZON, and PQL now
-- refuses that rather than quietly forecasting a lifetime total.
CREATE OR REPLACE TABLE snap_sales AS
SELECT row_number() OVER () AS id, s.id AS snapshot_id,
       t.occurred_at, li.qty::DOUBLE AS qty, li.net_sales::DOUBLE AS net_sales
FROM snapshots s
JOIN line_items li ON li.product_id = s.product_id
JOIN transactions t ON t.transaction_id = li.transaction_id
WHERE NOT COALESCE(li.voided, false);

SELECT (SELECT count(*) FROM snapshots) AS snapshots,
       (SELECT count(*) FROM snap_sales) AS snap_sales;

-- How many units will this product move in the next 30 days?
-- Read `baseline` next to `test`: it is persistence (repeat the last window),
-- and a model that cannot beat it is not worth deploying.
TRAIN MODEL demand
  PREDICT COUNT(snap_sales) FOR snapshots AT as_of HORIZON 30 DAYS
  OPTIONS (epochs = 120, hidden = 48, mean_cols = 5, seed = 1234);

-- Will it sell at all? (a slow-mover flag)
TRAIN MODEL moves
  PREDICT EXISTS(snap_sales) FOR snapshots AT as_of HORIZON 30 DAYS
  OPTIONS (epochs = 120, hidden = 48, mean_cols = 5);

SELECT * FROM pql_models();

-- One model, many questions.
SELECT 'one product (133)' AS slice, count(*) AS n, round(avg(prediction),1) AS avg_demand
FROM pql_exec('PREDICT COUNT(snap_sales) FOR snapshots WHERE product_id = 133 USING MODEL demand')
UNION ALL SELECT 'category 17', count(*), round(avg(prediction),1)
FROM pql_exec('PREDICT COUNT(snap_sales) FOR snapshots WHERE category_id = 17 USING MODEL demand')
UNION ALL SELECT 'cat 17 or 32', count(*), round(avg(prediction),1)
FROM pql_exec('PREDICT COUNT(snap_sales) FOR snapshots WHERE category_id = 17 OR category_id = 32 USING MODEL demand');

-- Predictions join back to your own tables. Timestamp literals work in filters.
SELECT s.product_id, round(p.prediction,1) AS predicted_next_30d
FROM pql_exec('PREDICT COUNT(snap_sales) FOR snapshots WHERE as_of = ''2026-04-01'' USING MODEL demand') p
JOIN snapshots s ON s.id = p.id
ORDER BY p.prediction DESC LIMIT 5;
