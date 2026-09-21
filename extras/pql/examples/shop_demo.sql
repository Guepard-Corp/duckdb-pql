-- PQL demo: a small shop database, then three models over it.
-- Run:  build/reldebug/duckdb -c ".read extras/pql/examples/shop_demo.sql"

CREATE OR REPLACE TABLE customers AS
  SELECT i AS id,
         TIMESTAMP '2025-06-01' + INTERVAL '1 day' * ((i*13)%200) AS last_seen,
         CASE WHEN i%3=0 THEN 'EU' WHEN i%3=1 THEN 'US' ELSE 'APAC' END AS region,
         (i%7)+1 AS tier,
         ((i%17)*3.5) AS lifetime_value
  FROM range(1500) t(i);

CREATE OR REPLACE TABLE orders AS
  SELECT row_number() OVER () AS id, c.id AS customer_id,
         c.last_seen - INTERVAL '1 day' * (k*7+1) AS placed_at,
         20.0 + (k*3) AS amount,
         CASE WHEN (c.id+k)%9=0 THEN 'refunded' ELSE 'paid' END AS status
  FROM customers c, range(20) r(k) WHERE k < (c.id % 18);

-- customers with a long history also order again inside the horizon
-- (id%9=0 never coincides with id%18>9, so the refund rule here is its own:
-- otherwise the refunds model below would have no positive example at all)
INSERT INTO orders
  SELECT 800000+c.id, c.id, c.last_seen + INTERVAL '1 day' * 12, 30.0,
         CASE WHEN c.id%4=0 THEN 'refunded' ELSE 'paid' END
  FROM customers c WHERE (c.id % 18) > 9;

-- 1. will this customer order again in the next 30 days?
TRAIN MODEL churn PREDICT EXISTS(orders) FOR customers AT last_seen HORIZON 30 DAYS
  OPTIONS (epochs = 100, hidden = 48);

-- 2. how much will they spend in the next 60?
TRAIN MODEL spend PREDICT SUM(orders.amount) FOR customers AT last_seen HORIZON 60 DAYS
  OPTIONS (epochs = 100, hidden = 48);

-- 3. an inner filter changes the question, not the population
TRAIN MODEL refunds PREDICT EXISTS(orders WHERE orders.status = 'refunded')
  FOR customers AT last_seen HORIZON 60 DAYS OPTIONS (epochs = 100, hidden = 48);

SELECT * FROM pql_models();

-- conditioning: the same model, different slices
PREDICT EXISTS(orders) FOR customers WHERE id IN (7, 19, 33, 41) USING MODEL churn;

SELECT 'EU'   AS region, round(avg(prediction),3) AS avg_p
FROM pql_exec('PREDICT EXISTS(orders) FOR customers WHERE region = ''EU'' USING MODEL churn')
UNION ALL
SELECT 'US',  round(avg(prediction),3)
FROM pql_exec('PREDICT EXISTS(orders) FOR customers WHERE region = ''US'' USING MODEL churn');

-- predictions compose with ordinary SQL
SELECT c.id, c.region, round(p.prediction,2) AS expected_spend_60d
FROM pql_exec('PREDICT SUM(orders.amount) FOR customers USING MODEL spend') p
JOIN customers c ON c.id = p.id
ORDER BY p.prediction DESC LIMIT 5;
