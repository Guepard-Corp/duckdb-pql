-- The dataset Part 1 of the README walks through. Load it, then paste the
-- statements from the tutorial as you read.
--
--   build/reldebug/duckdb -c ".read extras/pql/examples/tutorial.sql"

CREATE OR REPLACE TABLE customers AS
  SELECT i AS id,
         TIMESTAMP '2025-01-01' + INTERVAL '1 day' * ((i * 13) % 300) AS last_seen,
         ['EU', 'US', 'APAC'][1 + (i % 3)] AS region,
         (i % 7) + 1 AS tier,
         -- mostly explained by how much they ordered, but not entirely
         CASE WHEN (i % 18) < 6 THEN (i % 5) <> 0 ELSE (i % 5) = 0 END AS churned
  FROM range(1500) t(i);

CREATE OR REPLACE TABLE orders AS
  SELECT row_number() OVER () AS id, c.id AS customer_id,
         c.last_seen - INTERVAL '1 day' * (k * 7 + 1) AS placed_at,
         20.0 + (k * 3) AS amount,
         CASE WHEN (c.id + k) % 9 = 0 THEN 'refunded' ELSE 'paid' END AS status
  FROM customers c, range(20) r(k) WHERE k < (c.id % 18);

-- Customers with a long history are likelier to order again inside the horizon,
-- but not certain to. A target that follows exactly from a feature trains to a
-- perfect score and teaches nothing.
-- They keep the rhythm they had, which is what makes persistence a real
-- opponent for the spend forecast rather than a straw man.
INSERT INTO orders
  SELECT 900000 + c.id * 10 + k, c.id,
         c.last_seen + INTERVAL '1 day' * (k * 7 + 2), 20.0 + (k * 3), 'paid'
  FROM customers c, range(20) r(k)
  WHERE k < (c.id % 18)
    AND (((c.id % 18) > 9 AND (c.id * 7 % 10) < 7)    -- most heavy buyers return
      OR ((c.id % 18) <= 9 AND (c.id * 7 % 10) < 2)); -- and a few light ones do
