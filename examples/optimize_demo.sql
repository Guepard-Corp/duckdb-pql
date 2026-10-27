-- PQL demo: from a breakdown forecast to a maintenance plan.
-- Run:  build/release/duckdb -c ".read examples/optimize_demo.sql"

-- 800 trucks. A quarter are on the 'care' plan, and the risky ones on it come
-- back for repairs far less often.
CREATE OR REPLACE TABLE trucks AS
  SELECT i AS id,
         TIMESTAMP '2026-01-01' + INTERVAL '1 day' * ((i * 7) % 200) AS as_of,
         (i % 12) AS age,
         50 + (i * 37 % 150) AS service_cost,
         CASE WHEN i % 4 = 0 THEN 'care' ELSE 'basic' END AS plan
  FROM range(800) t(i);

CREATE OR REPLACE TABLE repairs AS
  SELECT row_number() OVER () AS id, t.id AS truck_id,
         t.as_of - INTERVAL '1 day' * (k * 9 + 1) AS at
  FROM trucks t, range(20) r(k) WHERE k < (t.id % 15);

INSERT INTO repairs
  SELECT 900000 + t.id, t.id, t.as_of + INTERVAL '1 day' * 5
  FROM trucks t WHERE (t.id % 15) > 8 AND (t.plan = 'basic' OR t.id % 3 = 0);

-- 1. will this truck need a repair in the next 30 days?
TRAIN MODEL breakdown PREDICT EXISTS(repairs) FOR trucks AT as_of HORIZON 30 DAYS
  OPTIONS (epochs = 60, hidden = 32);

-- 2. five service bays this week: a breakdown costs 4200, a service what the
--    truck's row says. The five trucks worth most, best first.
OPTIMIZE trucks
  MAXIMIZE SUM(prediction * 4200 - service_cost)
  SUBJECT TO COUNT(*) <= 5
  USING MODEL breakdown;

-- 3. a budget instead of bays, and no more than three of the oldest trucks.
--    EXPLAIN reports the totals; `bound` equal to `value` means proven best.
EXPLAIN OPTIMIZE trucks
  MAXIMIZE SUM(prediction * 4200 - service_cost)
  SUBJECT TO SUM(service_cost) <= 2000 AND COUNT(*) FILTER (WHERE age > 9) <= 3
  USING MODEL breakdown;

-- 4. a plan per truck: the model is asked what each plan would change, and
--    'care' costs 300 with room for 150 trucks on it.
EXPLAIN OPTIMIZE trucks
  CHOOSE plan IN ('basic', 'care')
  MINIMIZE SUM(prediction * 4200 + CASE plan WHEN 'care' THEN 300 ELSE 0 END)
  SUBJECT TO COUNT(*) FILTER (WHERE plan = 'care') <= 150
  USING MODEL breakdown;

-- 5. it is still SQL: which trucks would change plan?
SELECT t.plan AS today, p.plan AS planned, count(*) AS trucks
FROM pql_exec('OPTIMIZE trucks CHOOSE plan IN (''basic'', ''care'')
                 MINIMIZE SUM(prediction * 4200 + CASE plan WHEN ''care'' THEN 300 ELSE 0 END)
                 SUBJECT TO COUNT(*) FILTER (WHERE plan = ''care'') <= 150
                 USING MODEL breakdown') p
JOIN trucks t USING (id)
GROUP BY ALL ORDER BY ALL;
