-- Load the full sunnyside_v2 copy into DuckDB.
--   build/reldebug/duckdb /tmp/sunny.duckdb -c ".read extras/pql/examples/sunnyside_load.sql"
CREATE OR REPLACE TABLE categories          AS SELECT * FROM read_csv_auto('/tmp/sunnyside/categories.csv');
CREATE OR REPLACE TABLE customers           AS SELECT * FROM read_csv_auto('/tmp/sunnyside/customers.csv');
CREATE OR REPLACE TABLE dining_options      AS SELECT * FROM read_csv_auto('/tmp/sunnyside/dining_options.csv');
CREATE OR REPLACE TABLE locations           AS SELECT * FROM read_csv_auto('/tmp/sunnyside/locations.csv');
CREATE OR REPLACE TABLE modifiers           AS SELECT * FROM read_csv_auto('/tmp/sunnyside/modifiers.csv');
CREATE OR REPLACE TABLE price_points        AS SELECT * FROM read_csv_auto('/tmp/sunnyside/price_points.csv');
CREATE OR REPLACE TABLE devices             AS SELECT * FROM read_csv_auto('/tmp/sunnyside/devices.csv');
CREATE OR REPLACE TABLE products            AS SELECT * FROM read_csv_auto('/tmp/sunnyside/products.csv');
CREATE OR REPLACE TABLE transactions        AS SELECT * FROM read_csv_auto('/tmp/sunnyside/transactions.csv');
CREATE OR REPLACE TABLE line_items          AS SELECT * FROM read_csv_auto('/tmp/sunnyside/line_items.csv');
CREATE OR REPLACE TABLE payments            AS SELECT * FROM read_csv_auto('/tmp/sunnyside/payments.csv');
CREATE OR REPLACE TABLE payment_tenders     AS SELECT * FROM read_csv_auto('/tmp/sunnyside/payment_tenders.csv');
CREATE OR REPLACE TABLE line_item_modifiers AS SELECT * FROM read_csv_auto('/tmp/sunnyside/line_item_modifiers.csv');
CREATE OR REPLACE TABLE calendar_days       AS SELECT * FROM read_csv_auto('/tmp/sunnyside/calendar_days.csv');
CREATE OR REPLACE TABLE category_encodings  AS SELECT * FROM read_csv_auto('/tmp/sunnyside/category_encodings.csv');

SELECT 'categories' t, count(*) n FROM categories UNION ALL SELECT 'customers', count(*) FROM customers
UNION ALL SELECT 'products', count(*) FROM products UNION ALL SELECT 'transactions', count(*) FROM transactions
UNION ALL SELECT 'line_items', count(*) FROM line_items UNION ALL SELECT 'payments', count(*) FROM payments
UNION ALL SELECT 'payment_tenders', count(*) FROM payment_tenders
UNION ALL SELECT 'line_item_modifiers', count(*) FROM line_item_modifiers
UNION ALL SELECT 'calendar_days', count(*) FROM calendar_days ORDER BY 1;
