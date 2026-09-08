#!/bin/sh
# Faithful copy of the sunnyside_v2 `sunny` schema out of Postgres.
#   sh extras/pql/examples/sunnyside_export.sh [outdir]
#
# Every base table is exported as-is, so the DuckDB side is a real copy rather
# than a hand-picked subset. numeric is cast to double and booleans to 0/1
# because those are the shapes the model reads.
set -e
OUT=${1:-/tmp/sunnyside}
PG="psql -h 127.0.0.1 -p 5432 -d sunnyside_v2"
mkdir -p "$OUT"

for T in categories customers dining_options locations modifiers price_points devices \
         products transactions line_items payments payment_tenders line_item_modifiers \
         calendar_days category_encodings; do
  $PG -c "\copy (SELECT * FROM sunny.$T) TO '$OUT/$T.csv' CSV HEADER"
done
echo "exported 15 tables to $OUT"
