#!/bin/bash
# setup_sf1_nopk.sh
# Provision a NEW database tpch_sf1_nopk that mirrors tpch_sf1_pkonly but
# WITHOUT any PRIMARY KEY constraints (literal Kossmann §5.2).

set -euo pipefail
ROOT=${IS_QDBO_ROOT:-$(cd "$(dirname "$0")/../../workloads/index_selection"; pwd)}
: "${PGHOST:?Set PGHOST in config.env}"
: "${PGPORT:?Set PGPORT in config.env}"
: "${IS_QDBO_SOURCE_DB:?Set IS_QDBO_SOURCE_DB in config.env}"
: "${IS_QDBO_DB_NOPK:?Set IS_QDBO_DB_NOPK in config.env}"

DB=$IS_QDBO_DB_NOPK
SRC_DB=$IS_QDBO_SOURCE_DB

if psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB}'" -d postgres | grep -q 1; then
  echo "[$(date)] DB $DB already exists; skipping create."
else
  echo "[$(date)] CREATE DATABASE $DB ..."
  psql -d postgres -c "CREATE DATABASE ${DB};"

  echo "[$(date)] Loading schema (no PK indices)..."
  psql -d $DB -f $ROOT/configs/tpch_schema.sql

  echo "[$(date)] Loading data via COPY ... TO STDOUT | COPY ... FROM STDIN ..."
  for tbl in region nation part supplier partsupp customer orders lineitem; do
    start=$(date +%s)
    echo "  $tbl..."
    psql -d $SRC_DB -c "COPY $tbl TO STDOUT WITH (FORMAT csv, DELIMITER '|', NULL '')" \
      | psql -d $DB -c "COPY $tbl FROM STDIN WITH (FORMAT csv, DELIMITER '|', NULL '')"
    end=$(date +%s)
    echo "    $tbl: $((end - start))s"
  done

  echo "[$(date)] ALTER DATABASE settings (parallel + statement_timeout)..."
  psql -d postgres -c "ALTER DATABASE ${DB} SET max_parallel_workers_per_gather = 0;"
  psql -d postgres -c "ALTER DATABASE ${DB} SET statement_timeout = '3600s';"
fi

echo "[$(date)] Verifying NO primary keys..."
PK_COUNT=$(psql -d $DB -tAc "SELECT count(*) FROM pg_constraint WHERE contype='p'")
INDEX_COUNT=$(psql -d $DB -tAc "SELECT count(*) FROM pg_indexes WHERE schemaname='public'")
ROW_COUNT_LI=$(psql -d $DB -tAc "SELECT count(*) FROM lineitem")
SIZE=$(psql -d $DB -tAc "SELECT pg_size_pretty(pg_database_size('${DB}'))")

echo
echo "=== $DB summary ==="
echo "primary keys (contype=p): $PK_COUNT"
echo "user indexes (pg_indexes public): $INDEX_COUNT"
echo "lineitem rows: $ROW_COUNT_LI"
echo "DB size: $SIZE"
[ "$PK_COUNT" = "0" ] && echo "OK: zero PKs (strict no-PK)" || { echo "ERR: expected 0 PKs, got $PK_COUNT"; exit 1; }
[ "$INDEX_COUNT" = "0" ] && echo "OK: zero indexes" || echo "WARN: $INDEX_COUNT indexes present (should be 0)"
