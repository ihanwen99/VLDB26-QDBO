#!/bin/bash
# Loads TPC-H SF=1 data into a PostgreSQL database.
#
# Required env vars are documented in config.example.env. This script has no
# machine-specific path, host, port, or database defaults.

set -euo pipefail
: "${TPCH_DBGEN_DIR:?Set TPCH_DBGEN_DIR in config.env}"
: "${PGHOST:?Set PGHOST in config.env}"
: "${PGPORT:?Set PGPORT in config.env}"
: "${PGDATABASE:?Set PGDATABASE in config.env}"
DBGEN_DIR=$TPCH_DBGEN_DIR
DB=$PGDATABASE

if [ ! -x "$DBGEN_DIR/dbgen" ]; then
  echo "ERROR: TPC-H dbgen not found at $DBGEN_DIR/dbgen"
  echo "       Set TPCH_DBGEN_DIR or build it first."
  exit 1
fi

echo "[$(date)] Generating SF=1 data via $DBGEN_DIR/dbgen ..."
(cd "$DBGEN_DIR" && ./dbgen -s 1 -f)

echo "[$(date)] Loading schema into $DB at $PGHOST:$PGPORT ..."
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PGHOST=$PGHOST PGPORT=$PGPORT psql -d $DB -f $SCRIPT_DIR/tpch_schema.sql

for tbl in region nation supplier customer part partsupp orders lineitem; do
  echo "[$(date)] Copying $tbl ..."
  PGHOST=$PGHOST PGPORT=$PGPORT psql -d $DB -c "\\copy $tbl FROM $DBGEN_DIR/$tbl.tbl DELIMITER '|' CSV"
done

echo "[$(date)] Done."
