#!/bin/bash
# E2 replay only: build PG indexes + 3-run hot-cache replay × 19q × 44 cells
#   (4 budgets {10,20,40,50}% × 11 selectors; budgets per paper Table 4)
# Cap: per-query 60s + cumulative cap = 2 × baseline_total × 3-run (~649s)
# sol_cache dedup: hash(sorted(selected_ids)) -> reuse replay txt

set -uo pipefail
ROOT=${IS_QDBO_ROOT:-$(cd "$(dirname "$0")/../../workloads/index_selection"; pwd)}
: "${PGHOST:?Set PGHOST in config.env}"
: "${PGPORT:?Set PGPORT in config.env}"
: "${IS_QDBO_DB_NOPK:?Set IS_QDBO_DB_NOPK in config.env}"
REPO_ROOT=$(cd "$(dirname "$0")/../.."; pwd)
EXP=${IS_QDBO_EXP_DIR:-$REPO_ROOT/results/index_selection/sf1_nopk_19q_e2_runtime}
DB=$IS_QDBO_DB_NOPK
VENV_Q=${IS_QDBO_PYTHON:-python}
PG_ENDPOINT=$PGHOST
PG_PORT=$PGPORT
TIMEOUT_S=60
QUERIES_19=(1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 21 22)
EMBEDDINGS=(random greedy semanticINDEX_greedy semanticSLACK_greedy)
HEARTBEAT=$EXP/logs/e2_replay.log
mkdir -p "$EXP/logs" "$EXP/replay_cache"
now() { TZ='America/Los_Angeles' date '+%Y-%m-%dT%H:%M:%S%z'; }

# Compute cell cap from real baseline_per_q.json (2x × 3run)
CELL_WALL_CAP_MS=$($VENV_Q -c "
import json
b=json.load(open('$EXP/baseline_per_query.json'))
print(int(2 * sum(b.values()) * 3))
")
echo "[$(now)] === Task E2 REPLAY (cap: per-query 60s, cell cap=${CELL_WALL_CAP_MS}ms = $((CELL_WALL_CAP_MS/1000))s) ===" | tee -a "$HEARTBEAT"

hb_done=0
hb_inc() { hb_done=$((hb_done+1)); echo "[$(now)] cells=$hb_done $1" >> "$HEARTBEAT"; }

replay_one() {
  local SOL=$1 LABEL=$2
  local BD_DIR=$(dirname $(dirname "$SOL"))
  if [[ "$SOL" == *qdbo_*/sol_* ]]; then
    BD_DIR=$(dirname $(dirname "$SOL"))
  else
    BD_DIR=$(dirname "$SOL")
  fi
  local PCT_DIR=$(echo "$SOL" | grep -oE 'budget_[0-9]+pct' | head -1)
  local BD=$EXP/$PCT_DIR
  local RPL=$BD/replay
  mkdir -p "$RPL"
  local PCT=$(echo "$PCT_DIR" | grep -oE '[0-9]+')
  local RPL_TXT=$RPL/${LABEL}_b${PCT}.txt

  [ ! -f "$SOL" ] && return
  if [ -f "$RPL_TXT" ] && grep -q "Q22 run3:" "$RPL_TXT"; then
    echo "  [skip] $RPL_TXT done"
    hb_inc "B=${PCT}%-${LABEL}-cached"
    return
  fi

  # cache dedup
  local IDS_HASH=$($VENV_Q -c "
import json, hashlib
sol=json.load(open('$SOL'))
ids=tuple(sorted(sol.get('selected_ids', [])))
print(hashlib.sha1(repr(ids).encode()).hexdigest()[:16] + '_n' + str(len(ids)))
")
  local CACHE_TXT=$EXP/replay_cache/${IDS_HASH}.txt
  if [ -f "$CACHE_TXT" ] && grep -q "Q22 run3:" "$CACHE_TXT"; then
    cp "$CACHE_TXT" "$RPL_TXT"
    sed -i.bak "1i# replay_cache_hit: $IDS_HASH" "$RPL_TXT" && rm -f "$RPL_TXT.bak"
    echo "  [cache_hit] $LABEL B=${PCT}% reused $IDS_HASH"
    hb_inc "B=${PCT}%-${LABEL}-cachehit"
    return
  fi

  # Build DDL list
  local DDLS=$($VENV_Q <<PYEOF
import json
sol=json.load(open("$SOL")); prob=json.load(open("$BD/problem.json"))
ids=sol.get("selected_ids", [])
ddl_by_cid={c["candidate_id"]: c["ddl"] for c in prob["candidates"]}
cids=[e["candidate_id"] for e in prob["evaluations"]]
selected_cids=[cids[i] for i in ids if 0<=i<len(cids)]
for cid in selected_cids: print(ddl_by_cid[cid])
PYEOF
)
  local SQL_PREP=$(mktemp); local SQL_TEAR=$(mktemp)
  echo "BEGIN;" > "$SQL_PREP"
  while IFS= read -r ddl; do [ -z "$ddl" ] && continue; echo "$ddl;" >> "$SQL_PREP"; done <<<"$DDLS"
  echo "COMMIT;" >> "$SQL_PREP"
  echo "BEGIN;" > "$SQL_TEAR"
  while IFS= read -r ddl; do
    [ -z "$ddl" ] && continue
    name=$(echo "$ddl" | sed -nE 's/.*INDEX +"([^"]+)".*/\1/p')
    [ -n "$name" ] && echo "DROP INDEX IF EXISTS \"public\".\"$name\";" >> "$SQL_TEAR"
  done <<<"$DDLS"
  echo "COMMIT;" >> "$SQL_TEAR"

  PGHOST=$PG_ENDPOINT PGPORT=$PG_PORT psql -d $DB -f "$SQL_PREP" >/dev/null 2>&1 || {
    PGHOST=$PG_ENDPOINT PGPORT=$PG_PORT psql -d $DB -f "$SQL_TEAR" >/dev/null 2>&1
    rm -f "$SQL_PREP" "$SQL_TEAR"
    echo "  [ERR] index build failed for $LABEL B=${PCT}%"
    return
  }

  echo "# replay_cache_miss: $IDS_HASH" > "$RPL_TXT"
  echo "[$(now)] === ${LABEL} B=${PCT}% (per-q cap=${TIMEOUT_S}s, cell cap=${CELL_WALL_CAP_MS}ms) ===" >> "$RPL_TXT"

  local cell_cum_ms=0
  local cell_aborted=0
  for q in "${QUERIES_19[@]}"; do
    for run in 1 2 3; do
      if [ $cell_aborted -eq 1 ]; then
        echo "Q${q} run${run}: ${TIMEOUT_S}000ms cell_aborted_overcap" >> "$RPL_TXT"
        continue
      fi
      QSQL=$(sed -E "s/;[[:space:]]*$//" "$ROOT/queries/q${q}.sql")
      EA_OUT=$(PGHOST=$PG_ENDPOINT PGPORT=$PG_PORT PGOPTIONS="-c statement_timeout=${TIMEOUT_S}s" psql -d $DB -tAc "EXPLAIN (ANALYZE, TIMING ON, FORMAT TEXT) ${QSQL}" 2>/dev/null)
      rc=$?
      if [ $rc -ne 0 ]; then
        ms=$(( TIMEOUT_S * 1000 )); tag=capped
      else
        EA_MS=$(echo "$EA_OUT" | grep -oE "Execution Time: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
        if [ -z "$EA_MS" ]; then
          ms=$(( TIMEOUT_S * 1000 )); tag=ea_parse_fail
        else
          ms=$(printf "%.0f" "$EA_MS"); tag=ok
        fi
      fi
      cell_cum_ms=$(( cell_cum_ms + ms ))
      echo "Q${q} run${run}: ${ms}ms ${tag}" >> "$RPL_TXT"
      if [ $cell_cum_ms -gt $CELL_WALL_CAP_MS ]; then
        cell_aborted=1
        echo "[$(now)] $LABEL B=${PCT}% ABORT: cumulative ${cell_cum_ms}ms > ${CELL_WALL_CAP_MS}ms cap" | tee -a "$RPL_TXT"
      fi
    done
  done

  PGHOST=$PG_ENDPOINT PGPORT=$PG_PORT psql -d $DB -f "$SQL_TEAR" >/dev/null 2>&1
  rm -f "$SQL_PREP" "$SQL_TEAR"
  echo "[$(now)] === ${LABEL} B=${PCT}% DONE ===" >> "$RPL_TXT"
  cp "$RPL_TXT" "$CACHE_TXT"
  hb_inc "B=${PCT}%-${LABEL}"
}

for PCT in 10 20 40 50; do
  BD=$EXP/budget_${PCT}pct
  echo "[$(now)] === replay budget B=${PCT}% ==="
  replay_one "$BD/sol_greedy.json" "greedy"
  replay_one "$BD/sol_exact.json"  "exact"
  replay_one "$BD/sol_hybrid.json" "hybrid"
  for EMB in "${EMBEDDINGS[@]}"; do
    for K in 1 3; do
      replay_one "$BD/qdbo_${EMB}/sol_iter_${K}_trimmed.json" "qdbo_${EMB}_iter${K}"
    done
  done
done

# hb_done counts every cell that produced a replay txt (fresh, cache hit, or
# already complete). Zero means no cell got through: either no sol_*.json
# exists under $EXP (run sweep_solve.sh first) or every index build failed.
if [ "$hb_done" -eq 0 ]; then
  echo "[$(now)] ERROR: 0 cells replayed under $EXP; no sol_*.json found or all index builds failed (see $HEARTBEAT)" >&2
  exit 1
fi
echo "[$(now)] === ALL ${hb_done} CELLS DONE ===" >> "$HEARTBEAT"
echo "[$(now)] e2 replay complete"
