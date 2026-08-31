#!/bin/bash
ROOT=${IS_QDBO_ROOT:-$(cd "$(dirname "$0")/../../workloads/index_selection"; pwd)}
REPO_ROOT=$(cd "$(dirname "$0")/../.."; pwd)
EXP=${IS_QDBO_EXP_DIR:-$REPO_ROOT/results/index_selection/sf1_nopk_19q_e2_runtime}
REPO=${IS_QDBO_PIPELINE_DIR:-$REPO_ROOT}
VENV_PG=${IS_QDBO_PYTHON:-python}
VENV_Q=${IS_QDBO_PYTHON:-python}
NUM_READS=100
SEED=42
EMBEDDINGS=(random greedy semanticINDEX_greedy semanticSLACK_greedy)
TZ_PT() { TZ="America/Los_Angeles" date "+%H:%M:%S PT"; }

# Generate budgets.json from problem.json if absent (idempotent).
# sigma = sum(weights); per percent: units = max(1, int(sigma*pct/100)),
# mb = units * weight_unit_mb (=1 in the e2 instance, so units==mb).
# The 4 percents [10,20,40,50] are the storage budgets reported in the paper
# (Table 4) and match the solve loop below; the schema keys (sigma_units /
# budgets / {pct}pct / units / mb) match the reads in the loop.
if [ ! -f "$EXP/budgets.json" ]; then
  $VENV_Q <<PYEOF2
import json
p = json.load(open("$EXP/problem.json"))
sigma = sum(p["weights"])
unit = p["metadata"]["weight_unit_mb"]
out = {"sigma_units": sigma, "budgets": {}}
for pct in [10, 20, 40, 50]:
    bu = max(1, int(sigma * pct / 100))
    out["budgets"][f"{pct}pct"] = {"units": bu, "mb": bu * unit}
json.dump(out, open("$EXP/budgets.json", "w"), indent=2)
print(out)
PYEOF2
fi

for PCT in 10 20 40 50; do
  BD=$EXP/budget_${PCT}pct
  mkdir -p $BD
  BUD_UNITS=$($VENV_Q -c "import json; print(json.load(open(\"$EXP/budgets.json\"))[\"budgets\"][\"${PCT}pct\"][\"units\"])")
  BUD_MB=$($VENV_Q -c "import json; print(json.load(open(\"$EXP/budgets.json\"))[\"budgets\"][\"${PCT}pct\"][\"mb\"])")
  echo "[$(TZ_PT)] === B=${PCT}% (${BUD_UNITS}u = ${BUD_MB}MB) ==="

  if [ ! -f "$BD/problem.json" ]; then
    $VENV_Q <<PYEOF2
import json, copy
p=json.load(open("$EXP/problem.json")); p2=copy.deepcopy(p); p2["max_weight"]=$BUD_UNITS; p2["metadata"]["budget_mb"]=$BUD_MB
json.dump(p2, open("$BD/problem.json","w"), indent=2)
PYEOF2
  fi

  [ ! -f "$BD/04_qubo.json" ] && PYTHONPATH=$REPO $VENV_PG -m qdbo build-qubo \
    --problem $BD/problem.json --out $BD/04_qubo.json --include-ising 2>&1 | tail -1

  # Classical baselines
  [ ! -f "$BD/sol_greedy.json" ] && PYTHONPATH=$REPO $VENV_PG -m qdbo solve \
    --problem $BD/problem.json --qubo $BD/04_qubo.json \
    --out $BD/sol_greedy.json --solver greedy 2>&1 | tail -1

  [ ! -f "$BD/sol_exact.json" ] && PYTHONPATH=$REPO $VENV_PG -m qdbo solve \
    --problem $BD/problem.json --qubo $BD/04_qubo.json \
    --out $BD/sol_exact.json --solver exact-knapsack 2>&1 | tail -1

  # BQM-Solver via LeapHybridBQMSampler
  SOL_HYBRID=$BD/sol_hybrid.json
  if [ ! -f "$SOL_HYBRID" ]; then
    PYTHONPATH=$REPO $VENV_PG -m qdbo solve \
      --problem $BD/problem.json --qubo $BD/04_qubo.json --out $SOL_HYBRID \
      --solver dwave-hybrid --time-limit 5 2>&1 | tail -1
  fi

  # QDBO 4 emb x 2 iter
  for EMB in "${EMBEDDINGS[@]}"; do
    EMB_DIR=$BD/qdbo_${EMB}; mkdir -p $EMB_DIR
    if [ ! -f "$EMB_DIR/qdbo_iterative_summary.json" ]; then
      echo "[$(TZ_PT)] QDBO ${EMB} B=${PCT}%..."
      $VENV_Q $REPO_ROOT/scripts/index_selection/run_qdbo_iterative_on_index_qubo.py \
        --problem $BD/problem.json --qubo $BD/04_qubo.json \
        --reference-solution $SOL_HYBRID --out-dir $EMB_DIR \
        --iterations 1,3 --embedding $EMB --num-reads $NUM_READS \
        2>&1 | tail -2
      if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "  [WARN] QDBO ${EMB} failed"; fi
    fi
    # Trim QDBO solutions to feasibility per knapsack budget
    for K in 1 3; do
      ITER=$EMB_DIR/qdbo_iter_${K}.json
      TRIM=$EMB_DIR/sol_iter_${K}_trimmed.json
      if [ -f "$ITER" ] && [ ! -f "$TRIM" ]; then
        $VENV_Q <<PYEOF2
import json, sys
qd=json.load(open("$ITER")); row=qd.get("row",qd)
if row.get("status") != "ok" or "selected_ids" not in row:
    # A failed solver run must not be turned into a fake empty-but-feasible solution.
    sys.stderr.write("[WARN] skipping trim for $EMB iter${K}: solver run failed (%s)\n" % row.get("error_type", "unknown"))
    sys.exit(0)
ids=row["selected_ids"]
prob=json.load(open("$BD/problem.json"))
weights=prob["weights"]; profits=prob["profits"]; budget=prob["max_weight"]
if sum(weights[i] for i in ids) <= budget: keep=ids
else:
    pairs=sorted(ids, key=lambda i: -profits[i]/max(weights[i],1))
    keep=[]; w=0
    for i in pairs:
        if w + weights[i] <= budget: keep.append(i); w += weights[i]
out={"solver_label":"qdbo--iter${K}-trimmed","selected_ids":sorted(keep),
     "feasible":True,"weight":int(sum(weights[i] for i in keep)),
     "profit":int(sum(profits[i] for i in keep)),
     "solver_info":{"embedding":"","iter":${K},"num_reads":${NUM_READS},"post_hoc_trim":"profit_weight_greedy"}}
# status=="ok" rows come from a real DWaveSampler run, so default to "qpu".
out["solver_info"]["qpu_or_cpu"]=row.get("qpu_or_cpu","qpu")
json.dump(out, open("$TRIM","w"), indent=2)
PYEOF2
      fi
    done
  done
done

echo "[$(TZ_PT)] === all 4 budgets solve done ==="
