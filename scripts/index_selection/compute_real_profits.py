#!/usr/bin/env python3
"""compute_real_profits.py

Real EXPLAIN ANALYZE profit + real pg_relation_size weight for each candidate,
single psycopg2 connection (hot-cache aligned, capped EXPLAIN ANALYZE).

Per-candidate flow (within ONE persistent connection):
  1. BEGIN
  2. CREATE INDEX (ddl)
  3. SELECT pg_relation_size(real_index)         → weight
  4. for each query: EXPLAIN ANALYZE -> wall ms  → benefit (cap=60s)
  5. ROLLBACK (clean up — index never persisted)

Benefit per candidate = sum_q (baseline_ms - treatment_ms)
Profit = round(benefit / profit_unit_ms)
Weight = ceil(size_bytes / weight_unit_mb / 1024 / 1024)

Capped query is recorded as cap_seconds*1000 ms.

Args:
  --db
  --candidates
  --queries-dir
  --query-ids-file
  --baseline-json   per-query baseline ms (JSON dict {qid: ms}); if missing, computed first
  --out-problem
  --workload-name
  --weight-unit-mb  default 1
  --profit-unit-ms  default 100
  --cap-seconds     default 60
  --dsn             PostgreSQL DSN; defaults to PG_DSN or QDBO_DB_DSN
"""
import argparse, json, math, os, time, signal
from pathlib import Path
import psycopg2
from psycopg2 import extensions

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument(
    "--dsn", default=os.environ.get("PG_DSN") or os.environ.get("QDBO_DB_DSN")
)
ap.add_argument("--candidates", required=True)
ap.add_argument("--queries-dir", required=True)
ap.add_argument("--query-ids-file", required=True)
ap.add_argument("--baseline-json", default=None)
ap.add_argument("--out-problem", required=True)
ap.add_argument("--workload-name", required=True)
ap.add_argument("--weight-unit-mb", type=int, default=1)
ap.add_argument("--profit-unit-ms", type=float, default=100.0)
ap.add_argument("--cap-seconds", type=int, default=60, help="per-query cap")
ap.add_argument("--baseline-runs", type=int, default=3, help="baseline runs (3 for hot-cache)")
ap.add_argument("--candidate-runs", type=int, default=1, help="per-candidate profit-measurement runs (1 by user direction)")
ap.add_argument("--candidate-cap-multiplier", type=float, default=3.0,
                help="candidate aborts if cumulative wall-clock > multiplier * baseline_total")
ap.add_argument("--smoke-limit", type=int, default=0,
                help="if > 0, only process the first N candidates (for smoke)")
args = ap.parse_args()
if not args.dsn:
    ap.error("set PG_DSN/QDBO_DB_DSN or pass --dsn")

CAP_MS = args.cap_seconds * 1000

def run_query_with_cap(conn, cur, sql, cap_ms, in_txn=False, sp_name="sp_query"):
    """Run a query with per-statement timeout. Two modes:
    - in_txn=False (autocommit): each query is its own implicit txn; QueryCanceled doesn't poison anything.
    - in_txn=True: wrap in SAVEPOINT so a cancel/error doesn't kill the outer txn.
    Return (wall_ms, ok_or_capped_or_error)."""
    if in_txn:
        try:
            cur.execute(f"SAVEPOINT {sp_name}")
            cur.execute(f"SET LOCAL statement_timeout = '{cap_ms}'")
        except Exception:
            return cap_ms, "txn_aborted"
        t0 = time.time()
        try:
            cur.execute(sql)
            try: cur.fetchall()
            except psycopg2.ProgrammingError: pass
            elapsed = int((time.time() - t0) * 1000)
            cur.execute(f"RELEASE SAVEPOINT {sp_name}")
            return min(elapsed, cap_ms), "ok"
        except psycopg2.errors.QueryCanceled:
            try: cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            except: pass
            try: cur.execute(f"RELEASE SAVEPOINT {sp_name}")
            except: pass
            return cap_ms, "capped"
        except Exception as e:
            try: cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            except: pass
            try: cur.execute(f"RELEASE SAVEPOINT {sp_name}")
            except: pass
            return cap_ms, f"error:{type(e).__name__}"
    else:
        # autocommit mode: SET statement_timeout once, run query, no savepoint
        try:
            cur.execute(f"SET statement_timeout = '{cap_ms}'")
        except Exception:
            pass
        t0 = time.time()
        try:
            cur.execute(sql)
            try: cur.fetchall()
            except psycopg2.ProgrammingError: pass
            elapsed = int((time.time() - t0) * 1000)
            return min(elapsed, cap_ms), "ok"
        except psycopg2.errors.QueryCanceled:
            return cap_ms, "capped"
        except Exception as e:
            return cap_ms, f"error:{type(e).__name__}"

conn = psycopg2.connect(args.dsn, dbname=args.db)
conn.autocommit = True
cur = conn.cursor()
query_ids = [l.strip() for l in open(args.query_ids_file) if l.strip()]
queries_dir = Path(args.queries_dir)

def read_query(qid):
    return (queries_dir / f"{qid}.sql").read_text().replace("\n", " ").strip().rstrip(";")

# Stage 1: baseline (zero index, hot-cache 3-run, mean of run 2+3)
if args.baseline_json and Path(args.baseline_json).exists():
    baseline = {qid: float(v) for qid, v in json.loads(Path(args.baseline_json).read_text()).items()}
    print(f"[stage1] reused baseline from {args.baseline_json}", flush=True)
else:
    print(f"[stage1] running 19q baseline (zero indexes, {args.baseline_runs}-run, cap={args.cap_seconds}s)...", flush=True)
    baseline_runs = {q: [] for q in query_ids}
    for q in query_ids:
        for run in range(1, args.baseline_runs + 1):
            ms, tag = run_query_with_cap(conn, cur, read_query(q), CAP_MS, in_txn=False)
            baseline_runs[q].append((ms, tag))
            print(f"  Q{q} run{run}: {ms} ms {tag}", flush=True)
    baseline = {}
    for q in query_ids:
        # If 3-run, mean of run 2+3 (drop-first hot-cache); if 1-run, take that one
        if args.baseline_runs >= 3:
            baseline[q] = (baseline_runs[q][1][0] + baseline_runs[q][2][0]) / 2
        elif args.baseline_runs == 2:
            baseline[q] = baseline_runs[q][1][0]
        else:
            baseline[q] = baseline_runs[q][0][0]
    if args.baseline_json:
        Path(args.baseline_json).write_text(json.dumps(baseline, indent=2))
        print(f"[stage1] wrote baseline to {args.baseline_json}", flush=True)

# Stage 2: per-candidate: BEGIN; CREATE; size; replay 19q×1; ROLLBACK
cands_all = json.loads(Path(args.candidates).read_text())
cands = cands_all if args.smoke_limit == 0 else cands_all[:args.smoke_limit]
print(f"[stage2] processing {len(cands)} candidates (smoke_limit={args.smoke_limit})", flush=True)

evaluations = []
weights_int = []
profits_int = []
conn.autocommit = False

for i, cand in enumerate(cands):
    cid = cand["candidate_id"]; ddl = cand["ddl"]; name = cand["name"]
    saved_ms = 0.0
    per_q = {}
    n_capped = 0

    cur.execute("BEGIN")
    try:
        cur.execute(ddl)
    except Exception as e:
        cur.execute("ROLLBACK")
        print(f"[c{i:3d}] CREATE INDEX FAIL: {str(e)[:80]}", flush=True)
        weights_int.append(1); profits_int.append(0)
        evaluations.append({"candidate_id": cid, "name": name, "size_bytes": 0,
                            "size_mb": 0.0, "weight": 1, "profit": 0,
                            "raw_estimated_saved_ms": 0.0, "n_capped": 0,
                            "skip_reason": "create_index_failed"})
        continue

    # Real size
    cur.execute("SELECT pg_relation_size(%s)", (f'public."{name}"',))
    size_bytes = int(cur.fetchone()[0])
    weight = max(1, math.ceil(size_bytes / (args.weight_unit_mb * 1024 * 1024)))

    # 1-run EA per query.
    # Candidate aborts early if cumulative wall-clock > 3 * baseline_total
    # (bad candidates get a large negative profit, no infinite stalls.)
    baseline_total = sum(baseline.values())
    candidate_cap_ms = baseline_total * args.candidate_cap_multiplier
    cumulative_ms = 0.0
    aborted = False
    for qid in query_ids:
        if qid not in baseline: continue
        if aborted:
            # Penalize remaining queries: treat them as 3x baseline (bad candidate, big negative profit signal)
            penalty_ms = baseline[qid] * 3
            per_q[f"Q{qid}"] = {"baseline": baseline[qid], "treatment": penalty_ms,
                                "saved": baseline[qid] - penalty_ms, "tag": "skipped_overcap"}
            saved_ms += baseline[qid] - penalty_ms
            n_capped += 1
            continue
        ms, tag = run_query_with_cap(conn, cur, read_query(qid), CAP_MS, in_txn=True, sp_name=f"sp_{cid}_{qid}")
        if tag == "capped" or tag.startswith("error") or tag == "txn_aborted":
            n_capped += 1
        cumulative_ms += ms
        diff = baseline[qid] - ms
        per_q[f"Q{qid}"] = {"baseline": baseline[qid], "treatment": ms, "saved": diff, "tag": tag}
        saved_ms += diff
        # Check cumulative cap
        if cumulative_ms > candidate_cap_ms:
            aborted = True
            print(f"  [c{i}] ABORT after Q{qid}: cumulative {cumulative_ms:.0f}ms > {candidate_cap_ms:.0f}ms (3x baseline)", flush=True)

    cur.execute("ROLLBACK")  # drop the index — no persistent state
    profit_units = int(round(saved_ms / args.profit_unit_ms))
    weights_int.append(weight)
    profits_int.append(profit_units)
    evaluations.append({
        "candidate_id": cid, "name": name,
        "size_bytes": size_bytes, "size_mb": round(size_bytes/(1024*1024), 2),
        "weight": weight,
        "raw_estimated_saved_ms": saved_ms,
        "profit": profit_units,
        "n_capped": n_capped,
        "per_q": per_q,
    })
    print(f"[c{i:3d}/{len(cands)}] sz={size_bytes/1e6:6.1f}MB w={weight:4d} "
          f"saved_ms={saved_ms:+.0f} profit={profit_units:+d} caps={n_capped} {name[:50]}", flush=True)

conn.close()

problem = {
    "name": args.workload_name,
    "profits": profits_int, "weights": weights_int, "max_weight": 0,
    "candidates": cands, "evaluations": evaluations,
    "workload": {"queries": [f"Q{q}" for q in query_ids]},
    "metadata": {
        "weight_unit_mb": args.weight_unit_mb,
        "profit_unit_ms": args.profit_unit_ms,
        "n_input_candidates": len(cands), "n_kept_candidates": len(cands),
        "profit_definition": "real_explain_analyze_baseline_minus_treatment_summed_over_workload (cap_recorded_as_cap_ms)",
        "weight_definition": "ceil(real_pg_relation_size / weight_unit_mb / 1024 / 1024)",
        "size_source": "pg_relation_size",
        "prefilter": False, "negative_profit_kept": True,
        "cap_seconds": args.cap_seconds,
        "tools": "psycopg2 single connection; real CREATE INDEX in transaction with ROLLBACK",
        "smoke_limit": args.smoke_limit,
        "candidate_cap_multiplier": args.candidate_cap_multiplier,
        "candidate_cap_rule": f"abort candidate if cumulative wall-clock > {args.candidate_cap_multiplier}x baseline_total; remaining queries get treatment=3x baseline_q (bad signal -> negative profit -> QUBO avoids)",
    },
    "pairwise_interactions": [],
}
Path(args.out_problem).write_text(json.dumps(problem, indent=2))
n_pos = sum(1 for p in profits_int if p > 0)
n_neg = sum(1 for p in profits_int if p < 0)
n_zero = sum(1 for p in profits_int if p == 0)
print(f"[ok] wrote {args.out_problem} (n={len(cands)}, profit pos={n_pos} neg={n_neg} zero={n_zero}, Sigma_w={sum(weights_int)})")
