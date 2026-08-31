#!/usr/bin/env python3
"""
compute_table_csv.py

Aggregate replay/*.txt files under results/index_selection/<experiment_id>/
into machine-readable CSV tables. The aggregation formula matches the paper
Table 4 exactly:

  per_query_ms_replay   = mean(run2, run3)  # drop-first 3-run hot-cache
  per_query_ms_capped   = min(per_query_ms_replay, timeout_per_query_s * 1000)
  workload_total_ms     = sum_over_queries(per_query_ms_capped)
  workload_speedup      = baseline_total_ms / workload_total_ms

For QDBO cells the better of iter=1 / iter=3 is chosen per (advisor, budget)
when constructing a one-row-per-(advisor,budget) summary; this script emits
all rows including iteration so downstream callers can group as needed.

Output CSV schema (single full-column table, paper Table 4):
  advisor, budget_pct, iteration, n_indexes, ids_hash, total_ms,
  baseline_total_ms, speedup, cache_hit, aborted, source_path,
  selected_ids, note

  - n_indexes makes the DP-vs-QDBO index-count contrast (DP packs 71 vs
    QDBO ~50 at B=50%) readable straight off the table.
  - cache_hit / aborted are replay-quality diagnostics from sweep_replay.sh.
  - advisor carries the QDBO iteration in its label (qdbo_greedy_iter1) and
    the iteration column repeats it; classical solvers leave iteration blank.

All 11 selectors are emitted per budget: greedy_roi, exact_dp (DP knapsack),
bqm_solver_hybrid (D-Wave LeapHybridBQMSampler), and the 4 QDBO embeddings
(random, greedy, semanticINDEX_greedy, semanticSLACK_greedy) at iter 1 and 3.

The experiment directory ships only the 19-query baseline (Q2/Q17/Q20 are
excluded everywhere: per_query_baseline/, sweep_replay.sh QUERIES_19), so the
primary table aggregates over exactly the 19 queries the paper reports in
Table 4. --exclude-queries 2,17,20 additionally emits a defensive variant that
re-applies the exclusion explicitly; on the shipped 19q instance the two tables
are identical.

Only the 4 paper budgets {10, 20, 40, 50} percent are aggregated by default
(--budgets); stale budget_*pct dirs left in the experiment directory (e.g. an
older 5/15/25/60pct sweep) are skipped, so the default table is exactly 44 rows
(4 budgets x 11 selectors). Pass --budgets 5,10,15,20,25,40,50,60 for the full
sweep.

Usage:
  python scripts/index_selection/compute_table_csv.py \
      results/index_selection/sf1_nopk_19q_e2_runtime/ \
      --exclude-queries 2,17,20  # primary 19q table + explicit-exclusion variant
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPLAY_LINE_RE = re.compile(
    r"^Q(?P<q>\d+)\s+run(?P<run>\d+):\s+(?P<ms>\d+)ms\s+(?P<status>\S+)"
)
TIMEOUT_MS = 600_000

# The storage budgets reported in paper Table 4. build_rows_for_view filters
# to these by default so stale budget_*pct dirs (e.g. an older 5/15/25/60pct
# sweep left in the experiment directory) never leak into the canonical table.
# Overridable via --budgets for anyone who wants the full sweep.
PAPER_BUDGETS = {10, 20, 40, 50}


def parse_replay_file(path):
    """Return {query_id: mean(run2, run3) ms} with 600k cap applied."""
    by_q = {}
    with path.open() as f:
        for line in f:
            m = REPLAY_LINE_RE.match(line)
            if not m:
                continue
            q = int(m.group("q"))
            run = int(m.group("run"))
            ms = int(m.group("ms"))
            ms = min(ms, TIMEOUT_MS)
            if run in (2, 3):
                by_q.setdefault(q, []).append(ms)
    out = {}
    for q, runs in by_q.items():
        if not runs:
            continue
        out[q] = sum(runs) / len(runs)
    return out


def workload_total(per_q, queries=None):
    if queries is None:
        return sum(per_q.values())
    return sum(per_q.get(q, 0.0) for q in queries)


def find_solution_json(replay_path, exp_root):
    """Map replay/*.txt back to its sol_*.json.

    Replay file naming conventions:
      greedy_b<pct>.txt          -> budget_<pct>pct/sol_greedy.json
      exact_b<pct>.txt           -> budget_<pct>pct/sol_exact.json
      hybrid_b<pct>.txt          -> budget_<pct>pct/sol_hybrid.json
      qdbo_<emb>_iter<i>_b<pct>.txt -> budget_<pct>pct/qdbo_<emb>/sol_iter_<i>_trimmed.json
    """
    name = replay_path.stem
    budget_dir = replay_path.parent.parent
    if name.startswith("greedy_b"):
        return budget_dir / "sol_greedy.json"
    if name.startswith("exact_b"):
        return budget_dir / "sol_exact.json"
    if name.startswith("hybrid_b"):
        return budget_dir / "sol_hybrid.json"
    m = re.match(r"^qdbo_(.+)_iter(\d+)_b(\d+)$", name)
    if m:
        emb = m.group(1)
        it = m.group(2)
        return budget_dir / f"qdbo_{emb}" / f"sol_iter_{it}_trimmed.json"
    return None


def load_solution(sol_path):
    """Return sorted selected index ids for a sol_*.json (or [] if missing)."""
    if sol_path is None or not sol_path.exists():
        return []
    try:
        sol = json.loads(sol_path.read_text())
    except Exception:
        return []
    return sorted(sol.get("selected_ids", []))


def ids_hash16(ids):
    """16-hex-char sha1 of the sorted-id tuple. Matches the replay_cache key
    written by sweep_replay.sh (hashlib.sha1(repr(tuple(sorted(ids))))[:16])."""
    import hashlib

    return hashlib.sha1(repr(tuple(sorted(ids))).encode()).hexdigest()[:16]


def replay_quality_flags(replay_path):
    """(cache_hit, aborted) as 0/1 ints, read from the replay txt markers
    emitted by sweep_replay.sh."""
    cache_hit = 0
    aborted = 0
    try:
        text = replay_path.read_text()
    except Exception:
        return 0, 0
    if "replay_cache_hit:" in text:
        cache_hit = 1
    if "cell_aborted_overcap" in text or " ABORT:" in text:
        aborted = 1
    return cache_hit, aborted


def parse_replay_filename(name):
    """(advisor, budget_pct, iteration_or_None) or None."""
    m = re.match(r"^greedy_b(\d+)$", name)
    if m:
        return "greedy_roi", int(m.group(1)), None
    m = re.match(r"^exact_b(\d+)$", name)
    if m:
        return "exact_dp", int(m.group(1)), None
    m = re.match(r"^hybrid_b(\d+)$", name)
    if m:
        return "bqm_solver_hybrid", int(m.group(1)), None
    m = re.match(r"^qdbo_(.+)_iter(\d+)_b(\d+)$", name)
    if m:
        return f"qdbo_{m.group(1)}", int(m.group(3)), int(m.group(2))
    return None


def _baseline_from_txt(exp_root):
    """Fallback: rebuild {q: ms} from per_query_baseline/q*.txt.

    Prefers mean(run2, run3) (drop-first hot cache). For capped queries that
    never produced run2/run3 (e.g. q21 timed out at the 60s cap, only run1
    exists) fall back to the run1 value so the query is NOT silently dropped.
    """
    base_dir = exp_root / "per_query_baseline"
    if not base_dir.is_dir():
        return {}
    out = {}
    for f in sorted(base_dir.glob("q*.txt")):
        m = re.match(r"^q(\d+)", f.stem)
        if not m:
            continue
        q = int(m.group(1))
        per_q = parse_replay_file(f)
        if q in per_q:
            out[q] = per_q[q]
            continue
        # no run2/run3: use whatever runs exist (capped single run included)
        ms_runs = []
        for line in f.read_text().splitlines():
            mm = REPLAY_LINE_RE.match(line)
            if mm:
                ms_runs.append(min(int(mm.group("ms")), TIMEOUT_MS))
        if ms_runs:
            out[q] = sum(ms_runs) / len(ms_runs)
    return out


def load_baseline_per_q(exp_root):
    """Authoritative per-query zero-index baseline {q: ms}.

    Prefers the shipped baseline_per_query.json (the same file consumed by
    sweep_replay.sh and the values reported in the paper Table 4 baseline
    column). This is authoritative because capped queries such as q21 are
    recorded there as the 60s cap; re-deriving them from per_query_baseline/
    q*.txt would drop q21 (it never produced run2/run3), shrinking the set to
    18 queries and corrupting every speedup. Falls back to the q*.txt files
    only when the JSON is absent.
    """
    bj = exp_root / "baseline_per_query.json"
    if bj.exists():
        try:
            raw = json.loads(bj.read_text())
            out = {}
            for k, v in raw.items():
                mk = re.match(r"^q?(\d+)$", str(k))
                if mk:
                    out[int(mk.group(1))] = float(v)
            if out:
                return out
        except Exception:
            pass
    return _baseline_from_txt(exp_root)


def discover_baseline(exp_root):
    per_q = load_baseline_per_q(exp_root)
    if not per_q:
        raise FileNotFoundError(
            f"no baseline found in {exp_root} "
            f"(need baseline_per_query.json or per_query_baseline/q*.txt)"
        )
    return sum(per_q.values()), sorted(per_q.keys())


def build_per_query_baseline(exp_root, queries):
    """Return {q: ms} restricted to the requested query set."""
    per_q = load_baseline_per_q(exp_root)
    return {q: per_q[q] for q in queries if q in per_q}


def emit_table_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Full 13-column schema (paper Table 4). n_indexes lets the DP-vs-QDBO
    # index-count contrast (e.g. DP packs 71 vs QDBO ~50 at B=50%) be read
    # straight off the table; cache_hit/aborted/note are replay-quality
    # diagnostics. advisor encodes the QDBO iteration in its name AND the
    # iteration column carries it separately.
    fieldnames = [
        "advisor",
        "budget_pct",
        "iteration",
        "n_indexes",
        "ids_hash",
        "total_ms",
        "baseline_total_ms",
        "speedup",
        "cache_hit",
        "aborted",
        "source_path",
        "selected_ids",
        "note",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_rows_for_view(
    exp_root, queries_filter, baseline_total_ms, expected_queries=None, budgets=None
):
    """expected_queries: iterable of query ids that the replay file MUST cover.
    Cells missing any of these are skipped with a warning (incomplete replay).

    budgets: iterable of budget percents to keep (e.g. {10, 20, 40, 50}). Any
    budget_*pct directory whose percent is not in this set is skipped, so stale
    sweeps left in the experiment directory do not leak into the table. When
    None, every budget_*pct directory is included."""
    rows = []
    budget_set = set(budgets) if budgets is not None else None
    for budget_dir in sorted(exp_root.glob("budget_*pct")):
        m_bd = re.match(r"^budget_(\d+)pct$", budget_dir.name)
        if m_bd is None:
            continue
        dir_budget_pct = int(m_bd.group(1))
        if budget_set is not None and dir_budget_pct not in budget_set:
            continue
        replay_dir = budget_dir / "replay"
        if not replay_dir.is_dir():
            continue
        for replay_file in sorted(replay_dir.glob("*.txt")):
            parsed = parse_replay_filename(replay_file.stem)
            if parsed is None:
                continue
            advisor, budget_pct, iteration = parsed
            # For QDBO cells the iteration is folded into the advisor label
            # (qdbo_greedy_iter1) as well as kept in its own column.
            if iteration is not None:
                advisor = f"{advisor}_iter{iteration}"
            per_q = parse_replay_file(replay_file)
            if expected_queries is not None:
                missing = sorted(set(expected_queries) - set(per_q.keys()))
                if missing:
                    print(
                        f"[skip-incomplete] {replay_file.name}: missing Q{missing} "
                        f"(advisor={advisor}, B={budget_pct}, iter={iteration})"
                    )
                    continue
            total_ms = workload_total(per_q, queries_filter)
            if total_ms <= 0:
                continue
            sol_path = find_solution_json(replay_file, exp_root)
            ids = load_solution(sol_path)
            cache_hit, aborted = replay_quality_flags(replay_file)
            speedup = baseline_total_ms / total_ms if total_ms > 0 else 0.0
            rel_source = replay_file.relative_to(exp_root.parent)
            rows.append(
                {
                    "advisor": advisor,
                    "budget_pct": str(budget_pct),
                    "iteration": "" if iteration is None else str(iteration),
                    "n_indexes": str(len(ids)),
                    "ids_hash": ids_hash16(ids) if ids else "",
                    "total_ms": f"{total_ms:.1f}",
                    "baseline_total_ms": f"{baseline_total_ms:.1f}",
                    "speedup": f"{speedup:.4f}",
                    "cache_hit": str(cache_hit),
                    "aborted": str(aborted),
                    "source_path": str(rel_source),
                    "selected_ids": ";".join(str(i) for i in ids),
                    "note": "",
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", type=Path)
    ap.add_argument("--exclude-queries", default="", help="comma list, e.g. 2,17,20")
    ap.add_argument(
        "--budgets",
        default="10,20,40,50",
        help=(
            "comma list of budget percents to include (default 10,20,40,50 = "
            "paper Table 4). Stale budget_*pct dirs outside this set are skipped. "
            "Pass e.g. 5,10,15,20,25,40,50,60 to aggregate the full sweep."
        ),
    )
    # Single full-column output. --out (legacy aliases --out-22q/--out-19q) lets
    # a caller override the destination path.
    ap.add_argument("--out", "--out-19q", "--out-22q", dest="out", type=Path, default=None)
    args = ap.parse_args()

    exp_root = args.exp_root.resolve()
    if not exp_root.is_dir():
        print(f"missing dir: {exp_root}", file=sys.stderr)
        return 1

    baseline_total_ms, queries_all = discover_baseline(exp_root)
    print(f"[baseline] total={baseline_total_ms:.0f} ms over {len(queries_all)} queries")

    # The experiment ships only the 19-query baseline, so the query set is
    # already the paper's reported 19q subset. --exclude-queries re-applies the
    # Q2/Q17/Q20 exclusion defensively (a no-op on the shipped 19q instance, but
    # correct if a future fixture ships the full 22q baseline).
    excl_arg = args.exclude_queries.strip()
    excl = {int(x) for x in excl_arg.split(",") if x.strip()} if excl_arg else set()

    budgets_arg = args.budgets.strip()
    budgets = (
        {int(x) for x in budgets_arg.split(",") if x.strip()}
        if budgets_arg
        else set(PAPER_BUDGETS)
    )
    print(f"[budgets] keeping {sorted(budgets)}")
    queries_filter = sorted(q for q in queries_all if q not in excl)
    if excl:
        per_q_baseline = build_per_query_baseline(exp_root, queries_all)
        baseline_used = sum(per_q_baseline.get(q, 0.0) for q in queries_filter)
        print(
            f"[filtered] {len(queries_filter)} queries (excluded {sorted(excl)}), "
            f"baseline={baseline_used:.0f} ms"
        )
    else:
        baseline_used = baseline_total_ms

    out_path = args.out or (exp_root / "tables" / "table_nopk_19q.csv")
    rows = build_rows_for_view(
        exp_root,
        queries_filter=queries_filter,
        baseline_total_ms=baseline_used,
        expected_queries=set(queries_filter),
        budgets=budgets,
    )
    if not rows:
        print(
            f"[error] no rows assembled from {exp_root} for budgets "
            f"{sorted(budgets)}: either no budget_*pct/replay/*.txt outputs "
            f"exist, or every replay file was skipped as incomplete. "
            f"Run sweep_solve.sh then sweep_replay.sh first.",
            file=sys.stderr,
        )
        return 1
    emit_table_csv(rows, out_path)
    print(f"[ok] wrote {out_path} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
