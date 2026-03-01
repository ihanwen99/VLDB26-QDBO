#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Actual Quantum4DB Pipeline: (NL | CQM | BQM | ITER) over real query workloads (CEB/JOB)
# Author: Hanwen Liu
# Date: 2026/01/09
#
# Layout:
#   {BENCHMARK_ROOT}/{query_folder}
# Filter:
#   ONLY_QUERY_IDX filters by index in the sorted query folder list.


import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

from dwave.system import LeapHybridNLSampler

from backend.bqm_solver_execution import build_bqm, solve_bqm, read_out
from backend.nl_solver_execution import nl_query_optimization
from backend.cqm_solver_execution import build_cqm, solve_cqm
from backend.utils import process_input, parse_selectivities
from backend.utils import compute_db_cost, get_all_folders_in_target_directory_and_sorted

# ============================================================
# GLOBAL CONFIG (edit here)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
BENCHMARK_ROOT = Path(os.environ.get("QDBO_BENCHMARK_ROOT", DEFAULT_BENCHMARK_ROOT))

WORKLOADS = [
    ("CEB", str(BENCHMARK_ROOT / "CEB")),
    ("JOB", str(BENCHMARK_ROOT / "JOB")),
]

RESULTS_ROOT = Path(os.environ.get("QDBO_RESULTS_ROOT", PROJECT_ROOT))
OUT_DIR = str(RESULTS_ROOT / "results/actual_cpp_overhead/black-box-solvers")
LOG_DIR = str(RESULTS_ROOT / "results/actual_cpp_overhead/black-box-solvers/txt")
CSV_PREFIX = "cpp_iter_actual_cost"
LOG_PREFIX = "cpp_iter_actual_cost"


RUN_NL = False
RUN_CQM = False
RUN_BQM = False
RUN_ITER = True  # iterative solver

# ---- NL/CQM configs ----
NL_TIME_LIMIT_S = 1
CQM_TIME_LIMIT_S = 5
NL_SAMPLER_LABEL = "Quantum4DB NL_Solver (Actual)"
CQM_LABEL = "Quantum4DB CQM (Actual)"

# ---- BQM configs ----
SILENCE_BQM_OUTPUT = True  # suppress noisy prints inside BQM build/solve/read_out

# ---- iterative solver configs ----
ITER_TIMEOUT_S = 3600
try:
    from algorithm.iterative_solver import actual_query_blackbox
except Exception as e:
    print(f"[FATAL] cannot import actual_query_blackbox: {e}")
    RUN_ITER = False

CUSTOM_EMBEDDINGS = [
    # "seeded_neighbor_greedy",
    "random"
]
ITERATIONS_LIST: List[Optional[int]] = [3]
ITER_LABEL = "IterativeSolver (Actual)"

ROUNDS = 1

# Filter by index in query folder list
# None = run all queries
ONLY_QUERY_IDX: Optional[int] = None


# ============================================================


# -----------------------
# Utilities
# -----------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                return s.fileno()
        raise OSError("No fileno")


@contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    """
    Redirect sys.stdout/sys.stderr to /dev/null temporarily.
    Useful to silence noisy solvers (e.g., BQM internal prints).
    """
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _join_order_to_str(join_order: Any) -> str:
    try:
        if isinstance(join_order, list):
            return str(join_order)
        if hasattr(join_order, "tolist"):
            return str(join_order.tolist())
        return str(join_order)
    except Exception:
        return str(join_order)


def _normalize_join_order(join_order: Any) -> List[int]:
    """
    Normalize join_order into List[int] robustly.
    """
    if hasattr(join_order, "tolist"):
        join_order = join_order.tolist()
    if isinstance(join_order, (tuple, set)):
        join_order = list(join_order)
    if not isinstance(join_order, list):
        join_order = [join_order]
    out: List[int] = []
    for x in join_order:
        try:
            out.append(int(x))
        except Exception:
            # last resort: stringify then int
            out.append(int(str(x)))
    return out


def _embedding_to_str(emb: Any) -> str:
    if emb is None:
        return "None"
    name = getattr(emb, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(emb)


def find_query_folders(benchmark_root: str, only_query_idx: Optional[int] = None) -> List[str]:
    """
    Find query folders under a benchmark root. Filter to folders containing
    cardinalities.json and selectivities.json. Ordering uses the shared helper.
    """
    base = Path(benchmark_root)
    if not base.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_root}")

    query_dirs = get_all_folders_in_target_directory_and_sorted(directory=benchmark_root)
    query_dirs = [
        q for q in query_dirs
        if (Path(q) / "cardinalities.json").is_file() and (Path(q) / "selectivities.json").is_file()
    ]

    if only_query_idx is None:
        return query_dirs

    return [q for i, q in enumerate(query_dirs) if i == only_query_idx]


def run_blackbox_with_timeout(
        full_problem_path: str,
        emb: Any,
        iterations: Optional[int],
        timeout_s: int,
) -> Tuple[Any, Any, Optional[float], Any]:
    """
    Run actual_query_blackbox in a separate Python process with a hard timeout.
    Returns (join_order, db_cost, blackbox_time_s, timing_information).
    Raises TimeoutError on timeout, RuntimeError on blackbox failure.
    """
    env = os.environ.copy()
    env["Q4DB_FULL_PROBLEM_PATH"] = str(full_problem_path)
    env["Q4DB_CUSTOM_EMB"] = str(emb)
    env["Q4DB_ITERATIONS"] = "" if iterations is None else str(iterations)

    cmd = [
        sys.executable,
        "-c",
        r"""
import json, os, time

from algorithm.iterative_solver import actual_query_blackbox

full_problem_path = os.environ["Q4DB_FULL_PROBLEM_PATH"]
emb = os.environ["Q4DB_CUSTOM_EMB"]
iters_raw = os.environ.get("Q4DB_ITERATIONS", "")
iterations = None if iters_raw == "" else int(iters_raw)

t0 = time.time()
join_order, db_cost, timing = actual_query_blackbox(
    full_problem_path,
    emb,
    verbose=False,
    iterations=iterations,
)
t1 = time.time()
blackbox_time_s = t1 - t0

if hasattr(join_order, "tolist"):
    join_order = join_order.tolist()

print(timing)
print(json.dumps({
    "join_order": join_order,
    "db_cost": db_cost,
    "blackbox_time_s": blackbox_time_s,
    "timing_information": timing,
}, ensure_ascii=True, default=str))
""".strip(),
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"actual_query_blackbox timeout after {timeout_s}s") from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"actual_query_blackbox failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError("actual_query_blackbox produced no stdout")

    try:
        payload = json.loads(out_lines[-1])
        return (
            payload["join_order"],
            payload["db_cost"],
            payload.get("blackbox_time_s"),
            payload.get("timing_information"),
        )
    except Exception as e:
        raise RuntimeError(f"cannot parse blackbox output as json. stdout={proc.stdout!r}") from e


def _make_base_row(
        run_ts: str,
        benchmark: str,
        relative_path: str,
        i: int,
        query_dir: str,
        query_name: str,
        solver: str,
        round_id: int,
) -> Dict[str, Any]:
    return {
        "run_timestamp": run_ts,
        "row_timestamp": datetime.now().isoformat(timespec="seconds"),
        "benchmark": benchmark,
        "relative_path": relative_path,
        "query_idx": i,
        "query_folder": query_dir,
        "query_name": query_name,
        "solver": solver,
        "round": round_id,
        "solver_overhead_s": None,
        "solver_overhead_ms": None,
        "qpu_time_s": None,
        "qpu_time_ms": None,
        "db_cost": None,
        "join_order": None,
        "use_fallback_join_order": None,
        "notes": "",
        "error": "",
    }


# -----------------------
# Core runner
# -----------------------
def run_actual_cost(
        selected_benchmarks: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    _ensure_dir(OUT_DIR)
    _ensure_dir(LOG_DIR)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    total_retry_count = 0

    selected = None
    if selected_benchmarks:
        selected = [b.upper() for b in selected_benchmarks]
    workload_list = WORKLOADS
    if selected:
        workload_list = [w for w in WORKLOADS if w[0].upper() in selected]
        if not workload_list:
            raise ValueError(f"No workloads matched --benchmark={selected_benchmarks}")

    benchmark_tag = ""
    if selected and len(workload_list) == 1:
        benchmark_tag = "_" + workload_list[0][0].lower()

    nl_csv_path = os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_nl.csv") if RUN_NL else None
    cqm_csv_path = os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_cqm.csv") if RUN_CQM else None
    bqm_csv_path = os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_bqm.csv") if RUN_BQM else None
    iter_prefix = CSV_PREFIX
    if RUN_ITER:
        if len(CUSTOM_EMBEDDINGS) == 1:
            iter_prefix = f"{CUSTOM_EMBEDDINGS[0]}-{iter_prefix}"
        else:
            iter_prefix = f"multiemb-{iter_prefix}"
        if len(ITERATIONS_LIST) == 1:
            iter_prefix = f"{iter_prefix}-iter{ITERATIONS_LIST[0]}"
        else:
            iter_prefix = f"{iter_prefix}-multiiter"
    iter_csv_path = os.path.join(OUT_DIR, f"{iter_prefix}{benchmark_tag}_{run_ts}_iter.csv") if RUN_ITER else None
    log_prefix = (iter_prefix if RUN_ITER else CSV_PREFIX) + benchmark_tag
    log_path = os.path.join(LOG_DIR, f"{log_prefix}_{run_ts}.txt")

    log_file = open(log_path, mode="w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    workload_str = ", ".join([f"{name}={root}" for name, root in workload_list])
    print(f"[WORKLOADS] {workload_str}")
    print(f"[RUN] rounds={ROUNDS} RUN_NL={RUN_NL} RUN_CQM={RUN_CQM} RUN_BQM={RUN_BQM} RUN_ITER={RUN_ITER}")
    print(f"[NL]  time_limit_s={NL_TIME_LIMIT_S} label={NL_SAMPLER_LABEL}")
    print(f"[CQM] time_limit_s={CQM_TIME_LIMIT_S} label={CQM_LABEL}")
    print(f"[BQM] silence_output={SILENCE_BQM_OUTPUT} (parse_selectivities-based wiring)")
    print(f"[ITER] timeout_s={ITER_TIMEOUT_S} embeddings={len(CUSTOM_EMBEDDINGS)} iterations={ITERATIONS_LIST}")
    print(f"[FILTER] ONLY_QUERY_IDX={ONLY_QUERY_IDX} (None = all)")
    if nl_csv_path:
        print(f"[CSV][NL]   {nl_csv_path}")
    if cqm_csv_path:
        print(f"[CSV][CQM]  {cqm_csv_path}")
    if bqm_csv_path:
        print(f"[CSV][BQM]  {bqm_csv_path}")
    if iter_csv_path:
        print(f"[CSV][ITER] {iter_csv_path}")
    print(f"[LOG] {log_path}")

    nl_sampler = LeapHybridNLSampler() if RUN_NL else None

    base_header = [
        "run_timestamp",
        "row_timestamp",
        "benchmark",
        "relative_path",
        "query_idx",
        "query_name",
        "query_folder",
        "solver",
        "round",
        "solver_overhead_s",
        "solver_overhead_ms",
        "qpu_time_s",
        "qpu_time_ms",
        "db_cost",
        "join_order",
        "use_fallback_join_order",
        "notes",
        "error",
    ]
    iter_extra = [
        "custom_embedding",
        "iterations",
        "timing_metrics_json",
        "embedding_stats_json",
        "iter_total_ms_sum",
        "iter_total_ms_list",
        "pre_iter_overhead_ms",
        "in_pipeline_make_query_id_ms",
        "in_pipeline_get_join_ordering_problem_ms",
        "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms",
        "in_pipeline_SEBREMforBQM_ms",
        "in_pipeline_read_out_ms",
        "in_pipeline_TOTAL_ms",
        "total_overhead_ms",
        "inner_faithful_total_overhead_ms",
        "faithful_total_overhead_ms",
        "blackbox_time_ms_measure_end_to_end",
    ]

    nl_writer = cqm_writer = bqm_writer = iter_writer = None
    nl_fcsv = cqm_fcsv = bqm_fcsv = iter_fcsv = None

    try:
        if RUN_NL and nl_csv_path:
            nl_fcsv = open(nl_csv_path, "w", newline="", encoding="utf-8")
            nl_writer = csv.DictWriter(nl_fcsv, fieldnames=base_header)
            nl_writer.writeheader()
            nl_fcsv.flush()

        if RUN_CQM and cqm_csv_path:
            cqm_fcsv = open(cqm_csv_path, "w", newline="", encoding="utf-8")
            cqm_writer = csv.DictWriter(cqm_fcsv, fieldnames=base_header)
            cqm_writer.writeheader()
            cqm_fcsv.flush()

        if RUN_BQM and bqm_csv_path:
            bqm_fcsv = open(bqm_csv_path, "w", newline="", encoding="utf-8")
            bqm_writer = csv.DictWriter(bqm_fcsv, fieldnames=base_header)
            bqm_writer.writeheader()
            bqm_fcsv.flush()

        if RUN_ITER and iter_csv_path:
            iter_fcsv = open(iter_csv_path, "w", newline="", encoding="utf-8")
            iter_writer = csv.DictWriter(iter_fcsv, fieldnames=(base_header + iter_extra))
            iter_writer.writeheader()
            iter_fcsv.flush()

        for r in range(ROUNDS):
            print(f"\n<---- Round {r + 1}/{ROUNDS} ---->")

            for benchmark_name, benchmark_root in workload_list:
                folders = find_query_folders(benchmark_root, only_query_idx=ONLY_QUERY_IDX)
                print(f"\n[WORKLOAD] {benchmark_name} root={benchmark_root} queries={len(folders)}")

                for i, inst_dir in enumerate(folders):
                    inst_name = os.path.basename(inst_dir.rstrip("/"))
                    relative_path = os.path.relpath(inst_dir, benchmark_root)
                    query_label = f"{benchmark_name}/{relative_path}"

                    # Load input ONCE per instance if any NL/CQM/BQM needs it
                    cardinalities_content = None
                    selectivities_content = None
                    pred = pred_sel = None
                    card = None

                    need_process_input = (RUN_NL or RUN_CQM or RUN_BQM)
                    if need_process_input:
                        try:
                            cardinalities_content, selectivities_content = process_input(inst_dir)
                            card = cardinalities_content
                            # BQM uses pred/pred_sel; cheap to parse once for all solvers
                            pred, pred_sel = parse_selectivities(selectivities_content)
                        except Exception as e:
                            msg = f"process_input/parse_selectivities failed: {e}"
                            print(f"[ERR] [{i + 1}/{len(folders)}] {query_label} {msg}")
                            traceback.print_exc()

                            # Write error rows for enabled solvers
                            if RUN_NL and nl_writer is not None:
                                row = _make_base_row(
                                    run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "nl", r
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                nl_writer.writerow({k: row.get(k, None) for k in base_header})
                                nl_fcsv.flush()
                            if RUN_CQM and cqm_writer is not None:
                                row = _make_base_row(
                                    run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "cqm", r
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                cqm_writer.writerow({k: row.get(k, None) for k in base_header})
                                cqm_fcsv.flush()
                            if RUN_BQM and bqm_writer is not None:
                                row = _make_base_row(
                                    run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "bqm", r
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                bqm_writer.writerow({k: row.get(k, None) for k in base_header})
                                bqm_fcsv.flush()
                            continue

                    # ---------------- NL ----------------
                    if RUN_NL and nl_writer is not None:
                        row = _make_base_row(
                            run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "nl", r
                        )
                        try:
                            t0 = time.time()
                            nl_model, _ = nl_query_optimization(cardinalities_content, selectivities_content)
                            nl_sampler.sample(nl_model, label=NL_SAMPLER_LABEL, time_limit=NL_TIME_LIMIT_S)
                            join_order_arr = next(nl_model.iter_decisions()).state(0).astype(int)
                            t1 = time.time()

                            overhead_s = t1 - t0
                            overhead_ms = overhead_s * 1000.0

                            join_order = _normalize_join_order(join_order_arr)
                            db_cost = compute_db_cost(join_order, cardinalities_content, selectivities_content)

                            row.update({
                                "solver_overhead_s": overhead_s,
                                "solver_overhead_ms": overhead_ms,
                                "db_cost": db_cost,
                                "join_order": _join_order_to_str(join_order),
                            })

                            print(
                                f"[NL]   [{i + 1}/{len(folders)}] {query_label} "
                                f"overhead={overhead_s:.3f}s db_cost={db_cost}"
                            )
                        except Exception as e:
                            row["error"] = str(e)
                            print(f"[ERR][NL] [{i + 1}/{len(folders)}] {query_label}: {e}")
                            traceback.print_exc()

                        nl_writer.writerow({k: row.get(k, None) for k in base_header})
                        nl_fcsv.flush()

                    # ---------------- CQM ----------------
                    if RUN_CQM and cqm_writer is not None:
                        row = _make_base_row(
                            run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "cqm", r
                        )
                        try:
                            t0 = time.time()
                            cqm = build_cqm(cardinalities_content, selectivities_content)
                            join_order_raw, qpu_time = solve_cqm(cqm, time_limit=CQM_TIME_LIMIT_S)
                            t1 = time.time()

                            overhead_s = t1 - t0
                            overhead_ms = overhead_s * 1000.0

                            qpu_time_s = float(qpu_time) if qpu_time is not None else None
                            qpu_time_ms = (qpu_time_s * 1000.0) if qpu_time_s is not None else None

                            join_order = _normalize_join_order(join_order_raw)
                            db_cost = compute_db_cost(join_order, cardinalities_content, selectivities_content)

                            row.update({
                                "solver_overhead_s": overhead_s,
                                "solver_overhead_ms": overhead_ms,
                                "qpu_time_s": qpu_time_s,
                                "qpu_time_ms": qpu_time_ms,
                                "db_cost": db_cost,
                                "join_order": _join_order_to_str(join_order),
                            })

                            print(
                                f"[CQM]  [{i + 1}/{len(folders)}] {query_label} "
                                f"overhead={overhead_s:.3f}s qpu_s={qpu_time_s} db_cost={db_cost}"
                            )
                        except Exception as e:
                            row["error"] = str(e)
                            print(f"[ERR][CQM] [{i + 1}/{len(folders)}] {query_label}: {e}")
                            traceback.print_exc()

                        cqm_writer.writerow({k: row.get(k, None) for k in base_header})
                        cqm_fcsv.flush()

                    # ---------------- BQM ----------------
                    if RUN_BQM and bqm_writer is not None:
                        row = _make_base_row(
                            run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "bqm", r
                        )
                        try:
                            print("BQM Timing with build_bqm")
                            t0 = time.time()
                            bqm = build_bqm(card, pred, pred_sel)
                            with suppress_stdout_stderr(enabled=SILENCE_BQM_OUTPUT):
                                best_sample = solve_bqm(bqm)
                                join_order_raw, db_cost, use_fallback_join_order = read_out(
                                    best_sample, card, pred, pred_sel, {}
                                )
                            t1 = time.time()

                            overhead_s = t1 - t0
                            overhead_ms = overhead_s * 1000.0

                            join_order = _normalize_join_order(join_order_raw)

                            row.update({
                                "solver_overhead_s": overhead_s,
                                "solver_overhead_ms": overhead_ms,
                                "db_cost": db_cost,
                                "join_order": _join_order_to_str(join_order),
                                "use_fallback_join_order": bool(use_fallback_join_order),
                            })

                            print(
                                f"[BQM]  [{i + 1}/{len(folders)}] {query_label} "
                                f"overhead={overhead_s:.3f}s db_cost={db_cost} fallback={bool(use_fallback_join_order)}"
                            )
                        except Exception as e:
                            row["error"] = str(e)
                            print(f"[ERR][BQM] [{i + 1}/{len(folders)}] {query_label}: {e}")
                            traceback.print_exc()

                        bqm_writer.writerow({k: row.get(k, None) for k in base_header})
                        bqm_fcsv.flush()

                    # ---------------- ITERATIVE SOLVER ----------------
                    if RUN_ITER and iter_writer is not None:
                        full_problem_path = inst_dir
                        for emb in CUSTOM_EMBEDDINGS:
                            for iters in ITERATIONS_LIST:
                                row = _make_base_row(
                                    run_ts, benchmark_name, relative_path, i, inst_dir, inst_name, "iter", r
                                )
                                row.update({
                                    "custom_embedding": _embedding_to_str(emb),
                                    "iterations": iters,
                                })

                                try:
                                    max_attempts = 3
                                    attempt = 0
                                    while True:
                                        try:
                                            t0 = time.time()
                                            join_order_raw, db_cost, blackbox_time_s, timing_information = run_blackbox_with_timeout(
                                                full_problem_path=full_problem_path,
                                                emb=emb,
                                                iterations=iters,
                                                timeout_s=ITER_TIMEOUT_S,
                                            )
                                            t1 = time.time()
                                            break
                                        except Exception as e:
                                            attempt += 1
                                            if attempt >= max_attempts:
                                                raise
                                            total_retry_count += 1
                                            print(
                                                f"[RETRY][ITER] [{i + 1}/{len(folders)}] {query_label} "
                                                f"emb={_embedding_to_str(emb)} iters={iters} "
                                                f"attempt={attempt}/{max_attempts - 1} err={e}"
                                            )

                                    # wall_time_s: end-to-end blackbox call time (subprocess + python overhead)
                                    # blackbox_time_s: internal timing reported by actual_query_blackbox

                                    wall_time_s = t1 - t0
                                    wall_time_ms = wall_time_s * 1000.0

                                    join_order = _normalize_join_order(join_order_raw)

                                    timing_metrics_json = json.dumps(timing_information, ensure_ascii=True, default=str)
                                    embedding_stats = None
                                    iter_total_ms_list = None
                                    iter_total_ms_sum = None
                                    pre_iter_overhead_ms = None
                                    total_overhead_ms = None
                                    bqm_to_hj_ms = None
                                    in_pipeline_timing_ms = {
                                        "in_pipeline_make_query_id_ms": None,
                                        "in_pipeline_get_join_ordering_problem_ms": None,
                                        "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms": None,
                                        "in_pipeline_SEBREMforBQM_ms": None,
                                        "in_pipeline_read_out_ms": None,
                                        "in_pipeline_TOTAL_ms": None,
                                    }
                                    if isinstance(timing_information, list) and timing_information:
                                        iter_entries = [
                                            d for d in timing_information
                                            if isinstance(d, dict) and "iteration" in d and "latency_total_measured_ms" in d
                                        ]
                                        iter_total_ms_list = [d["latency_total_measured_ms"] for d in iter_entries]
                                        if iter_total_ms_list:
                                            iter_total_ms_sum = float(sum(iter_total_ms_list))
                                        summary_entries = [
                                            d for d in timing_information
                                            if isinstance(d, dict) and d.get("__tag__") == "SEBREMforBQM_summary"
                                        ]
                                        if summary_entries:
                                            summary = summary_entries[-1]
                                            overall = summary.get("overall_latency_ms", {})
                                            if isinstance(overall, dict):
                                                if overall.get("bqm_to_hJ_arrays") is not None:
                                                    bqm_to_hj_ms = float(overall.get("bqm_to_hJ_arrays"))
                                                pre_iter_overhead_ms = (
                                                    float(overall.get("embed_no_chains_drop_missing_cpp_internal", 0.0))
                                                    + float(overall.get("build_variable_order_and_hardware_indices", 0.0))
                                                    + float(overall.get("compute_variable_interaction_from_mapping", 0.0))
                                                )
                                                if iter_total_ms_sum is not None:
                                                    total_overhead_ms = pre_iter_overhead_ms + iter_total_ms_sum
                                        first_metric = timing_information[0]
                                        if isinstance(first_metric, dict):
                                            embedding_stats = first_metric.get("embedding_stats")
                                        summary_entries = [
                                            d for d in timing_information
                                            if isinstance(d, dict) and d.get("__tag__") == "actual_query_blackbox_summary"
                                        ]
                                        if summary_entries:
                                            summary = summary_entries[-1]
                                            timings_s = summary.get("timings_s", {})
                                            if isinstance(timings_s, dict):
                                                if timings_s.get("make_query_id") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_make_query_id_ms"] = float(timings_s["make_query_id"]) * 1000.0
                                                if timings_s.get("get_join_ordering_problem") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_get_join_ordering_problem_ms"] = float(timings_s["get_join_ordering_problem"]) * 1000.0
                                                if timings_s.get("generate_Fujitsu_QUBO_for_left_deep_trees") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"] = float(timings_s["generate_Fujitsu_QUBO_for_left_deep_trees"]) * 1000.0
                                                if timings_s.get("SEBREMforBQM") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_SEBREMforBQM_ms"] = float(timings_s["SEBREMforBQM"]) * 1000.0
                                                if timings_s.get("read_out") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_read_out_ms"] = float(timings_s["read_out"]) * 1000.0
                                                if timings_s.get("TOTAL") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_TOTAL_ms"] = float(timings_s["TOTAL"]) * 1000.0
                                    elif isinstance(timing_information, dict):
                                        embedding_stats = timing_information.get("embedding_stats")
                                        if timing_information.get("__tag__") == "actual_query_blackbox_summary":
                                            timings_s = timing_information.get("timings_s", {})
                                            if isinstance(timings_s, dict):
                                                if timings_s.get("make_query_id") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_make_query_id_ms"] = float(timings_s["make_query_id"]) * 1000.0
                                                if timings_s.get("get_join_ordering_problem") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_get_join_ordering_problem_ms"] = float(timings_s["get_join_ordering_problem"]) * 1000.0
                                                if timings_s.get("generate_Fujitsu_QUBO_for_left_deep_trees") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"] = float(timings_s["generate_Fujitsu_QUBO_for_left_deep_trees"]) * 1000.0
                                                if timings_s.get("SEBREMforBQM") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_SEBREMforBQM_ms"] = float(timings_s["SEBREMforBQM"]) * 1000.0
                                                if timings_s.get("read_out") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_read_out_ms"] = float(timings_s["read_out"]) * 1000.0
                                                if timings_s.get("TOTAL") is not None:
                                                    in_pipeline_timing_ms["in_pipeline_TOTAL_ms"] = float(timings_s["TOTAL"]) * 1000.0

                                    embedding_stats_json = None
                                    if embedding_stats is not None:
                                        embedding_stats_json = json.dumps(
                                            embedding_stats, ensure_ascii=True, default=str
                                        )
                                    faithful_total_overhead_ms = None
                                    inner_faithful_total_overhead_ms = None
                                    if total_overhead_ms is not None:
                                        inner_faithful_total_overhead_ms = total_overhead_ms
                                        faithful_total_overhead_ms = total_overhead_ms
                                        if bqm_to_hj_ms is not None:
                                            inner_faithful_total_overhead_ms += bqm_to_hj_ms
                                            faithful_total_overhead_ms += bqm_to_hj_ms
                                        in_pipeline_generate_ms = in_pipeline_timing_ms.get(
                                            "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"
                                        )
                                        if in_pipeline_generate_ms is not None:
                                            faithful_total_overhead_ms += in_pipeline_generate_ms
                                    row.update({
                                        "solver_overhead_s": (total_overhead_ms / 1000.0) if total_overhead_ms is not None else None,
                                        "solver_overhead_ms": total_overhead_ms,
                                        "timing_metrics_json": timing_metrics_json,
                                        "embedding_stats_json": embedding_stats_json,
                                        "iter_total_ms_sum": iter_total_ms_sum,
                                        "iter_total_ms_list": json.dumps(iter_total_ms_list, ensure_ascii=True) if iter_total_ms_list is not None else None,
                                        "pre_iter_overhead_ms": pre_iter_overhead_ms,
                                        "total_overhead_ms": total_overhead_ms,
                                        "blackbox_time_ms_measure_end_to_end": (blackbox_time_s * 1000.0) if blackbox_time_s is not None else None,
                                        "faithful_total_overhead_ms": faithful_total_overhead_ms,
                                        "in_pipeline_make_query_id_ms": in_pipeline_timing_ms["in_pipeline_make_query_id_ms"],
                                        "in_pipeline_get_join_ordering_problem_ms": in_pipeline_timing_ms["in_pipeline_get_join_ordering_problem_ms"],
                                        "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms": in_pipeline_timing_ms["in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"],
                                        "in_pipeline_SEBREMforBQM_ms": in_pipeline_timing_ms["in_pipeline_SEBREMforBQM_ms"],
                                        "in_pipeline_read_out_ms": in_pipeline_timing_ms["in_pipeline_read_out_ms"],
                                        "in_pipeline_TOTAL_ms": in_pipeline_timing_ms["in_pipeline_TOTAL_ms"],
                                        "inner_faithful_total_overhead_ms": inner_faithful_total_overhead_ms,
                                        "db_cost": db_cost,
                                        "join_order": _join_order_to_str(join_order),
                                    })

                                    blackbox_s_fmt = f"{blackbox_time_s:.3f}" if blackbox_time_s is not None else "None"
                                    total_overhead_fmt = (
                                        f"{total_overhead_ms / 1000.0:.3f}s"
                                        if total_overhead_ms is not None else "None"
                                    )
                                    faithful_overhead_fmt = (
                                        f"{faithful_total_overhead_ms / 1000.0:.3f}s"
                                        if faithful_total_overhead_ms is not None else "None"
                                    )
                                    inner_faithful_overhead_fmt = (
                                        f"{inner_faithful_total_overhead_ms / 1000.0:.3f}s"
                                        if inner_faithful_total_overhead_ms is not None else "None"
                                    )
                                    pre_overhead_fmt = (
                                        f"{pre_iter_overhead_ms / 1000.0:.3f}s"
                                        if pre_iter_overhead_ms is not None else "None"
                                    )
                                    sum_iter_ms_fmt = (
                                        f"{int(round(iter_total_ms_sum))}ms"
                                        if iter_total_ms_sum is not None else "None"
                                    )
                                    iter_list_fmt = "None"
                                    if iter_total_ms_list is not None:
                                        iter_list_int = [int(round(v)) for v in iter_total_ms_list]
                                        iter_list_fmt = json.dumps(iter_list_int, ensure_ascii=True)
                                    print(
                                        f"\n[ITER] [{i + 1}/{len(folders)}] {query_label} "
                                        f"emb={row['custom_embedding']} iters={iters} "
                                        f"total_overhead={total_overhead_fmt} "
                                        f"inner_faithful_overhead={inner_faithful_overhead_fmt} "
                                        f"faithful_overhead={faithful_overhead_fmt} blackbox_s={blackbox_s_fmt} | "
                                        f"pre_overhead={pre_overhead_fmt} "
                                        f"sum_iter={sum_iter_ms_fmt} "
                                        f"iter_list={iter_list_fmt} "
                                        f"walltime_overhead={wall_time_s:.3f}s db_cost={db_cost}"
                                    )
                                    if embedding_stats_json is not None:
                                        print(f"[ITER][STATS] {embedding_stats_json}")
                                    print(f"[ITER][TIMING] {timing_metrics_json}")
                                    print(
                                        "[ITER][TIMING][DERIVED] "
                                        f"inner_faithful_overhead_ms={inner_faithful_total_overhead_ms} "
                                        f"faithful_overhead_ms={faithful_total_overhead_ms}"
                                    )
                                except Exception as e:
                                    row["error"] = str(e)
                                    print(
                                        f"[ERR][ITER] [{i + 1}/{len(folders)}] {query_label} "
                                        f"emb={row['custom_embedding']} iters={iters}: {e}"
                                    )
                                    traceback.print_exc()

                                iter_writer.writerow({k: row.get(k, None) for k in (base_header + iter_extra)})
                                iter_fcsv.flush()

    finally:
        for f in (nl_fcsv, cqm_fcsv, bqm_fcsv, iter_fcsv):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        try:
            log_file.close()
        except Exception:
            pass

    print(
        f"[DONE] NL={nl_csv_path} CQM={cqm_csv_path} BQM={bqm_csv_path} "
        f"ITER={iter_csv_path} LOG={log_path} RETRIES={total_retry_count}"
    )
    return nl_csv_path, cqm_csv_path, bqm_csv_path, iter_csv_path, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Actual Quantum4DB pipeline over CEB/JOB with optional iterative solver overrides."
    )
    parser.add_argument(
        "--solvers",
        help="Comma-separated solvers to run: nl,cqm,bqm,iter (overrides RUN_* flags).",
    )
    parser.add_argument(
        "--iterations",
        help="Override ITERATIONS_LIST for RUN_ITER (int or 'none').",
    )
    parser.add_argument(
        "--embedding",
        help="Override CUSTOM_EMBEDDINGS for RUN_ITER (single embedding name).",
    )
    parser.add_argument(
        "--benchmark",
        help="Run a single benchmark (e.g., CEB or JOB).",
    )
    args = parser.parse_args()

    if args.iterations is not None:
        raw = str(args.iterations).strip().lower()
        ITERATIONS_LIST = [None] if raw in ("none", "null", "") else [int(raw)]

    if args.embedding is not None:
        CUSTOM_EMBEDDINGS = [str(args.embedding)]

    if args.solvers is not None:
        raw = [s.strip().lower() for s in str(args.solvers).split(",") if s.strip()]
        allowed = {"nl", "cqm", "bqm", "iter"}
        unknown = [s for s in raw if s not in allowed]
        if unknown:
            raise ValueError(f"Unknown solvers in --solvers: {unknown}")
        RUN_NL = "nl" in raw
        RUN_CQM = "cqm" in raw
        RUN_BQM = "bqm" in raw
        RUN_ITER = "iter" in raw

    selected_benchmarks = None
    if args.benchmark is not None:
        selected_benchmarks = [str(args.benchmark)]

    run_actual_cost(selected_benchmarks=selected_benchmarks)
