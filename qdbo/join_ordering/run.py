#!/usr/bin/env python3


import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

from dwave.system import LeapHybridNLSampler

from qdbo import config
from qdbo.join_ordering.bqm import (
    build_bqm,
    solve_bqm,
    decode_join_order,
)
from qdbo.join_ordering.inputs import load_statistics, parse_predicate_selectivities
from qdbo.join_ordering.inputs import list_folders_by_number
from qdbo.join_ordering.inputs import read_query_sql
from qdbo.join_ordering.postgres import explain_analyze_with_hint
from qdbo.join_ordering.pipeline import (
    Tee,
    format_embedding,
    ensure_dir,
    build_iter_csv_prefix,
    format_join_order,
    normalize_join_order,
    open_csv_writer,
    run_cqm,
    run_nl,
    run_solve_with_timeout,
    summarize_iter_timing,
    suppress_stdout_stderr,
)


BENCHMARK_ROOT = config.BENCHMARK_ROOT

WORKLOADS = [
    ("CEB", str(BENCHMARK_ROOT / "CEB")),
    ("JOB", str(BENCHMARK_ROOT / "JOB")),
]

RESULTS_ROOT = config.RESULTS_ROOT
OUT_DIR = str(RESULTS_ROOT / "join_ordering/actual_cpp_overhead/black-box-solvers")
LOG_DIR = str(RESULTS_ROOT / "join_ordering/actual_cpp_overhead/black-box-solvers/txt")
CSV_PREFIX = "cpp_iter_actual_cost"


RUN_NL = False
RUN_CQM = False
RUN_BQM = False
RUN_ITER = True

NL_TIME_LIMIT_S = 1
CQM_TIME_LIMIT_S = 5
NL_SAMPLER_LABEL = "QDBO NL Solver (Actual)"
CQM_LABEL = "QDBO CQM (Actual)"

ITER_TIMEOUT_S = 3600
CUSTOM_EMBEDDINGS = ["random"]
ITERATIONS_LIST: List[Optional[int]] = [3]

ROUNDS = 1

EXECUTE_TIMEOUT_MS = 600000


def find_query_folders(benchmark_root: str) -> List[str]:
    base = Path(benchmark_root)
    if not base.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_root}")

    query_dirs = list_folders_by_number(directory=benchmark_root)
    query_dirs = [
        q
        for q in query_dirs
        if (Path(q) / "cardinalities.json").is_file()
        and (Path(q) / "selectivities.json").is_file()
    ]
    return query_dirs


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


exec_extra = [
    "leading_hint",
    "planning_ms",
    "exec_ms",
    "baseline_planning_ms",
    "baseline_exec_ms",
    "exec_error",
]


def _append_exec_error(row: Dict[str, Any], msg: str) -> None:
    prev = row.get("exec_error") or ""
    row["exec_error"] = f"{prev}; {msg}" if prev else msg


def _execute_plans_and_attach_metrics(row: Dict[str, Any], query_dir: str) -> None:
    raw_order = row.get("join_order")
    if not raw_order:
        return
    try:
        join_order = (
            json.loads(raw_order) if isinstance(raw_order, str) else list(raw_order)
        )
    except Exception as e:
        row["exec_error"] = (
            f"cannot parse join_order {raw_order!r}: {type(e).__name__}: {e}"
        )
        return

    try:
        query = read_query_sql(query_dir)
    except Exception as e:
        row["exec_error"] = f"read_query_sql failed: {type(e).__name__}: {e}"
        return

    try:
        planning_ms, exec_ms, _explain, hint = explain_analyze_with_hint(
            join_order, query, EXECUTE_TIMEOUT_MS
        )
        row["leading_hint"] = hint
        row["planning_ms"] = planning_ms
        row["exec_ms"] = exec_ms
    except Exception as e:
        _append_exec_error(row, f"hinted execution failed: {type(e).__name__}: {e}")

    try:
        from qdbo.join_ordering.postgres import explain_analyze

        _explain, base_exec_ms, base_planning_ms = explain_analyze(
            query, EXECUTE_TIMEOUT_MS
        )
        row["baseline_exec_ms"] = base_exec_ms
        row["baseline_planning_ms"] = base_planning_ms
    except Exception as e:
        _append_exec_error(row, f"baseline execution failed: {type(e).__name__}: {e}")

    if row.get("exec_ms") is None and row.get("baseline_exec_ms") is None:
        _append_exec_error(
            row,
            "no execution time returned (check the database connection, "
            "JOB_DSN in qdbo/config.py, and that pg_hint_plan is installed)",
        )


def run_actual_cost(
    selected_benchmarks: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)

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

    nl_csv_path = (
        os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_nl.csv")
        if RUN_NL
        else None
    )
    cqm_csv_path = (
        os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_cqm.csv")
        if RUN_CQM
        else None
    )
    bqm_csv_path = (
        os.path.join(OUT_DIR, f"{CSV_PREFIX}{benchmark_tag}_{run_ts}_bqm.csv")
        if RUN_BQM
        else None
    )
    iter_prefix = (
        build_iter_csv_prefix(CSV_PREFIX, CUSTOM_EMBEDDINGS, ITERATIONS_LIST)
        if RUN_ITER
        else CSV_PREFIX
    )
    iter_csv_path = (
        os.path.join(OUT_DIR, f"{iter_prefix}{benchmark_tag}_{run_ts}_iter.csv")
        if RUN_ITER
        else None
    )
    log_prefix = (iter_prefix if RUN_ITER else CSV_PREFIX) + benchmark_tag
    log_path = os.path.join(LOG_DIR, f"{log_prefix}_{run_ts}.txt")

    log_file = open(log_path, mode="w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    workload_str = ", ".join([f"{name}={root}" for name, root in workload_list])
    print(f"[WORKLOADS] {workload_str}")
    print(
        f"[RUN] rounds={ROUNDS} RUN_NL={RUN_NL} RUN_CQM={RUN_CQM} RUN_BQM={RUN_BQM} RUN_ITER={RUN_ITER}"
    )
    print(f"[NL]  time_limit_s={NL_TIME_LIMIT_S} label={NL_SAMPLER_LABEL}")
    print(f"[CQM] time_limit_s={CQM_TIME_LIMIT_S} label={CQM_LABEL}")
    print(
        f"[ITER] timeout_s={ITER_TIMEOUT_S} embeddings={len(CUSTOM_EMBEDDINGS)} iterations={ITERATIONS_LIST}"
    )
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
        "in_pipeline_iterative_qa_solve_ms",
        "in_pipeline_read_out_ms",
        "in_pipeline_TOTAL_ms",
        "total_overhead_ms",
        "inner_faithful_total_overhead_ms",
        "faithful_total_overhead_ms",
        "solve_time_ms_measure_end_to_end",
    ]

    base_fields = base_header + exec_extra
    iter_fields = base_header + iter_extra + exec_extra

    nl_writer = cqm_writer = bqm_writer = iter_writer = None
    nl_fcsv = cqm_fcsv = bqm_fcsv = iter_fcsv = None

    try:
        if RUN_NL and nl_csv_path:
            nl_fcsv, nl_writer = open_csv_writer(nl_csv_path, base_fields)

        if RUN_CQM and cqm_csv_path:
            cqm_fcsv, cqm_writer = open_csv_writer(cqm_csv_path, base_fields)

        if RUN_BQM and bqm_csv_path:
            bqm_fcsv, bqm_writer = open_csv_writer(bqm_csv_path, base_fields)

        if RUN_ITER and iter_csv_path:
            iter_fcsv, iter_writer = open_csv_writer(iter_csv_path, iter_fields)

        for r in range(ROUNDS):
            print(f"\n<---- Round {r + 1}/{ROUNDS} ---->")

            for benchmark_name, benchmark_root in workload_list:
                folders = find_query_folders(benchmark_root)
                print(
                    f"\n[WORKLOAD] {benchmark_name} root={benchmark_root} queries={len(folders)}"
                )

                for i, inst_dir in enumerate(folders):
                    inst_name = os.path.basename(inst_dir.rstrip("/"))
                    relative_path = os.path.relpath(inst_dir, benchmark_root)
                    query_label = f"{benchmark_name}/{relative_path}"

                    cardinalities_content = None
                    selectivities_content = None
                    pred = pred_sel = None
                    card = None

                    need_load_statistics = RUN_NL or RUN_CQM or RUN_BQM
                    if need_load_statistics:
                        try:
                            cardinalities_content, selectivities_content = (
                                load_statistics(inst_dir)
                            )
                            card = cardinalities_content
                            pred, pred_sel = parse_predicate_selectivities(
                                selectivities_content
                            )
                        except Exception as e:
                            msg = f"load_statistics/parse_predicate_selectivities failed: {e}"
                            print(f"[ERR] [{i + 1}/{len(folders)}] {query_label} {msg}")
                            traceback.print_exc()

                            if RUN_NL and nl_writer is not None:
                                row = _make_base_row(
                                    run_ts,
                                    benchmark_name,
                                    relative_path,
                                    i,
                                    inst_dir,
                                    inst_name,
                                    "nl",
                                    r,
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                nl_writer.writerow(
                                    {k: row.get(k, None) for k in base_fields}
                                )
                                nl_fcsv.flush()
                            if RUN_CQM and cqm_writer is not None:
                                row = _make_base_row(
                                    run_ts,
                                    benchmark_name,
                                    relative_path,
                                    i,
                                    inst_dir,
                                    inst_name,
                                    "cqm",
                                    r,
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                cqm_writer.writerow(
                                    {k: row.get(k, None) for k in base_fields}
                                )
                                cqm_fcsv.flush()
                            if RUN_BQM and bqm_writer is not None:
                                row = _make_base_row(
                                    run_ts,
                                    benchmark_name,
                                    relative_path,
                                    i,
                                    inst_dir,
                                    inst_name,
                                    "bqm",
                                    r,
                                )
                                row["notes"] = "process_input_failed"
                                row["error"] = str(e)
                                bqm_writer.writerow(
                                    {k: row.get(k, None) for k in base_fields}
                                )
                                bqm_fcsv.flush()
                            continue

                    if RUN_NL and nl_writer is not None:
                        row = _make_base_row(
                            run_ts,
                            benchmark_name,
                            relative_path,
                            i,
                            inst_dir,
                            inst_name,
                            "nl",
                            r,
                        )
                        run_nl(
                            row,
                            cardinalities_content,
                            selectivities_content,
                            nl_sampler,
                            NL_SAMPLER_LABEL,
                            NL_TIME_LIMIT_S,
                            f"[{i + 1}/{len(folders)}] {query_label}",
                        )

                        _execute_plans_and_attach_metrics(row, inst_dir)
                        nl_writer.writerow({k: row.get(k, None) for k in base_fields})
                        nl_fcsv.flush()

                    if RUN_CQM and cqm_writer is not None:
                        row = _make_base_row(
                            run_ts,
                            benchmark_name,
                            relative_path,
                            i,
                            inst_dir,
                            inst_name,
                            "cqm",
                            r,
                        )
                        run_cqm(
                            row,
                            cardinalities_content,
                            selectivities_content,
                            CQM_TIME_LIMIT_S,
                            f"[{i + 1}/{len(folders)}] {query_label}",
                        )

                        _execute_plans_and_attach_metrics(row, inst_dir)
                        cqm_writer.writerow({k: row.get(k, None) for k in base_fields})
                        cqm_fcsv.flush()

                    if RUN_BQM and bqm_writer is not None:
                        row = _make_base_row(
                            run_ts,
                            benchmark_name,
                            relative_path,
                            i,
                            inst_dir,
                            inst_name,
                            "bqm",
                            r,
                        )
                        try:
                            print("BQM Timing with build_bqm")
                            t0 = time.time()
                            bqm = build_bqm(card, pred, pred_sel)
                            with suppress_stdout_stderr():
                                best_sample = solve_bqm(bqm)
                                join_order_raw, db_cost, use_fallback_join_order = (
                                    decode_join_order(
                                        best_sample, card, pred, pred_sel, {}
                                    )
                                )
                            t1 = time.time()

                            overhead_s = t1 - t0
                            overhead_ms = overhead_s * 1000.0

                            join_order = normalize_join_order(join_order_raw)

                            row.update(
                                {
                                    "solver_overhead_s": overhead_s,
                                    "solver_overhead_ms": overhead_ms,
                                    "db_cost": db_cost,
                                    "join_order": format_join_order(join_order),
                                    "use_fallback_join_order": bool(
                                        use_fallback_join_order
                                    ),
                                }
                            )

                            print(
                                f"[BQM]  [{i + 1}/{len(folders)}] {query_label} "
                                f"overhead={overhead_s:.3f}s db_cost={db_cost} fallback={bool(use_fallback_join_order)}"
                            )
                        except Exception as e:
                            row["error"] = str(e)
                            print(
                                f"[ERR][BQM] [{i + 1}/{len(folders)}] {query_label}: {e}"
                            )
                            traceback.print_exc()

                        _execute_plans_and_attach_metrics(row, inst_dir)
                        bqm_writer.writerow({k: row.get(k, None) for k in base_fields})
                        bqm_fcsv.flush()

                    if RUN_ITER and iter_writer is not None:
                        full_problem_path = inst_dir
                        for emb in CUSTOM_EMBEDDINGS:
                            for iters in ITERATIONS_LIST:
                                row = _make_base_row(
                                    run_ts,
                                    benchmark_name,
                                    relative_path,
                                    i,
                                    inst_dir,
                                    inst_name,
                                    "iter",
                                    r,
                                )
                                row.update(
                                    {
                                        "custom_embedding": format_embedding(emb),
                                        "iterations": iters,
                                    }
                                )

                                try:
                                    max_attempts = 3
                                    attempt = 0
                                    while True:
                                        try:
                                            t0 = time.time()
                                            (
                                                join_order_raw,
                                                db_cost,
                                                solve_time_s,
                                                timing_information,
                                            ) = run_solve_with_timeout(
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
                                                f"emb={format_embedding(emb)} iters={iters} "
                                                f"attempt={attempt}/{max_attempts - 1} err={e}"
                                            )

                                    wall_time_s = t1 - t0

                                    join_order = normalize_join_order(join_order_raw)

                                    timing_metrics_json = json.dumps(
                                        timing_information,
                                        ensure_ascii=True,
                                        default=str,
                                    )
                                    iter_timing = summarize_iter_timing(
                                        timing_information
                                    )
                                    embedding_stats = iter_timing["embedding_stats"]
                                    iter_total_ms_list = iter_timing[
                                        "iter_total_ms_list"
                                    ]
                                    iter_total_ms_sum = iter_timing["iter_total_ms_sum"]
                                    pre_iter_overhead_ms = iter_timing[
                                        "pre_iter_overhead_ms"
                                    ]
                                    total_overhead_ms = iter_timing["total_overhead_ms"]
                                    bqm_to_hj_ms = iter_timing["bqm_to_hj_ms"]
                                    in_pipeline_timing_ms = iter_timing[
                                        "in_pipeline_timing_ms"
                                    ]
                                    embedding_stats_json = None
                                    if embedding_stats is not None:
                                        embedding_stats_json = json.dumps(
                                            embedding_stats,
                                            ensure_ascii=True,
                                            default=str,
                                        )
                                    faithful_total_overhead_ms = None
                                    inner_faithful_total_overhead_ms = None
                                    if total_overhead_ms is not None:
                                        inner_faithful_total_overhead_ms = (
                                            total_overhead_ms
                                        )
                                        faithful_total_overhead_ms = total_overhead_ms
                                        if bqm_to_hj_ms is not None:
                                            inner_faithful_total_overhead_ms += (
                                                bqm_to_hj_ms
                                            )
                                            faithful_total_overhead_ms += bqm_to_hj_ms
                                        in_pipeline_generate_ms = in_pipeline_timing_ms.get(
                                            "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"
                                        )
                                        if in_pipeline_generate_ms is not None:
                                            faithful_total_overhead_ms += (
                                                in_pipeline_generate_ms
                                            )
                                    row.update(
                                        {
                                            "solver_overhead_s": (
                                                total_overhead_ms / 1000.0
                                            )
                                            if total_overhead_ms is not None
                                            else None,
                                            "solver_overhead_ms": total_overhead_ms,
                                            "timing_metrics_json": timing_metrics_json,
                                            "embedding_stats_json": embedding_stats_json,
                                            "iter_total_ms_sum": iter_total_ms_sum,
                                            "iter_total_ms_list": json.dumps(
                                                iter_total_ms_list, ensure_ascii=True
                                            )
                                            if iter_total_ms_list is not None
                                            else None,
                                            "pre_iter_overhead_ms": pre_iter_overhead_ms,
                                            "total_overhead_ms": total_overhead_ms,
                                            "solve_time_ms_measure_end_to_end": (
                                                solve_time_s * 1000.0
                                            )
                                            if solve_time_s is not None
                                            else None,
                                            "faithful_total_overhead_ms": faithful_total_overhead_ms,
                                            "in_pipeline_make_query_id_ms": in_pipeline_timing_ms[
                                                "in_pipeline_make_query_id_ms"
                                            ],
                                            "in_pipeline_get_join_ordering_problem_ms": in_pipeline_timing_ms[
                                                "in_pipeline_get_join_ordering_problem_ms"
                                            ],
                                            "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms": in_pipeline_timing_ms[
                                                "in_pipeline_generate_Fujitsu_QUBO_for_left_deep_trees_ms"
                                            ],
                                            "in_pipeline_iterative_qa_solve_ms": in_pipeline_timing_ms[
                                                "in_pipeline_iterative_qa_solve_ms"
                                            ],
                                            "in_pipeline_read_out_ms": in_pipeline_timing_ms[
                                                "in_pipeline_read_out_ms"
                                            ],
                                            "in_pipeline_TOTAL_ms": in_pipeline_timing_ms[
                                                "in_pipeline_TOTAL_ms"
                                            ],
                                            "inner_faithful_total_overhead_ms": inner_faithful_total_overhead_ms,
                                            "db_cost": db_cost,
                                            "join_order": format_join_order(join_order),
                                        }
                                    )

                                    solve_s_fmt = (
                                        f"{solve_time_s:.3f}"
                                        if solve_time_s is not None
                                        else "None"
                                    )
                                    total_overhead_fmt = (
                                        f"{total_overhead_ms / 1000.0:.3f}s"
                                        if total_overhead_ms is not None
                                        else "None"
                                    )
                                    faithful_overhead_fmt = (
                                        f"{faithful_total_overhead_ms / 1000.0:.3f}s"
                                        if faithful_total_overhead_ms is not None
                                        else "None"
                                    )
                                    inner_faithful_overhead_fmt = (
                                        f"{inner_faithful_total_overhead_ms / 1000.0:.3f}s"
                                        if inner_faithful_total_overhead_ms is not None
                                        else "None"
                                    )
                                    pre_overhead_fmt = (
                                        f"{pre_iter_overhead_ms / 1000.0:.3f}s"
                                        if pre_iter_overhead_ms is not None
                                        else "None"
                                    )
                                    sum_iter_ms_fmt = (
                                        f"{int(round(iter_total_ms_sum))}ms"
                                        if iter_total_ms_sum is not None
                                        else "None"
                                    )
                                    iter_list_fmt = "None"
                                    if iter_total_ms_list is not None:
                                        iter_list_int = [
                                            int(round(v)) for v in iter_total_ms_list
                                        ]
                                        iter_list_fmt = json.dumps(
                                            iter_list_int, ensure_ascii=True
                                        )
                                    print(
                                        f"\n[ITER] [{i + 1}/{len(folders)}] {query_label} "
                                        f"emb={row['custom_embedding']} iters={iters} "
                                        f"total_overhead={total_overhead_fmt} "
                                        f"inner_faithful_overhead={inner_faithful_overhead_fmt} "
                                        f"faithful_overhead={faithful_overhead_fmt} solve_s={solve_s_fmt} | "
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

                                _execute_plans_and_attach_metrics(row, inst_dir)
                                iter_writer.writerow(
                                    {k: row.get(k, None) for k in iter_fields}
                                )
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
        description="QDBO join-ordering pipeline over CEB/JOB workloads."
    )
    parser.add_argument(
        "--solvers",
        help="Comma-separated solvers to run: nl,cqm,bqm,iter",
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations of the iterative solver",
    )
    parser.add_argument(
        "--embedding",
        help="Variable-to-qubit mapping strategy",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark to run: JOB or CEB",
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
