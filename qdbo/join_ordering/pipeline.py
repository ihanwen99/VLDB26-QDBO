import csv
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdbo.join_ordering.cqm import build_cqm, solve_cqm
from qdbo.join_ordering.join_order import compute_db_cost
from qdbo.join_ordering.nl import build_left_deep_nl_model


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


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def format_join_order(join_order: Any) -> str:
    try:
        if isinstance(join_order, list):
            return str(join_order)
        if hasattr(join_order, "tolist"):
            return str(join_order.tolist())
        return str(join_order)
    except Exception:
        return str(join_order)


def normalize_join_order(join_order: Any) -> List[int]:
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
            out.append(int(str(x)))
    return out


def format_embedding(emb: Any) -> str:
    if emb is None:
        return "None"
    name = getattr(emb, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(emb)


def build_iter_csv_prefix(
    csv_prefix: str, embeddings: List[str], iterations_list: List[Optional[int]]
) -> str:
    if len(embeddings) == 1:
        prefix = f"{embeddings[0]}-{csv_prefix}"
    else:
        prefix = f"multiemb-{csv_prefix}"
    if len(iterations_list) == 1:
        return f"{prefix}-iter{iterations_list[0]}"
    return f"{prefix}-multiiter"


def open_csv_writer(path: str, fieldnames: List[str]):
    fcsv = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()
    fcsv.flush()
    return fcsv, writer


def run_nl(
    row: Dict[str, Any],
    cardinalities,
    selectivities,
    sampler,
    label: str,
    time_limit: int,
    progress: str,
) -> None:
    try:
        t0 = time.time()
        nl_model, _ = build_left_deep_nl_model(cardinalities, selectivities)
        sampler.sample(nl_model, label=label, time_limit=time_limit)
        join_order_arr = next(nl_model.iter_decisions()).state(0).astype(int)
        t1 = time.time()

        overhead_s = t1 - t0
        overhead_ms = overhead_s * 1000.0

        join_order = normalize_join_order(join_order_arr)
        db_cost = compute_db_cost(join_order, cardinalities, selectivities)

        row.update(
            {
                "solver_overhead_s": overhead_s,
                "solver_overhead_ms": overhead_ms,
                "db_cost": db_cost,
                "join_order": format_join_order(join_order),
            }
        )

        print(f"[NL]   {progress} overhead={overhead_s:.3f}s db_cost={db_cost}")
    except Exception as e:
        row["error"] = str(e)
        print(f"[ERR][NL] {progress}: {e}")
        traceback.print_exc()


def run_cqm(
    row: Dict[str, Any], cardinalities, selectivities, time_limit: int, progress: str
) -> None:
    try:
        t0 = time.time()
        cqm = build_cqm(cardinalities, selectivities)
        join_order_raw, qpu_time = solve_cqm(cqm, time_limit=time_limit)
        t1 = time.time()

        overhead_s = t1 - t0
        overhead_ms = overhead_s * 1000.0

        qpu_time_ms = float(qpu_time) if qpu_time is not None else None
        qpu_time_s = (qpu_time_ms / 1000.0) if qpu_time_ms is not None else None

        join_order = normalize_join_order(join_order_raw)
        db_cost = compute_db_cost(join_order, cardinalities, selectivities)

        row.update(
            {
                "solver_overhead_s": overhead_s,
                "solver_overhead_ms": overhead_ms,
                "qpu_time_s": qpu_time_s,
                "qpu_time_ms": qpu_time_ms,
                "db_cost": db_cost,
                "join_order": format_join_order(join_order),
            }
        )

        print(
            f"[CQM]  {progress} overhead={overhead_s:.3f}s qpu_s={qpu_time_s} db_cost={db_cost}"
        )
    except Exception as e:
        row["error"] = str(e)
        print(f"[ERR][CQM] {progress}: {e}")
        traceback.print_exc()


IN_PIPELINE_KEYS = [
    "make_query_id",
    "get_join_ordering_problem",
    "generate_Fujitsu_QUBO_for_left_deep_trees",
    "iterative_qa_solve",
    "read_out",
    "TOTAL",
]


def summarize_iter_timing(timing_information: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "embedding_stats": None,
        "iter_total_ms_list": None,
        "iter_total_ms_sum": None,
        "pre_iter_overhead_ms": None,
        "total_overhead_ms": None,
        "bqm_to_hj_ms": None,
        "in_pipeline_timing_ms": {
            f"in_pipeline_{k}_ms": None for k in IN_PIPELINE_KEYS
        },
    }
    if not (isinstance(timing_information, list) and timing_information):
        return out
    iter_entries = [
        d
        for d in timing_information
        if isinstance(d, dict) and "iteration" in d and "latency_total_measured_ms" in d
    ]
    out["iter_total_ms_list"] = [d["latency_total_measured_ms"] for d in iter_entries]
    if out["iter_total_ms_list"]:
        out["iter_total_ms_sum"] = float(sum(out["iter_total_ms_list"]))
    summary_entries = [
        d
        for d in timing_information
        if isinstance(d, dict) and d.get("__tag__") == "iterative_qa_solve_summary"
    ]
    if summary_entries:
        overall = summary_entries[-1].get("overall_latency_ms", {})
        if isinstance(overall, dict):
            if overall.get("bqm_to_hJ_arrays") is not None:
                out["bqm_to_hj_ms"] = float(overall.get("bqm_to_hJ_arrays"))
            out["pre_iter_overhead_ms"] = (
                float(overall.get("embed_no_chains_drop_missing_cpp_internal", 0.0))
                + float(overall.get("build_variable_order_and_hardware_indices", 0.0))
                + float(overall.get("compute_variable_interaction_from_mapping", 0.0))
            )
            if out["iter_total_ms_sum"] is not None:
                out["total_overhead_ms"] = (
                    out["pre_iter_overhead_ms"] + out["iter_total_ms_sum"]
                )
    first_metric = timing_information[0]
    if isinstance(first_metric, dict):
        out["embedding_stats"] = first_metric.get("embedding_stats")
    summary_entries = [
        d
        for d in timing_information
        if isinstance(d, dict)
        and d.get("__tag__") == "solve_join_ordering_query_summary"
    ]
    if summary_entries:
        timings_s = summary_entries[-1].get("timings_s", {})
        if isinstance(timings_s, dict):
            for k in IN_PIPELINE_KEYS:
                if timings_s.get(k) is not None:
                    out["in_pipeline_timing_ms"][f"in_pipeline_{k}_ms"] = (
                        float(timings_s[k]) * 1000.0
                    )
    return out


def run_solve_with_timeout(
    full_problem_path: str,
    emb: Any,
    iterations: Optional[int],
    timeout_s: int,
) -> Tuple[Any, Any, Optional[float], Any]:
    env = os.environ.copy()
    env["QDBO_FULL_PROBLEM_PATH"] = str(full_problem_path)
    env["QDBO_CUSTOM_EMB"] = str(emb)
    env["QDBO_ITERATIONS"] = "" if iterations is None else str(iterations)

    cmd = [
        sys.executable,
        "-c",
        r"""
import json, os, time

from qdbo.join_ordering.join_order import solve_join_ordering_query

full_problem_path = os.environ["QDBO_FULL_PROBLEM_PATH"]
emb = os.environ["QDBO_CUSTOM_EMB"]
iters_raw = os.environ.get("QDBO_ITERATIONS", "")
iterations = None if iters_raw == "" else int(iters_raw)

t0 = time.time()
join_order, db_cost, timing = solve_join_ordering_query(
    full_problem_path,
    emb,
    iterations=iterations,
)
t1 = time.time()
solve_time_s = t1 - t0

if hasattr(join_order, "tolist"):
    join_order = join_order.tolist()

print(json.dumps({
    "join_order": join_order,
    "db_cost": db_cost,
    "solve_time_s": solve_time_s,
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
        raise TimeoutError(
            f"solve_join_ordering_query timeout after {timeout_s}s"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"solve_join_ordering_query failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError("solve_join_ordering_query produced no stdout")

    try:
        payload = json.loads(out_lines[-1])
        return (
            payload["join_order"],
            payload["db_cost"],
            payload.get("solve_time_s"),
            payload.get("timing_information"),
        )
    except Exception as e:
        raise RuntimeError(
            f"cannot parse solve output as json. stdout={proc.stdout!r}"
        ) from e
