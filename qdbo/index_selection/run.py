from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import dimod

from qdbo import config
from qdbo.index_selection.qubo import build_index_selection_qubo
from qdbo.iterative_solver import solve_bqm_with_iterative_qa


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def _sort_qubo_terms(
    qubo: Mapping[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    return {(int(i), int(j)): float(v) for (i, j), v in sorted(qubo.items())}


def _compute_selection_weight(ids: Iterable[int], weights: List[float]) -> float:
    return float(sum(weights[i] for i in ids))


def _compute_selection_profit(
    ids: Iterable[int],
    profits: List[float],
    pairwise: Optional[Mapping[str, float]] = None,
) -> float:
    selected = set(ids)
    total = float(sum(profits[i] for i in selected))
    for key, value in (pairwise or {}).items():
        i, j = [int(part) for part in key.split(",")]
        if i in selected and j in selected:
            total += float(value)
    return total


def _compute_qubo_energy(
    Q: Mapping[Tuple[int, int], float], bits_by_var: Mapping[int, int]
) -> float:
    return float(
        sum(
            q * bits_by_var.get(i, 0) * bits_by_var.get(j, 0) for (i, j), q in Q.items()
        )
    )


def _summarize_timing(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, list):
        return {}
    iter_entries = [
        item
        for item in metrics
        if isinstance(item, dict)
        and "iteration" in item
        and "latency_total_measured_ms" in item
    ]
    summary_entries = [
        item
        for item in metrics
        if isinstance(item, dict)
        and item.get("__tag__") == "iterative_qa_solve_summary"
    ]
    return {
        "iteration_latencies_ms": [
            float(item["latency_total_measured_ms"]) for item in iter_entries
        ],
        "iteration_latency_sum_ms": float(
            sum(float(item["latency_total_measured_ms"]) for item in iter_entries)
        ),
        "qdbo_summary": summary_entries[-1] if summary_entries else None,
    }


BUDGET_PERCENTS = [10, 20, 40, 50]
ITERATIONS = [1, 3]
NUM_READS = 100
BETA = 1.0
STEP = 0.01


def solve_budget(budget_percent: int, embedding: str) -> int:
    problem_path = Path(config.INDEX_SELECTION_PROBLEM)
    out_dir = (
        Path(config.INDEX_SELECTION_RESULTS_DIR)
        / f"budget_{budget_percent}pct"
        / f"qdbo_{embedding}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = _read_json(problem_path)
    problem["max_weight"] = max(1, int(sum(problem["weights"]) * budget_percent / 100))
    qubo, qubo_meta = build_index_selection_qubo(
        problem["profits"],
        problem["weights"],
        problem["max_weight"],
        pairwise_interactions=problem.get("pairwise_interactions") or None,
    )
    Q = _sort_qubo_terms(qubo)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    var_order = list(bqm.variables)

    profits = [float(x) for x in problem["profits"]]
    weights = [float(x) for x in problem["weights"]]
    budget = float(problem["max_weight"])
    pairwise = problem.get("pairwise_interactions") or {}
    n_index_vars = qubo_meta["n_index_vars"]
    n_slack_vars = qubo_meta["n_slack_vars"]

    index_split = {
        "INDEX": var_order[:n_index_vars],
        "SLACK": var_order[n_index_vars : n_index_vars + n_slack_vars],
    }
    split_meta = {
        "source": "qdbo_index_qubo",
        "n_index_vars": n_index_vars,
        "n_slack_vars": n_slack_vars,
        "n_total_vars": len(var_order),
    }
    semantic_query_meta = {"index_split": index_split, "split_meta": split_meta}

    def _query_meta_for(strategy: str) -> Optional[Dict[str, Any]]:
        return semantic_query_meta if strategy.lower().startswith("semantic") else None

    rows = []
    raw_results = []
    iteration_list = ITERATIONS
    max_iterations = max(iteration_list)

    start = time.time()
    try:
        rel_ent, samples, best_hist, best_sample, best_obj, num_calls, metrics = (
            solve_bqm_with_iterative_qa(
                bqm.copy(),
                None,
                BETA,
                max_iterations,
                STEP,
                NUM_READS,
                embedding,
                query_id="qdbo_index_qubo",
                query_meta=_query_meta_for(embedding),
            )
        )
    except Exception as exc:
        wall = time.time() - start
        for iterations in iteration_list:
            row = {
                "iterations": iterations,
                "embedding": embedding,
                "num_reads": NUM_READS,
                "beta": BETA,
                "step": STEP,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_time_s": wall,
            }
            rows.append(row)
            raw_results.append({"iterations": iterations, "row": row})
            _write_json(out_dir / f"qdbo_iter_{iterations}.json", raw_results[-1])
        return 1

    wall = time.time() - start
    summary_entry = next(
        (
            m
            for m in metrics
            if isinstance(m, dict) and m.get("__tag__") == "iterative_qa_solve_summary"
        ),
        None,
    )
    best_so_far_history = (
        summary_entry.get("best_sample_so_far_history", []) if summary_entry else []
    )

    for iterations in iteration_list:
        hist_idx = min(iterations - 1, len(best_so_far_history) - 1)
        if hist_idx >= 0 and best_so_far_history:
            iter_best_sample = best_so_far_history[hist_idx]
            iter_best_obj = summary_entry["best_so_far_search_objective"][hist_idx]
            iter_num_calls = hist_idx + 1
        else:
            iter_best_sample = best_sample.tolist() if best_sample is not None else []
            iter_best_obj = best_obj
            iter_num_calls = num_calls

        bits_by_var = {
            int(var): int(iter_best_sample[pos]) for pos, var in enumerate(var_order)
        }
        selected_ids = [i for i in range(n_index_vars) if bits_by_var.get(i, 0) == 1]
        selected_weight = _compute_selection_weight(selected_ids, weights)
        selected_profit = _compute_selection_profit(selected_ids, profits, pairwise)
        feasible = selected_weight <= budget
        row: Dict[str, Any] = {
            "iterations": iterations,
            "embedding": embedding,
            "num_reads": NUM_READS,
            "beta": BETA,
            "step": STEP,
        }
        row.update(
            {
                "status": "ok",
                "selected_ids": selected_ids,
                "selected_weight": selected_weight,
                "selected_profit": selected_profit,
                "feasible": feasible,
                "qubo_energy": _compute_qubo_energy(Q, bits_by_var),
                "qdbo_best_objective": float(iter_best_obj),
                "num_calls": int(iter_num_calls),
                "wall_time_s": wall,
                "best_objective_history": [
                    float(x) if not math.isnan(float(x)) else None
                    for x in best_hist[:iterations]
                ],
            }
        )
        row.update(_summarize_timing(metrics))
        raw_results.append(
            {"iterations": iterations, "row": row, "timing_metrics": metrics}
        )
        rows.append(row)
        _write_json(out_dir / f"qdbo_iter_{iterations}.json", raw_results[-1])

    summary = {
        "problem": str(problem_path),
        "budget": problem["max_weight"],
        "solver": "VLDB26-QDBO iterative_qa_solve over DWaveSampler QPU",
        "rows": rows,
    }
    _write_json(out_dir / "qdbo_iterative_summary.json", summary)

    csv_path = out_dir / "qdbo_iterative_summary.csv"
    fieldnames = [
        "iterations",
        "status",
        "embedding",
        "num_reads",
        "selected_ids",
        "selected_weight",
        "selected_profit",
        "feasible",
        "qubo_energy",
        "qdbo_best_objective",
        "num_calls",
        "wall_time_s",
        "error_type",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            compact = dict(row)
            if isinstance(compact.get("selected_ids"), list):
                compact["selected_ids"] = ",".join(
                    str(x) for x in compact["selected_ids"]
                )
            writer.writerow({key: compact.get(key) for key in fieldnames})

    print(json.dumps(summary, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the QDBO iterative D-Wave QPU solver on the TPC-H index selection QUBO."
    )
    parser.add_argument(
        "--embedding",
        default="random",
        choices=[
            "random",
            "greedy",
            "semanticINDEX_greedy",
            "semanticSLACK_greedy",
        ],
    )
    args = parser.parse_args()
    status = 0
    for budget_percent in BUDGET_PERCENTS:
        status = solve_budget(budget_percent, args.embedding) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
