#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def _qubo_from_json(doc: Mapping[str, Any]) -> Dict[Tuple[int, int], float]:
    return {(int(item["i"]), int(item["j"])): float(item["value"]) for item in doc["Q"]}


def _selection_weight(ids: Iterable[int], weights: List[float]) -> float:
    return float(sum(weights[i] for i in ids))


def _selection_profit(ids: Iterable[int], profits: List[float], pairwise: Optional[Mapping[str, float]] = None) -> float:
    selected = set(ids)
    total = float(sum(profits[i] for i in selected))
    for key, value in (pairwise or {}).items():
        i, j = [int(part) for part in key.split(",")]
        if i in selected and j in selected:
            total += float(value)
    return total


def _qubo_energy(Q: Mapping[Tuple[int, int], float], bits_by_var: Mapping[int, int]) -> float:
    return float(sum(q * bits_by_var.get(i, 0) * bits_by_var.get(j, 0) for (i, j), q in Q.items()))


def _reference_solution(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {"error": f"missing reference solution: {path}"}
    return _read_json(path)


def _timing_summary(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, list):
        return {}
    iter_entries = [
        item for item in metrics
        if isinstance(item, dict) and "iteration" in item and "latency_total_measured_ms" in item
    ]
    summary_entries = [
        item for item in metrics
        if isinstance(item, dict) and item.get("__tag__") == "SEBREMforBQM_summary"
    ]
    return {
        "iteration_latencies_ms": [float(item["latency_total_measured_ms"]) for item in iter_entries],
        "iteration_latency_sum_ms": float(sum(float(item["latency_total_measured_ms"]) for item in iter_entries)),
        "qdbo_summary": summary_entries[-1] if summary_entries else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the QDBO iterative D-Wave QPU solver on qdbo index QUBO.")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--qubo", required=True)
    parser.add_argument("--reference-solution")
    parser.add_argument("--qdbo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", default="1,2,3")
    parser.add_argument(
        "--embedding",
        default="random",
        choices=[
            # Non-semantic strategies (apply to any BQM).
            "random",
            "greedy",
            # Semantic strategies for the index-selection (knapsack-style) QUBO.
            # INDEX prioritises the binary index-decision variables; SLACK
            # prioritises the budget slack bits. These four are the embeddings
            # reported in the paper (Table 4).
            "semanticINDEX_greedy",
            "semanticSLACK_greedy",
        ],
    )
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    qdbo_root = Path(args.qdbo_root).resolve()
    sys.path.insert(0, str(qdbo_root))

    import dimod
    from qdbo.core.iterative_solver import SEBREMforBQM

    problem_path = Path(args.problem)
    qubo_path = Path(args.qubo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = _read_json(problem_path)
    qubo_doc = _read_json(qubo_path)
    reference = _reference_solution(Path(args.reference_solution) if args.reference_solution else None)
    Q = _qubo_from_json(qubo_doc)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    var_order = list(bqm.variables)

    profits = [float(x) for x in problem["profits"]]
    weights = [float(x) for x in problem["weights"]]
    budget = float(problem["max_weight"])
    pairwise = problem.get("pairwise_interactions") or {}
    n_index_vars = int(qubo_doc.get("meta", {}).get("n_index_vars", len(profits)))
    n_slack_vars = int(qubo_doc.get("meta", {}).get("n_slack_vars", max(0, len(var_order) - n_index_vars)))

    # Precomputed semantic split for index-selection BQMs. var_order follows
    # the QUBO meta contract: [x_0..x_{n-1}, s_0..s_{m-1}].
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
        """Return query_meta needed by SEBREMforBQM for the given strategy.

        Semantic strategies require a split; non-semantic strategies do not.
        """
        return semantic_query_meta if strategy.lower().startswith("semantic") else None

    rows = []
    raw_results = []
    iteration_list = sorted(int(part.strip()) for part in args.iterations.split(",") if part.strip())
    max_iterations = max(iteration_list)

    # Run SEBREMforBQM once with max iterations; extract intermediate bests
    # from best_sample_so_far_history to avoid redundant QPU calls.
    start = time.time()
    try:
        rel_ent, samples, best_hist, best_sample, best_obj, num_calls, metrics = SEBREMforBQM(
            bqm.copy(),
            None,
            args.beta,
            max_iterations,
            args.step,
            args.num_reads,
            args.embedding,
            return_timing_metric=True,
            query_id="qdbo_index_qubo",
            query_meta=_query_meta_for(args.embedding),
        )
    except Exception as exc:
        # If the single run fails, record error for all iteration values
        wall = time.time() - start
        for iterations in iteration_list:
            row = {"iterations": iterations, "embedding": args.embedding,
                   "num_reads": args.num_reads, "beta": args.beta, "step": args.step,
                   "status": "error", "error_type": type(exc).__name__,
                   "error": str(exc), "wall_time_s": wall}
            rows.append(row)
            raw_results.append({"iterations": iterations, "row": row})
            _write_json(out_dir / f"qdbo_iter_{iterations}.json", raw_results[-1])
        # Do NOT write qdbo_iterative_summary.json here: sweep_solve.sh uses that
        # file as the "cell already done" marker, so a failed run must leave it
        # absent and surface as a non-zero exit code.
        return 1

    if metrics is not None:
        wall = time.time() - start
        # Extract best_sample_so_far_history from the summary entry in metrics
        summary_entry = next((m for m in metrics if isinstance(m, dict) and m.get("__tag__") == "SEBREMforBQM_summary"), None)
        best_so_far_history = summary_entry.get("best_sample_so_far_history", []) if summary_entry else []

        for iterations in iteration_list:
            # Use best_so_far at the end of iteration (iterations-1) index
            hist_idx = min(iterations - 1, len(best_so_far_history) - 1)
            if hist_idx >= 0 and best_so_far_history:
                iter_best_sample = best_so_far_history[hist_idx]
            else:
                iter_best_sample = best_sample.tolist() if best_sample is not None else []

            bits_by_var = {int(var): int(iter_best_sample[pos]) for pos, var in enumerate(var_order)}
            selected_ids = [i for i in range(n_index_vars) if bits_by_var.get(i, 0) == 1]
            selected_weight = _selection_weight(selected_ids, weights)
            selected_profit = _selection_profit(selected_ids, profits, pairwise)
            feasible = selected_weight <= budget
            row: Dict[str, Any] = {
                "iterations": iterations,
                "embedding": args.embedding,
                "num_reads": args.num_reads,
                "beta": args.beta,
                "step": args.step,
            }
            row.update({
                    "status": "ok",
                    "selected_ids": selected_ids,
                    "selected_weight": selected_weight,
                    "selected_profit": selected_profit,
                    "feasible": feasible,
                    "qubo_energy": _qubo_energy(Q, bits_by_var),
                    "qdbo_best_objective": float(best_obj),
                    "num_calls": int(num_calls),
                    "wall_time_s": wall,
                    "best_objective_history": [float(x) if not math.isnan(float(x)) else None for x in best_hist[:iterations]],
                })
            row.update(_timing_summary(metrics))
            raw_results.append({"iterations": iterations, "row": row, "timing_metrics": metrics})
            rows.append(row)
            _write_json(out_dir / f"qdbo_iter_{iterations}.json", raw_results[-1])

    summary = {
        "problem": str(problem_path),
        "qubo": str(qubo_path),
        "qdbo_root": str(qdbo_root),
        "solver": "VLDB26-QDBO SEBREMforBQM over DWaveSampler QPU",
        "reference_solution": reference,
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
                compact["selected_ids"] = ",".join(str(x) for x in compact["selected_ids"])
            writer.writerow({key: compact.get(key) for key in fieldnames})

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
