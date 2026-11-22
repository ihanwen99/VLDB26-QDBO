import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
from dimod import Vartype
from dwave.system import DWaveSampler

from qdbo.embedding.cpp_embed import embed_no_chains_drop_missing_cpp
from qdbo.output import save_embedding_json
from qdbo import config

GLOBAL_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_EMBEDDING_DIR = config.EMBEDDING_DIR


def is_semantic_strategy(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("semantic")


def split_bqm_variables_by_kind(card, pred_sel, bqm):
    R = len(card)
    P = len(pred_sel)
    if R < 3:
        raise ValueError(f"num_relations must be >=3, got {R}")
    J = R - 2
    N = len(bqm.variables)

    nV = R * J
    nP = P * J
    base = nV + nP
    if base > N:
        raise ValueError(
            f"Index split invalid: V+PRED={base} > N={N} (R={R},P={P},J={J})"
        )

    slack_total = N - base
    if slack_total % J != 0:
        raise ValueError(
            f"Index split invalid: slack_total={slack_total} not divisible by J={J}. "
            f"Maybe variable order changed or extra aux vars exist."
        )
    S = slack_total // J

    vars_list = list(bqm.variables)

    idx_v = list(range(nV))
    idx_pred = list(range(nV, nV + nP))
    idx_slack = list(range(nV + nP, N))

    split = {
        "V": [vars_list[i] for i in idx_v],
        "PRED": [vars_list[i] for i in idx_pred],
        "SLACK": [vars_list[i] for i in idx_slack],
    }

    meta = {
        "R": R,
        "P": P,
        "J": J,
        "N": N,
        "nV": nV,
        "nP": nP,
        "slack_total": slack_total,
        "S_per_join": S,
    }
    return split, meta


def build_variable_order_and_hardware_indices(orig_bqm, mapping):
    var_order = [v for v in orig_bqm.variables if v in mapping]
    hw_for_var = [mapping[v] for v in var_order]
    return var_order, hw_for_var


def compute_variable_interactions_from_mapping(orig_bqm, target_bqm, mapping):
    var_order = [v for v in orig_bqm.variables if v in mapping]
    var_index = {v: i for i, v in enumerate(var_order)}
    qubit_to_var_idx = {}
    for v, q in mapping.items():
        if v in var_index:
            qubit_to_var_idx[q] = var_index[v]

    pairs = []
    for q1, q2 in target_bqm.quadratic:
        i = qubit_to_var_idx.get(q1)
        j = qubit_to_var_idx.get(q2)
        if i is None or j is None or i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.append([i, j])

    if not pairs:
        return np.empty((0, 2), dtype=int), var_order
    return np.array(pairs, dtype=int), var_order


def update_bqm_in_spin_space(
    bqm_bin, grad_re_h, grad_re_j, step, variable_interaction, var_order, mapping
):
    bqm_spin = bqm_bin.change_vartype(Vartype.SPIN, inplace=False)
    bqm_spin = update_embedded_ising_model(
        bqm_spin, grad_re_h, grad_re_j, step, variable_interaction, var_order, mapping
    )
    bqm_bin_next = bqm_spin.change_vartype(Vartype.BINARY, inplace=False)
    return bqm_bin_next


def bqm_to_hj_arrays(orig_bqm, var_order):
    h_dict, J_dict, offset = orig_bqm.to_ising()
    idx = {v: i for i, v in enumerate(var_order)}
    N = len(var_order)
    h = np.zeros(N, dtype=float)
    J = np.zeros((N, N), dtype=float)

    for v, bias in h_dict.items():
        i = idx.get(v)
        if i is not None:
            h[i] = float(bias)

    if J_dict:
        ijb = [(idx.get(u), idx.get(v), float(bias)) for (u, v), bias in J_dict.items()]
        ijb = [(i, j, b) for i, j, b in ijb if i is not None and j is not None]
        if ijb:
            i_idx, j_idx, b_idx = zip(*ijb)
            J[i_idx, j_idx] += b_idx

    return h, J, float(offset)


def compute_ising_energies_from_spins(spins_var, h, J):
    part_h = spins_var.T @ h
    part_J = np.einsum("ns,nm,ms->s", spins_var, J, spins_var, optimize=True)
    return part_h + part_J


def compute_relative_entropy_gradient(
    spin_solutions, probabilities, gvec, beta, variable_interaction
):
    n_variables = spin_solutions.shape[0]
    corr1 = spin_solutions @ probabilities
    corr2 = spin_solutions @ np.diag(probabilities) @ spin_solutions.T
    vec_corr2 = corr2.flatten()
    indices = variable_interaction[:, 0] * n_variables + variable_interaction[:, 1]

    corr2_coupler = vec_corr2[indices]
    hvec = probabilities * (np.log(probabilities) + beta * gvec)
    prod_vec = (
        spin_solutions[variable_interaction[:, 0], :]
        * spin_solutions[variable_interaction[:, 1], :]
    )
    average_log = (np.log(probabilities) + beta * gvec) @ probabilities
    grad_re_h = -beta * (spin_solutions @ hvec - corr1 * average_log)
    grad_re_j = -beta * (prod_vec @ hvec - corr2_coupler * average_log)
    return grad_re_h, grad_re_j


def update_embedded_ising_model(
    bqm, grad_re_h, grad_re_j, step, variable_interaction, var_order, mapping
):
    for i, var in enumerate(var_order):
        q = mapping[var]
        bqm.add_linear(q, -step * float(grad_re_h[i]))
    for idx, (i, j) in enumerate(variable_interaction):
        q_i = mapping[var_order[i]]
        q_j = mapping[var_order[j]]
        if (q_i, q_j) in bqm.quadratic or (q_j, q_i) in bqm.quadratic:
            bqm.add_quadratic(q_i, q_j, -step * float(grad_re_j[idx]))

    return bqm


def solve_bqm_with_iterative_qa(
    bqm,
    partial_objective,
    beta,
    n_iterations,
    step,
    num_reads,
    custom_embedding,
    query_id: str = "unknown",
    query_meta: Optional[dict] = None,
):
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    def _now():
        return time.perf_counter()

    def _ms(dt_sec: float) -> float:
        return dt_sec * 1000.0

    def _record_elapsed_ms(store: dict, key: str, t0: float):
        store[key] = _ms(_now() - t0)

    overall_latency_ms = {}
    per_iter_latency_sum_ms = {}

    t_total0 = _now()

    t_split = _now()
    index_split = None
    split_meta = None
    if is_semantic_strategy(custom_embedding):
        if query_meta is None:
            raise ValueError(
                "semantic embedding requires query_meta. Provide either "
                "{index_split: {...}, split_meta: {...}} for a precomputed split, "
                "or {card: [...], pred_sel: [...]} for a join-ordering BQM."
            )

        if "index_split" in query_meta:
            index_split = query_meta["index_split"]
            split_meta = query_meta.get("split_meta", {})
        else:
            card = query_meta["card"]
            pred_sel = query_meta["pred_sel"]

            t = _now()
            index_split, split_meta = split_bqm_variables_by_kind(card, pred_sel, bqm)
            _record_elapsed_ms(
                overall_latency_ms, "semantic_split_bqm_indices_by_query", t
            )

    _record_elapsed_ms(overall_latency_ms, "stage_semantic_split_total", t_split)

    with DWaveSampler() as sampler:
        t_setup0 = _now()

        t = _now()
        orig_bqm = bqm.copy()
        _record_elapsed_ms(overall_latency_ms, "bqm_copy", t)

        t = _now()
        target_bqm, mapping, stats = embed_no_chains_drop_missing_cpp(
            bqm, sampler, strategy=custom_embedding, index_split=index_split
        )
        embed_elapsed_ms = _ms(_now() - t)
        overall_latency_ms["embed_no_chains_drop_missing_cpp_internal"] = float(
            stats.get("find_mapping_time_ms", 0.0)
        ) + float(stats.get("embed_bqm_time_ms", 0.0))
        overall_latency_ms["embed_no_chains_drop_missing_cpp"] = embed_elapsed_ms

        mapping_time_s = float(embed_elapsed_ms) / 1000.0

        bqm = target_bqm

        t = _now()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{query_id}__{custom_embedding}__{ts}__pid{os.getpid()}"
        _record_elapsed_ms(overall_latency_ms, "build_run_id", t)

        t = _now()
        save_embedding_json(
            base_dir=str(DEFAULT_EMBEDDING_DIR / GLOBAL_TIME),
            custom_embedding=custom_embedding,
            function_name="solve_join_ordering_query",
            query_id=query_id,
            run_id=run_id,
            orig_bqm=orig_bqm,
            target_bqm=target_bqm,
            mapping=mapping,
            stats=stats,
            extra_meta={
                "num_reads": num_reads,
                "beta": beta,
                "step": step,
                "n_iterations": n_iterations,
                "index_split": index_split if index_split else {},
                "split_meta": split_meta if split_meta else {},
            },
        )
        _record_elapsed_ms(overall_latency_ms, "save_embedding_json", t)

        t = _now()
        var_order, hw_for_var = build_variable_order_and_hardware_indices(
            orig_bqm, mapping
        )
        _record_elapsed_ms(
            overall_latency_ms, "build_variable_order_and_hardware_indices", t
        )

        t = _now()
        variable_interaction, _ = compute_variable_interactions_from_mapping(
            orig_bqm, bqm, mapping
        )
        _record_elapsed_ms(
            overall_latency_ms, "compute_variable_interaction_from_mapping", t
        )

        t = _now()
        h_full, J_full, _ = bqm_to_hj_arrays(orig_bqm, var_order)
        _record_elapsed_ms(overall_latency_ms, "bqm_to_hJ_arrays", t)

        _record_elapsed_ms(
            overall_latency_ms, "stage_setup_and_embedding_total", t_setup0
        )

        t = _now()
        best_objective_history = np.zeros(n_iterations)
        relative_entropy = np.zeros(n_iterations)
        per_iteration_best_samples = []
        best_sample_so_far_history = []
        best_objective_so_far_history = []

        if partial_objective is not None:
            best_objective_so_far = partial_objective + 1.0
        else:
            best_objective_so_far = float("inf")
        best_sample_overall = None

        complete_timing_metrics = []
        _record_elapsed_ms(overall_latency_ms, "stage_init_arrays_and_state_total", t)

        samples_raw = None
        for iteration in range(n_iterations):
            iter_latency_ms = {}
            iter_t0 = _now()

            t = _now()
            response = sampler.sample(bqm, num_reads=num_reads)
            _record_elapsed_ms(iter_latency_ms, "sampler.sample", t)

            t = _now()
            resp_order = list(response.variables)
            samples01 = response.record.sample
            counts = response.record.num_occurrences.astype(np.float64)
            _record_elapsed_ms(iter_latency_ms, "unpack_response_arrays", t)

            t = _now()
            timing_metrics = {}

            visible_future = getattr(response, "visible_future", None)
            for field in (
                "time_created",
                "time_received",
                "time_solved",
                "time_resolved",
            ):
                timing_metrics[field] = getattr(visible_future, field, None)

            service_timing = response.info.get("timing", {})
            for field in (
                "qpu_sampling_time",
                "qpu_anneal_time_per_sample",
                "qpu_readout_time_per_sample",
                "qpu_access_time",
                "qpu_access_overhead_time",
                "qpu_programming_time",
                "qpu_delay_time_per_sample",
                "post_processing_overhead_time",
                "total_post_processing_time",
            ):
                timing_metrics[field] = service_timing.get(field)

            timing_metrics["mapping_time_s"] = float(mapping_time_s)
            timing_metrics["embedding_stats"] = stats

            _record_elapsed_ms(iter_latency_ms, "extract_response_timing_info", t)

            t = _now()
            probabilities = counts / np.sum(counts)
            prob_safe = np.clip(probabilities, 1e-12, 1.0)
            _record_elapsed_ms(iter_latency_ms, "compute_probabilities", t)

            t = _now()
            spins_hw = 2 * samples01 - 1
            resp_index = {q: j for j, q in enumerate(resp_order)}
            cols = [resp_index[q] for q in hw_for_var]
            spins_var = spins_hw[:, cols].T
            samples_raw = samples01
            _record_elapsed_ms(iter_latency_ms, "convert_and_reorder_samples", t)

            t = _now()
            gvec = compute_ising_energies_from_spins(spins_var, h_full, J_full)
            current_best_objective = float(np.min(gvec))
            best_objective_history[iteration] = current_best_objective
            _record_elapsed_ms(iter_latency_ms, "ising_energy_from_spins_and_best", t)

            t = _now()
            best_idx = int(np.argmin(gvec))
            iteration_best_sample = samples01[best_idx, cols]
            per_iteration_best_samples.append(
                iteration_best_sample.astype(int).tolist()
            )
            if current_best_objective < best_objective_so_far:
                best_objective_so_far = current_best_objective
                best_sample_overall = iteration_best_sample
            if best_sample_overall is not None:
                best_sample_so_far_history.append(
                    best_sample_overall.astype(int).tolist()
                )
            else:
                best_sample_so_far_history.append([])
            best_objective_so_far_history.append(float(best_objective_so_far))
            _record_elapsed_ms(iter_latency_ms, "update_best_tracker", t)

            t = _now()
            relative_entropy[iteration] = -np.sum(
                prob_safe * np.log(prob_safe)
            ) + beta * np.dot(gvec, prob_safe)
            _record_elapsed_ms(iter_latency_ms, "compute_relative_entropy", t)

            if iteration > 0:
                t = _now()
                grad_re_h, grad_re_j = compute_relative_entropy_gradient(
                    spins_var, prob_safe, gvec, beta, variable_interaction
                )
                _record_elapsed_ms(iter_latency_ms, "re_gradient", t)

                t = _now()
                bqm = update_bqm_in_spin_space(
                    bqm,
                    grad_re_h,
                    grad_re_j,
                    step,
                    variable_interaction,
                    var_order,
                    mapping,
                )
                _record_elapsed_ms(iter_latency_ms, "update_in_spin_then_back", t)

            t = _now()
            early_stop = (partial_objective is not None) and (
                best_objective_so_far < partial_objective
            )
            _record_elapsed_ms(iter_latency_ms, "early_stopping_check", t)

            iter_total_ms = _ms(_now() - iter_t0)
            iter_blocks_sum_ms = float(sum(iter_latency_ms.values()))

            timing_metrics["iteration"] = iteration
            timing_metrics["latency_ms"] = iter_latency_ms
            timing_metrics["latency_blocks_sum_ms"] = iter_blocks_sum_ms
            timing_metrics["latency_total_measured_ms"] = iter_total_ms
            timing_metrics["latency_unaccounted_ms"] = max(
                0.0, iter_total_ms - iter_blocks_sum_ms
            )

            complete_timing_metrics.append(timing_metrics)

            for k, v in iter_latency_ms.items():
                per_iter_latency_sum_ms[k] = per_iter_latency_sum_ms.get(
                    k, 0.0
                ) + float(v)

            if early_stop:
                break

        number_of_calls = iteration + 1

        total_ms = _ms(_now() - t_total0)
        summary = {
            "__tag__": "iterative_qa_solve_summary",
            "query_id": query_id,
            "custom_embedding": custom_embedding,
            "n_iterations_requested": int(n_iterations),
            "n_iterations_executed": int(number_of_calls),
            "overall_latency_ms": overall_latency_ms,
            "per_iter_latency_sum_ms": per_iter_latency_sum_ms,
            "total_ms": total_ms,
            "mapping_time_s": float(mapping_time_s),
            "embedding_stats": stats,
            "var_order": [str(v) for v in var_order],
            "hw_for_var": [int(q) for q in hw_for_var],
            "per_iteration_best_samples": per_iteration_best_samples,
            "best_sample_so_far_history": best_sample_so_far_history,
            "per_iteration_search_objective": [
                float(x) for x in best_objective_history[:number_of_calls]
            ],
            "best_so_far_search_objective": best_objective_so_far_history,
        }
        complete_timing_metrics.append(summary)

        return (
            relative_entropy,
            samples_raw,
            best_objective_history,
            best_sample_overall,
            best_objective_so_far,
            number_of_calls,
            complete_timing_metrics,
        )
