import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pytz
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler

from backend.ProblemGenerator import get_join_ordering_problem, generate_Fujitsu_QUBO_for_left_deep_trees
from algorithm.embedding.cpp_embed import embed_no_chains_drop_missing_cpp
from backend.utils import save_embedding_json, make_query_id
from backend.generate_qubo import generate_maxcut_qubo

la = pytz.timezone("America/Los_Angeles")
GLOBAL_TIME = datetime.now(la).strftime("%Y%m%d_%H%M%S")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "persistent_custom_embedding"


#########################################
############# New Functions #############
#########################################

def is_semantic_strategy(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("semantic")


def split_bqm_indices_by_query(card, pred_sel, bqm):
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
        raise ValueError(f"Index split invalid: V+PRED={base} > N={N} (R={R},P={P},J={J})")

    slack_total = N - base
    if slack_total % J != 0:
        raise ValueError(
            f"Index split invalid: slack_total={slack_total} not divisible by J={J}. "
            f"Maybe variable order changed or extra aux vars exist."
        )
    S = slack_total // J

    # IMPORTANT: convert index ranges -> variable labels (as they appear in bqm.variables)
    vars_list = list(bqm.variables)

    idx_v = list(range(0, nV))
    idx_pred = list(range(nV, nV + nP))
    idx_slack = list(range(nV + nP, N))

    split = {
        "V": [vars_list[i] for i in idx_v],
        "PRED": [vars_list[i] for i in idx_pred],
        "SLACK": [vars_list[i] for i in idx_slack],
    }

    meta = {
        "R": R, "P": P, "J": J, "N": N,
        "nV": nV, "nP": nP,
        "slack_total": slack_total,
        "S_per_join": S
    }
    return split, meta


def build_variable_order_and_hardware_indices(orig_bqm, mapping):
    """
    Build a stable variable order and aligned hardware indices
    Returns:
        - var_order: list of logical variable labels (ordered as in orig_bqm)
        - hw_for_var: list of hardware qubits aligned to var_order
    """
    var_order = [v for v in orig_bqm.variables if v in mapping]  # Keep only variables successfully mapped to hardware
    hw_for_var = [mapping[v] for v in var_order]  # Convert each logical variable to its mapped hardware qubit
    return var_order, hw_for_var  # Return ordered variables and corresponding qubits


def compute_variable_interaction_from_mapping(orig_bqm, target_bqm, mapping):
    """
    Build variable-space coupler pairs that exist on hardware
    - Constructs variable_interaction (pairs of indices in var_order) but keeps only pairs that correspond
        to actual quadratic couplers present on the embedded hardware BQM (target_bqm), mirroring MATLAB VariableInteraction.
    """
    var_order = [v for v in orig_bqm.variables if v in mapping]  # Logical variables that were embedded
    var_index = {v: i for i, v in enumerate(var_order)}  # Map logical variable -> var_order index
    qubit_to_var_idx = {}
    for v, q in mapping.items():
        if v in var_index:
            qubit_to_var_idx[q] = var_index[v]

    pairs = []  # Accumulator for index pairs (i, j) in var_order space
    for q1, q2 in target_bqm.quadratic:  # Iterate only existing hardware couplers
        i = qubit_to_var_idx.get(q1)
        j = qubit_to_var_idx.get(q2)
        if i is None or j is None or i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.append([i, j])
    return np.array(pairs, dtype=int), var_order  # Return Nx2 array of pairs and the var_order used


def update_in_spin_then_back(bqm_bin,
                             grad_re_h, grad_re_j, step,
                             variable_interaction, var_order, mapping):
    """
    Update a binary BQM by switching to SPIN domain, applying (h,J) gradient, then switching back
    """
    # This implements "sample in QUBO/BINARY, update in Ising/SPIN, convert back to BINARY"
    bqm_spin = bqm_bin.change_vartype(Vartype.SPIN, inplace=False)  # Convert BINARY BQM to SPIN (energy-equivalent up to offset)
    bqm_spin = update_ising_model_embedded(bqm_spin, grad_re_h, grad_re_j, step,  # Apply gradient update in SPIN (h,J) semantics
                                           variable_interaction, var_order, mapping)  # Use only existing hardware couplers
    bqm_bin_next = bqm_spin.change_vartype(Vartype.BINARY, inplace=False)  # Convert back to BINARY for the next sampling round
    return bqm_bin_next  # Return updated BINARY BQM


def bqm_to_hJ_arrays(orig_bqm, var_order):
    """
    Converts the original logical BQM into Ising parameters (h,J) and aligns them to var_order.
    Returns:
      h: (N,) array of linear SPIN biases
      J: (N,N) symmetric array of quadratic SPIN couplers
      offset: constant energy shift (not used for gradients/comparisons)
    """
    h_dict, J_dict, offset = orig_bqm.to_ising()  # Convert to Ising dictionaries and energy offset (dimod handles conversion)
    idx = {v: i for i, v in enumerate(var_order)}  # Map variable label -> aligned index
    N = len(var_order)  # Number of embedded variables
    h = np.zeros(N, dtype=float)  # Linear term array
    J = np.zeros((N, N), dtype=float)  # Quadratic term matrix

    for v, bias in h_dict.items():  # Fill h entries where variable exists in var_order
        i = idx.get(v)
        if i is not None:
            h[i] = float(bias)

    if J_dict:
        ijb = [(idx.get(u), idx.get(v), float(bias)) for (u, v), bias in J_dict.items()]
        ijb = [(i, j, b) for i, j, b in ijb if i is not None and j is not None]
        if ijb:
            i_idx, j_idx, b_idx = zip(*ijb)
            J[i_idx, j_idx] += b_idx
            J[j_idx, i_idx] += b_idx

    return h, J, float(offset)  # Return arrays and offset


def ising_energy_from_spins(spins_var, h, J):
    """
    Compute G(s) = s^T J s + h^T s for variable-space spins
    - spins_var: (N, S) in {-1, +1}, columns are samples in variable order
    - h: (N,), J: (N,N) symmetric Ising parameters aligned with spins_var rows
    """
    part_h = spins_var.T @ h  # (S,) compute h^T s for each sample
    part_J = np.einsum('ns,nm,ms->s', spins_var, J, spins_var, optimize=True)  # (S,) compute s^T J s compactly
    return part_h + part_J  # Return per-sample energies


#########################################

# ---------------------------
# Readout Functions Similar/Same as Digital Inspired (vldb24) Paper
# ---------------------------

def get_raw_join_order(cost_vector):  # Rank indices by descending cost (custom readout heuristic)
    # Example:
    #   cost_vector        [0 6 1 3 6]
    #   argsort ascending  [0 2 3 1 4]
    #   reversed           [4 1 3 2 0]  (from large to small)
    join_order = np.argsort(cost_vector).tolist()  # Indices sorted ascending by cost
    join_order.reverse()  # Reverse to descending order
    return join_order  # Return ranking list


def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=False):  # Compute per-step left-deep costs
    int_costs = []  # Accumulator for intermediate cardinalities/costs
    join_order = join_order.copy()  # Defensive copy
    if join_order[0] > join_order[1]:  # Enforce deterministic ordering on first two relations
        join_order[0], join_order[1] = join_order[1], join_order[0]  # Swap to ensure join_order[0] < join_order[1]
    prev_join_result = card[join_order[0]]  # Cardinality of the first relation
    for j in range(1, len(card) - 1):  # Iterate through join positions
        jo_hash = str(join_order[0:j + 1])  # Key for memoization of intermediate cardinality
        if jo_hash in card_dict:  # If cached
            int_card = card_dict[jo_hash]  # Retrieve cached intermediate
        else:  # Otherwise compute selectivity
            sel = get_selectivity_for_new_relation(join_order, j, pred, pred_sel)  # Combined selectivity with earlier relations
            int_card = prev_join_result * card[join_order[j]] * sel  # Estimated intermediate cardinality
            card_dict[jo_hash] = int_card  # Cache the value
        prev_join_result = int_card  # Update previous result
        int_costs.append(int_card)  # Record this step's cost
    if verbose:  # Optional printing
        print(int_costs)  # Show intermediate costs
    return int_costs  # Return list of intermediate costs


def get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict, verbose=False):  # Sum intermediate join costs
    total_costs = 0  # Accumulator for total cost
    int_costs = get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=verbose)  # Compute partial costs
    for cost in int_costs:  # Sum all intermediate costs
        total_costs = total_costs + cost  # Add intermediate cost
    return total_costs  # Return total cost


def get_selectivity_for_new_relation(join_order, j, pred, pred_sel):  # Compute selectivity when adding relation j
    sel = 1  # Start with neutral selectivity
    new_relation = join_order[j]  # Relation to add
    for i in range(j):  # Check predicates between the new relation and all earlier ones
        relation = join_order[i]  # Earlier relation
        if (relation, new_relation) in pred:  # If predicate stored as (i,j)
            sel = sel * pred_sel[pred.index((relation, new_relation))]  # Multiply by corresponding selectivity
        elif (new_relation, relation) in pred:  # Or stored reversed
            sel = sel * pred_sel[pred.index((new_relation, relation))]  # Multiply by corresponding selectivity
    return sel  # Return combined selectivity


def postprocess_join_order(raw_join_order, cost_vector, num_relations, pred):  # Connectivity-aware postprocessing of join order
    join_order = [raw_join_order[0]]  # Start from the top-ranked relation
    while len(join_order) < num_relations:  # Until we pick all relations
        applicable_predicates = [pred_tuple for t in join_order for pred_tuple in pred if t in pred_tuple]  # Predicates touching current set
        neighborhood_indices = [t for t in set(sum(applicable_predicates, ())) if t not in join_order]  # Neighbors not yet included
        if len(neighborhood_indices) != 0:  # If there are neighbors
            best_neighbor_relation = neighborhood_indices[np.argmax(cost_vector[neighborhood_indices])]  # Pick the neighbor with max cost
            join_order.append(best_neighbor_relation)  # Append neighbor
        else:  # No neighbors, fall back to global best not yet used
            global_indices = [x for x in raw_join_order if x not in join_order]  # Remaining relations by global order
            best_global_relation = global_indices[np.argmax(cost_vector[global_indices])]  # Best among remaining
            join_order.append(best_global_relation)  # Append
    return join_order  # Return refined order


def read_out(sample, card, pred, pred_sel, card_dict):
    """
    Convert the annealer output to actual join order
    :param sample: [0, 1, 0, 1, ...]
    :return: [join_order, int(cost), whether_is_fallback_join_order ]
    """
    weight_vector = np.arange(1, len(card) - 1)  # Create weights [1,2,...,n-2]
    weight_vector = weight_vector[len(card) - 3::-1]  # Reverse and trim to match partial rows

    num_relations = len(card)  # Number of relations
    bitstring = sample[:len(card) * (len(card) - 2)]  # sample here <== bitstrings[i] in DA code
    partial_bitstrings = np.array_split(bitstring, len(card))  # Split into rows per relation
    cost_vector = np.array(partial_bitstrings).dot(weight_vector)  # Weighted sum per relation

    raw_join_order = get_raw_join_order(cost_vector)  # Rank by cost descending
    costs = get_costs_for_leftdeep_tree(raw_join_order, card, pred, pred_sel, card_dict)  # Compute cost of raw order
    solution = [raw_join_order, int(costs), False]  # Prepare primary solution

    fallback_join_order = postprocess_join_order(raw_join_order, cost_vector, num_relations, pred)  # Connectivity-aware fallback
    fallback_costs = get_costs_for_leftdeep_tree(fallback_join_order, card, pred, pred_sel, card_dict)  # Cost of fallback order
    fallback_solution = [fallback_join_order, int(fallback_costs), True]  # Prepare fallback solution
    if costs < fallback_costs:  # Choose better of the two
        return solution  # Return primary if better
    return fallback_solution  # Else return fallback


# ---------------------------
# Helper Functions
# ---------------------------
def qubo_to_ising(Q):
    """
    Converts a QUBO matrix to Ising model parameters h and J.
    Assumes Q is symmetric upper-triangularized beforehand. #TODO: Double Check
    """
    N = Q.shape[0]
    h = np.zeros(N)
    J = np.zeros((N, N))

    for i in range(N):
        h[i] = Q[i, i] + 0.5 * np.sum(Q[i, :]) - 0.5 * Q[i, i]

    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = Q[i, j] / 4

    return h, J


# ---------------------------
# Gradient Functions & Update Ising Models
# ---------------------------

def re_gradient(spin_solutions, probabilities, gvec, beta, variable_interaction):
    """
    Computes the gradient of the relative entropy.

    spin_solutions: array of shape (n_variables, n_samples)
    probabilities: array of length n_samples
    gvec: vector of objective values for each sample
    variable_interaction: array of shape (n_couplers, 2) with pairs (i, j)
    """

    n_variables = spin_solutions.shape[0]  # Number of variables N
    corr1 = spin_solutions @ probabilities  # (N,) E[s] first moments across samples
    corr2 = spin_solutions @ np.diag(probabilities) @ spin_solutions.T  # (N,N) E[s s^T] second moments
    vec_corr2 = corr2.flatten()  # Vectorize second-moment matrix
    indices = variable_interaction[:, 0] * n_variables + variable_interaction[:, 1]  # Row-major indices for (i,j)

    # Flatten corr2 (row-major) and extract only entries for the given couplers
    corr2_coupler = vec_corr2[indices]  # Extract only entries that correspond to active couplers
    hvec = probabilities * (np.log(probabilities) + beta * gvec)  # (S,) per-sample weights for gradient
    prod_vec = (spin_solutions[variable_interaction[:, 0], :] *  # (M,S) pairwise spin products s_i * s_j
                spin_solutions[variable_interaction[:, 1], :])  # (M,S) aligned by variable_interaction
    average_log = (np.log(probabilities) + beta * gvec) @ probabilities  # Scalar: E[log p + beta G]
    grad_re_h = -beta * (spin_solutions @ hvec - corr1 * average_log)  # (N,) gradient wrt h
    grad_re_j = -beta * (prod_vec @ hvec - corr2_coupler * average_log)  # (M,) gradient wrt couplers J_ij for existing couplers
    return grad_re_h, grad_re_j  # Return gradients


def update_ising_model_embedded(bqm, grad_re_h, grad_re_j, step, variable_interaction, var_order, mapping):
    """
    Map variable-space gradients (aligned to var_order) back to the hardware-space target_bqm.
    - bqm: target_bqm (hardware-labeled variables)
    - grad_re_h: linear gradient per variable (aligned to var_order)
    - grad_re_j: quadratic gradient per variable pair (aligned to variable_interaction)
    - variable_interaction: (i, j) index pairs for var_order
    - var_order: logical variable labels in order
    - mapping: logical variable -> hardware qubit
    """
    for i, var in enumerate(var_order):  # Update linear terms
        q = mapping[var]  # Hardware qubit for logical variable
        bqm.add_linear(q, -step * float(grad_re_h[i]))  # Gradient descent on h (SPIN) mapped to qubit bias
    for idx, (i, j) in enumerate(variable_interaction):  # Update quadratic terms
        q_i = mapping[var_order[i]]  # Hardware qubit for variable i
        q_j = mapping[var_order[j]]  # Hardware qubit for variable j
        if (q_i, q_j) in bqm.quadratic or (q_j, q_i) in bqm.quadratic:  # Only update if hardware coupler exists
            bqm.add_quadratic(q_i, q_j, -step * float(grad_re_j[idx]))  # Gradient descent on J_ij (SPIN) mapped to coupler
        # If the coupler is absent, skip (variable_interaction should already reflect presence)  # Clarification
    return bqm  # Return updated SPIN-domain BQM


# ---------------------------
# Main SEBREM Function
# ---------------------------

def SEBREMforQUBO(Qfull, partial_objective, beta, n_iterations, step, num_reads, custom_embedding, return_timing_metric):
    """
    Wrapper around SEBREMforBQM for QUBO input.
    """
    bqm = BinaryQuadraticModel.from_qubo(Qfull)  # Convert the QUBO matrix to a Binary Quadratic Model
    return SEBREMforBQM(bqm, partial_objective, beta, n_iterations, step, num_reads, custom_embedding, return_timing_metric)


def SEBREMforBQM(bqm, partial_objective, beta, n_iterations, step, num_reads, custom_embedding,
                 return_timing_metric=False, query_id: str = "unknown", query_meta: Optional[dict] = None):
    """
    Core SEBREM loop working directly on a BQM.
    """

    # -----------------------
    # Timing helpers
    # -----------------------
    def _now():
        return time.perf_counter()

    def _ms(dt_sec: float) -> float:
        return dt_sec * 1000.0

    def _t_end_ms(store: dict, key: str, t0: float):
        store[key] = _ms(_now() - t0)

    overall_latency_ms = {}  # non-iteration stage timings (ms)
    per_iter_latency_sum_ms = {}  # aggregate per-iteration block sums across iterations (ms)

    t_total0 = _now()

    # -----------------------
    # Stage A: semantic split (optional)
    # -----------------------
    t_stageA = _now()
    index_split = None
    split_meta = None
    if is_semantic_strategy(custom_embedding):
        if query_meta is None:
            raise ValueError("semantic embedding requires query_meta with card and pred_sel (or R,P).")
        card = query_meta["card"]
        pred_sel = query_meta["pred_sel"]

        t = _now()
        index_split, split_meta = split_bqm_indices_by_query(card, pred_sel, bqm)
        _t_end_ms(overall_latency_ms, "semantic_split_bqm_indices_by_query", t)

        print("[semantic] index_split meta:", split_meta)
    _t_end_ms(overall_latency_ms, "stage_semantic_split_total", t_stageA)

    # -----------------------
    # Stage B: setup sampler + embedding
    # -----------------------
    print(f"custom_embedding: {custom_embedding}")
    with DWaveSampler() as sampler:
        t_setup0 = _now()

        # copy
        t = _now()
        orig_bqm = bqm.copy()
        _t_end_ms(overall_latency_ms, "bqm_copy", t)

        # embedding mapping (IMPORTANT: restore mapping_time_s correctly)
        t = _now()
        target_bqm, mapping, stats = embed_no_chains_drop_missing_cpp(
            bqm, sampler, strategy=custom_embedding, index_split=index_split
        )
        embed_elapsed_ms = _ms(_now() - t)
        overall_latency_ms["embed_no_chains_drop_missing_cpp_internal"] = (
                float(stats.get("find_mapping_time_ms", 0.0))
                + float(stats.get("embed_bqm_time_ms", 0.0))
        )
        overall_latency_ms["embed_no_chains_drop_missing_cpp"] = embed_elapsed_ms

        # keep the original "mapping_time_s" contract (seconds, float, non-None)
        mapping_time_s = float(embed_elapsed_ms) / 1000.0

        bqm = target_bqm
        print(f"mapping: {mapping}")
        print(f"stats: {stats}")
        print(f"mapping_time_s: {mapping_time_s:.6f}")

        # run_id + save json
        t = _now()
        ts = datetime.now(la).strftime("%Y%m%d_%H%M%S")
        run_id = f"{query_id}__{custom_embedding}__{ts}__pid{os.getpid()}"
        _t_end_ms(overall_latency_ms, "build_run_id", t)

        t = _now()
        saved_path = save_embedding_json(
            base_dir=str(DEFAULT_EMBEDDING_DIR / GLOBAL_TIME),
            custom_embedding=custom_embedding,
            function_name="actual_query_blackbox",
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
        _t_end_ms(overall_latency_ms, "save_embedding_json", t)
        print(f"[saved] embedding json -> {saved_path}")

        # build var order + interactions + arrays
        t = _now()
        var_order, hw_for_var = build_variable_order_and_hardware_indices(orig_bqm, mapping)
        _t_end_ms(overall_latency_ms, "build_variable_order_and_hardware_indices", t)

        t = _now()
        variable_interaction, _ = compute_variable_interaction_from_mapping(orig_bqm, bqm, mapping)
        _t_end_ms(overall_latency_ms, "compute_variable_interaction_from_mapping", t)
        print("variable_interaction length: ", len(variable_interaction))

        t = _now()
        h_full, J_full, _ = bqm_to_hJ_arrays(orig_bqm, var_order)
        _t_end_ms(overall_latency_ms, "bqm_to_hJ_arrays", t)

        _t_end_ms(overall_latency_ms, "stage_setup_and_embedding_total", t_setup0)

        # -----------------------
        # Initialize history storage
        # -----------------------
        t = _now()
        best_objective_history = np.zeros(n_iterations)
        relative_entropy = np.zeros(n_iterations)

        if partial_objective is not None:
            best_objective_so_far = partial_objective + 1.0
        else:
            best_objective_so_far = float("inf")
        best_sample_overall = None

        complete_timing_metrics = []
        _t_end_ms(overall_latency_ms, "stage_init_arrays_and_state_total", t)

        # -----------------------
        # Main optimization loop
        # -----------------------
        samples_raw = None  # ensure defined even if n_iterations==0
        for iteration in range(n_iterations):
            iter_latency_ms = {}
            iter_t0 = _now()

            print("\n\n======= Iteration: {} =======".format(iteration))
            print(f"bqm:{bqm}")

            # ---- Block 1: quantum sampling call
            t = _now()
            response = sampler.sample(bqm, num_reads=num_reads)
            _t_end_ms(iter_latency_ms, "sampler.sample", t)

            # ---- Block 2: unpack response
            t = _now()
            resp_order = list(response.variables)
            samples01 = response.record.sample
            energies = response.record.energy
            counts = response.record.num_occurrences.astype(np.float64)
            _t_end_ms(iter_latency_ms, "unpack_response_arrays", t)

            # ---- Timing Metrics (KEEP ORIGINAL FIELDS)
            t = _now()
            timing_metrics = {}

            ## 1) Calling Stack (service-provided)
            timing_metrics["time_created"] = response.visible_future.time_created
            timing_metrics["time_received"] = response.visible_future.time_received
            timing_metrics["time_solved"] = response.visible_future.time_solved
            timing_metrics["time_resolved"] = response.visible_future.time_resolved

            ## 2) Quantum Service (service-provided)
            timing_metrics["qpu_sampling_time"] = response.info["timing"]["qpu_sampling_time"]
            timing_metrics["qpu_anneal_time_per_sample"] = response.info["timing"]["qpu_anneal_time_per_sample"]
            timing_metrics["qpu_readout_time_per_sample"] = response.info["timing"]["qpu_readout_time_per_sample"]
            timing_metrics["qpu_access_time"] = response.info["timing"]["qpu_access_time"]
            timing_metrics["qpu_access_overhead_time"] = response.info["timing"]["qpu_access_overhead_time"]
            timing_metrics["qpu_programming_time"] = response.info["timing"]["qpu_programming_time"]
            timing_metrics["qpu_delay_time_per_sample"] = response.info["timing"]["qpu_delay_time_per_sample"]
            timing_metrics["post_processing_overhead_time"] = response.info["timing"]["post_processing_overhead_time"]
            timing_metrics["total_post_processing_time"] = response.info["timing"]["total_post_processing_time"]

            # IMPORTANT: keep original fields your outer code expects
            timing_metrics["mapping_time_s"] = float(mapping_time_s)  # seconds, not None
            timing_metrics["embedding_stats"] = stats  # dict

            _t_end_ms(iter_latency_ms, "extract_response_timing_info", t)

            # ---- Block 4: probability compute
            t = _now()
            probabilities = counts / np.sum(counts)
            prob_safe = np.clip(probabilities, 1e-12, 1.0)
            _t_end_ms(iter_latency_ms, "compute_probabilities", t)

            # ---- Block 5: 0/1 -> spin, reorder hw->var
            t = _now()
            spins_hw = 2 * samples01 - 1
            resp_index = {q: j for j, q in enumerate(resp_order)}
            cols = [resp_index[q] for q in hw_for_var]
            spins_var = spins_hw[:, cols].T
            samples_raw = samples01
            _t_end_ms(iter_latency_ms, "convert_and_reorder_samples", t)

            # ---- Block 6: compute energies
            t = _now()
            gvec = ising_energy_from_spins(spins_var, h_full, J_full)
            current_best_objective = float(np.min(gvec))
            best_objective_history[iteration] = current_best_objective
            _t_end_ms(iter_latency_ms, "ising_energy_from_spins_and_best", t)

            # ---- Block 7: update best
            t = _now()
            if current_best_objective < best_objective_so_far:
                best_objective_so_far = current_best_objective
                best_idx = int(np.argmin(gvec))
                best_sample_overall = samples01[best_idx, cols]
            _t_end_ms(iter_latency_ms, "update_best_tracker", t)

            # ---- Block 8: relative entropy
            t = _now()
            relative_entropy[iteration] = -np.sum(prob_safe * np.log(prob_safe)) + beta * np.dot(gvec, prob_safe)
            _t_end_ms(iter_latency_ms, "compute_relative_entropy", t)

            # ---- Block 9: gradient + update
            if iteration > 0:
                t = _now()
                grad_re_h, grad_re_j = re_gradient(spins_var, prob_safe, gvec, beta, variable_interaction)
                _t_end_ms(iter_latency_ms, "re_gradient", t)

                t = _now()
                bqm = update_in_spin_then_back(
                    bqm,
                    grad_re_h, grad_re_j, step,
                    variable_interaction, var_order, mapping
                )
                _t_end_ms(iter_latency_ms, "update_in_spin_then_back", t)

            # ---- Block 10: early stopping check
            t = _now()
            early_stop = (partial_objective is not None) and (best_objective_so_far < partial_objective)
            _t_end_ms(iter_latency_ms, "early_stopping_check", t)

            # ---- finalize iteration metrics (NEW, additive)
            iter_total_ms = _ms(_now() - iter_t0)
            iter_blocks_sum_ms = float(sum(iter_latency_ms.values()))

            timing_metrics["iteration"] = iteration
            timing_metrics["latency_ms"] = iter_latency_ms
            timing_metrics["latency_blocks_sum_ms"] = iter_blocks_sum_ms
            timing_metrics["latency_total_measured_ms"] = iter_total_ms
            timing_metrics["latency_unaccounted_ms"] = max(0.0, iter_total_ms - iter_blocks_sum_ms)

            complete_timing_metrics.append(timing_metrics)

            for k, v in iter_latency_ms.items():
                per_iter_latency_sum_ms[k] = per_iter_latency_sum_ms.get(k, 0.0) + float(v)

            print(f"==> energies: {energies}")
            print(f"====> probabilities: {probabilities}")
            print(f"[iter {iteration}] measured_total_ms={iter_total_ms:.3f}, blocks_sum_ms={iter_blocks_sum_ms:.3f}, unaccounted_ms={timing_metrics['latency_unaccounted_ms']:.3f}")

            if early_stop:
                break

        number_of_calls = iteration + 1
        print("\n======= Finished Iteration =======\n")

        # -----------------------
        # Append a final summary dict (NEW, additive)
        # -----------------------
        total_ms = _ms(_now() - t_total0)
        summary = {
            "__tag__": "SEBREMforBQM_summary",
            "query_id": query_id,
            "custom_embedding": custom_embedding,
            "n_iterations_requested": int(n_iterations),
            "n_iterations_executed": int(number_of_calls),
            "overall_latency_ms": overall_latency_ms,
            "per_iter_latency_sum_ms": per_iter_latency_sum_ms,
            "SEBREM_total_ms": total_ms,

            # OPTIONAL but helpful for external parsing (doesn't break old code)
            "mapping_time_s": float(mapping_time_s),
            "embedding_stats": stats,
        }
        complete_timing_metrics.append(summary)

        if return_timing_metric:
            return (relative_entropy, samples_raw, best_objective_history,
                    best_sample_overall, best_objective_so_far, number_of_calls, complete_timing_metrics)
        return (relative_entropy, samples_raw, best_objective_history,
                best_sample_overall, best_objective_so_far, number_of_calls)


def basic_test():
    # Dummy QUBO for demonstration
    Q_example = np.random.rand(10, 10)
    Q_example = np.triu(Q_example) + np.triu(Q_example.T, 1)
    return Q_example


# ---------------------------
# Example Usage (to be adapted as needed)
# ---------------------------

def test_fqo_single(custom_embedding):
    Q_example = generate_maxcut_qubo(10)

    partial_objective = None
    beta = 1.0
    n_iterations = 10
    step = 0.01
    num_reads = 100

    # Run SEBREM optimization
    rel_ent, samples, best_obj_hist, best_sample, best_obj, num_calls = SEBREMforQUBO(
        Q_example, partial_objective, beta, n_iterations, step, num_reads, custom_embedding
    )

    print("Optimization finished:")
    print("Best Objective:", best_obj)
    print("Best Sample:", best_sample)
    print("Number of Calls:", num_calls)

    return rel_ent, samples, best_obj_hist, best_sample, best_obj, num_calls


def test_fqo_wrapper(
        bqm,
        partial_objective,
        iterations,
        custom_embedding,
        *,
        num_reads: int = 100,
        beta: float = 1.0,
        step: float = 0.01,
        return_history: bool = False,
        history_csv_path: Optional[str] = None,
):
    """
    Thin wrapper around SEBREMforBQM that optionally returns and/or persists
    the per-iteration best objective trajectory.

    Parameters
    ----------
    bqm : BinaryQuadraticModel
    partial_objective : float | None
    iterations : int
    custom_embedding : bool
    num_reads : int
    beta : float
    step : float
    return_history : bool
        If True, return a third value: best_obj_hist (np.ndarray).
    history_csv_path : str | None
        If provided, write a CSV at this path with columns: [iteration, best_obj].

    Returns
    -------
    (best_obj, best_sample) or (best_obj, best_sample, best_obj_hist)
    """
    # Run SEBREM optimization
    rel_ent, samples, best_obj_hist, best_sample, best_obj, num_calls = SEBREMforBQM(
        bqm, partial_objective, beta, iterations, step, num_reads, custom_embedding)

    # Optional: persist the trajectory as CSV
    if history_csv_path is not None:
        import csv, os
        os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "best_obj"])
            for t, val in enumerate(best_obj_hist):
                # cast to float for json/csv friendliness of numpy types
                w.writerow([t, float(val)])

    # Optional: return the trajectory to the caller
    if return_history:
        return best_obj, best_sample, best_obj_hist

    return best_obj, best_sample


def create_output_folder():
    la_time = datetime.now(pytz.timezone("America/Los_Angeles"))
    folder_name = la_time.strftime("Evaluations/%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_script_copy(output_dir):
    script_path = os.path.realpath(__file__)
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))


# ---------- Encapsulation ----------
def actual_query_blackbox(full_problem_path, custom_embedding: str, verbose=False,
                          iterations: Optional[int] = None):
    # --- simple timer helpers ---
    timings = {}
    t0_total = time.perf_counter()

    def _mark(key: str, t_start: float):
        timings[key] = time.perf_counter() - t_start

    # 1) make_query_id
    t = time.perf_counter()
    query_id = make_query_id(full_problem_path)
    _mark("make_query_id", t)

    # 2) load problem
    t = time.perf_counter()
    card, pred, pred_sel = get_join_ordering_problem(full_problem_path, generated_problems=True)
    _mark("get_join_ordering_problem", t)

    if verbose:
        print(f"Card: {card}")
        print(f"Pred: {pred}")
        print(f"Pred_sel: {pred_sel}")

    # 3) build QUBO/BQM
    t = time.perf_counter()
    bqm = generate_Fujitsu_QUBO_for_left_deep_trees(
        card, pred, pred_sel, 0.63, 2, penalty_scaling=2
    )
    _mark("generate_Fujitsu_QUBO_for_left_deep_trees", t)

    if verbose:
        print(bqm)

    # Configuration
    partial_objective = None
    beta = 1.0
    n_iterations = 10 if iterations is None else int(iterations)
    step = 0.01
    num_reads = 100

    # 4) run iterative solver
    t = time.perf_counter()
    rel_ent, samples, best_obj_hist, best_sample, best_obj, num_calls, complete_timing_metrics = SEBREMforBQM(
        bqm, partial_objective, beta, n_iterations, step, num_reads, custom_embedding,
        return_timing_metric=True, query_id=query_id, query_meta={"card": card, "pred_sel": pred_sel}
    )
    _mark("SEBREMforBQM", t)

    if verbose:
        print("Optimization finished:")
        print("Best Objective:", best_obj)
        print("Best Sample:", best_sample)
        print("Number of Calls:", num_calls)

        # Pretty print timing metrics
        print("\n" + "=" * 80)
        print("TIMING METRICS (from SEBREMforBQM)")
        print("=" * 80)
        for idx, metrics in enumerate(complete_timing_metrics):
            print(f"\n--- Iteration {idx} ---")
            print(json.dumps(metrics, indent=2, default=str))

    # 5) decode solution
    t = time.perf_counter()
    solution = read_out(best_sample, card, pred, pred_sel, {})
    join_order, db_cost, use_fallback_join_order = solution
    _mark("read_out", t)

    # total
    timings["TOTAL"] = time.perf_counter() - t0_total

    # --- print summary ---
    print("\n" + "=" * 80)
    print("LATENCY SUMMARY (seconds)")
    print("=" * 80)
    keys_in_order = [
        "make_query_id",
        "get_join_ordering_problem",
        "generate_Fujitsu_QUBO_for_left_deep_trees",
        "SEBREMforBQM",
        "read_out",
        "TOTAL",
    ]
    for k in keys_in_order:
        if k in timings:
            print(f"{k:45s}: {timings[k]:.6f}s")

    summary_entry = {
        "__tag__": "actual_query_blackbox_summary",
        "timings_s": {k: timings.get(k) for k in keys_in_order if k in timings},
    }
    if isinstance(complete_timing_metrics, list):
        complete_timing_metrics.append(summary_entry)
    elif isinstance(complete_timing_metrics, dict):
        complete_timing_metrics = [complete_timing_metrics, summary_entry]
    else:
        complete_timing_metrics = [summary_entry]

    return join_order, db_cost, complete_timing_metrics


def test_fqo_actual_query(custom_embedding, iterations: Optional[int] = None):
    full_problem_path = os.environ.get("QDBO_SAMPLE_PROBLEM", "")
    if not full_problem_path:
        full_problem_path = str(PROJECT_ROOT / "Problems/Original-Problems/benchmarks/job/q1")
    if not os.path.exists(full_problem_path):
        print(
            "[SKIP] QDBO_SAMPLE_PROBLEM not set and default sample problem path not found: "
            f"{full_problem_path}"
        )
        return
    join_order, db_cost, complete_timing_metrics = actual_query_blackbox(full_problem_path, custom_embedding, verbose=True, iterations=iterations)
    print(f"joinOrder: {join_order}")
    print(f"dbCost: {db_cost}")
    print(f"complete_timing_metrics: {complete_timing_metrics}")


if __name__ == "__main__":
    test_fqo_actual_query(custom_embedding="semanticV_greedy", iterations=1)
    # test_fqo_actual_query(custom_embedding="semanticV_random", iterations=1)
    test_fqo_actual_query(custom_embedding="semanticPRED_greedy", iterations=1)
    # test_fqo_actual_query(custom_embedding="semanticPRED_random", iterations=1)
    test_fqo_actual_query(custom_embedding="semanticSLACK_greedy", iterations=1)
    # test_fqo_actual_query(custom_embedding="semanticSLACK_random", iterations=1)

    # test_fqo_actual_query(custom_embedding="random", iterations=1)
    # test_fqo_actual_query(custom_embedding="random", iterations=10)
    #
    # test_fqo_actual_query(custom_embedding="seeded_neighbor_greedy")
    # test_fqo_actual_query(custom_embedding="seeded_neighbor_stochastic")
