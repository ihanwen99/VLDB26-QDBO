import random
from typing import Dict, Hashable, Tuple, Optional, List

import dimod

from algorithm.embedding.embedding_utils import _logical_degrees, _physical_degrees, _build_adj


# ---------- mapping strategies (No chain always) ----------

def _build_mapping_random(logical_vars: List[Hashable],
                          nodelist: List[int],
                          rng: Optional[random.Random] = None) -> Dict[Hashable, int]:
    """Random injective mapping: variable -> unique qubit."""
    if rng is None:
        rng = random.Random()
    qs = list(nodelist)
    rng.shuffle(qs)
    return {v: qs[i] for i, v in enumerate(logical_vars)}


def _build_mapping_greedy(bqm: dimod.BinaryQuadraticModel,
                          nodelist: List[int],
                          edgelist: List[Tuple[int, int]]) -> Dict[Hashable, int]:
    """
    Degree-aware greedy mapping:
        For each variable, find the best qubit to maximize the number of coupler.
    TODO: This can be improved by using a weighted score plus algorithm improvement

    Workflow:
    - Order logical vars by logical degree (desc).
    - Order physical qubits by physical degree (desc).
    - Place each var on the unused qubit that maximizes the number of kept
      couplers to already-placed neighbors.
    """
    logical_vars: List[Hashable] = list(bqm.variables)

    log_deg = _logical_degrees(bqm)
    phys_deg = _physical_degrees(nodelist, edgelist)

    vars_order = sorted(logical_vars, key=lambda v: log_deg[v], reverse=True)
    qubits_order = sorted(nodelist, key=lambda q: phys_deg[q], reverse=True)

    adj = _build_adj(nodelist, edgelist)
    mapping: Dict[Hashable, int] = {}
    used: set = set()

    # score: how many neighbor couplers would be realized if v -> q
    def score_place(v, q) -> int:
        s = 0
        for u in mapping:
            if (u, v) in bqm.quadratic or (v, u) in bqm.quadratic:
                if mapping[u] in adj[q]:
                    s += 1
        return s

    for v in vars_order:
        best_q = None
        best_s = -1
        for q in qubits_order:
            if q in used:
                continue
            s = score_place(v, q)
            if s > best_s:
                best_s = s
                best_q = q
        if best_q is None:
            # Fallback: pick any unused qubit
            for q in qubits_order:
                if q not in used:
                    best_q = q
                    break
        mapping[v] = best_q
        used.add(best_q)

    return mapping


def _build_mapping_seeded_neighbor_greedy(
        bqm,
        nodelist: List[int],
        edgelist: List[Tuple[int, int]],
        *,
        starting_qubit: Optional[int] = None,
) -> Dict[Hashable, int]:
    """
    Build an injective mapping (variable -> unique qubit, no chain) using a
    seeded-neighborhood greedy strategy: (Matlab -> GreedyEmbedding Version)

    1) Seed variable:
       Pick the logical variable whose total absolute coupling sum is maximal:
           seed_var = argmax_v sum_u |J_{u,v}|.
    2) Seed qubit:
       If `starting_qubit` is provided, use it; otherwise pick the physical qubit
       with the highest degree in the hardware graph.
    3) Iterative placement:
       Maintain the candidate qubit pool as the union of neighbors of all
       already-assigned qubits (excluding used ones). For each candidate qubit q,
       find the unassigned variable v that maximizes the preserved absolute
       coupling to already-placed variables, i.e.,
           score(v -> q) = sum_{u in placed} |J_{u,v}|  if  q adjacent to mapping[u].
       Choose the (v, q) pair that achieves the best column score. If multiple
       candidate columns tie, break ties by picking the q whose index has the
       smallest absolute difference to the most recently assigned qubit index
       (i.e., abs(q - last_q)).
    4) Continue until all variables are assigned.

    Returns:
        mapping: Dict[logical_var, physical_qubit]
    """
    logical_vars: List[Hashable] = list(bqm.variables)

    # Precompute |J| neighbor map and per-variable absolute coupling totals
    nbrs = {v: {} for v in logical_vars}  # v -> {u: |J_{v,u}|}
    abs_weight_sum = {v: 0.0 for v in logical_vars}
    for (u, v), J in bqm.quadratic.items():
        w = abs(float(J))
        nbrs[u][v] = w
        nbrs[v][u] = w
        abs_weight_sum[u] += w
        abs_weight_sum[v] += w

    # Seed variable: max total |J|
    seed_var = max(logical_vars, key=lambda v: abs_weight_sum[v])

    # Physical graph helpers
    adj = _build_adj(nodelist, edgelist)
    phys_deg = _physical_degrees(nodelist, edgelist)

    # Seed qubit: user-provided or highest-degree qubit
    if starting_qubit is None:
        starting_qubit = max(nodelist, key=lambda q: phys_deg[q])
    elif starting_qubit not in adj:
        raise ValueError("starting_qubit is not present in the provided nodelist.")

    # State
    mapping: Dict[Hashable, int] = {}
    variables_left = set(logical_vars)
    variables_left.remove(seed_var)
    variables_assigned = [seed_var]  # order of placement

    used_qubits = set([starting_qubit])
    qubits_assigned = [starting_qubit]  # order of placement

    # Candidate qubits = union of neighbors of all assigned qubits (minus used)
    def current_neighbor_pool() -> List[int]:
        pool = set()
        for q in qubits_assigned:
            pool.update(adj[q])
        pool.difference_update(used_qubits)
        return sorted(pool)

    def distance_to_last(q: int) -> int:
        last_q = qubits_assigned[-1]
        return abs(q - last_q)

    # Commit seed placement
    mapping[seed_var] = starting_qubit

    # Main loop
    neighbors = current_neighbor_pool()
    while variables_left:
        if not neighbors:
            # Fallback: allow any unused qubit
            neighbors = sorted([q for q in nodelist if q not in used_qubits])
            if not neighbors:
                raise RuntimeError("No available qubits left to complete the mapping.")

        best_col = None
        best_col_score = float("-inf")
        best_col_row_var = None

        for q in neighbors:
            col_best_score = float("-inf")
            col_best_var = None

            # Try each remaining variable
            for v in variables_left:
                # Score = sum |J_{u,v}| over already-placed u whose mapped qubit is adjacent to q
                s = 0.0
                for u in variables_assigned:
                    if mapping[u] in adj[q]:
                        s += nbrs[v].get(u, 0.0)

                if s > col_best_score:
                    col_best_score = s
                    col_best_var = v

            # Track best column with tie-breaking on distance to last qubit
            if col_best_score > best_col_score:
                best_col_score = col_best_score
                best_col = q
                best_col_row_var = col_best_var
            elif col_best_score == best_col_score and best_col is not None:
                if distance_to_last(q) < distance_to_last(best_col):
                    best_col = q
                    best_col_row_var = col_best_var

        # Commit chosen (var, qubit)
        new_q = best_col
        new_v = best_col_row_var if best_col_row_var is not None else next(iter(variables_left))

        mapping[new_v] = new_q
        variables_left.remove(new_v)
        variables_assigned.append(new_v)
        used_qubits.add(new_q)
        qubits_assigned.append(new_q)

        neighbors = current_neighbor_pool()

    return mapping


# TODO: whether this can be merged with the previous function
# Hanwen Comment on Jan 14, This function has been ignored
def _build_mapping_seeded_neighbor_stochastic(
        bqm,
        nodelist: List[int],
        edgelist: List[Tuple[int, int]],
        *,
        starting_qubit: Optional[int] = None,
        rng: Optional[random.Random] = None,
) -> Dict[Hashable, int]:
    """
    Build an injective mapping (variable -> unique qubit, chain=1) using a
    seeded neighborhood *stochastic-greedy* strategy: (Matlab: StochasticGreedyEmbedding)

    Algorithm:
      1) Seed variable:
         Choose the logical variable with maximal total absolute coupling:
             seed_var = argmax_v sum_u |J_{u,v}|.
      2) Seed qubit:
         If `starting_qubit` is provided, use it; otherwise pick the physical
         qubit with the highest degree.
      3) Iterative placement:
         Maintain the candidate qubit set as the union of neighbors of all
         already-assigned qubits (excluding used ones). For each candidate
         qubit q (a column), find the unassigned variable v that maximizes:
             score(v -> q) = sum_{u in placed} |J_{u,v}|
                              where q is adjacent to mapping[u].
         Let S be the set of columns achieving the maximal column score.
         Randomly choose one column from S, then place the corresponding
         best-scoring variable on that column's qubit.
         If the neighbor pool is empty, fall back to any unused qubit.
      4) Repeat until all variables are assigned.

    Returns:
      mapping: Dict[logical_var, physical_qubit]
    """
    if rng is None:
        rng = random.Random()

    # Logical variables and absolute-weight neighbor map
    logical_vars: List[Hashable] = list(bqm.variables)

    nbrs = {v: {} for v in logical_vars}  # v -> {u: |J_{v,u}|}
    abs_weight_sum = {v: 0.0 for v in logical_vars}
    for (u, v), J in bqm.quadratic.items():
        w = abs(float(J))
        nbrs[u][v] = w
        nbrs[v][u] = w
        abs_weight_sum[u] += w
        abs_weight_sum[v] += w

    # Seed variable: maximal total |J|
    seed_var = max(logical_vars, key=lambda v: abs_weight_sum[v])

    # Physical graph helpers
    adj = _build_adj(nodelist, edgelist)
    phys_deg = _physical_degrees(nodelist, edgelist)

    # Seed qubit: user-provided or highest-degree qubit
    if starting_qubit is None:
        starting_qubit = max(nodelist, key=lambda q: phys_deg[q])
    elif starting_qubit not in adj:
        raise ValueError("starting_qubit is not present in nodelist.")

    # State
    mapping: Dict[Hashable, int] = {}
    variables_left = set(logical_vars)
    variables_left.remove(seed_var)
    variables_assigned = [seed_var]  # order of placement

    used_qubits = set([starting_qubit])
    qubits_assigned = [starting_qubit]  # order of placement

    def neighbor_pool() -> List[int]:
        """Union of neighbors of all assigned qubits minus used qubits."""
        pool = set()
        for q in qubits_assigned:
            pool.update(adj[q])
        pool.difference_update(used_qubits)
        return sorted(pool)

    # Commit seed
    mapping[seed_var] = starting_qubit
    candidates = neighbor_pool()

    # Main loop
    while variables_left:
        if not candidates:
            # Fallback: allow any unused qubit
            candidates = sorted([q for q in nodelist if q not in used_qubits])
            if not candidates:
                raise RuntimeError("No available qubits left to complete the mapping.")

        # For each candidate column (qubit), compute its best row (variable) score
        col_best_scores = {}  # q -> (best_score, best_var)
        best_column_score = float("-inf")

        for q in candidates:
            best_s = float("-inf")
            best_v = None
            for v in variables_left:
                # score = sum_{u in placed} |J_{u,v}| if q adjacent to mapping[u]
                s = 0.0
                for u in variables_assigned:
                    if mapping[u] in adj[q]:
                        s += nbrs[v].get(u, 0.0)
                if s > best_s:
                    best_s = s
                    best_v = v
            col_best_scores[q] = (best_s, best_v)
            if best_s > best_column_score:
                best_column_score = best_s

        # Among columns with maximal score, choose one at random
        top_columns = [q for q, (s, _) in col_best_scores.items() if s == best_column_score]
        chosen_q = rng.choice(top_columns)
        chosen_v = col_best_scores[chosen_q][1] if col_best_scores[chosen_q][1] is not None else next(
            iter(variables_left))

        # Commit the choice
        mapping[chosen_v] = chosen_q
        variables_left.remove(chosen_v)
        variables_assigned.append(chosen_v)
        used_qubits.add(chosen_q)
        qubits_assigned.append(chosen_q)

        # Refresh candidate pool
        candidates = neighbor_pool()

    return mapping


# =========================
# NEW: semantic priority then fill
# =========================

def _extend_mapping_random(
        remaining_vars: List[Hashable],
        remaining_qubits: List[int],
        *,
        rng: Optional[random.Random] = None,
) -> Dict[Hashable, int]:
    if rng is None:
        rng = random.Random()
    qs = list(remaining_qubits)
    rng.shuffle(qs)
    if len(qs) < len(remaining_vars):
        raise ValueError("Not enough remaining qubits to finish random mapping.")
    return {v: qs[i] for i, v in enumerate(remaining_vars)}


def _extend_mapping_greedy(
        bqm: dimod.BinaryQuadraticModel,
        remaining_vars: List[Hashable],
        nodelist: List[int],
        edgelist: List[Tuple[int, int]],
        *,
        seed_mapping: Dict[Hashable, int],
        used_qubits: set,
) -> Dict[Hashable, int]:
    """
    Continue greedy mapping starting from an existing partial mapping.
    """
    log_deg = _logical_degrees(bqm)
    phys_deg = _physical_degrees(nodelist, edgelist)

    vars_order = sorted(remaining_vars, key=lambda v: log_deg.get(v, 0), reverse=True)
    qubits_order = sorted(nodelist, key=lambda q: phys_deg[q], reverse=True)

    adj = _build_adj(nodelist, edgelist)

    mapping = dict(seed_mapping)

    def score_place(v, q) -> int:
        s = 0
        for u in mapping:
            if (u, v) in bqm.quadratic or (v, u) in bqm.quadratic:
                if mapping[u] in adj[q]:
                    s += 1
        return s

    for v in vars_order:
        best_q = None
        best_s = -1
        for q in qubits_order:
            if q in used_qubits:
                continue
            s = score_place(v, q)
            if s > best_s:
                best_s = s
                best_q = q
        if best_q is None:
            for q in qubits_order:
                if q not in used_qubits:
                    best_q = q
                    break
        mapping[v] = best_q
        used_qubits.add(best_q)

    return mapping


def _build_mapping_semantic_group_then_fill(
        bqm: dimod.BinaryQuadraticModel,
        nodelist: List[int],
        edgelist: List[Tuple[int, int]],
        *,
        priority_vars: List[Hashable],
        remainder: str,  # "random" | "greedy"
        rng: Optional[random.Random] = None,
) -> Dict[Hashable, int]:
    """
    First map `priority_vars` onto the highest-connectivity (top-degree) physical qubits.
    Within this top-qubit pool, use a greedy-like `score_place` heuristic to choose placements.
    Then fill the remaining variables using the chosen `remainder` method ("random" or "greedy").

    Optimizations (scheme 2):
      - score_place iterates only over logical neighbors of v (O(deg(v)) instead of O(|mapping|)).
      - remainder greedy uses an adaptive top-K physical-qubit scan (K scales with #vars), avoiding
        scanning the full hardware nodelist when it is very large.
    """
    if rng is None:
        rng = random.Random()

    logical_vars: List[Hashable] = list(bqm.variables)
    logical_set = set(logical_vars)

    # Deduplicate priority_vars while keeping order
    seen: set = set()
    prio: List[Hashable] = []
    for v in priority_vars:
        if v in logical_set and v not in seen:
            prio.append(v)
            seen.add(v)

    remaining_vars: List[Hashable] = [v for v in logical_vars if v not in seen]

    if len(nodelist) < len(logical_vars):
        raise ValueError("Not enough physical qubits for an embedding.")

    # Precompute once
    log_deg = _logical_degrees(bqm)
    phys_deg = _physical_degrees(nodelist, edgelist)
    adj = _build_adj(nodelist, edgelist)

    # Build logical adjacency: v -> list of logical neighbors
    logical_adj: Dict[Hashable, List[Hashable]] = {v: [] for v in logical_vars}
    for (u, v) in bqm.quadratic:
        logical_adj[u].append(v)
        logical_adj[v].append(u)

    # Physical qubits ordered by degree (desc)
    qubits_order: List[int] = sorted(nodelist, key=lambda q: phys_deg[q], reverse=True)

    # -------------------------
    # Phase A: priority vars
    # -------------------------
    top_qubits: List[int] = qubits_order[:len(prio)]
    used_qubits: set = set()

    prio_order: List[Hashable] = sorted(prio, key=lambda v: log_deg.get(v, 0), reverse=True)
    mapping: Dict[Hashable, int] = {}

    def score_place(v, q) -> int:
        # Only check logical neighbors of v
        s = 0
        for u in logical_adj.get(v, []):
            if u in mapping and mapping[u] in adj[q]:
                s += 1
        return s

    for v in prio_order:
        best_q = None
        best_s = -1

        for q in top_qubits:
            if q in used_qubits:
                continue
            s = score_place(v, q)
            if s > best_s:
                best_s = s
                best_q = q

        if best_q is None:
            # Fallback: pick any unused qubit within top_qubits
            for q in top_qubits:
                if q not in used_qubits:
                    best_q = q
                    break

        mapping[v] = best_q
        used_qubits.add(best_q)

    # -------------------------
    # Phase B: fill remainder
    # -------------------------
    if not remaining_vars:
        return mapping

    if remainder == "random":
        remaining_qubits = [q for q in qubits_order if q not in used_qubits]
        mapping.update(_extend_mapping_random(remaining_vars, remaining_qubits, rng=rng))
        return mapping

    if remainder != "greedy":
        raise ValueError(f"Unknown remainder method: {remainder!r} (expected 'random' or 'greedy').")

    # Adaptive top-K scan parameters (scheme 2)
    total_vars = len(prio) + len(remaining_vars)
    # K scales with the problem size; cap by hardware size.
    # - For small problems, 128 is enough.
    # - For larger problems, scan ~8x variables.
    K = min(len(nodelist), max(128, 8 * max(1, total_vars)))

    # Candidate qubits to scan for the remainder stage
    cand_qubits = [q for q in qubits_order[:K] if q not in used_qubits]

    # If too few candidates remain (rare if K is reasonable), fall back to full list
    if len(cand_qubits) < len(remaining_vars):
        cand_qubits = [q for q in qubits_order if q not in used_qubits]

    # Greedy fill, continuing from the current mapping, but scanning only cand_qubits
    # and using neighbor-only scoring.
    rem_order = sorted(remaining_vars, key=lambda v: log_deg.get(v, 0), reverse=True)
    for v in rem_order:
        best_q = None
        best_s = -1

        for q in cand_qubits:
            # cand_qubits is already filtered by used_qubits, but keep it safe
            if q in used_qubits:
                continue
            s = score_place(v, q)
            if s > best_s:
                best_s = s
                best_q = q

        if best_q is None:
            # Fallback: any unused qubit on hardware
            for q in qubits_order:
                if q not in used_qubits:
                    best_q = q
                    break

        mapping[v] = best_q
        used_qubits.add(best_q)

    return mapping
