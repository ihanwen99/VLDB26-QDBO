import random
import time
from typing import Dict, Hashable, Tuple, Optional, List

import dimod
import numpy as np
from dwave.system import DWaveSampler

from algorithm.embedding.embedding_utils import _norm_edge, _check_mapping
from algorithm.embedding.mapping_strategy import (
    _build_mapping_random,
    _build_mapping_greedy,
    _build_mapping_seeded_neighbor_greedy,
    _build_mapping_seeded_neighbor_stochastic,
    _build_mapping_semantic_group_then_fill,  # NEW
)
from backend.generate_qubo import generate_maxcut_qubo

from typing import Optional, Tuple
from algorithm.embedding.cpp_embed import embed_no_chains_drop_missing_cpp


def _parse_semantic_strategy(strategy: str) -> Optional[Tuple[str, str]]:
    """
    Parse ONLY:
      semanticV_random / semanticV_greedy
      semanticPRED_random / semanticPRED_greedy
      semanticSLACK_random / semanticSLACK_greedy

    Also accepts (optional):
      semanticV / semanticPRED / semanticSLACK  -> default remainder="random"

    Returns:
      (priority_key, remainder) where:
        priority_key in {"V","PRED","SLACK"}
        remainder in {"random","greedy"}
    """
    s = (strategy or "").strip()
    sl = s.lower()
    if not sl.startswith("semantic"):
        return None

    # remove prefix "semantic" and optional leading underscore
    rest = s[len("semantic"):].lstrip("_")
    if not rest:
        return None

    # split group and remainder using ONLY "_"
    if "_" in rest:
        group_part, remainder = rest.split("_", 1)
        remainder = remainder.strip().lower()
    else:
        group_part = rest
        remainder = "random"  # default

    group_part = group_part.strip().lower()

    if group_part in ("v", "join", "j"):
        key = "V"
    elif group_part in ("pred", "p"):
        key = "PRED"
    elif group_part in ("slack", "s"):
        key = "SLACK"
    else:
        return None

    if remainder not in ("random", "greedy"):
        remainder = "random"

    return key, remainder


def _build_mapping_injective(
        bqm: dimod.BinaryQuadraticModel,
        nodelist: List[int],
        edgelist: List[Tuple[int, int]],
        strategy: str,
        *,
        rng: Optional[random.Random] = None,
        index_split: Optional[Dict[str, List[Hashable]]] = None,  # NEW
) -> Dict[Hashable, int]:
    """
    Build an injective mapping (variable -> unique qubit) with no qubit chain.
    This function routes to specific strategies.
    """
    logical_vars: List[Hashable] = list(bqm.variables)

    if len(nodelist) < len(logical_vars):
        raise ValueError("Not enough physical qubits for an embedding.")

    parsed = _parse_semantic_strategy(strategy)
    if parsed is not None:
        if index_split is None:
            raise ValueError(
                f"{strategy!r} requires index_split with keys 'V','PRED','SLACK'. "
                f"Got index_split=None."
            )
        priority_key, remainder = parsed
        priority_vars = index_split.get(priority_key, [])
        return _build_mapping_semantic_group_then_fill(
            bqm, nodelist, edgelist,
            priority_vars=priority_vars,
            remainder=remainder,
            rng=rng,
        )

    if strategy == "random":
        return _build_mapping_random(logical_vars, nodelist, rng)
    elif strategy == "greedy":
        return _build_mapping_greedy(bqm, nodelist, edgelist)
    elif strategy == "seeded_neighbor_greedy":
        return _build_mapping_seeded_neighbor_greedy(bqm, nodelist, edgelist)
    elif strategy == "seeded_neighbor_stochastic":
        return _build_mapping_seeded_neighbor_stochastic(bqm, nodelist, edgelist, rng=rng)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Use 'greedy'/'random'/'seeded_neighbor_*' "
            f"or semanticV/semanticPRED/semanticSLACK with _random/_greedy."
        )


# ---------- main embedding (no qubit chain, drop missing couplers) ----------

def embed_no_chains_drop_missing(
        bqm: dimod.BinaryQuadraticModel,
        sampler: DWaveSampler,
        strategy: str,
        *,
        mapping: Optional[Dict[Hashable, int]] = None,
        rng: Optional[random.Random] = None,
        index_split: Optional[Dict[str, List[Hashable]]] = None,  # NEW
) -> Tuple[dimod.BinaryQuadraticModel, Dict[Hashable, int], Dict[str, int]]:
    """
    No qubit chain embedding with "keep if coupler exists, otherwise drop".
    """
    nodelist: List[int] = sampler.nodelist
    edgelist: List[Tuple[int, int]] = sampler.edgelist

    # 1) Build / validate mapping
    if mapping is None:
        t_mapping_start = time.perf_counter_ns()
        mapping = _build_mapping_injective(
            bqm, nodelist, edgelist,
            strategy=strategy,
            rng=rng,
            index_split=index_split,
        )
        t_mapping_end = time.perf_counter_ns()
        find_mapping_ms = (t_mapping_end - t_mapping_start) / 1e6
    else:
        find_mapping_ms = 0.0
        _check_mapping(bqm, mapping)

    # 2) Relabel variables to qubit IDs
    t_embed_start = time.perf_counter_ns()
    target_bqm = bqm.relabel_variables(mapping, inplace=False)

    # 3) Drop missing couplers
    phys_edge_set = {_norm_edge(a, b) for (a, b) in edgelist}
    to_drop = [(p, q) for (p, q) in target_bqm.quadratic if _norm_edge(p, q) not in phys_edge_set]
    for (p, q) in to_drop:
        target_bqm.remove_interaction(p, q)
    t_embed_end = time.perf_counter_ns()
    embed_bqm_ms = (t_embed_end - t_embed_start) / 1e6

    stats = {
        "edge_remaining_rate": f"{len(target_bqm.quadratic) / len(bqm.quadratic) * 100:.2f}%",
        "variables": len(target_bqm.variables),
        "kept_edges": len(target_bqm.quadratic),
        "dropped": len(to_drop),
        "original": len(bqm.quadratic),
        "find_mapping_time_ms": f"{find_mapping_ms:.2f}",
        "embed_bqm_time_ms": f"{embed_bqm_ms:.2f}",
    }
    return target_bqm, mapping, stats


def test_perform_embedding(bqm, sampler, test_sampling):
    # ---- Fake index_split for semantic testing ----
    # Split variables by index: first 30% -> V, middle 40% -> PRED, last 30% -> SLACK
    vars_list = list(bqm.variables)  # preserves bqm's variable order
    N = len(vars_list)
    nV = int(round(N * 0.30))
    nP = int(round(N * 0.40))
    # Make sure we don't exceed N due to rounding
    if nV + nP > N:
        nP = max(0, N - nV)
    nS = N - nV - nP

    fake_index_split = {
        "V": vars_list[:nV],
        "PRED": vars_list[nV:nV + nP],
        "SLACK": vars_list[nV + nP:],
    }
    print("[test] fake_index_split sizes:",
          {k: len(v) for k, v in fake_index_split.items()},
          "N=", N)

    strategies = [
        "random",
        "greedy",
        "seeded_neighbor_greedy",
        "seeded_neighbor_stochastic",

        # ---- Semantic test strategies (require index_split) ----
        "semanticV_random",
        "semanticV_greedy",
        "semanticPRED_random",
        "semanticPRED_greedy",
        "semanticSLACK_random",
        "semanticSLACK_greedy",
    ]

    for strategy in strategies:
        # Only semantic* strategies need index_split
        if strategy.lower().startswith("semantic"):
            target_bqm, mapping, stats = embed_no_chains_drop_missing(
                bqm, sampler, strategy=strategy, index_split=fake_index_split
            )
        else:
            target_bqm, mapping, stats = embed_no_chains_drop_missing(
                bqm, sampler, strategy=strategy
            )

        print(f"\n<-------------strategy: {strategy}------------->")
        print("mapping :", mapping)
        print("stats   :", stats)
        print("same vartype:", target_bqm.vartype == bqm.vartype)
        print("same offset :", target_bqm.offset == bqm.offset)
        print("source bqm  :", bqm)
        print("target bqm  :", target_bqm)

        if test_sampling:
            # for num_reads in [10, 100, 500, 1000, 2000, 3000]:
            for num_reads in [10]:
                t_sampling_start = time.perf_counter_ns()
                response = sampler.sample(target_bqm, num_reads=num_reads)
                samples_raw = np.array([list(sample.values()) for sample in response.samples()])
                energies = response.record.energy
                counts = response.record.num_occurrences
                probabilities = counts / np.sum(counts)
                t_sampling_end = time.perf_counter_ns()
                print(
                    f"num_reads: {num_reads}\t, "
                    f"Sampling Time (ms): {((t_sampling_end - t_sampling_start) / 1e6):.2f}"
                )


# ---------- Example ----------
if __name__ == '__main__':
    TEST_SAMPLING = True
    TEST_SIMPLE_QUBO = False

    if TEST_SIMPLE_QUBO:
        Q = {('a', 'b'): -1.2, ('b', 'c'): +0.7, ('a', 'c'): -0.4}
    else:
        Q = generate_maxcut_qubo(100)

    bqm = dimod.BQM.from_qubo(Q)
    sampler = DWaveSampler()

    test_perform_embedding(bqm, sampler, TEST_SAMPLING)
