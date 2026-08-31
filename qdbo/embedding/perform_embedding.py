import random
import time
from typing import Dict, Hashable, Tuple, Optional, List

import dimod
from dwave.system import DWaveSampler

from qdbo.embedding.embedding_utils import _norm_edge, _check_mapping
from qdbo.embedding.mapping_strategy import (
    _build_mapping_random,
    _build_mapping_greedy,
    _build_mapping_seeded_neighbor_greedy,
    _build_mapping_semantic_group_then_fill,
)

from typing import Optional, Tuple
from qdbo.embedding.cpp_embed import embed_no_chains_drop_missing_cpp


def _parse_semantic_strategy(strategy: str) -> Optional[Tuple[str, str]]:
    """
    Parse semantic embedding strategy strings of the form:
      semantic<GROUP>_<REMAINDER>

    Recognised GROUP aliases:
      V / join / j         -> "V"     (join-position vars in join-ordering BQMs)
      PRED / p             -> "PRED"  (predicate vars in join-ordering BQMs)
      INDEX / ix / idx     -> "INDEX" (selection vars in knapsack-style BQMs,
                                         e.g. index advising)
      SLACK / s            -> "SLACK" (budget/encoding slack vars)

    Recognised REMAINDER values: "random" or "greedy". Defaults to "random"
    when omitted.

    Returns:
      (priority_key, remainder) where:
        priority_key in {"V", "PRED", "INDEX", "SLACK"}
        remainder    in {"random", "greedy"}
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
    elif group_part in ("index", "ix", "idx"):
        # Generic selection-style group for non-join-ordering BQMs.
        key = "INDEX"
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
        index_split: Optional[Dict[str, List[Hashable]]] = None,
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
        index_split: Optional[Dict[str, List[Hashable]]] = None,
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
