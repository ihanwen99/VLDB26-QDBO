import ctypes
import os
from typing import Dict, Hashable, List, Optional, Tuple

import dimod
import numpy as np

LIB_NAME = "cpp_embed_lib.so"
LIB_PATH = os.path.join(os.path.dirname(__file__), LIB_NAME)


class CppEmbedError(RuntimeError):
    pass


def _parse_semantic_strategy(strategy: str) -> Optional[Tuple[str, str]]:
    s = (strategy or "").strip()
    sl = s.lower()
    if not sl.startswith("semantic"):
        return None

    rest = s[len("semantic"):].lstrip("_")
    if not rest:
        return None

    if "_" in rest:
        group_part, remainder = rest.split("_", 1)
        remainder = remainder.strip().lower()
    else:
        group_part = rest
        remainder = "random"

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


def _load_lib() -> ctypes.CDLL:
    if not os.path.exists(LIB_PATH):
        raise CppEmbedError(
            f"C++ library not found at {LIB_PATH}. Build it first with: "
            f"clang++ -O3 -std=c++17 -shared -fPIC -o {LIB_PATH} {os.path.join(os.path.dirname(__file__), 'cpp_embed.cpp')}"
        )
    lib = ctypes.CDLL(LIB_PATH)

    lib.embed_no_chains_drop_missing_cpp.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.embed_no_chains_drop_missing_cpp.restype = ctypes.c_int

    lib.embed_last_error.argtypes = []
    lib.embed_last_error.restype = ctypes.c_char_p

    return lib


def embed_no_chains_drop_missing_cpp(
    bqm: dimod.BinaryQuadraticModel,
    sampler,
    strategy: str,
    *,
    rng_seed: int = 0,
    index_split: Optional[Dict[str, List[Hashable]]] = None,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[Hashable, int], Dict[str, int]]:
    nodelist = list(sampler.nodelist)
    edgelist = list(sampler.edgelist)

    variables = list(bqm.variables)
    n_vars = len(variables)

    var_to_idx = {v: i for i, v in enumerate(variables)}

    quad_u = []
    quad_v = []
    quad_w = []
    for (u, v), w in bqm.quadratic.items():
        quad_u.append(var_to_idx[u])
        quad_v.append(var_to_idx[v])
        quad_w.append(float(w))

    linear_bias = [float(bqm.linear[v]) for v in variables]

    quad_u_arr = np.asarray(quad_u, dtype=np.int32)
    quad_v_arr = np.asarray(quad_v, dtype=np.int32)
    quad_w_arr = np.asarray(quad_w, dtype=np.float64)
    linear_arr = np.asarray(linear_bias, dtype=np.float64)

    node_arr = np.asarray(nodelist, dtype=np.int32)
    edge_u_arr = np.asarray([a for a, _ in edgelist], dtype=np.int32)
    edge_v_arr = np.asarray([b for _, b in edgelist], dtype=np.int32)

    if strategy.lower().startswith("semantic"):
        if index_split is None:
            raise ValueError("semantic strategy requires index_split")
        parsed = _parse_semantic_strategy(strategy)
        if parsed is None:
            raise ValueError(f"Invalid semantic strategy: {strategy!r}")
        priority_key, _ = parsed
        priority_vars = index_split.get(priority_key, [])
        priority_idx = [var_to_idx[v] for v in priority_vars if v in var_to_idx]
    else:
        priority_idx = []

    priority_arr = np.asarray(priority_idx, dtype=np.int32)

    out_mapping = np.empty(n_vars, dtype=np.int32)
    out_linear_qubit = np.empty(n_vars, dtype=np.int32)
    out_linear_bias = np.empty(n_vars, dtype=np.float64)

    out_keep_u = np.empty(len(quad_u_arr), dtype=np.int32)
    out_keep_v = np.empty(len(quad_u_arr), dtype=np.int32)
    out_keep_w = np.empty(len(quad_u_arr), dtype=np.float64)

    kept_count = ctypes.c_int(0)
    mapping_ms = ctypes.c_double(0.0)
    embed_ms = ctypes.c_double(0.0)

    lib = _load_lib()
    rc = lib.embed_no_chains_drop_missing_cpp(
        ctypes.c_int(n_vars),
        quad_u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        quad_v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        quad_w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(quad_u_arr)),
        linear_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        node_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(node_arr)),
        edge_u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        edge_v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(edge_u_arr)),
        strategy.encode("utf-8"),
        priority_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) if len(priority_arr) else None,
        ctypes.c_int(len(priority_arr)),
        ctypes.c_uint(rng_seed),
        out_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_linear_qubit.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_linear_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_keep_u.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_keep_v.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_keep_w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(kept_count),
        ctypes.byref(mapping_ms),
        ctypes.byref(embed_ms),
    )

    if rc != 0:
        err = lib.embed_last_error().decode("utf-8", errors="ignore")
        raise CppEmbedError(f"C++ embed failed (code {rc}): {err}")

    kept = kept_count.value
    quad_dict = {}
    for i in range(kept):
        quad_dict[(int(out_keep_u[i]), int(out_keep_v[i]))] = float(out_keep_w[i])

    linear_dict = {}
    for i in range(n_vars):
        q = int(out_linear_qubit[i])
        linear_dict[q] = float(out_linear_bias[i])

    target_bqm = dimod.BinaryQuadraticModel(linear_dict, quad_dict, bqm.offset, bqm.vartype)

    mapping = {variables[i]: int(out_mapping[i]) for i in range(n_vars)}

    stats = {
        "edge_remaining_rate": f"{len(quad_dict) / max(1, len(quad_u_arr)) * 100:.2f}%",
        "variables": len(linear_dict),
        "kept_edges": len(quad_dict),
        "dropped": len(quad_u_arr) - len(quad_dict),
        "original": len(quad_u_arr),
        "find_mapping_time_ms": f"{mapping_ms.value:.2f}",
        "embed_bqm_time_ms": f"{embed_ms.value:.2f}",
    }

    return target_bqm, mapping, stats
