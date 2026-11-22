from __future__ import annotations
import math
from typing import Dict, List, Mapping, Optional, Tuple

Qubo = Dict[Tuple[int, int], float]


def canonicalize_qubo(Q: Mapping[Tuple[int, int], float]) -> Qubo:
    out = {}
    for (i, j), v in Q.items():
        a, b = (i, j) if i <= j else (j, i)
        out[(a, b)] = out.get((a, b), 0.0) + float(v)
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def build_index_selection_qubo(
    profits: List[int | float],
    weights: List[int | float],
    budget: int | float,
    *,
    penalty: Optional[float] = None,
    pairwise_interactions: Optional[Mapping[str, float]] = None,
):
    if len(profits) != len(weights):
        raise ValueError("profits and weights length mismatch")
    if not profits:
        raise ValueError("at least one candidate is required")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if any(float(w) <= 0 for w in weights):
        raise ValueError("weights must be positive")
    n = len(profits)
    W = int(math.ceil(float(budget)))
    m = max(1, math.ceil(math.log2(W + 1)))
    p = [float(v) for v in profits]
    w = [int(math.ceil(float(x))) for x in weights]
    if penalty is None:
        pos = sum(max(0.0, x) for x in p) + sum(
            max(0.0, float(x)) for x in (pairwise_interactions or {}).values()
        )
        penalty = max(1.0, 2.0 * pos + 1.0)
    coeffs = w + [2**j for j in range(m)]
    Q = {}
    for k, a in enumerate(coeffs):
        Q[(k, k)] = float(penalty) * (a * a - 2 * W * a) - (p[k] if k < n else 0.0)
    for i in range(n + m):
        for j in range(i + 1, n + m):
            Q[(i, j)] = 2.0 * float(penalty) * coeffs[i] * coeffs[j]
    for key, g in (pairwise_interactions or {}).items():
        i, j = [int(x) for x in key.split(",")]
        a, b = (i, j) if i < j else (j, i)
        Q[(a, b)] = Q.get((a, b), 0.0) - float(g)
    meta = {
        "n_index_vars": n,
        "n_slack_vars": m,
        "n_total_vars": n + m,
        "budget": W,
        "penalty": float(penalty),
        "variable_order": [f"x_{i}" for i in range(n)] + [f"s_{j}" for j in range(m)],
        "formula": "min -sum_i v_i*x_i -sum_ij g_ij*x_i*x_j + P*(sum_i w_i*x_i + sum_j 2^j*s_j - W)^2",
    }
    return canonicalize_qubo(Q), meta
