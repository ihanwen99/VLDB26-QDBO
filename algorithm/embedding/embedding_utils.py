from typing import Dict, Hashable, Iterable, Tuple

import dimod


# ---------- utilities ----------

def _norm_edge(a: int, b: int) -> Tuple[int, int]:
    """Return a normalized (ordered) edge tuple (min, max)."""
    return (a, b) if a <= b else (b, a)


def _logical_degrees(bqm: dimod.BinaryQuadraticModel) -> Dict[Hashable, int]:
    """Compute degrees of logical variables in the BQM quadratic graph."""
    deg = {v: 0 for v in bqm.variables}
    for (u, v) in bqm.quadratic:
        deg[u] += 1
        deg[v] += 1
    return deg


def _physical_degrees(nodelist: Iterable[int],
                      edgelist: Iterable[Tuple[int, int]]) -> Dict[int, int]:
    """Compute degrees of physical qubits in the hardware graph."""
    deg = {q: 0 for q in nodelist}
    for a, b in edgelist:
        deg[a] += 1
        deg[b] += 1
    return deg


def _build_adj(nodelist: Iterable[int],
               edgelist: Iterable[Tuple[int, int]]) -> Dict[int, set]:
    """Build adjacency sets for the physical graph."""
    adj = {q: set() for q in nodelist}
    for a, b in edgelist:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _check_mapping(bqm, mapping):
    # Basic validation of injectivity and coverage
    if set(mapping.keys()) != set(bqm.variables):
        raise ValueError("Provided mapping does not cover all logical variables.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Provided mapping is not injective (duplicate qubits).")
