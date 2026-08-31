from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from .models import SolverResult
from .qubo import (
    Qubo,
    build_index_selection_qubo,
    decode_selected,
    normalize_qubo,
    qubo_from_jsonable,
    selection_profit,
    selection_weight,
)


def _result(name, ids, profits, weights, budget, pairwise=None, energy=None, raw=None):
    ids=sorted(set(int(i) for i in ids)); bits=[1 if i in ids else 0 for i in range(len(profits))]
    w=selection_weight(ids,weights); p=selection_profit(ids,profits,pairwise)
    return SolverResult(name, ids, bits, int(round(w)), int(round(p)), energy, w<=budget, raw or {}, float(p))


def solve_exact_knapsack(profits: List[int|float], weights: List[int|float], budget: int|float, pairwise_interactions: Optional[Mapping[str,float]]=None) -> SolverResult:
    """Exact 0/1 knapsack via DP. O(n * W) time and memory.
    pairwise_interactions are NOT included in DP search but added to final profit reporting if present."""
    n=len(profits)
    import math as _m
    W=int(_m.ceil(float(budget)))
    w=[int(_m.ceil(float(x))) for x in weights]
    p_=[float(v) for v in profits]
    INF_NEG=float("-inf")
    dp=[[INF_NEG]*(W+1) for _ in range(n+1)]
    dp[0]=[0.0]*(W+1)
    for i in range(1,n+1):
        wi=w[i-1]; pi=p_[i-1]
        row=dp[i]; prev=dp[i-1]
        for b in range(W+1):
            best_skip=prev[b]
            if wi<=b and prev[b-wi]!=INF_NEG:
                best_take=prev[b-wi]+pi
                row[b]=best_take if best_take>best_skip else best_skip
            else:
                row[b]=best_skip
    best_ids=[]
    b=W
    for i in range(n,0,-1):
        if dp[i][b]!=dp[i-1][b]:
            best_ids.append(i-1)
            b-=w[i-1]
    best_ids.sort()
    return _result("exact-knapsack", best_ids, profits, weights, budget, pairwise_interactions, raw={"note":"DP O(nW), n="+str(n)+", W="+str(W)})


def solve_greedy_roi(profits, weights, budget) -> SolverResult:
    order=sorted(range(len(profits)), key=lambda i:(float(profits[i])/float(weights[i]), float(profits[i])), reverse=True); ids=[]; w=0.0
    for i in order:
        if w+float(weights[i]) <= float(budget): ids.append(i); w+=float(weights[i])
    return _result("greedy-roi", ids, profits, weights, budget, raw={"order":order})


def _best_feasible_from_sampleset(sampleset, n_index_vars, profits, weights, budget, pairwise=None, solver_name="qubo-solver"):
    try: n_total=max(int(v) for v in sampleset.variables)+1
    except Exception: n_total=n_index_vars
    best_key=None; best=None
    for row in sampleset.data(["sample","energy","num_occurrences"]):
        bits=[int(row.sample.get(i,0)) for i in range(n_total)]; ids=decode_selected(bits,n_index_vars); w=selection_weight(ids,weights); p=selection_profit(ids,profits,pairwise)
        if w<=budget:
            key=(p,-w,-float(row.energy), int(row.num_occurrences))
            if best_key is None or key>best_key: best_key=key; best=(ids,float(row.energy))
    if best is None:
        first=sampleset.first; bits=[int(first.sample.get(i,0)) for i in range(n_total)]; ids=decode_selected(bits,n_index_vars)
        while selection_weight(ids,weights)>budget and ids:
            worst=min(ids, key=lambda i:(float(profits[i])/float(weights[i]), -float(weights[i]))); ids.remove(worst)
        return _result(solver_name, ids, profits, weights, budget, pairwise, float(first.energy), {"fallback":"lowest_energy_repaired","sampleset_info":dict(sampleset.info)})
    ids,energy=best; return _result(solver_name, ids, profits, weights, budget, pairwise, energy, {"sampleset_info":dict(sampleset.info)})


def solve_qubo_with_dwave_hybrid(Q: Qubo, n_index_vars:int, profits, weights, budget, *, time_limit:Optional[int]=None, label="qdbo-index-selection", pairwise_interactions=None) -> SolverResult:
    try:
        import dimod
        try:
            from dwave.system import LeapHybridBQMSampler as _HybridSampler
        except Exception:
            from dwave.system import LeapHybridSampler as _HybridSampler
    except Exception as exc: raise RuntimeError("dwave-system required; install dwave-ocean-sdk and configure credentials") from exc
    kwargs={"label":label}
    if time_limit is not None: kwargs["time_limit"]=time_limit
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    return _best_feasible_from_sampleset(_HybridSampler().sample(bqm, **kwargs), n_index_vars, profits, weights, budget, pairwise_interactions, "dwave-hybrid-bqm")


def solve_problem_dict(
    problem: Mapping[str, Any],
    *,
    solver: str,
    qubo_doc: Optional[Mapping[str, Any]] = None,
    penalty: Optional[float] = None,
    normalize: Optional[float] = None,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    label: str = "qdbo-index-selection",
    chain_strength: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve a JSON-style index-selection problem with the requested backend."""

    profits = list(problem["profits"])
    weights = list(problem["weights"])
    budget = problem["max_weight"]
    pairwise = problem.get("pairwise_interactions") or None

    if solver == "exact-knapsack":
        return solve_exact_knapsack(profits, weights, budget, pairwise).to_dict()
    if solver == "greedy":
        return solve_greedy_roi(profits, weights, budget).to_dict()

    if qubo_doc:
        Q = qubo_from_jsonable(qubo_doc["Q"])
        meta = dict(qubo_doc.get("meta", {}))
    else:
        Q, meta = build_index_selection_qubo(
            profits,
            weights,
            budget,
            penalty=penalty,
            pairwise_interactions=pairwise,
        )
    if normalize is not None:
        Q, scale = normalize_qubo(Q, normalize)
        meta["normalization_factor_at_solve_time"] = scale
    n_index_vars = int(meta.get("n_index_vars", len(profits)))

    if solver == "dwave-hybrid":
        return solve_qubo_with_dwave_hybrid(
            Q,
            n_index_vars,
            profits,
            weights,
            budget,
            time_limit=time_limit,
            label=label,
            pairwise_interactions=pairwise,
        ).to_dict()
    raise ValueError(f"unknown solver: {solver}")
