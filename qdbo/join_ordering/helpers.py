"""Join-ordering helpers and the iterative QDBO entry point."""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np

from qdbo.core.iterative_solver import SEBREMforBQM
from qdbo.backend.utils import make_query_id
from qdbo.join_ordering.backend.ProblemGenerator import (
    generate_Fujitsu_QUBO_for_left_deep_trees,
    get_join_ordering_problem,
)


def get_raw_join_order(cost_vector):
    """Rank relation indices by descending readout cost."""
    return np.argsort(cost_vector)[::-1].tolist()


def get_intermediate_costs_for_join_order(
    join_order, card, pred, pred_sel, card_dict, verbose=False
):
    """Return estimated intermediate cardinalities for a left-deep plan."""
    join_order = join_order.copy()
    if join_order[0] > join_order[1]:
        join_order[0], join_order[1] = join_order[1], join_order[0]

    intermediate_costs = []
    previous_cardinality = card[join_order[0]]
    for position in range(1, len(card) - 1):
        prefix_key = str(join_order[: position + 1])
        if prefix_key in card_dict:
            cardinality = card_dict[prefix_key]
        else:
            selectivity = get_selectivity_for_new_relation(
                join_order, position, pred, pred_sel
            )
            cardinality = (
                previous_cardinality * card[join_order[position]] * selectivity
            )
            card_dict[prefix_key] = cardinality
        previous_cardinality = cardinality
        intermediate_costs.append(cardinality)

    if verbose:
        print(intermediate_costs)
    return intermediate_costs


def get_costs_for_leftdeep_tree(
    join_order, card, pred, pred_sel, card_dict, verbose=False
):
    """Return the sum of intermediate cardinalities for a left-deep plan."""
    return sum(
        get_intermediate_costs_for_join_order(
            join_order,
            card,
            pred,
            pred_sel,
            card_dict,
            verbose=verbose,
        )
    )


def get_selectivity_for_new_relation(join_order, position, pred, pred_sel):
    """Combine predicates connecting a new relation to the current plan."""
    selectivity = 1
    new_relation = join_order[position]
    for earlier_position in range(position):
        relation = join_order[earlier_position]
        forward = (relation, new_relation)
        reverse = (new_relation, relation)
        if forward in pred:
            selectivity *= pred_sel[pred.index(forward)]
        elif reverse in pred:
            selectivity *= pred_sel[pred.index(reverse)]
    return selectivity


def postprocess_join_order(raw_join_order, cost_vector, num_relations, pred):
    """Prefer the highest-ranked connected relation at each join step."""
    join_order = [raw_join_order[0]]
    while len(join_order) < num_relations:
        touching = [
            edge for relation in join_order for edge in pred if relation in edge
        ]
        neighbors = [
            relation
            for relation in set(sum(touching, ()))
            if relation not in join_order
        ]
        if neighbors:
            next_relation = neighbors[np.argmax(cost_vector[neighbors])]
        else:
            remaining = [
                relation for relation in raw_join_order if relation not in join_order
            ]
            next_relation = remaining[np.argmax(cost_vector[remaining])]
        join_order.append(next_relation)
    return join_order


def read_out(sample, card, pred, pred_sel, card_dict):
    """Decode annealer bits into the lower-cost of raw and connected plans."""
    weight_vector = np.arange(1, len(card) - 1)[len(card) - 3 :: -1]
    bitstring = sample[: len(card) * (len(card) - 2)]
    cost_vector = np.array(np.array_split(bitstring, len(card))).dot(weight_vector)

    raw_order = get_raw_join_order(cost_vector)
    raw_cost = get_costs_for_leftdeep_tree(raw_order, card, pred, pred_sel, card_dict)
    raw_solution = [raw_order, int(raw_cost), False]

    connected_order = postprocess_join_order(raw_order, cost_vector, len(card), pred)
    connected_cost = get_costs_for_leftdeep_tree(
        connected_order, card, pred, pred_sel, card_dict
    )
    connected_solution = [connected_order, int(connected_cost), True]
    return raw_solution if raw_cost < connected_cost else connected_solution


def actual_query_blackbox(
    full_problem_path,
    custom_embedding: str,
    verbose: bool = False,
    iterations: Optional[int] = None,
):
    """Build, solve, decode, and time one join-ordering problem."""
    timings = {}
    total_start = time.perf_counter()

    def mark(key: str, started: float) -> None:
        timings[key] = time.perf_counter() - started

    started = time.perf_counter()
    query_id = make_query_id(full_problem_path)
    mark("make_query_id", started)

    started = time.perf_counter()
    card, pred, pred_sel = get_join_ordering_problem(
        full_problem_path, generated_problems=True
    )
    mark("get_join_ordering_problem", started)

    started = time.perf_counter()
    bqm = generate_Fujitsu_QUBO_for_left_deep_trees(
        card, pred, pred_sel, 0.63, 2, penalty_scaling=2
    )
    mark("generate_Fujitsu_QUBO_for_left_deep_trees", started)

    started = time.perf_counter()
    (
        _relative_entropy,
        _samples,
        _best_objective_history,
        best_sample,
        best_objective,
        number_of_calls,
        timing_metrics,
    ) = SEBREMforBQM(
        bqm,
        partial_objective=None,
        beta=1.0,
        n_iterations=10 if iterations is None else int(iterations),
        step=0.01,
        num_reads=100,
        custom_embedding=custom_embedding,
        return_timing_metric=True,
        query_id=query_id,
        query_meta={"card": card, "pred_sel": pred_sel},
        verbose=verbose,
    )
    mark("SEBREMforBQM", started)

    started = time.perf_counter()
    join_order, database_cost, _used_fallback = read_out(
        best_sample, card, pred, pred_sel, {}
    )
    mark("read_out", started)
    timings["TOTAL"] = time.perf_counter() - total_start

    ordered_keys = [
        "make_query_id",
        "get_join_ordering_problem",
        "generate_Fujitsu_QUBO_for_left_deep_trees",
        "SEBREMforBQM",
        "read_out",
        "TOTAL",
    ]
    summary = {
        "__tag__": "actual_query_blackbox_summary",
        "timings_s": {key: timings[key] for key in ordered_keys},
    }
    if isinstance(timing_metrics, list):
        timing_metrics.append(summary)
    elif isinstance(timing_metrics, dict):
        timing_metrics = [timing_metrics, summary]
    else:
        timing_metrics = [summary]

    if verbose:
        print(f"Card: {card}")
        print(f"Pred: {pred}")
        print(f"Pred_sel: {pred_sel}")
        print(f"Best objective: {best_objective}")
        print(f"Best sample: {best_sample}")
        print(f"Number of calls: {number_of_calls}")
        print(json.dumps(summary, indent=2))

    return join_order, database_cost, timing_metrics
