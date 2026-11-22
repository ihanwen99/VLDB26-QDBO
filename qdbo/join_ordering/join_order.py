from __future__ import annotations

import time
from typing import Optional

import numpy as np

from qdbo.iterative_solver import solve_bqm_with_iterative_qa
from qdbo.output import make_query_id
from qdbo.join_ordering.qubo import (
    build_left_deep_bqm,
    load_join_ordering_problem,
)


def rank_relations_by_cost(cost_vector):
    return np.argsort(cost_vector)[::-1].tolist()


def compute_intermediate_costs(join_order, card, pred, pred_sel, card_dict):
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
            selectivity = compute_selectivity_for_new_relation(
                join_order, position, pred, pred_sel
            )
            cardinality = (
                previous_cardinality * card[join_order[position]] * selectivity
            )
            card_dict[prefix_key] = cardinality
        previous_cardinality = cardinality
        intermediate_costs.append(cardinality)

    return intermediate_costs


def compute_left_deep_cost(join_order, card, pred, pred_sel, card_dict):
    return sum(
        compute_intermediate_costs(
            join_order,
            card,
            pred,
            pred_sel,
            card_dict,
        )
    )


def compute_selectivity_for_new_relation(join_order, position, pred, pred_sel):
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


def build_connected_join_order(raw_join_order, cost_vector, num_relations, pred):
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


def decode_join_order(sample, card, pred, pred_sel, card_dict):
    weight_vector = np.arange(1, len(card) - 1)[len(card) - 3 :: -1]
    bitstring = sample[: len(card) * (len(card) - 2)]
    cost_vector = np.array(np.array_split(bitstring, len(card))).dot(weight_vector)

    raw_order = rank_relations_by_cost(cost_vector)
    raw_cost = compute_left_deep_cost(raw_order, card, pred, pred_sel, card_dict)
    raw_solution = [raw_order, int(raw_cost), False]

    connected_order = build_connected_join_order(
        raw_order, cost_vector, len(card), pred
    )
    connected_cost = compute_left_deep_cost(
        connected_order, card, pred, pred_sel, card_dict
    )
    connected_solution = [connected_order, int(connected_cost), True]
    return raw_solution if raw_cost < connected_cost else connected_solution


def compute_db_cost(join_order: list, cardinalities, selectivities):
    total_cost = 0

    num_relations = len(join_order)
    curr_rid = join_order[0]
    intermediate_cardinality = cardinalities[curr_rid]
    involved_relations = [curr_rid]

    for i in range(1, num_relations):
        right_rid = join_order[i]
        current_selectivity = 1
        for left_rid in involved_relations:
            if abs(selectivities[left_rid][right_rid] - 1) == 0.00000001:
                return "JOIN UNPROPER RELATIONS..."
            current_selectivity *= selectivities[left_rid][right_rid]
        intermediate_cardinality = (
            intermediate_cardinality * cardinalities[right_rid] * current_selectivity
        )
        total_cost += intermediate_cardinality
        involved_relations.append(right_rid)

    return total_cost


def solve_join_ordering_query(
    full_problem_path,
    custom_embedding: str,
    iterations: Optional[int] = None,
):
    timings = {}
    total_start = time.perf_counter()

    def record_elapsed_time(key: str, started: float) -> None:
        timings[key] = time.perf_counter() - started

    started = time.perf_counter()
    query_id = make_query_id(full_problem_path)
    record_elapsed_time("make_query_id", started)

    started = time.perf_counter()
    card, pred, pred_sel = load_join_ordering_problem(full_problem_path)
    record_elapsed_time("get_join_ordering_problem", started)

    started = time.perf_counter()
    bqm = build_left_deep_bqm(card, pred, pred_sel, 0.63, 2, penalty_scaling=2)
    record_elapsed_time("generate_Fujitsu_QUBO_for_left_deep_trees", started)

    started = time.perf_counter()
    (
        _relative_entropy,
        _samples,
        _best_objective_history,
        best_sample,
        _best_objective,
        _number_of_calls,
        timing_metrics,
    ) = solve_bqm_with_iterative_qa(
        bqm,
        partial_objective=None,
        beta=1.0,
        n_iterations=10 if iterations is None else int(iterations),
        step=0.01,
        num_reads=100,
        custom_embedding=custom_embedding,
        query_id=query_id,
        query_meta={"card": card, "pred_sel": pred_sel},
    )
    record_elapsed_time("iterative_qa_solve", started)

    started = time.perf_counter()
    join_order, database_cost, _used_fallback = decode_join_order(
        best_sample, card, pred, pred_sel, {}
    )
    record_elapsed_time("read_out", started)
    timings["TOTAL"] = time.perf_counter() - total_start

    ordered_keys = [
        "make_query_id",
        "get_join_ordering_problem",
        "generate_Fujitsu_QUBO_for_left_deep_trees",
        "iterative_qa_solve",
        "read_out",
        "TOTAL",
    ]
    summary = {
        "__tag__": "solve_join_ordering_query_summary",
        "timings_s": {key: timings[key] for key in ordered_keys},
    }
    timing_metrics.append(summary)

    return join_order, database_cost, timing_metrics
