from math import ceil, log

import dimod
from dwave.system import LeapHybridCQMSampler


def solve_cqm(cqm: dimod.ConstrainedQuadraticModel, time_limit: int):
    sampler = LeapHybridCQMSampler()
    solution = sampler.sample_cqm(
        cqm, label="Join Order Optimization", time_limit=time_limit
    )
    solution = solution.filter(lambda row: row.is_feasible)

    best_sample = solution.first.sample

    all_relations = set()
    for key, val in best_sample.items():
        if key.startswith("roj_") and val == 1.0:
            _, relation, _, _ = key.split("_")
            all_relations.add(relation)

    join_pairs = {}
    for key, val in best_sample.items():
        if key.startswith("pred_") and val == 1.0:
            parts = key.split("_")
            rel1, rel2 = parts[1], parts[2]
            join_index = int(parts[-1])
            join_pairs[(rel1, rel2)] = join_index

    sorted_pairs = sorted(join_pairs.items(), key=lambda x: x[1])

    final_join_order = []
    for (rel1, rel2), join_index in sorted_pairs:
        if rel1 not in final_join_order:
            final_join_order.append(rel1)
        if rel2 not in final_join_order:
            final_join_order.append(rel2)

    for relation in all_relations:
        if relation not in final_join_order:
            final_join_order.append(relation)
    num_relations = sum(
        1 for key in best_sample if key.startswith("roj_") and key.endswith("_join_1")
    )
    for relation in range(num_relations):
        if str(relation) not in final_join_order:
            final_join_order.append(str(relation))
    total_qpu_time = solution.info["qpu_access_time"] / 1e3

    return final_join_order, total_qpu_time


def build_cqm(
    cardinalities_content, selectivities_content
) -> dimod.ConstrainedQuadraticModel:
    relations = [i for i in range(len(cardinalities_content))]

    log_cardinalities = {
        relation: round(log(card), 2)
        for relation, card in zip(relations, cardinalities_content)
    }

    log_selectivities = {}
    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            if selectivities_content[i][j] != 1.0:
                log_selectivities[(relations[i], relations[j])] = round(
                    log(selectivities_content[i][j]), 2
                )

    predicates = list(log_selectivities.keys())

    cqm = dimod.ConstrainedQuadraticModel()

    pred_vars = {
        (pred, j): dimod.Binary(f"pred_{pred[0]}_{pred[1]}_join_{j}")
        for j in range(1, len(relations))
        for pred in predicates
    }
    roj_vars = {
        (r, j): dimod.Binary(f"roj_{r}_join_{j}")
        for j in range(1, len(relations) + 1)
        for r in relations
    }
    num_joins = len(relations) - 2

    for r in relations:
        for j in range(1, num_joins):
            cqm.add_constraint(
                roj_vars[(r, j)] - roj_vars[(r, j + 1)] <= 0, label=f"cont_use_{r}_{j}"
            )

    for j in range(1, num_joins + 1):
        cqm.add_constraint(
            sum(roj_vars[(r, j)] for r in relations) == j + 1,
            label=f"incr_operands_{j}",
        )

    for pred, j in pred_vars.keys():
        relation1, relation2 = pred
        cqm.add_constraint(
            (pred_vars[(pred, j)] - roj_vars[(relation1, j)]) <= 0,
            label=f"pred_app_{pred}_{j}_rel1",
        )
        cqm.add_constraint(
            (pred_vars[(pred, j)] - roj_vars[(relation2, j)]) <= 0,
            label=f"pred_app_{pred}_{j}_rel2",
        )

    max_theta_t = 100
    N = ceil(log(max_theta_t, 2))

    stij_vars = {
        (j, i): dimod.Binary(f"st_{j}_{i}")
        for j in range(1, len(relations) + 1)
        for i in range(N)
    }

    buffertj_expressions = {}
    for j in range(1, len(relations) + 1):
        buffertj_expressions[j] = dimod.quicksum(
            2**i * stij_vars[(j, i)] for i in range(N)
        )

    for j in range(1, len(relations) + 1):
        log_int_card_j_expr = dimod.quicksum(
            log_cardinalities[r] * roj_vars[(r, j)] for r in relations
        )

        log_int_card_j_expr += dimod.quicksum(
            log_selectivities.get((r1, r2), 0) * pred_vars.get(((r1, r2), j), 0)
            for r1 in relations
            for r2 in relations
            if r1 != r2 and ((r1, r2), j) in pred_vars
        )

        buffertj_expr = buffertj_expressions[j]

        penalty_expr = (buffertj_expr - log_int_card_j_expr) ** 2

        cqm.set_objective(cqm.objective + penalty_expr * max_theta_t)
    return cqm
