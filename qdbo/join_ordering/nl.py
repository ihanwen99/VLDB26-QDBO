import math

from dwave.optimization.model import Model

from qdbo.output import return_result_and_elapsed_time


@return_result_and_elapsed_time
def build_left_deep_nl_model(cardinalities: list, selectivities: list) -> Model:
    model = Model()

    num_relations = len(cardinalities)
    join_order = model.list(num_relations)

    cardinalities = model.constant(cardinalities)
    selectivities = model.constant(selectivities)

    total_cost = model.constant(0)
    curr_rid = join_order[0]
    intermediate_cardinality = cardinalities[curr_rid]
    involved_relations = [curr_rid]

    for i in range(1, num_relations):
        right_rid = join_order[i]
        current_selectivity = model.constant(1)
        for left_rid in involved_relations:
            if selectivities[left_rid][right_rid] == 1:
                adjusted_selectivity = math.inf
            else:
                adjusted_selectivity = selectivities[left_rid][right_rid]
            current_selectivity *= adjusted_selectivity
        intermediate_cardinality = (
            intermediate_cardinality * cardinalities[right_rid] * current_selectivity
        )
        total_cost += intermediate_cardinality
        involved_relations.append(right_rid)

    model.minimize(total_cost)
    model.lock()
    return model
