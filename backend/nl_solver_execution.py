import math

from dwave.optimization.model import Model
from dwave.system import LeapHybridNLSampler

from backend.utils import measure_time_return


@measure_time_return
def nl_query_optimization(
        cardinalities: list,
        selectivities: list) -> Model:
    """
    Using database cost model to define the cost of NL-solver
    Returns:
    """
    model = Model()

    num_relations = len(cardinalities)
    # Init the variable as join order, which comes from the #relations
    # [2,1,0] -> Represents join R3 and R2 first, and then with R1
    join_order = model.list(num_relations)

    cardinalities = model.constant(cardinalities)
    selectivities = model.constant(selectivities)

    total_cost = model.constant(0)
    curr_rid = join_order[0]  # Current relation id
    intermediate_cardinality = cardinalities[curr_rid]  # The beginning cardinality
    involved_relations = [curr_rid]

    for i in range(1, num_relations):
        right_rid = join_order[i]  # Right side - relation id (incoming join)
        current_selectivity = model.constant(1)
        for left_rid in involved_relations:
            if selectivities[left_rid][right_rid] == 1:
                adjusted_selectivity = math.inf
            else:
                adjusted_selectivity = selectivities[left_rid][right_rid]
            current_selectivity *= adjusted_selectivity
            # current_selectivity *= selectivities[left_rid][right_rid]
        intermediate_cardinality = intermediate_cardinality * cardinalities[right_rid] * current_selectivity
        total_cost += intermediate_cardinality

    # Overall optimization target - total_cost
    model.minimize(total_cost)
    model.lock()
    return model



@measure_time_return
def nl_initialize_sampler():
    return LeapHybridNLSampler()


@measure_time_return
def nl_solver_sample(sampler, nl_model, time_limit, label="Quantum4DB NL_Solver"):
    return sampler.sample(nl_model, label=label, time_limit=time_limit)


@measure_time_return
def nl_fetch_join_order(nl_model):
    return next(nl_model.iter_decisions()).state(0).astype(int)
