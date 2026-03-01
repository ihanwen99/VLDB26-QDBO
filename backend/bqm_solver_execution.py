import numpy as np
from dwave.system import LeapHybridBQMSampler

from backend.ProblemGenerator import generate_Fujitsu_QUBO_for_left_deep_trees


def get_raw_join_order(cost_vector):  # Rank indices by descending cost (custom readout heuristic)
    # Example:
    #   cost_vector        [0 6 1 3 6]
    #   argsort ascending  [0 2 3 1 4]
    #   reversed           [4 1 3 2 0]  (from large to small)
    join_order = np.argsort(cost_vector).tolist()  # Indices sorted ascending by cost
    join_order.reverse()  # Reverse to descending order
    return join_order  # Return ranking list


def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=False):  # Compute per-step left-deep costs
    int_costs = []  # Accumulator for intermediate cardinalities/costs
    join_order = join_order.copy()  # Defensive copy
    if join_order[0] > join_order[1]:  # Enforce deterministic ordering on first two relations
        join_order[0], join_order[1] = join_order[1], join_order[0]  # Swap to ensure join_order[0] < join_order[1]
    prev_join_result = card[join_order[0]]  # Cardinality of the first relation
    for j in range(1, len(card) - 1):  # Iterate through join positions
        jo_hash = str(join_order[0:j + 1])  # Key for memoization of intermediate cardinality
        if jo_hash in card_dict:  # If cached
            int_card = card_dict[jo_hash]  # Retrieve cached intermediate
        else:  # Otherwise compute selectivity
            sel = get_selectivity_for_new_relation(join_order, j, pred, pred_sel)  # Combined selectivity with earlier relations
            int_card = prev_join_result * card[join_order[j]] * sel  # Estimated intermediate cardinality
            card_dict[jo_hash] = int_card  # Cache the value
        prev_join_result = int_card  # Update previous result
        int_costs.append(int_card)  # Record this step's cost
    if verbose:  # Optional printing
        print(int_costs)  # Show intermediate costs
    return int_costs  # Return list of intermediate costs


def get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict, verbose=False):  # Sum intermediate join costs
    total_costs = 0  # Accumulator for total cost
    int_costs = get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=verbose)  # Compute partial costs
    for cost in int_costs:  # Sum all intermediate costs
        total_costs = total_costs + cost  # Add intermediate cost
    return total_costs  # Return total cost


def get_selectivity_for_new_relation(join_order, j, pred, pred_sel):  # Compute selectivity when adding relation j
    sel = 1  # Start with neutral selectivity
    new_relation = join_order[j]  # Relation to add
    for i in range(j):  # Check predicates between the new relation and all earlier ones
        relation = join_order[i]  # Earlier relation
        if (relation, new_relation) in pred:  # If predicate stored as (i,j)
            sel = sel * pred_sel[pred.index((relation, new_relation))]  # Multiply by corresponding selectivity
        elif (new_relation, relation) in pred:  # Or stored reversed
            sel = sel * pred_sel[pred.index((new_relation, relation))]  # Multiply by corresponding selectivity
    return sel  # Return combined selectivity


def postprocess_join_order(raw_join_order, cost_vector, num_relations, pred):  # Connectivity-aware postprocessing of join order
    join_order = [raw_join_order[0]]  # Start from the top-ranked relation
    while len(join_order) < num_relations:  # Until we pick all relations
        applicable_predicates = [pred_tuple for t in join_order for pred_tuple in pred if t in pred_tuple]  # Predicates touching current set
        neighborhood_indices = [t for t in set(sum(applicable_predicates, ())) if t not in join_order]  # Neighbors not yet included
        if len(neighborhood_indices) != 0:  # If there are neighbors
            best_neighbor_relation = neighborhood_indices[np.argmax(cost_vector[neighborhood_indices])]  # Pick the neighbor with max cost
            join_order.append(best_neighbor_relation)  # Append neighbor
        else:  # No neighbors, fall back to global best not yet used
            global_indices = [x for x in raw_join_order if x not in join_order]  # Remaining relations by global order
            best_global_relation = global_indices[np.argmax(cost_vector[global_indices])]  # Best among remaining
            join_order.append(best_global_relation)  # Append
    return join_order  # Return refined order


def read_out(sample, card, pred, pred_sel, card_dict):
    """
    Convert the annealer output to actual join order
    :param sample: [0, 1, 0, 1, ...]
    :return: [join_order, int(cost), whether_is_fallback_join_order ]
    """
    weight_vector = np.arange(1, len(card) - 1)  # Create weights [1,2,...,n-2]
    weight_vector = weight_vector[len(card) - 3::-1]  # Reverse and trim to match partial rows

    num_relations = len(card)  # Number of relations
    bitstring = sample[:len(card) * (len(card) - 2)]  # sample here <== bitstrings[i] in DA code
    partial_bitstrings = np.array_split(bitstring, len(card))  # Split into rows per relation
    cost_vector = np.array(partial_bitstrings).dot(weight_vector)  # Weighted sum per relation

    raw_join_order = get_raw_join_order(cost_vector)  # Rank by cost descending
    costs = get_costs_for_leftdeep_tree(raw_join_order, card, pred, pred_sel, card_dict)  # Compute cost of raw order
    solution = [raw_join_order, int(costs), False]  # Prepare primary solution

    fallback_join_order = postprocess_join_order(raw_join_order, cost_vector, num_relations, pred)  # Connectivity-aware fallback
    fallback_costs = get_costs_for_leftdeep_tree(fallback_join_order, card, pred, pred_sel, card_dict)  # Cost of fallback order
    fallback_solution = [fallback_join_order, int(fallback_costs), True]  # Prepare fallback solution
    if costs < fallback_costs:  # Choose better of the two
        return solution  # Return primary if better
    return fallback_solution  # Else return fallback


def build_bqm(card, pred, pred_sel):
    penalty_scaling = 2
    approximation_precisions = (4, 2, [0.63])
    ap, num_decimal_pos, thres = approximation_precisions
    bqm = generate_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=penalty_scaling)
    return bqm


def solve_bqm(bqm):
    solution = LeapHybridBQMSampler().sample(bqm)
    best_dict = solution.first.sample
    var_order = list(bqm.variables)
    best_vec = np.array([int(best_dict[v]) for v in var_order], dtype=int)
    return best_vec
