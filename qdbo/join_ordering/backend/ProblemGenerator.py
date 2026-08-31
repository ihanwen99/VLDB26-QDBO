import json
import math
import os

import dimod
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

from qdbo.join_ordering.backend.jo_utils import parse_selectivities

def load_from_path(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file) as file:
            data = json.load(file)
            return data


def format_loaded_pred(pred):
    form_pred = []
    for p in pred:
        form_pred.append(tuple(p))
    return form_pred


def get_join_ordering_problem(problem_path, generated_problems=True):
    if generated_problems:
        card = load_from_path(problem_path + '/cardinalities.json')
        sel = load_from_path(problem_path + '/selectivities.json')
        pred, pred_sel = parse_selectivities(sel)
        return card, pred, pred_sel
    else:
        card = load_from_path(problem_path + "/card.txt")
        pred = format_loaded_pred(load_from_path(problem_path + "/pred.txt"))
        pred_sel = load_from_path(problem_path + "/pred_sel.txt")
        return card, pred, pred_sel


def get_log_values(coeff, num_decimal_pos, use_rounding=True):
    if use_rounding:
        log_coeff = np.around(np.log10(coeff), num_decimal_pos)
    else:
        log_coeff = np.log10(coeff)
    return log_coeff.tolist()


def get_binary_slack_coeff(num_slack, precision):
    slack_coeff = []
    for i in range(num_slack):
        slack_coeff.append(pow(2, i))
    slack_coeff = [x * precision for x in slack_coeff]
    return slack_coeff


def get_binary_slack_variables_for_bound(model, bound, num_decimal_pos):
    precision = pow(0.1, num_decimal_pos)
    num_slack = int(math.floor(np.log2(bound / precision))) + 1
    slack = model.binary_var_list(num_slack)
    return slack, get_binary_slack_coeff(num_slack, precision)


def generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, log_thres, num_decimal_pos, penalty_scaling=1,
                                           minimum_penalty_weight=20):
    card = get_log_values(card, num_decimal_pos)
    pred_sel = get_log_values(pred_sel, num_decimal_pos)

    model = Model('docplex_model')

    num_relations = len(card)
    num_pred = len(pred_sel)
    num_joins = len(card) - 2

    v = model.binary_var_matrix(num_relations, num_joins)

    b = np.arange(2, num_joins + 2).tolist()

    # Incentivise that the right number of relations is present for every join (i.e., 2 for join 1, 3 for join 2, ...)
    H_A = model.sum((b[j] - model.sum(v[(t, j)] for t in range(num_relations))) ** 2 for j in range(num_joins))

    # Incentivise that, once joined, a relation is always part of subosequent joins
    H_B = model.sum(
        model.sum(v[(t, j - 1)] - v[(t, j - 1)] * v[(t, j)] for j in range(1, num_joins)) for t in range(num_relations))

    # Incentivise that a predicate is only applicable for a join if both associated relations are present
    pred_vars = model.binary_var_matrix(num_pred, num_joins)
    H_pred_a = model.sum(
        model.sum(pred_vars[(p, j)] - pred_vars[(p, j)] * v[(pred[p][0], j)] for p in range(num_pred)) for j in
        range(num_joins))
    H_pred_b = model.sum(
        model.sum(pred_vars[(p, j)] - pred_vars[(p, j)] * v[(pred[p][1], j)] for p in range(num_pred)) for j in
        range(num_joins))
    H_pred = H_pred_a + H_pred_b

    H_cost = 0
    penalty_weight = 0

    # Intermediate cardinality calculation
    for j in range(num_joins):
        penalty_weight = penalty_weight + 1
        slack, slack_coeff = get_binary_slack_variables_for_bound(model, log_thres, num_decimal_pos)
        H_thres = (model.sum(slack_coeff[s] * slack[s] for s in range(len(slack))) - (
                model.sum(card[t] * v[(t, j)] for t in range(num_relations)) + model.sum(
            pred_sel[p] * pred_vars[(p, j)] for p in range(num_pred)))) ** 2
        H_cost = H_cost + H_thres

    penalty_weight = penalty_weight * penalty_scaling
    if penalty_weight < minimum_penalty_weight:
        penalty_weight = minimum_penalty_weight

    H_valid = H_A + H_B + H_pred

    H = penalty_weight * H_valid + H_cost

    model.minimize(H)

    qubo = from_docplex_mp(model)

    return qubo, penalty_weight


def generate_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=1):
    ibmq_qubo, penalty_weight = generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=penalty_scaling)
    dwave_qubo = dimod.as_bqm(ibmq_qubo.objective.linear.to_array(), ibmq_qubo.objective.quadratic.to_array(), ibmq_qubo.objective.constant, dimod.BINARY)
    return dwave_qubo
