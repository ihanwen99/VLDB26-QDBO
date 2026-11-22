import numpy as np
from dwave.system import LeapHybridBQMSampler

from qdbo.join_ordering.qubo import (
    build_left_deep_bqm,
)
from qdbo.join_ordering.join_order import decode_join_order

__all__ = ["build_bqm", "solve_bqm", "decode_join_order"]


def build_bqm(card, pred, pred_sel):
    return build_left_deep_bqm(
        card,
        pred,
        pred_sel,
        0.63,
        2,
        penalty_scaling=2,
    )


def solve_bqm(bqm):
    solution = LeapHybridBQMSampler().sample(bqm)
    best_sample = solution.first.sample
    return np.array([int(best_sample[v]) for v in bqm.variables], dtype=int)
