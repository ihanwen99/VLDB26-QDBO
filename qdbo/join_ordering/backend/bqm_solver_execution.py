import numpy as np
from dwave.system import LeapHybridBQMSampler

from qdbo.join_ordering.backend.ProblemGenerator import (
    generate_Fujitsu_QUBO_for_left_deep_trees,
)
from qdbo.join_ordering.helpers import read_out

__all__ = ["build_bqm", "solve_bqm", "read_out"]


def build_bqm(card, pred, pred_sel):
    """Build the left-deep join-ordering BQM used by the paper pipeline."""
    return generate_Fujitsu_QUBO_for_left_deep_trees(
        card,
        pred,
        pred_sel,
        0.63,
        2,
        penalty_scaling=2,
    )


def solve_bqm(bqm):
    """Solve a BQM through Leap Hybrid and return bits in BQM variable order."""
    solution = LeapHybridBQMSampler().sample(bqm)
    best_sample = solution.first.sample
    return np.array([int(best_sample[v]) for v in bqm.variables], dtype=int)
