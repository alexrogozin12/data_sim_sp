import sys
sys.path.append("../")

import numpy as np
from oracles.saddle import create_robust_linear_oracle, OracleLinearComb


def gen_oracles_for_sliding(n_one: int, d: int, num_summands: int, noise: float,
                            regcoef_x: float, regcoef_y: float, seed=0):
    np.random.seed(seed)
    A_one = np.random.randn(n_one, d)
    A = np.tile(A_one.T, num_summands).T
    A[n_one:] += noise * np.random.randn(n_one * (num_summands - 1), d)

    b_one = np.random.randn(n_one)
    b = np.tile(b_one, num_summands)
    b[n_one:] += noise * np.random.randn(n_one * (num_summands - 1))

    oracle_sum = create_robust_linear_oracle(A, b, num_summands * regcoef_x,
                                             num_summands * regcoef_y, normed=False)
    oracle_sum = OracleLinearComb([oracle_sum], [1. / num_summands])
    oracle_phi = create_robust_linear_oracle(A_one, b_one, regcoef_x, regcoef_y, normed=False)
    oracle_g = OracleLinearComb([oracle_sum, oracle_phi], [1., -1.])

    return oracle_sum, oracle_phi, oracle_g


