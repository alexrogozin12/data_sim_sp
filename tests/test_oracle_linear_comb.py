import sys
sys.path.append("../")

import numpy as np
from oracles.saddle import OracleLinearComb, create_robust_linear_oracle, ArrayPair
from utils import grad_finite_diff_saddle


def test_oracle_linear_comb():
    np.random.seed(0)
    d = 10
    A_one, A_two = np.random.randn(40, d), np.random.randn(70, d)
    b_one, b_two = np.random.randn(40), np.random.randn(70)
    oracle_one = create_robust_linear_oracle(A_one, b_one, 0.1, 0.2, normed=True)
    oracle_two = create_robust_linear_oracle(A_two, b_two, 0.5, 1.0, normed=True)

    coefs = 3. * np.random.rand(2)
    oracle = OracleLinearComb([oracle_one, oracle_two], [coefs[0], coefs[1]])

    for _ in range(20):
        z = ArrayPair(np.random.rand(d), np.random.rand(d))

        oracle_grad_x = oracle.grad_x(z)
        oracle_grad_y = oracle.grad_y(z)

        diff_grad_x, diff_grad_y = grad_finite_diff_saddle(oracle.func, z, eps=1e-7)

        assert np.allclose(oracle_grad_x, diff_grad_x, atol=1e-4)
        assert np.allclose(oracle_grad_y, diff_grad_y, atol=1e-4)
