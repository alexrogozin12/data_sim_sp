import sys

sys.path.append("../")

import numpy as np
from oracles.robust_linear import create_robust_linear_oracle
from utils import grad_finite_diff_saddle


def test_robust_linear_oracle():
    np.random.seed(0)
    n, d = 50, 8
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef=0.1)

    for _ in range(30):
        x, delta = np.random.rand(d), np.random.rand(d)
        oracle_grad_x = oracle.grad_x(x, delta)
        oracle_grad_y = oracle.grad_delta(x, delta)

        diff_grad_x, diff_grad_y = grad_finite_diff_saddle(oracle.func, x, delta, eps=1e-7)

        assert np.allclose(oracle_grad_x, diff_grad_x)
        assert np.allclose(oracle_grad_y, diff_grad_y)
