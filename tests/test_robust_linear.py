import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import ArrayPair, create_robust_linear_oracle
from utils import grad_finite_diff_saddle


def test_robust_linear_oracle_normed():
    np.random.seed(0)
    n, d = 50, 8
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef_x=0.1, regcoef_delta=0.5, normed=True)

    for _ in range(20):
        z = ArrayPair(np.random.rand(d), np.random.rand(d))

        oracle_grad_x = oracle.grad_x(z)
        oracle_grad_y = oracle.grad_y(z)

        diff_grad_x, diff_grad_y = grad_finite_diff_saddle(oracle.func, z, eps=1e-7)

        assert np.allclose(oracle_grad_x, diff_grad_x)
        assert np.allclose(oracle_grad_y, diff_grad_y)


def test_robust_linear_oracle_not_normed():
    np.random.seed(0)
    n, d = 50, 8
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef_x=0.1, regcoef_delta=0.5, normed=False)

    for _ in range(10):
        z = ArrayPair(np.random.rand(d), np.random.rand(d))

        oracle_grad_x = oracle.grad_x(z)
        oracle_grad_y = oracle.grad_y(z)

        diff_grad_x, diff_grad_y = grad_finite_diff_saddle(oracle.func, z, eps=1e-7)

        assert np.allclose(oracle_grad_x, diff_grad_x)
        assert np.allclose(oracle_grad_y, diff_grad_y)
