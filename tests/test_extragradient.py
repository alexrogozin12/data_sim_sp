import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import create_robust_linear_oracle, RobustLinearOracle, ArrayPair
from oracles.saddle.saddle_simple import ScalarProdOracle
from methods.saddle import Extragradient


def create_random_robust_linear_oracle(n: int, d: int) -> RobustLinearOracle:
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef=0.1)
    return oracle


def test_extragradient_step():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0)
    method.step()


def test_extragradient_run_robust_linear():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0)
    method.run(max_iter=20)


def test_extragradient_run_scalar_prod():
    np.random.seed(0)
    d = 20
    oracle = ScalarProdOracle()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.5, z_0)
    method.run(max_iter=1000)
    assert method.hist['z_star'].dot(method.hist['z_star']) <= 1e-8
