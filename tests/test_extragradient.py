import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import create_robust_linear_oracle, RobustLinearOracle, ArrayPair
from oracles.saddle.saddle_simple import ScalarProdOracle
from methods.saddle import Extragradient, Logger, ConstraintsL2


def create_random_robust_linear_oracle(n: int, d: int) -> RobustLinearOracle:
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef_x=0.1, regcoef_delta=0.5, normed=True)
    return oracle


def test_extragradient_step():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0, tolerance=None, stopping_criteria=None, logger=None)
    method.step()


def test_extragradient_run_robust_linear():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0, tolerance=None, stopping_criteria=None, logger=None)
    method.run(max_iter=20)


def test_extragradient_run_scalar_prod():
    np.random.seed(0)
    d = 20
    oracle = ScalarProdOracle()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    logger = Logger()
    method = Extragradient(oracle, 0.5, z_0, tolerance=None, stopping_criteria=None, logger=logger)
    method.run(max_iter=1000)
    assert logger.z_star.dot(logger.z_star) <= 1e-8


def test_extragradient_run_scalar_prod_constrained():
    np.random.seed(0)
    d = 20
    oracle = ScalarProdOracle()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    logger = Logger()
    constraints = ConstraintsL2(1., 2.)
    method = Extragradient(oracle, 0.5, z_0, tolerance=None, stopping_criteria=None, logger=logger,
                           constraints=constraints)
    method.run(max_iter=1000)
    assert logger.z_star.dot(logger.z_star) <= 1e-8
