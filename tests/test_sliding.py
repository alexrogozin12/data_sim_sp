import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import create_robust_linear_oracle, RobustLinearOracle, ArrayPair
from oracles.saddle.saddle_simple import ScalarProdOracle, SquareDiffOracle
from methods.saddle import Extragradient, SaddleSliding, extragradient_solver


def test_sliding_simple():
    np.random.seed(0)
    d = 20
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracle_g = ScalarProdOracle(coef=0.01)
    oracle_phi = SquareDiffOracle(coef_x=0.5, coef_y=0.5)
    L, mu, delta = 1., 1., 0.01
    eta = min(1. / (2 * delta), 1 / (6 * mu))
    e = min(0.25, 1 / (64 / (eta * mu) + 64 * eta * L**2 / mu))
    eta_inner = 0.5 / (eta * L + 1)
    T_inner = int((1 + eta * L) * np.log10(1 / e))

    method = SaddleSliding(
        oracle_g=oracle_g,
        oracle_phi=oracle_phi,
        stepsize_outer=eta,
        stepsize_inner=eta_inner,
        inner_solver=extragradient_solver,
        inner_iterations=T_inner,
        z_0=z_0,
        trace=True
    )
    method.run(max_iter=100)
    assert method.hist['z_star'].dot(method.hist['z_star']) <= 1e-2
