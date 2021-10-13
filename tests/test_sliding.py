import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import ArrayPair
from oracles.saddle.saddle_simple import ScalarProdOracle, SquareDiffOracle
from methods.saddle import SaddleSliding, DecentralizedSaddleSliding, extragradient_solver, Logger
from utils import gen_mix_mat, compute_lam_2


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

    logger = Logger()
    method = SaddleSliding(
        oracle_g=oracle_g,
        oracle_phi=oracle_phi,
        stepsize_outer=eta,
        stepsize_inner=eta_inner,
        inner_solver=extragradient_solver,
        inner_iterations=T_inner,
        z_0=z_0,
        logger=logger
    )
    method.run(max_iter=100)
    assert logger.z_star.dot(logger.z_star) <= 1e-2


def test_decentralized_sliding_simple():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    mix_mat = gen_mix_mat(num_nodes)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracles = [SquareDiffOracle(coef_x=m/num_nodes, coef_y=1-m/num_nodes)
               for m in range(1, num_nodes + 1)]
    L = 2.
    mu = (num_nodes + 1) / num_nodes
    delta = (num_nodes - 1) / num_nodes
    gamma = min(1. / (7 * delta), 1 / (12 * mu))  # outer step-size
    e = 0.5 / (2 + 12 * gamma**2 * delta**2 + 4 / (gamma * mu) + (8 * gamma * delta**2) / mu)
    gamma_inner = 0.5 / (gamma * L + 1)
    T_inner = int((1 + gamma * L) * np.log10(1 / e))
    lam = compute_lam_2(mix_mat)
    gossip_step = (1 - np.sqrt(1 - lam**2)) / (1 + np.sqrt(1 - lam**2))

    logger = Logger()
    method = DecentralizedSaddleSliding(
        oracles=oracles,
        stepsize_outer=gamma,
        stepsize_inner=gamma_inner,
        inner_iterations=T_inner,
        con_iters_grad=20,
        con_iters_pt=20,
        mix_mat=mix_mat,
        gossip_step=gossip_step,
        z_0=z_0,
        logger=logger,
        constraints=None
    )
    method.run(max_iter=500)
    assert logger.z_star.dot(logger.z_star) <= 1e-2
