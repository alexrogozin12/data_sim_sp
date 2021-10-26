import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import ArrayPair
from oracles.saddle.saddle_simple import SquareDiffOracle
from methods.saddle import DecentralizedExtragradientCon, Logger
from utils import gen_mix_mat, compute_lam_2


def test_decentralized_extragradient():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    mix_mat = gen_mix_mat(num_nodes)
    lam = compute_lam_2(mix_mat)

    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracles = [SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
               for m in range(1, num_nodes + 1)]
    L = 2.
    mu = (num_nodes + 1) / num_nodes

    gamma = 1 / (4 * L)
    gossip_step = (1 - np.sqrt(1 - lam ** 2)) / (1 + np.sqrt(1 - lam ** 2))
    eps = 1e-4
    eps_0 = eps * mu * gamma * (1 + gamma * L)**2
    con_iters = int(5 * np.sqrt(1 / (1 - lam)) * np.log(1 / eps_0))
    logger = Logger()

    method = DecentralizedExtragradientCon(
        oracles=oracles,
        stepsize=gamma,
        con_iters=con_iters,
        mix_mat=mix_mat,
        gossip_step=gossip_step,
        z_0=z_0,
        logger=logger,
        constraints=None
    )
    method.run(max_iter=100)
    assert logger.z_star.dot(logger.z_star) <= 0.05
