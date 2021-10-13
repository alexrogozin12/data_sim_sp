import sys

sys.path.append("../")

import numpy as np
from oracles.saddle import ArrayPair
from oracles.saddle.saddle_simple import SquareDiffOracle
from methods.saddle import DecentralizedExtragradientGT, Logger
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
    gamma = mu

    eta = max((1 - lam)**2 * gamma / (500 * L), (1 - lam)**(4/3) * mu**(1/3) / (40 * L**(4/3)))
    eta = min(eta, (1 - lam)**2 / (22 * L))
    logger = Logger()

    method = DecentralizedExtragradientGT(
        oracles=oracles,
        stepsize=eta,
        mix_mat=mix_mat,
        z_0=z_0,
        logger=logger,
        constraints=None
    )
    method.run(max_iter=10000)
    assert logger.z_star.dot(logger.z_star) <= 0.05
