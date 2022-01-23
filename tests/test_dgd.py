import sys

sys.path.append("../")

import numpy as np
from oracles.minimization import QuadraticOracle, OracleLinearComb
from methods.saddle import Logger
from methods.minimization import DecentralizedGD
from utils import gen_mix_mat


def test_decentralized_gd():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    mix_mat = gen_mix_mat(num_nodes)

    z_0 = np.random.rand(d)
    oracles = [QuadraticOracle(m / num_nodes * np.diag(np.ones(d)), np.zeros(d))
               for m in range(1, num_nodes + 1)]

    method = DecentralizedGD(
        oracles=oracles,
        stepsize=0.01,
        mix_mat=mix_mat,
        z_0=z_0,
        trace=True,
    )
    method.run(max_iter=1000)
    assert method.hist["x_star"].dot(method.hist["x_star"]) <= 0.05
