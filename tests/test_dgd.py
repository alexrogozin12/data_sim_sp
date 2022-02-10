import sys

sys.path.append("../")

import numpy as np
from oracles.minimization import QuadraticOracle
from methods.minimization import DecentralizedGD
from methods.minimization.logger import LoggerDecentralized
from utils import gen_mix_mat


def test_decentralized_gd():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    mix_mat = gen_mix_mat(num_nodes)

    x_0 = np.random.rand(num_nodes * d).reshape(num_nodes, d)
    oracles = [QuadraticOracle(m / num_nodes * np.diag(np.ones(d)), np.zeros(d))
               for m in range(1, num_nodes + 1)]

    logger = LoggerDecentralized()
    method = DecentralizedGD(
        oracle_list=oracles,
        stepsize=0.01,
        mix_mat=mix_mat,
        x_0=x_0,
        logger=logger,
        mix_mat_repr="simple"
    )
    method.run(max_iter=1000)
    assert np.all((method.x ** 2).sum(axis=1) <= 0.05)

    logger_kron = LoggerDecentralized()
    method_kron = DecentralizedGD(
        oracle_list=oracles,
        stepsize=0.01,
        mix_mat=np.kron(mix_mat, np.eye(d).astype(np.float32)),
        x_0=x_0,
        logger=logger_kron,
        mix_mat_repr="kronecker"
    )
    method_kron.run(max_iter=1000)
    assert np.all((method_kron.x ** 2).sum(axis=1) <= 0.05)
