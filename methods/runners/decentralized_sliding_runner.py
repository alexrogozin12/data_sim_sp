import sys

sys.path.append("../")

import numpy as np
from typing import List
from methods.saddle import Logger, DecentralizedSaddleSliding
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle
from .utils import compute_lam_2


class DecentralizedSaddleSlidingRunner(object):
    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            L: float,
            mu: float,
            delta: float,
            mix_mat: np.ndarray,
            r_x: float,
            r_y: float,
            eps: float,
            logger: Logger
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.delta = delta
        self.mix_mat = mix_mat
        self.r_x = r_x
        self.r_y = r_y
        self.eps = eps
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        self.gamma = min(1. / (7 * self.delta), 1 / (12 * self.mu))  # outer step-size
        self.e = 0.5 / (2 + 12 * self.gamma ** 2 * self.delta ** 2 + 4 / (self.gamma * self.mu) + (
                    8 * self.gamma * self.delta ** 2) / self.mu)
        self.gamma_inner = 0.5 / (self.gamma * self.L + 1)
        self.T_inner = int((1 + self.gamma * self.L) * np.log10(1 / self.e))
        self._lam = compute_lam_2(self.mix_mat)
        self.gossip_step = (1 - np.sqrt(1 - self._lam ** 2)) / (1 + np.sqrt(1 - self._lam ** 2))

        self._omega = 2 * np.sqrt(self.r_x**2 + self.r_y**2)
        self._g = 0.  # upper bound on gradient at optimum; let it be 0 for now
        self._rho = 1 - self._lam
        self._num_nodes = len(self.oracles)
        self.con_iters_grad = int(1 / np.sqrt(self._rho) * \
            np.log(
                (self.gamma*2 + self.gamma / self.mu) * self._num_nodes *
                (self.L * self._omega + self._g)**2 /
                (self.eps * self.gamma * self.mu)
            ))
        self.con_iters_pt = int(1 / np.sqrt(self._rho) * \
            np.log(
                (1 + self.gamma**2 * self.L**2 + self.gamma * self.L**2 / self.mu) *
                self._num_nodes * self._omega**2 /
                (self.eps * self.gamma * self.mu)
            ))
        self._params_computed = True

    def create_method(self, z_0: ArrayPair):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedSaddleSliding(
            oracles=self.oracles,
            stepsize_outer=self.gamma,
            stepsize_inner=self.gamma_inner,
            inner_iterations=self.T_inner,
            con_iters_grad=self.con_iters_grad,
            con_iters_pt=self.con_iters_pt,
            mix_mat=self.mix_mat,
            gossip_step=self.gossip_step,
            z_0=z_0,
            logger=self.logger,
            constraints=None
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
