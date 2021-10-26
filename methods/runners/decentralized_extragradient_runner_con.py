import sys

sys.path.append("../")

import numpy as np
from typing import List
from methods.saddle import DecentralizedExtragradientCon, Logger, ConstraintsL2
from oracles.saddle import BaseSmoothSaddleOracle
from .utils import compute_lam_2


class DecentralizedExtragradientConRunner(object):
    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            L: float,
            mu: float,
            mix_mat: np.ndarray,
            r_x: float,
            r_y: float,
            eps: float,
            logger: Logger
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.mix_mat = mix_mat
        self.constraints = ConstraintsL2(r_x, r_y)
        self.eps = eps
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        self._lam = compute_lam_2(self.mix_mat)
        self.gamma = 1 / (4 * self.L)
        self.gossip_step = (1 - np.sqrt(1 - self._lam ** 2)) / (1 + np.sqrt(1 - self._lam ** 2))
        eps_0 = self.eps * self.mu * self.gamma * (1 + self.gamma * self.L) ** 2
        self.con_iters = int(np.sqrt(1 / (1 - self._lam)) * np.log(1 / eps_0))
        # rough estimate on con_iters (lower than actual)
        self._params_computed = True

    def create_method(self, z_0):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedExtragradientCon(
            oracles=self.oracles,
            stepsize=self.gamma,
            con_iters=self.con_iters,
            mix_mat=self.mix_mat,
            gossip_step=self.gossip_step,
            z_0=z_0,
            logger=self.logger,
            constraints=self.constraints
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
