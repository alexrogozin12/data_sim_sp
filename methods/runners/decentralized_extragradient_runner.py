import sys

sys.path.append("../")

import numpy as np
from typing import List
from methods.saddle import DecentralizedExtragradientGT, Logger
from oracles.saddle import BaseSmoothSaddleOracle
from .utils import compute_lam_2


class DecentralizedExtragradientGTRunner(object):
    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            L: float,
            mu: float,
            gamma: float,
            mix_mat: np.ndarray,
            logger: Logger
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.gamma = gamma
        self.mix_mat = mix_mat
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        self._lam = compute_lam_2(self.mix_mat)
        eta = max((1 - self._lam) ** 2 * self.gamma / (500 * self.L),
                  (1 - self._lam) ** (4 / 3) * self.mu ** (1 / 3) / (40 * self.L ** (4 / 3)))
        self.eta = min(eta, (1 - self._lam) ** 2 / (22 * self.L))
        self._params_computed = True

    def create_method(self, z_0):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedExtragradientGT(
            oracles=self.oracles,
            stepsize=self.eta,
            mix_mat=self.mix_mat,
            z_0=z_0,
            logger=self.logger,
            constraints=None
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
