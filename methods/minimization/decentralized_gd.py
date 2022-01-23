import numpy as np

from typing import List, Optional
from oracles.minimization import BaseSmoothOracle, OracleLinearComb
from .base import BaseMethod


class DecentralizedGD(BaseMethod):
    def __init__(
            self,
            oracles: List[BaseSmoothOracle],
            stepsize: float,
            mix_mat: np.ndarray,
            z_0: np.ndarray,
            trace: bool
    ):
        self._num_nodes = len(oracles)
        oracle_sum = OracleLinearComb(oracles, [1 / self._num_nodes] * self._num_nodes)
        super().__init__(oracle_sum, z_0, None, trace)
        self.oracle_list = oracles
        self.stepsize = stepsize
        self.mix_mat = mix_mat
        self.z_list = np.tile(z_0.copy(), self._num_nodes).reshape(self._num_nodes, z_0.shape[0])
        self.x_k = self.z_list.mean(axis=0)

    def step(self):
        self.z_list = self.mix_mat.dot(self.z_list) - \
                      self.stepsize * self.oracle_grad_list(self.z_list)
        self.x_k = self.z_list.mean(axis=0)

    def oracle_grad_list(self, z: np.ndarray) -> np.ndarray:
        res = np.empty_like(z)
        for i in range(z.shape[0]):
            res[i] = self.oracle_list[i].grad(z[i])
        return res
