import numpy as np

from typing import List, Optional
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle, OracleLinearComb
from methods.saddle import Logger
from .base import BaseMethod


class DecentralizedGD(BaseMethod):
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        stepsize: float,
        mix_mat: np.ndarray,
        z_0: ArrayPair,
        logger: Optional[Logger]
    ):
        self._num_nodes = len(oracles)
        oracle_sum = OracleLinearComb(oracles, [1 / self._num_nodes] * self._num_nodes)
        super().__init__(oracle_sum, z_0, None, None, logger)
        self.oracle_list = oracles
        self.stepsize = stepsize
        self.mix_mat = mix_mat
        self.z_list = ArrayPair(
            np.tile(z_0.x.copy(), self._num_nodes).reshape(self._num_nodes, z_0.x.shape[0]),
            np.tile(z_0.y.copy(), self._num_nodes).reshape(self._num_nodes, z_0.y.shape[0])
        )

    def step(self):
        self.z_list = self.mul_by_mix_mat(self.z_list) - \
                      self.stepsize * self.oracle_grad_list(self.z_list)
        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))

    def oracle_grad_list(self, z: ArrayPair) -> ArrayPair:
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def mul_by_mix_mat(self, z: ArrayPair):
        return ArrayPair(self.mix_mat.dot(z.x), self.mix_mat.dot(z.y))

