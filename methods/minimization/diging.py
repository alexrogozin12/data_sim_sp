import numpy as np

from typing import List, Optional
from oracles.minimization import BaseSmoothOracle, OracleLinearComb
from .base import BaseDecentralizedMethod
from .logger import LoggerDecentralized


class Diging(BaseDecentralizedMethod):
    def __init__(
        self,
        oracle_list: List[BaseSmoothOracle],
        stepsize: float,
        mix_mat: np.ndarray,
        x_0: np.ndarray,
        logger: LoggerDecentralized,
        mix_mat_repr: str
    ):
        super().__init__(oracle_list, x_0, logger)
        self.stepsize = stepsize
        self.mix_mat = mix_mat
        if mix_mat_repr not in ["simple", "kronecker"]:
            raise ValueError(
                "Matrix representation type should be 'simple' or 'kronecker', got '{}'"
                    .format(self.mix_mat_repr))
        self.mix_mat_repr = mix_mat_repr  # should be "kronecker" or "simple"
        self.y = self.grad_list(x_0)

    def step(self):
        x_new = self.mul_by_mix_mat(self.x) - self.stepsize * self.y
        self.y = self.mul_by_mix_mat(self.y) + self.grad_list(x_new) - self.grad_list(self.x)
        self.x = x_new.copy()

    def mul_by_mix_mat(self, x: np.ndarray):
        if self.mix_mat_repr == "simple":
            return self.mix_mat.dot(x)
        else:
            n, d = x.shape
            return self.mix_mat.dot(x.flatten()).reshape(n, d)
