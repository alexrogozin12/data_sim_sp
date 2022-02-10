import numpy as np

from typing import List, Optional
from oracles.minimization import BaseSmoothOracle, OracleLinearComb
from .base import BaseDecentralizedMethod
from .logger import LoggerDecentralized


class DecentralizedGD(BaseDecentralizedMethod):
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

    def step(self):
        if self.mix_mat_repr == "simple":
            x_mixed = self.mix_mat.dot(self.x)
        else:
            n, d = self.x.shape
            x_mixed = self.mix_mat.dot(self.x.flatten()).reshape(n, d)
        self.x = x_mixed - self.stepsize * self.grad_list(self.x)
