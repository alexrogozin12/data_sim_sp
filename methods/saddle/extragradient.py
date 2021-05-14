import numpy as np

from collections import defaultdict
from datetime import datetime
from typing import Callable
from oracles.saddle import RobustLinearOracle
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle
from .base import BaseSaddleMethod


class Extragradient(BaseSaddleMethod):
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            stepsize: float,
            z_0: ArrayPair,
            trace: bool = True):
        super().__init__(oracle, z_0, trace)
        self.stepsize = stepsize

    def step(self):
        w = self.z - self.oracle.grad(self.z) * self.stepsize
        self.z = self.z - self.oracle.grad(w) * self.stepsize


def extragradient_solver(oracle: BaseSmoothSaddleOracle, stepsize: float, z_0: ArrayPair,
                         num_iter: int) -> ArrayPair:
    method = Extragradient(oracle, stepsize, z_0, trace=True)
    method.run(max_iter=num_iter)
    return method.hist['z_star']