import numpy as np

from collections import defaultdict
from datetime import datetime
from typing import Optional
from oracles.saddle import RobustLinearOracle
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle
from .base import BaseSaddleMethod
from .logger import Logger


class Extragradient(BaseSaddleMethod):
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            stepsize: float,
            z_0: ArrayPair,
            tolerance: Optional[float],
            stopping_criteria: Optional[str],
            logger: Optional[Logger]):
        super().__init__(oracle, z_0, tolerance, stopping_criteria, logger)
        self.stepsize = stepsize

    def step(self):
        w = self.z - self.oracle.grad(self.z) * self.stepsize
        self.z = self.z - self.oracle.grad(w) * self.stepsize


def extragradient_solver(oracle: BaseSmoothSaddleOracle, stepsize: float, z_0: ArrayPair,
                         num_iter: int, tolerance: Optional[float] = None,
                         stopping_criteria: Optional[str] = None,
                         logger: Optional[Logger] = None) -> ArrayPair:
    method = Extragradient(oracle, stepsize, z_0, tolerance, stopping_criteria, logger)
    method.run(max_iter=num_iter)
    return method.z
