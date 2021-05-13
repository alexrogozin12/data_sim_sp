import numpy as np

from collections import defaultdict
from datetime import datetime
from typing import Callable
from oracles.saddle import RobustLinearOracle
from oracles.saddle import ArrayPair
from .base import BaseSaddleMethod


class Extragradient(BaseSaddleMethod):
    def __init__(
            self,
            oracle: RobustLinearOracle,
            stepsize: float,
            z_0: ArrayPair,
            trace: bool = True):
        super().__init__(oracle, z_0, trace)
        self.stepsize = stepsize

    def step(self):
        w = self.z - self.oracle.grad(self.z.x, self.z.y) * self.stepsize
        self.z = self.z - self.oracle.grad(w.x, w.y) * self.stepsize
