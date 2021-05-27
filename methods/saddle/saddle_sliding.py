import numpy as np

from typing import Callable, Optional
from oracles.saddle import BaseSmoothSaddleOracle, ArrayPair
from methods.saddle import Logger
from datetime import datetime
from collections import defaultdict
from .base import BaseSaddleMethod
from .constraints import ConstraintsL2


class SaddlePointOracleRegularizer(BaseSmoothSaddleOracle):
    def __init__(self, oracle: BaseSmoothSaddleOracle, eta: float, v: ArrayPair):
        self.oracle = oracle
        self.eta = eta
        self.v = v

    def func(self, z: ArrayPair) -> float:
        return self.eta * self.oracle.func(z) + 0.5 * (z.x - self.v.x).dot(z.x - self.v.x) - \
               0.5 * (z.y - self.v.y).dot(z.y - self.v.y)

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        return self.eta * self.oracle.grad_x(z) + z.x - self.v.x

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return self.eta * self.oracle.grad_y(z) + self.v.y - z.y


class SaddleSliding(BaseSaddleMethod):
    def __init__(
            self,
            oracle_g: BaseSmoothSaddleOracle,
            oracle_phi: BaseSmoothSaddleOracle,
            stepsize_outer: float,
            stepsize_inner: float,
            inner_solver: Callable,
            inner_iterations: int,
            z_0: ArrayPair,
            logger: Optional[Logger],
            constraints: Optional[ConstraintsL2] = None
    ):
        super().__init__(oracle_g, z_0, None, None, logger)
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_solver = inner_solver
        self.inner_iterations = inner_iterations
        self.constraints = constraints

    def step(self):
        v = self.z - self.oracle_g.grad(self.z) * self.stepsize_outer
        u = self.solve_subproblem(v)
        self.z = u + self.stepsize_outer * (self.oracle_g.grad(self.z) - self.oracle_g.grad(u))

    def solve_subproblem(self, v: ArrayPair) -> ArrayPair:
        suboracle = SaddlePointOracleRegularizer(self.oracle_phi, self.stepsize_outer, v)
        return self.inner_solver(
            suboracle,
            self.stepsize_inner, v, num_iter=self.inner_iterations, constraints=self.constraints)
