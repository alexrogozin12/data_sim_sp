import numpy as np

from typing import Callable
from oracles.saddle import BaseSmoothSaddleOracle, ArrayPair


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


class SaddleSliding(object):
    def __init__(
            self,
            oracle_g: BaseSmoothSaddleOracle,
            oracle_phi: BaseSmoothSaddleOracle,
            eta: float,
            inner_solver: Callable,
            z_0: ArrayPair,
            trace: bool = True
    ):
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.eta = eta
        self.inner_solver = inner_solver
        self.z = z_0
        self.trace = trace

    def step(self):
        v = self.z - self.oracle_g.grad(self.z) * self.eta
        u = self.solve_subproblem(v)
        self.z = u + self.eta * (self.oracle_g.grad(self.z) - self.oracle_g.grad(u))

    def solve_subproblem(self, v: ArrayPair) -> ArrayPair:
        suboracle = SaddlePointOracleRegularizer(self.oracle_phi, self.eta, v)
        return self.inner_solver(suboracle)
