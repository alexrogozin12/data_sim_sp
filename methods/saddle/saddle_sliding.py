import numpy as np

from typing import Callable
from oracles.saddle import BaseSmoothSaddleOracle, ArrayPair
from methods.saddle import Extragradient
from datetime import datetime
from collections import defaultdict



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
            stepsize_outer: float,
            stepsize_inner: float,
            inner_solver: Callable,
            inner_iterations: int,
            z_0: ArrayPair,
            trace: bool = True
    ):
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_solver = inner_solver
        self.inner_iterations = inner_iterations
        self.z = z_0
        self.trace = trace

    def step(self):
        v = self.z - self.oracle_g.grad(self.z) * self.stepsize_outer
        u = self.solve_subproblem(v)
        self.z = u + self.stepsize_outer * (self.oracle_g.grad(self.z) - self.oracle_g.grad(u))

    def solve_subproblem(self, v: ArrayPair) -> ArrayPair:
        suboracle = SaddlePointOracleRegularizer(self.oracle_phi, self.stepsize_outer, v)
        return self.inner_solver(suboracle, self.stepsize_inner, v, self.inner_iterations)

    def run(self, max_iter, max_time=None):
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, 'hist'):
            self.hist = defaultdict(list)
        if not hasattr(self, 'time'):
            self.time = 0.

        self._absolute_time = datetime.now()
        for iter_count in range(max_iter):
            if self.time > max_time:
                break
            if self.trace:
                self._update_history()
            self.step()

        self.hist['z_star'] = self.z.copy()

    def _update_history(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now
        self.hist['func'].append(self.oracle_g.func(self.z))
        self.hist['time'].append(self.time)
