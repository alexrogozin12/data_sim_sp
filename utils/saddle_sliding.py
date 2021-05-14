import numpy as np
from methods.saddle import SaddleSliding, extragradient_solver
from oracles.saddle import ArrayPair


class SaddleSlidingRunner(object):
    def __init__(self, oracle_g, oracle_phi, L: float, mu: float, delta: float):
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.L = L
        self.mu = mu
        self.delta = delta

    def create_method(self, z_0: ArrayPair):
        eta = min(1. / (2 * self.delta), 1 / (6 * self.mu))
        e = min(0.25, 1 / (64 / (eta * self.mu) + 64 * eta * self.L ** 2 / self.mu))
        eta_inner = 0.5 / (eta * self.L + 1)
        T_inner = int((1 + eta * self.L) * np.log10(1 / e))

        self.method = SaddleSliding(
            oracle_g=self.oracle_g,
            oracle_phi=self.oracle_phi,
            stepsize_outer=eta,
            stepsize_inner=eta_inner,
            inner_solver=extragradient_solver,
            inner_iterations=T_inner,
            z_0=z_0,
            trace=True
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
