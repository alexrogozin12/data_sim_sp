import numpy as np
from typing import Callable
from .base import BaseSmoothSaddleOracle
from oracles.saddle.base import ArrayPair


class RobustLinearOracle(BaseSmoothSaddleOracle):
    def __init__(self, matvec_Ax: Callable, matvec_ATx: Callable, b: np.ndarray, regcoef: float):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef

        self._n = self.b.shape[0]
        self._ones = np.ones_like(b)

    def func(self, z: ArrayPair) -> float:
        under_norm = self.matvec_Ax(z.x) + z.y.dot(z.x) * self._ones - self.b
        return under_norm.dot(under_norm) / (2 * self._n) + self.regcoef * z.x.dot(z.x) / 2.

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        w = self.matvec_Ax(z.x) + z.y.dot(z.x) * self._ones - self.b
        return (self.matvec_ATx(w) + self._ones.dot(w) * z.y) / self._n + self.regcoef * z.x

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return z.y.dot(z.x) * z.x + self._ones.dot(self.matvec_Ax(z.x)) * z.x / self._n - \
               self._ones.dot(self.b) * z.x / self._n


def create_robust_linear_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    return RobustLinearOracle(matvec_Ax, matvec_ATx, b, regcoef)
