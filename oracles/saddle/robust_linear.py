import numpy as np
from typing import Callable
from .base import BaseSmoothSaddleOracle
from oracles.saddle.base import ArrayPair


class RobustLinearOracle(BaseSmoothSaddleOracle):
    def __init__(self, matvec_Ax: Callable, matvec_ATx: Callable, b: np.ndarray, regcoef_x: float,
                 regcoef_delta: float):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef_x = regcoef_x
        self.regcoef_delta = regcoef_delta

        self._n = self.b.shape[0]
        self._ones = np.ones_like(b)

    def func(self, z: ArrayPair) -> float:
        x, delta = z.tuple()
        under_norm = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return under_norm.dot(under_norm) / (2 * self._n) + self.regcoef_x * x.dot(x) / 2. - \
               self.regcoef_delta * delta.dot(delta) / 2.

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        x, delta = z.tuple()
        w = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return (self.matvec_ATx(w) + self._ones.dot(w) * delta) / self._n + self.regcoef_x * x

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        x, delta = z.tuple()
        return delta.dot(x) * x + self._ones.dot(self.matvec_Ax(x)) * x / self._n - \
               self._ones.dot(self.b) * x / self._n - self.regcoef_delta * delta


def create_robust_linear_oracle(A, b: np.ndarray, regcoef_x: float, regcoef_delta: float):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    return RobustLinearOracle(matvec_Ax, matvec_ATx, b, regcoef_x, regcoef_delta)
