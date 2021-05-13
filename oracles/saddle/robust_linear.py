import numpy as np
from typing import Callable
from .base import BaseSmoothSaddleOracle


class RobustLinearOracle(BaseSmoothSaddleOracle):
    def __init__(self, matvec_Ax: Callable, matvec_ATx: Callable, b: np.ndarray, regcoef: float):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef

        self._n = self.b.shape[0]
        self._ones = np.ones_like(b)

    def func(self, x: np.ndarray, delta: np.ndarray) -> float:
        under_norm = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return under_norm.dot(under_norm) / (2 * self._n) + self.regcoef * x.dot(x) / 2.

    def grad_x(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        z = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return (self.matvec_ATx(z) + self._ones.dot(z) * delta) / self._n + self.regcoef * x

    def grad_y(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return delta.dot(x) * x + self._ones.dot(self.matvec_Ax(x)) * x / self._n - \
               self._ones.dot(self.b) * x / self._n


def create_robust_linear_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    return RobustLinearOracle(matvec_Ax, matvec_ATx, b, regcoef)
