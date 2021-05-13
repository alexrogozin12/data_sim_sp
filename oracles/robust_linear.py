import numpy as np
from scipy.special import expit
from typing import Callable
from oracles import BaseSmoothOracle


class RobustLinearOracle(object):
    def __init__(self, matvec_Ax: Callable, matvec_ATx: Callable, b: np.ndarray, regcoef: float):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef
        self._dim = self.b.shape[0]
        self._ones = np.ones_like(b)

    def func(self, x: np.ndarray, delta: np.ndarray):
        under_norm = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return under_norm.dot(under_norm) / (2 * self._dim) + self.regcoef * x.dot(x) / 2.

    def grad_x(self, x: np.ndarray, delta: np.ndarray):
        y = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        return (self.matvec_ATx(y) + self._ones.dot(y) * delta) / self._dim + self.regcoef * x

    def grad_delta(self, x: np.ndarray, delta: np.ndarray):
        return delta.dot(x) * x + self._ones.dot(self.matvec_Ax(x)) * x / self._dim - \
               self._ones.dot(self.b) * x / self._dim


def create_robust_linear_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    return RobustLinearOracle(matvec_Ax, matvec_ATx, b, regcoef)
