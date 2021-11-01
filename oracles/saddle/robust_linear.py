import numpy as np
from typing import Callable
from .base import BaseSmoothSaddleOracle
from oracles.saddle.base import ArrayPair


class RobustLinearOracle(BaseSmoothSaddleOracle):
    """
    Oracle for Robust linear regression.

    Parameters
    ----------
    matvec_Ax: Callable
        Multiplication by feature matrix A.

    matvec_ATx: Callable
        Multiplication by A.T.

    matmat_ATsA: Callable
        Computes A.T diag(s) A, where diag(s) is a diagonal matrix with values of s on diagonal.

    b: np.ndarray
        Vector of labels.

    regcoef_x: float
        Regularization coefficient for x.

    regcoef_delta: float
        Regularization coefficient for delta.

    normed: bool
        If True, compute mean squared error over the dataset; else compute sum squared error.
    """
    def __init__(self, matvec_Ax: Callable, matvec_ATx: Callable, b: np.ndarray, regcoef_x: float,
                 regcoef_delta: float, normed: bool):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef_x = regcoef_x
        self.regcoef_delta = regcoef_delta

        self._n = self.b.shape[0]
        self.normed = normed
        self._ones = np.ones_like(b)

    def func(self, z: ArrayPair) -> float:
        x, delta = z.tuple()
        under_norm = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        den = self._n if self.normed else 1.
        return under_norm.dot(under_norm) / (2 * den) + self.regcoef_x * x.dot(x) / 2. - \
               self.regcoef_delta * delta.dot(delta) / 2.

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        x, delta = z.tuple()
        w = self.matvec_Ax(x) + delta.dot(x) * self._ones - self.b
        den = self._n if self.normed else 1.
        return (self.matvec_ATx(w) + self._ones.dot(w) * delta) / den + self.regcoef_x * x

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        x, delta = z.tuple()
        den = self._n if self.normed else 1.
        return self._n / den * delta.dot(x) * x + self._ones.dot(self.matvec_Ax(x)) * x / den - \
               self._ones.dot(self.b) * x / den - self.regcoef_delta * delta


def create_robust_linear_oracle(A, b: np.ndarray, regcoef_x: float, regcoef_delta: float,
                                normed: bool):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    return RobustLinearOracle(matvec_Ax, matvec_ATx, b, regcoef_x, regcoef_delta, normed)
