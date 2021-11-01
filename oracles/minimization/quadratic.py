import numpy as np
import scipy
from .base import BaseSmoothOracle


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function 1/2 x.T A x - b.T x.

    Parameters
    ----------
    A: np.ndarray

    b: np.ndarray
    """
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def hess_mat_prod(self, x, S):
        return self.A.dot(S)
