import numpy as np
from scipy.special import expit
from .base import BaseSmoothOracle


class LinearRegressionL2Oracle(BaseSmoothOracle):
    """
    Linear regression oracle with L2 regularization.
    1/2 \|Ax - b\|_2^2 + regcoef * \|x\|_2^2.

    Parameters
    ----------
    A: np.ndarray
        Feature matrix

    b: np.ndarray
        Target vector
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, regcoef: float):
        self.A = A
        self.b = b
        self.regcoef = regcoef

    def func(self, x: np.ndarray):
        residual = self.A.dot(x) - self.b
        return 0.5 * residual.dot(residual)

    def grad(self, x: np.ndarray):
        residual = self.A.dot(x) - self.b
        return self.A.T.dot(residual)
