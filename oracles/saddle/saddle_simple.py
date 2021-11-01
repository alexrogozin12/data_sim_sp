import numpy as np
from .base import BaseSmoothSaddleOracle, ArrayPair


class ScalarProdOracle(BaseSmoothSaddleOracle):
    """
    Simple oracle for Scalar product. Computes coef * z.T z.

    Parameters
    ----------
    coef: float
    """
    def __init__(self, coef: float = 1.):
        super().__init__()
        self.coef = coef

    def func(self, z: ArrayPair) -> float:
        return self.coef * z.x.dot(z.y)

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        return self.coef * z.y

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return self.coef * z.x


class SquareDiffOracle(BaseSmoothSaddleOracle):
    """
    Oracle implementing coef_x * x.T x + coef_y * y.T y.

    Parameters
    ----------
    coef_x: float

    coef_y: float
    """
    def __init__(self, coef_x: float = 1., coef_y: float = 1.):
        super().__init__()
        self.coef_x = coef_x
        self.coef_y = coef_y

    def func(self, z: ArrayPair) -> float:
        x, y = z.tuple()
        return self.coef_x * x.dot(x) - self.coef_y * y.dot(y)

    def grad_x(self, z: ArrayPair) -> float:
        x, y = z.tuple()
        return 2 * self.coef_x * x

    def grad_y(self, z: ArrayPair) -> float:
        x, y = z.tuple()
        return -2 * self.coef_y * y
