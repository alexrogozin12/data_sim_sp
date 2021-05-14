import numpy as np
from typing import Tuple


class ArrayPair(object):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @property
    def shape_x(self):
        return self.x.shape

    @property
    def shape_y(self):
        return self.y.shape

    def __add__(self, other: "ArrayPair"):
        return ArrayPair(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "ArrayPair"):
        return ArrayPair(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return ArrayPair(self.x * other, self.y * other)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def copy(self):
        return ArrayPair(self.x.copy(), self.y.copy())

    def dot(self, other: "ArrayPair"):
        return self.x.dot(other.x) + self.y.dot(other.y)

    def tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y


class BaseSmoothSaddleOracle(object):
    """
    Base class for implementation of oracles for saddle point problems.
    """

    def func(self, z: ArrayPair) -> float:
        raise NotImplementedError('func() is not implemented.')

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        raise NotImplementedError('grad_x() is not implemented.')

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        raise NotImplementedError('grad_y() oracle is not implemented.')

    def grad(self, z: ArrayPair) -> ArrayPair:
        grad_x = self.grad_x(z)
        grad_y = self.grad_y(z)
        return ArrayPair(grad_x, -grad_y)
