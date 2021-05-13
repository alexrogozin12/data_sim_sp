import numpy as np


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

    def copy(self):
        return ArrayPair(self.x.copy(), self.y.copy())

    def dot(self, other: "ArrayPair"):
        return self.x.dot(other.x) + self.y.dot(other.y)


class BaseSmoothSaddleOracle(object):
    """
    Base class for implementation of oracles for saddle point problems.
    """

    def func(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError('func() is not implemented.')

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError('grad_x() is not implemented.')

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError('grad_y() oracle is not implemented.')

    def grad(self, x: np.ndarray, delta: np.ndarray) -> ArrayPair:
        grad_x = self.grad_x(x, delta)
        grad_delta = self.grad_y(x, delta)
        return ArrayPair(grad_x, -grad_delta)
