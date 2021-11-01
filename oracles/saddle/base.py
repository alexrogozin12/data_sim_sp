import numpy as np
from typing import List, Tuple


class ArrayPair(object):
    """
    Stores a pair of np.ndarrays representing x and y variables in a saddle-point problem.

    Parameters
    ----------
    x: np.ndarray

    y: np.ndarray
    """
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

    def norm(self):
        return np.sqrt(self.dot(self))

    def tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    @staticmethod
    def zeros(*args, **kwargs) -> "ArrayPair":
        """
        Same args as in np.zeros()
        """
        return ArrayPair(np.zeros(*args, **kwargs), np.zeros(*args, **kwargs))

    @staticmethod
    def zeros_like(*args, **kwargs) -> "ArrayPair":
        """
        Same args as in np.zeros_like()
        """
        return ArrayPair(np.zeros_like(*args, **kwargs), np.zeros_like(*args, **kwargs))


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


class OracleLinearComb(BaseSmoothSaddleOracle):
    """
    Implements linear combination of several saddle point oracles with given coefficients.
    Resulting oracle = sum_{m=1}^M coefs[m] * oracles[m].

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]

    coefs: List[float]
    """

    def __init__(self, oracles: List[BaseSmoothSaddleOracle], coefs: List[float]):
        if len(oracles) != len(coefs):
            raise ValueError("Numbers of oracles and coefs should be equal!")
        self.oracles = oracles
        self.coefs = coefs

    def func(self, z: ArrayPair) -> float:
        res = 0
        for oracle, coef in zip(self.oracles, self.coefs):
            res += oracle.func(z) * coef
        return res

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        res = self.oracles[0].grad_x(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad_x(z) * coef
        return res

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        res = self.oracles[0].grad_y(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad_y(z) * coef
        return res

    def grad(self, z: ArrayPair) -> ArrayPair:
        res = self.oracles[0].grad(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad(z) * coef
        return res
