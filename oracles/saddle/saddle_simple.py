import numpy as np
from .base import BaseSmoothSaddleOracle, ArrayPair


class ScalarProdOracle(BaseSmoothSaddleOracle):
    def __init__(self):
        pass

    def func(self, z: ArrayPair) -> float:
        return z.x.dot(z.y)

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        return z.y

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return z.x
