import numpy as np
from .base import BaseSmoothSaddleOracle


class ScalarProdOracle(BaseSmoothSaddleOracle):
    def __init__(self):
        pass

    def func(self, x: np.ndarray, y: np.ndarray) -> float:
        return x.dot(y)

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x
