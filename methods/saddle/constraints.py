import numpy as np
from .base import ArrayPair


class ConstraintsL2(object):
    """
    Applies L2-norm constraints. Bounds x and y to Euclidean balls with radiuses r_x and r_y,
    respectively
    """
    def __init__(self, r_x: float, r_y: float):
        self.r_x = r_x
        self.r_y = r_y

    def apply(self, z: ArrayPair):
        x_norm = np.linalg.norm(z.x)
        y_norm = np.linalg.norm(z.y)
        if x_norm >= self.r_x:
            z.x = z.x / x_norm * self.r_x
        if y_norm >= self.r_y:
            z.y = z.y / y_norm * self.r_y
