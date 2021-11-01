import numpy as np
from .base import ArrayPair


class ConstraintsL2(object):
    """
    Applies L2-norm constraints. Bounds x and y to Euclidean balls with radiuses r_x and r_y,
    respectively (inplace).

    Parameters
    ----------
    r_x: float
        Bound on x in L2 norm.

    r_y: float
        Bound on y in L2 norm.
    """
    def __init__(self, r_x: float, r_y: float):
        self.r_x = r_x
        self.r_y = r_y

    def apply(self, z: ArrayPair):
        """
        Applies L2 constraints to z (inplace).

        Parameters
        ----------
        z: ArrayPair
        """

        x_norm = np.linalg.norm(z.x)
        y_norm = np.linalg.norm(z.y)
        if x_norm >= self.r_x:
            z.x = z.x / x_norm * self.r_x
        if y_norm >= self.r_y:
            z.y = z.y / y_norm * self.r_y

    def apply_per_row(self, z_list: ArrayPair):
        """
        Applies L2 constraints to each row of z_list (inplace).

        Parameters
        ----------
        z_list: ArrayPair
        """

        for i in range(z_list.x.shape[0]):
            x_norm = np.linalg.norm(z_list.x[i])
            if x_norm >= self.r_x:
                z_list.x[i] = z_list.x[i] / x_norm * self.r_x

        for i in range(z_list.y.shape[0]):
            y_norm = np.linalg.norm(z_list.y[i])
            if y_norm >= self.r_y:
                z_list.y[i] = z_list.y[i] / y_norm * self.r_y
