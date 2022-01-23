import numpy as np
from typing import List


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class OracleLinearComb(BaseSmoothOracle):
    """
    Implements linear combination of several oracles with given coefficients.
    Resulting oracle = sum_{m=1}^M coefs[m] * oracles[m].

    Parameters
    ----------
    oracles: List[BaseSmoothOracle]

    coefs: List[float]
    """

    def __init__(self, oracles: List[BaseSmoothOracle], coefs: List[float]):
        if len(oracles) != len(coefs):
            raise ValueError("Numbers of oracles and coefs should be equal!")
        self.oracles = oracles
        self.coefs = coefs

    def func(self, x: np.ndarray) -> float:
        res = 0
        for oracle, coef in zip(self.oracles, self.coefs):
            res += oracle.func(x) * coef
        return res

    def grad(self, x: np.ndarray) -> np.ndarray:
        res = self.oracles[0].grad(x) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad(x) * coef
        return res
