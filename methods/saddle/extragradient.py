import numpy as np

from typing import Optional
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle
from .base import BaseSaddleMethod
from .logger import Logger
from .constraints import ConstraintsL2


class Extragradient(BaseSaddleMethod):
    """
    Non-distributed Extragradient method.

    oracle: BaseSmoothSaddleOracle
        Oracle of the objective function.

    stepsize: float
        Stepsize of Extragradient method.

    z_0: ArrayPair
        Initial guess.

    tolerance: Optional[float]
        Accuracy required for stopping criteria.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            stepsize: float,
            z_0: ArrayPair,
            tolerance: Optional[float],
            stopping_criteria: Optional[str],
            logger: Optional[Logger],
            constraints: Optional[ConstraintsL2] = None):
        super().__init__(oracle, z_0, tolerance, stopping_criteria, logger)
        self.stepsize = stepsize
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)

    def step(self):
        w = self.z - self.oracle.grad(self.z) * self.stepsize
        self.constraints.apply(w)
        self.grad = self.oracle.grad(w)
        self.z = self.z - self.grad * self.stepsize
        self.constraints.apply(self.z)


def extragradient_solver(oracle: BaseSmoothSaddleOracle, stepsize: float, z_0: ArrayPair,
                         num_iter: int, tolerance: Optional[float] = None,
                         stopping_criteria: Optional[str] = None,
                         logger: Optional[Logger] = None,
                         constraints: ConstraintsL2 = None) -> ArrayPair:
    """
    Solve the problem with standard Extragradient method up to a desired accuracy.
    """

    method = Extragradient(oracle, stepsize, z_0, tolerance, stopping_criteria, logger, constraints)
    method.run(max_iter=num_iter)
    return method.z
