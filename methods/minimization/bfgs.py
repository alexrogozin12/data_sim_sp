import numpy as np

from .line_search import get_line_search_tool
from .base import BaseMethod


class BFGS(BaseMethod):
    """
    BFGS optimization method.

    oracle: BaseSmoothOracle
        Oracle corresponding to the objective function.

    x_0: np.ndarray
        Initial guess.

    H_0: np.ndarray
        Initial guess for inverse Hessian

    tolerance: float
        Accuracy required for stopping criteria.

    line_search_options: dict
        Options for line search.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. See BaseMethod docs for details.

    trace: bool
        If True, saves the history of the method during its iterations.
    """
    def __init__(self, oracle, x_0, H_0=None, tolerance=1e-4,
                 line_search_options=None, stopping_criteria='grad_rel',
                 trace=True):

        super(BFGS, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.x_k = x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        if H_0 is None:
            self.H_k = np.eye(x_0.shape[0], dtype=x_0.dtype)
        else:
            self.H_k = H_0.copy()

        if self.line_search_tool._method == 'Constant':
            self.alpha_0 = self.line_search_tool.c
        else:
            self.alpha_0 = 1.

        self.grad_k = self.oracle.grad(self.x_k)

    def step(self):
        self.x_k_old = self.x_k.copy()
        self.grad_k_old = self.grad_k.copy()

        d_k = -self.H_k.dot(self.grad_k)
        alpha_k = self.line_search_tool.line_search(
            self.oracle, self.x_k, d_k, self.alpha_0
        )
        self.x_k += alpha_k * d_k
        self.grad_k = self.oracle.grad(self.x_k)
        self.update_H()

    def update_H(self):
        self.s_k = (self.x_k - self.x_k_old).reshape((self.x_k.shape[0], 1))
        self.y_k = (self.grad_k - self.grad_k_old).reshape((self.x_k.shape[0], 1))
        den = self.s_k.flatten().dot(self.y_k.flatten())
        self.H_k -= self.H_k.dot(self.y_k).dot(self.s_k.T) / (den + 1e-8)  # right multiply
        self.H_k -= (self.H_k.dot(self.y_k).dot(self.s_k.T) / (den + 1e-8)).T  # left multiply
        self.H_k += self.s_k.dot(self.s_k.T) / (den + 1e-8)
