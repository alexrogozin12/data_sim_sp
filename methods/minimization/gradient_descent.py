import numpy as np

from numpy.linalg import LinAlgError
from .line_search import get_line_search_tool
from .base import BaseMethod


class GradientDescent(BaseMethod):
    """
    Simple Gradient Descent methods with various line-search options.

    oracle: BaseSmoothOracle
        Oracle corresponding to the objective function.

    x_0: np.ndarray
        Initial guess.

    tolerance: float
        Accuracy required for stopping criteria.

    line_search_options: dict
        Options for line search.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. See BaseMethod docs for details.

    trace: bool
        If True, saves the history of the method during its iterations.
    """
    def __init__(self, oracle, x_0, tolerance=1e-5, line_search_options=None,
                 stopping_criteria='grad_rel', trace=True):

        super(GradientDescent, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.x_k = self.x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        try:
            self.alpha_k = self.line_search_tool.alpha_0
        except AttributeError:
            self.alpha_k = self.line_search_tool.c

    def step(self):
        self.grad_k = self.oracle.grad(self.x_k)
        self.alpha_k = self.line_search_tool.line_search(
            self.oracle, self.x_k, -self.grad_k, self.alpha_k)
        if self.alpha_k is None:
            return
        self.x_k = self.x_k - self.alpha_k * self.grad_k
