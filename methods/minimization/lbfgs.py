import numpy as np

from collections import deque
from .line_search import get_line_search_tool
from .base import BaseMethod


class LBFGS(BaseMethod):
    """
    Limited memory BFGS optimization method.

    oracle: BaseSmoothOracle
        Oracle corresponding to the objective function.

    x_0: np.ndarray
        Initial guess.

    tolerance: float
        Accuracy required for stopping criteria.

    memory_size: int
        Memory size of LBFGS.

    line_search_options: dict
        Options for line search.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. See BaseMethod docs for details.

    trace: bool
        If True, saves the history of the method during its iterations.
    """
    def __init__(self, oracle, x_0, tolerance=1e-4, memory_size=10,
                 line_search_options=None, stopping_criteria='grad_rel',
                 trace=True):

        super(LBFGS, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.x_k = self.x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.memory_size = memory_size
        self.lbfgs_queue = deque(maxlen=memory_size)
        if self.line_search_tool._method == 'Constant':
            self.alpha_0 = self.line_search_tool.c
        else:
            self.alpha_0 = 1.

    def step(self):
        self.grad_k = self.oracle.grad(self.x_k)

        self.old_x_k = self.x_k
        self.old_grad_k = self.grad_k

        try:
            s, y = self.lbfgs_queue.pop()
            self.lbfgs_queue.append((s, y))
            gamma_0 = y.dot(s) / y.dot(y)
            d_k = self.lbfgs_mul(-self.grad_k, list(self.lbfgs_queue), gamma_0)
        except IndexError:
            d_k = -self.grad_k

        alpha_k = self.line_search_tool.line_search(
            self.oracle, self.x_k, d_k, self.alpha_0
        )

        if alpha_k is None:
            return

        self.x_k = self.x_k + alpha_k * d_k
        self.lbfgs_queue.append((self.x_k - self.old_x_k,
                                 self.oracle.grad(self.x_k) - self.old_grad_k))

    def lbfgs_mul(self, v, memory, gamma_0):
        if len(memory) == 0:
            return gamma_0 * v
        s, y = memory[-1]
        v1 = v - s.dot(v) / y.dot(s) * y
        z = self.lbfgs_mul(v1, memory[:-1], gamma_0)
        return z + (s.dot(v) - y.dot(z)) / y.dot(s) * s
