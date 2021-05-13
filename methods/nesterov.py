import numpy as np
import scipy

from methods.line_search import get_line_search_tool
from methods.base import BaseMethod


class Nesterov(BaseMethod):
    def __init__(self, oracle, x_0, tolerance=1e-5, line_search_options=None, 
                 stopping_criteria='grad_rel', trace=True):
        super(Nesterov, self).__init__(oracle, x_0, stopping_criteria, trace)
        self.x_k = self.x_0.copy()
        self.y_k = self.x_0.copy()
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.tolerance = tolerance
        self.lam = 0.5
    
    def step(self):
        lam_old = self.lam
        self.lam = (1. + np.sqrt(1. + 4. * lam_old**2)) / 2.
        gamma = (1 - lam_old) / self.lam
        
        y_k_old = self.y_k.copy()
        self.grad_k = self.oracle.grad(self.x_k)
        alpha = self.line_search_tool.line_search(self.oracle, self.x_k, -self.grad_k)
        self.y_k = self.x_k - alpha * self.grad_k
        self.x_k = (1 - gamma) * self.y_k + gamma * y_k_old
