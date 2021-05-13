import numpy as np
import scipy

from numpy.linalg import LinAlgError
from datetime import datetime
from collections import defaultdict
from methods.line_search import get_line_search_tool
from methods.base import BaseMethod


class GradientDescent(BaseMethod):
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
