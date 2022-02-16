import numpy as np
from typing import Optional
from .base import BaseDecentralizedMethod


class LoggerDecentralized(object):
    def __init__(self, x_true: Optional[np.ndarray] = None):
        self.func_avg = []
        self.time = []
        self.sq_dist_to_con = []
        self.x_true = None
        if x_true is not None:
            self.x_true = x_true.copy()
            self.sq_dist_avg_to_opt = []

    def start(self, method: BaseDecentralizedMethod):
        pass

    def end(self, method: BaseDecentralizedMethod):
        pass

    def step(self, method: BaseDecentralizedMethod):
        self.func_avg.append(np.mean(method.func_list(method.x)))
        self.time.append(method.time)
        self.sq_dist_to_con.append(
            ((method.x - method.x.mean(axis=0)) ** 2).sum() / method.x.shape[0]
        )
        if self.x_true is not None:
            self.sq_dist_avg_to_opt.append(((method.x.mean(axis=0) - self.x_true) ** 2).sum())
