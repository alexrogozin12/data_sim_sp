import sys

sys.path.append("../")

import numpy as np
from collections import defaultdict
from datetime import datetime
from oracles.saddle import BaseSmoothSaddleOracle, ArrayPair
from typing import Optional
from .logger import Logger


class BaseSaddleMethod(object):
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            z_0: ArrayPair,
            tolerance: Optional[float],
            stopping_criteria: Optional[str],
            logger: Optional[Logger]
    ):
        self.oracle = oracle
        self.z = z_0.copy()
        self.tolerance = tolerance
        self.logger = logger
        if stopping_criteria == 'grad_rel':
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == 'grad_abs':
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        else:
            raise ValueError('Unknown stopping criteria type: "{}"' \
                             .format(stopping_criteria))

    def run(self, max_iter, max_time=None):
        self.grad_norm_0 = self.z.norm()
        if self.logger is not None:
            self.logger.start(self)
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, 'time'):
            self.time = 0.

        self._absolute_time = datetime.now()
        for iter_count in range(max_iter):
            if self.time > max_time:
                break
            self._update_time()
            if self.logger is not None:
                self.logger.step(self)
            self.step()

        if self.logger is not None:
            self.logger.end(self)

    def _update_time(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now

    def step(self):
        raise NotImplementedError('step() not implemented!')

    def stopping_criteria_grad_relative(self):
        return self.grad.dot(self.grad) <= self.tolerance * self.grad_norm_0 ** 2

    def stopping_criteria_grad_absolute(self):
        return self.grad.dot(self.grad) <= self.tolerance

    def stopping_criteria_none(self):
        return False
