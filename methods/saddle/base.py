import sys

sys.path.append("../")

import numpy as np
import numpy.linalg as npla
from collections import defaultdict
from datetime import datetime
from oracles.saddle import BaseSmoothSaddleOracle, ArrayPair


class BaseSaddleMethod(object):
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            z_0: ArrayPair,
            trace: bool = True
    ):
        self.oracle = oracle
        self.z = z_0.copy()
        self.trace = trace

    def run(self, max_iter, max_time=None):
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, 'hist'):
            self.hist = defaultdict(list)
        if not hasattr(self, 'time'):
            self.time = 0.

        self._absolute_time = datetime.now()
        for iter_count in range(max_iter):
            if self.time > max_time:
                break
            if self.trace:
                self._update_history()
            self.step()

        self.hist['z_star'] = self.z.copy()

    def _update_history(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now
        self.hist['func'].append(self.oracle.func(self.z))
        self.hist['time'].append(self.time)

    def step(self):
        raise NotImplementedError('step() not implemented!')
