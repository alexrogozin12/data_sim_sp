import numpy as np
import numpy.linalg as npla
from collections import defaultdict
from datetime import datetime
from typing import Optional


class BaseMethod(object):
    """
    Base class for minimization methods.

    Parameters
    ----------
    oracle: BaseSmoothOracle
        Oracle corresponding to the objective function.

    x_0: np.ndarray
        Initial guess.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps
        "func_abs": terminate if f(x_k) <= eps (implicitly assumes that f* = 0)

    trace: bool
        If True, saves the history of the method during its iterations.
    """

    def __init__(self, oracle, x_0, stopping_criteria, trace):
        self.oracle = oracle
        self.x_0 = x_0.copy()
        self.trace = trace
        if stopping_criteria == 'grad_rel':
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == 'grad_abs':
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == 'func_abs':
            self.stopping_criteria = self.stopping_criteria_func_absolute
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        else:
            raise ValueError('Unknown stopping criteria type: "{}"' \
                             .format(stopping_criteria))

    def run(self, max_iter=10, max_time=1200):
        """
        Run method for a maximum of max_iter iterations and max_time seconds.

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations

        max_time: int
            Maximum running time
        """

        if not hasattr(self, 'hist'):
            self.hist = defaultdict(list)
        if not hasattr(self, 'time'):
            self.time = 0.

        self._absolute_time = datetime.now()
        try:
            for iter_count in range(max_iter):
                if self.time > max_time:
                    break
                if self.trace:
                    self._update_history()
                self.step()

                if self.stopping_criteria():
                    break
        except KeyboardInterrupt:
            print('Run interrupted at iter #{}'.format(iter_count))

        self.hist['x_star'] = self.x_k.copy()

    def _update_history(self):
        """
        Updates self.hist: saves time, function values and gradient norm.
        """

        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now
        self.hist['func'].append(self.oracle.func(self.x_k))
        self.hist['time'].append(self.time)
        if not hasattr(self, 'grad_k'):
            self.grad_k = self.oracle.grad(self.x_k)
        self.hist['grad_norm'].append(npla.norm(self.grad_k))

    def step(self):
        raise NotImplementedError('step() not implemented!')

    def stopping_criteria_grad_relative(self):
        return npla.norm(self.grad_k) ** 2 <= self.tolerance * self.grad_norm_0 ** 2

    def stopping_criteria_grad_absolute(self):
        return npla.norm(self.grad_k) ** 2 <= self.tolerance

    def stopping_criteria_func_absolute(self):
        return self.oracle.func(self.x_k) < self.tolerance

    def stopping_criteria_none(self):
        return False


class BaseDecentralizedMethod(object):
    def __init__(self, oracle_list, x_0: np.ndarray, logger: "LoggerDecentralized"):
        self.oracle_list = oracle_list
        self.x = x_0.copy()
        self.logger = logger

    def run(self, max_iter: int, max_time: Optional[float] = None):
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
            self.logger.step(self)
            self.logger.end(self)

    def _update_time(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now

    def step(self):
        raise NotImplementedError('step() not implemented!')

    def func_list(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.oracle_list[i].func(x[i]) for i in range(len(self.oracle_list))])

    def grad_list(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([self.oracle_list[i].grad(x[i]) for i in range(len(self.oracle_list))])
