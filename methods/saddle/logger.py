from typing import Optional
from .base import ArrayPair


class Logger(object):
    """
    Instrument for saving the method history during its iterations.

    Parameters
    ----------
    z_true: Optional[ArrayPair]
        Exact solution of the problem. If specified, logs distance to solution.
    """

    def __init__(self, z_true: Optional[ArrayPair] = None):
        self.func = []
        self.time = []
        self.z_true = z_true
        if z_true is not None:
            self.dist_to_opt = []

    def start(self, method: "BaseSaddleMethod"):
        pass

    def step(self, method: "BaseSaddleMethod"):
        self.func.append(method.oracle.func(method.z))
        self.time.append(method.time)
        if self.z_true is not None:
            self.dist_to_opt.append((method.z - self.z_true).dot(method.z - self.z_true))

    def end(self, method: "BaseSaddleMethod"):
        self.z_star = method.z.copy()

    @property
    def num_steps(self):
        return len(self.func)


class LoggerDecentralized(Logger):
    """
    Instrument for saving method history during its iterations for decentralized methods.
    Additionally logs distance to consensus.

    Parameters
    ----------
    z_true: Optional[ArrayPair]
        Exact solution of the problem. If specified, logs distance to solution.
    """

    def __init__(self, z_true: Optional[ArrayPair] = None):
        super().__init__(z_true)
        self.dist_to_con = []

    def step(self, method: "BaseSaddleMethod"):
        super().step(method)
        self.dist_to_con.append(
            ((method.z_list.x - method.z_list.x.mean(axis=0)) ** 2).sum() /
            method.z_list.x.shape[0] +
            ((method.z_list.x - method.z_list.x.mean(axis=0)) ** 2).sum() /
            method.z_list.y.shape[0]
        )
