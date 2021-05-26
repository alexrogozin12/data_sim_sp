from typing import Optional
from .base import ArrayPair


class Logger(object):
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
