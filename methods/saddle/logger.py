# from .base import BaseSaddleMethod


class Logger(object):
    def __init__(self):
        self.func = []
        self.time = []

    def start(self, method: "BaseSaddleMethod"):
        pass

    def step(self, method: "BaseSaddleMethod"):
        self.func.append(method.oracle.func(method.z))
        self.time.append(method.time)

    def end(self, method: "BaseSaddleMethod"):
        self.z_star = method.z.copy()
