import numpy as np

from typing import Callable, List, Optional
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle, OracleLinearComb
from methods.saddle import Logger, extragradient_solver
from .base import BaseSaddleMethod
from .constraints import ConstraintsL2


class SaddlePointOracleRegularizer(BaseSmoothSaddleOracle):
    """
    Wrapper around saddle point oracle with additional regularization:
    eta * F(z.x, z.y) + 1/2 ||z.x - v.x||^2 - 1/2 ||z.y - v.y||^2.

    Parameters
    ----------
    oracle: BaseSmoothSaddleOracle
        Oracle to be wrapped.

    eta: float
        Scaling parameter for the wrapped oracle.

    v: vector for computing regularization parameters.
    """

    def __init__(self, oracle: BaseSmoothSaddleOracle, eta: float, v: ArrayPair):
        self.oracle = oracle
        self.eta = eta
        self.v = v

    def func(self, z: ArrayPair) -> float:
        return self.eta * self.oracle.func(z) + 0.5 * (z.x - self.v.x).dot(z.x - self.v.x) - \
               0.5 * (z.y - self.v.y).dot(z.y - self.v.y)

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        return self.eta * self.oracle.grad_x(z) + z.x - self.v.x

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return self.eta * self.oracle.grad_y(z) + self.v.y - z.y


class SaddleSliding(BaseSaddleMethod):
    """
    Centralized gradient sliding for saddle-point problems
    (Algorithm 1 in https://arxiv.org/abs/2107.10706).

    Parameters
    ----------
    oracle_g: BaseSmoothSaddleOracle
        Oracle representing sum_{m=1}^M f_m(x, y).

    oracle_phi: BaseSmoothSaddleOracle
        Oracle representing f_1(x, y).

    stepsize_outer: float
        Stepsize in outer loop.

    stepsize_inner: float
        Stepsize in inner loop.

    inner_solver: Callable
        Solver for inner problem.

    inner_iterations: int
        Number of iterations for solving the inner subproblem.

    z_0: ArrayPair
        Initial guess.

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """
    def __init__(
            self,
            oracle_g: BaseSmoothSaddleOracle,
            oracle_phi: BaseSmoothSaddleOracle,
            stepsize_outer: float,
            stepsize_inner: float,
            inner_solver: Callable,
            inner_iterations: int,
            z_0: ArrayPair,
            logger: Optional[Logger],
            constraints: Optional[ConstraintsL2] = None
    ):
        super().__init__(oracle_g, z_0, None, None, logger)
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_solver = inner_solver
        self.inner_iterations = inner_iterations
        self.constraints = constraints

    def step(self):
        v = self.z - self.oracle_g.grad(self.z) * self.stepsize_outer
        u = self.solve_subproblem(v)
        self.z = u + self.stepsize_outer * (self.oracle_g.grad(self.z) - self.oracle_g.grad(u))

    def solve_subproblem(self, v: ArrayPair) -> ArrayPair:
        suboracle = SaddlePointOracleRegularizer(self.oracle_phi, self.stepsize_outer, v)
        return self.inner_solver(
            suboracle,
            self.stepsize_inner, v, num_iter=self.inner_iterations, constraints=self.constraints)


class DecentralizedSaddleSliding(BaseSaddleMethod):
    """
    Decentralized gradient sliding for saddle-point problems
    (Algorithm 2 in https://arxiv.org/abs/2107.10706).

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]
        List of oracles corresponding to network nodes.

    stepsize_outer: float
        Stepsize in outer loop.

    stepsize_inner: float
        Stepsize in inner loop.

    inner_solver: Callable
        Solver for inner problem.

    inner_iterations: int
        Number of iterations for solving the inner subproblem.

    con_iters_grad: int
        Number of consensus iterations for mixing the gradients.

    con_iters_pt: int
        Number of consensus iterations for mixing the points.

    mix_mat: np.ndarray
        Mixing matrix

    gossip_step: float
        Stepsize for consensus subroutine.

    z_0: ArrayPair
        Initial guess.

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """
    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            stepsize_outer: float,
            stepsize_inner: float,
            inner_iterations: int,
            con_iters_grad: int,
            con_iters_pt: int,
            mix_mat: np.ndarray,
            gossip_step: float,
            z_0: ArrayPair,
            logger=Optional[Logger],
            constraints: Optional[ConstraintsL2] = None
    ):
        self._num_nodes = len(oracles)
        oracle_sum = OracleLinearComb(oracles, [1 / self._num_nodes] * self._num_nodes)
        super().__init__(oracle_sum, z_0, None, None, logger)
        self.oracle_list = oracles
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_iterations = inner_iterations
        self.con_iters_grad = con_iters_grad
        self.con_iters_pt = con_iters_pt
        self.mix_mat = mix_mat
        self.gossip_step = gossip_step
        self.constraints = constraints
        self.z_list = ArrayPair(
            np.tile(z_0.x.copy(), self._num_nodes).reshape(self._num_nodes, z_0.x.shape[0]),
            np.tile(z_0.y.copy(), self._num_nodes).reshape(self._num_nodes, z_0.y.shape[0])
        )

    def step(self):
        grad_list_z = self.oracle_grad_list(self.z_list)
        grad_av_z = self.acc_gossip(grad_list_z, self.con_iters_grad)
        m = np.random.randint(0, self._num_nodes, size=1)[0]
        grad_z_m = ArrayPair(grad_list_z.x[m], grad_list_z.y[m])
        z = ArrayPair(self.z_list.x[m], self.z_list.y[m])
        grad_av_z_m = ArrayPair(grad_av_z.x[m], grad_av_z.y[m])
        v = z - self.stepsize_outer * (grad_av_z_m - grad_z_m)
        u = self.solve_subproblem(m, v)

        u_list = ArrayPair(
            np.zeros((self._num_nodes, self.z.x.shape[0])),
            np.zeros((self._num_nodes, self.z.y.shape[0]))
        )
        u_list.x[m] = u.x
        u_list.y[m] = u.y
        u_list = self._num_nodes * self.acc_gossip(u_list, self.con_iters_pt)

        grad_av_u = self.acc_gossip(self.oracle_grad_list(u_list), self.con_iters_grad)
        grad_av_u_m = ArrayPair(grad_av_u.x[m], grad_av_u.y[m])
        z = u + self.stepsize_outer * (grad_av_z_m - grad_z_m - grad_av_u_m +
                                       self.oracle_list[m].grad(u))
        z_list = ArrayPair(
            np.zeros((self._num_nodes, self.z.x.shape[0])),
            np.zeros((self._num_nodes, self.z.y.shape[0]))
        )
        z_list.x[m] = z.x
        z_list.y[m] = z.y
        z_list = self._num_nodes * self.acc_gossip(z_list, self.con_iters_pt)
        for i in range(len(z_list.x)):
            z = ArrayPair(z_list.x[i], z_list.y[i])
            if self.constraints is not None:
                z_constr = self.constraints.apply(z)
            else:
                z_constr = z
            self.z_list.x[i] = z_constr.x
            self.z_list.y[i] = z_constr.y

        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))

    def solve_subproblem(self, m: int, v: ArrayPair):
        suboracle = SaddlePointOracleRegularizer(self.oracle_list[m], self.stepsize_outer, v)
        return extragradient_solver(suboracle,
                                    self.stepsize_inner, v, num_iter=self.inner_iterations,
                                    constraints=self.constraints)

    def oracle_grad_list(self, z: ArrayPair):
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def acc_gossip(self, z: ArrayPair, n_iters: int):
        z = z.copy()
        z_old = z.copy()
        for _ in range(n_iters):
            z_new = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
            z_new.x = (1 + self.gossip_step) * self.mix_mat.dot(z.x) - self.gossip_step * z_old.x
            z_new.y = (1 + self.gossip_step) * self.mix_mat.dot(z.y) - self.gossip_step * z_old.y
            z_old = z.copy()
            z = z_new.copy()
        return z
