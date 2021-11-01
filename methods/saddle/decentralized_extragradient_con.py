import numpy as np

from typing import List, Optional
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle, OracleLinearComb
from methods.saddle import Logger
from .base import BaseSaddleMethod
from .constraints import ConstraintsL2


class DecentralizedExtragradientCon(BaseSaddleMethod):
    """
    Decentralized Extragradient method with consensus subroutine
    (https://arxiv.org/pdf/2010.13112.pdf)

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]
        List of oracles corresponding to network nodes.

    stepsize: float
        Stepsize of Extragradient method.

    con_iters: int
        Number of iterations in consensus subroutine.

    mix_mat: np.ndarray
        Mixing matrix.

    gossip_step: float
        Step-size in consensus subroutine algorithm.

    z_0: ArrayPair
        Initial guess (similar at each node).

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """

    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            stepsize: float,
            con_iters: int,
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
        self.stepsize = stepsize
        self.con_iters = con_iters
        self.mix_mat = mix_mat
        self.gossip_step = gossip_step
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)
        self.z_list = ArrayPair(
            np.tile(z_0.x.copy(), self._num_nodes).reshape(self._num_nodes, z_0.x.shape[0]),
            np.tile(z_0.y.copy(), self._num_nodes).reshape(self._num_nodes, z_0.y.shape[0])
        )

    def step(self):
        z_half = self.z_list - self.stepsize * self.oracle_grad_list(self.z_list)
        z_half = self.acc_gossip(z_half, self.con_iters)
        self.constraints.apply_per_row(z_half)
        self.z_list = self.z_list - self.stepsize * self.oracle_grad_list(z_half)
        self.z_list = self.acc_gossip(self.z_list, self.con_iters)
        self.constraints.apply_per_row(self.z_list)
        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))

    def oracle_grad_list(self, z: ArrayPair) -> ArrayPair:
        """
        Compute oracle gradients at each computational network node.

        Parameters
        ----------
        z: ArrayPair
            Point at which the gradients are computed.

        Returns
        -------
        grad: ArrayPair
        """
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def acc_gossip(self, z: ArrayPair, n_iters: int):
        """
        Accelerated consensus subroutine.

        Parameters
        ----------
        z: ArrayPair
            Initial values at nodes.

        n_iters: int
            Number of consensus iterations.

        Returns
        -------
        z_mixed: ArrayPair
            Values at nodes after consensus subroutine.
        """
        z = z.copy()
        z_old = z.copy()
        for _ in range(n_iters):
            z_new = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
            z_new.x = (1 + self.gossip_step) * self.mix_mat.dot(z.x) - self.gossip_step * z_old.x
            z_new.y = (1 + self.gossip_step) * self.mix_mat.dot(z.y) - self.gossip_step * z_old.y
            z_old = z.copy()
            z = z_new.copy()
        return z
