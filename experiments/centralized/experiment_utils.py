import sys

sys.path.append("../")

import numpy as np
import scipy.linalg as sla
import numpy.linalg as npla
from typing import Optional
from methods.saddle import ConstraintsL2
from oracles.saddle import create_robust_linear_oracle, OracleLinearComb
from methods.saddle import SaddleSliding, extragradient_solver, Logger, Extragradient
from oracles.saddle import ArrayPair, BaseSmoothSaddleOracle
from sklearn.model_selection import train_test_split


class SaddleSlidingRunner(object):
    def __init__(self, oracle_g, oracle_phi, logger: Logger, L: float, mu: float, delta: float,
                 r_x: float, r_y: float):
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.L = L
        self.mu = mu
        self.delta = delta
        self.logger = logger
        self.r_x = r_x
        self.r_y = r_y

    def create_method(self, z_0: ArrayPair):
        self.eta = min(1. / (2 * self.delta), 1 / (6 * self.mu))
        self.e = min(0.25, 1 / (64 / (self.eta * self.mu) + 64 * self.eta * self.L ** 2 / self.mu))
        self.eta_inner = 0.5 / (self.eta * self.L + 1)
        self.T_inner = int((1 + self.eta * self.L) * np.log10(1 / self.e))

        self.method = SaddleSliding(
            oracle_g=self.oracle_g,
            oracle_phi=self.oracle_phi,
            stepsize_outer=self.eta,
            stepsize_inner=self.eta_inner,
            inner_solver=extragradient_solver,
            inner_iterations=self.T_inner,
            z_0=z_0,
            logger=self.logger,
            constraints=ConstraintsL2(self.r_x, self.r_y)
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)


def gen_matrices(n_one: int, d: int, mean: float, std: float, noise: float,
                 num_summands: int, seed=0):
    np.random.seed(seed)
    A_one = mean + std * np.random.randn(n_one, d)
    A = np.tile(A_one.T, num_summands).T
    A[n_one:] += noise * np.random.randn(n_one * (num_summands - 1), d)

    b_one = mean + std * np.random.randn(n_one)
    b = np.tile(b_one, num_summands)
    b[n_one:] += noise * np.random.randn(n_one * (num_summands - 1))

    return A, A_one, b, b_one


def gen_oracles_for_sliding(A, A_one, b, b_one, num_summands: int, regcoef_x: float,
                            regcoef_y: float):
    oracle_sum = create_robust_linear_oracle(A, b, num_summands * regcoef_x,
                                             num_summands * regcoef_y, normed=False)
    oracle_sum = OracleLinearComb([oracle_sum], [1. / num_summands])
    oracle_phi = create_robust_linear_oracle(A_one, b_one, regcoef_x, regcoef_y, normed=False)
    oracle_g = OracleLinearComb([oracle_sum, oracle_phi], [1., -1.])

    return oracle_sum, oracle_phi, oracle_g


def compute_L_delta_mu(A, A_one, b, r_x: float, r_y: float, regcoef_x: float, regcoef_y: float,
                       num_summands: int):
    lam = sla.svd(A.T.dot(A))[1].max()
    A_dot_one = npla.norm(A.sum(axis=0))
    L_xx = lam + 2 * r_y * A_dot_one + r_y ** 2
    L_yy = A.shape[0] * r_x ** 2 + regcoef_y ** 2
    L_xy = 2 * A.shape[0] * r_x * r_y + 2 * A_dot_one + b.sum()
    L = 2 * max(L_xx, L_yy, L_xy) / num_summands

    lam_delta = sla.svd(A.T.dot(A) / num_summands - A_one.T.dot(A_one))[1].max()
    A_g_dot_one = np.abs(A_dot_one / num_summands - npla.norm(A_one.sum(axis=0)))
    delta_xx = lam_delta + 2 * r_y * A_g_dot_one
    delta_yy = 0
    delta_xy = 2 * A_g_dot_one
    delta = 2 * max(np.abs(delta_xx), np.abs(delta_yy), np.abs(delta_xy))

    mu = min(regcoef_x, regcoef_y)

    return L, delta, mu


def solve_with_extragradient(
        oracle: BaseSmoothSaddleOracle, stepsize: float, r_x: float, r_y: float,
        z_0: ArrayPair, tolerance: Optional[float], num_iter: int, max_time: Optional[float],
        z_true: Optional[ArrayPair] = None) -> Logger:
    logger_extragradient = Logger(z_true)
    extragradient = Extragradient(
        oracle=oracle,
        stepsize=stepsize,
        z_0=z_0,
        tolerance=tolerance,
        stopping_criteria='grad_abs',
        constraints=ConstraintsL2(r_x, r_y),
        logger=logger_extragradient
    )
    extragradient.run(max_iter=num_iter, max_time=max_time)
    z_true = logger_extragradient.z_star
    print('steps performed: ', logger_extragradient.num_steps)
    print('grad norm: {:.4e}'.format(oracle.grad(z_true).norm()))
    print()

    return logger_extragradient


def run_experiment(n_one: int, d: int, mat_mean: float, mat_std: float, noise: float,
                   num_summands: int, regcoef_x: float, regcoef_y: float, r_x: float, r_y: float,
                   num_iter_solution: int, max_time_solution: int, tolerance_solution: float,
                   num_iter_experiment: int):
    A, A_one, b, b_one = gen_matrices(n_one, d, mean=mat_mean, std=mat_std, noise=noise,
                                      num_summands=num_summands, seed=0)
    oracle_sum, oracle_phi, oracle_g = gen_oracles_for_sliding(
        A, A_one, b, b_one, num_summands, regcoef_x, regcoef_y)

    L, delta, mu = compute_L_delta_mu(A, A_one, b, r_x, r_y, regcoef_x, regcoef_y, num_summands)
    print('L = {:.3f}, delta = {:.3f}, mu = {:.3f}'.format(L, delta, mu))

    z_0 = ArrayPair.zeros(d)

    print('Solving with extragradient...')
    z_true = solve_with_extragradient(
        oracle_sum, 1. / L, r_x, r_y, z_0, tolerance=tolerance_solution,
        num_iter=num_iter_solution, max_time=max_time_solution).z_star

    print('Running extragradient again...')
    logger_extragradient_again = solve_with_extragradient(
        oracle_sum, 1. / L, r_x, r_y, z_0, tolerance=0, max_time=None,
        num_iter=num_iter_experiment, z_true=z_true)

    print('Running Sliding...')
    runner = SaddleSlidingRunner(
        oracle_g=oracle_g,
        oracle_phi=oracle_phi,
        logger=Logger(z_true=z_true),
        L=L,
        mu=mu,
        delta=delta,
        r_x=r_x,
        r_y=r_y
    )
    runner.create_method(ArrayPair.zeros(A.shape[1]))
    print('T_inner = {}'.format(runner.method.inner_iterations))
    print()
    runner.run(num_iter_experiment)

    return runner, logger_extragradient_again


def run_experiment_real(A, b, num_summands: int, regcoef_x: float, regcoef_y: float, r_x: float,
                        r_y: float, num_iter_experiment: int, z_true: ArrayPair):
    _, A_one, _, b_one = train_test_split(A, b, test_size=1. / num_summands, random_state=0,
                                          shuffle=True)
    d = A.shape[1]
    oracle_sum, oracle_phi, oracle_g = gen_oracles_for_sliding(
        A, A_one, b, b_one, num_summands, regcoef_x, regcoef_y)

    L, delta, mu = compute_L_delta_mu(A, A_one, b, r_x, r_y, regcoef_x, regcoef_y, num_summands)
    print('L = {:.3f}, delta = {:.3f}, mu = {:.3f}'.format(L, delta, mu))

    z_0 = ArrayPair.zeros(d)

    print('Running extragradient...')
    logger_extragradient_again = solve_with_extragradient(
        oracle_sum, 1. / L, r_x, r_y, z_0, tolerance=0, max_time=None,
        num_iter=num_iter_experiment, z_true=z_true)

    print('Running Sliding...')
    runner = SaddleSlidingRunner(
        oracle_g=oracle_g,
        oracle_phi=oracle_phi,
        logger=Logger(z_true=z_true),
        L=L,
        mu=mu,
        delta=delta,
        r_x=r_x,
        r_y=r_y
    )
    runner.create_method(ArrayPair.zeros(A.shape[1]))
    print('T_inner = {}'.format(runner.method.inner_iterations))
    print()
    runner.run(num_iter_experiment)

    return runner, logger_extragradient_again


def create_oracles_for_sliding_real(A, b, regcoef_x, regcoef_y, num_summands, seed=0):
    oracle_sum = create_robust_linear_oracle(A, b, regcoef_x, regcoef_y, normed=False)
    oracle_sum = OracleLinearComb([oracle_sum], [1. / num_summands])
    _, A_small, _, b_small = train_test_split(A, b, test_size=1. / num_summands, random_state=seed,
                                              shuffle=True)
    oracle_phi = create_robust_linear_oracle(A_small, b_small, regcoef_x, regcoef_y, normed=False)
    oracle_g = OracleLinearComb([oracle_sum, oracle_phi], [1., -1.])

    return oracle_sum, oracle_phi, oracle_g
