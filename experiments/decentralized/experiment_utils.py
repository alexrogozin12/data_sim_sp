import sys

sys.path.append("../../")

import networkx as nx
import numpy as np
import scipy.linalg as sla
import numpy.linalg as npla
from sklearn.model_selection import StratifiedKFold
from typing import Iterable, Optional, Tuple
from methods.saddle import ConstraintsL2
from oracles.saddle import create_robust_linear_oracle, OracleLinearComb, ArrayPair, \
    BaseSmoothSaddleOracle
from methods.saddle import Logger, Extragradient, LoggerDecentralized
from methods.runners import DecentralizedExtragradientGTRunner, DecentralizedSaddleSlidingRunner, \
    DecentralizedExtragradientConRunner


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


def compute_robust_linear_normed_L(A: np.ndarray, b: np.ndarray, r_x: float, r_y: float,
                                   regcoef_x: float, regcoef_y: float) -> float:
    """
    Compute Lipschitz constant for Robust Linear Regression function
    (normed, i.e. 1 / num_samples * (...) + regcoef_x/2 ||x||^2 - regcoef_y/2 ||y||^2).
    """

    lam = sla.svd(A.T.dot(A))[1].max() / A.shape[0]
    A_dot_one = npla.norm(A.mean(axis=0))
    L_xx = lam + 2 * r_y * A_dot_one + r_y ** 2 + regcoef_x
    L_yy = r_x ** 2 + regcoef_y
    L_xy = 2 * r_x * r_y + 2 * A_dot_one * r_x + b.mean()
    L = 2 * max(L_xx, L_yy, L_xy)
    return L


def compute_robust_linear_normed_delta(A: np.ndarray, Am: np.ndarray, r_x: float, r_y: float,
                                       num_parts: int) -> float:
    """
    Compute similarity coefficient delta between whole dataset A and its part Am.
    """

    lam = sla.svd(A.T.dot(A) / num_parts - Am.T.dot(Am))[1].max() / Am.shape[0]
    A_dot_one = npla.norm(A.mean(axis=0) - Am.mean(axis=0))
    delta_xx = lam + 2 * r_y * A_dot_one
    delta_yy = 0
    delta_xy = 2 * r_x * A_dot_one
    delta = 2 * max(delta_xx, delta_yy, delta_xy)
    return delta


def compute_robust_linear_normed_L_delta_mu(
        A: np.ndarray, b: np.ndarray, part_sizes: Optional[Iterable], n_parts: Optional[int],
        r_x: float, r_y: float, regcoef_x: float, regcoef_y: float) -> Tuple[float, float, float]:

    if part_sizes is None and n_parts is None:
        raise ValueError('Please specify either part_sizes or n_parts')
    if part_sizes is not None and n_parts is not None:
        raise ValueError('Only one of part_sizes and n_parts should be specified')
    if part_sizes is None:
        size = A.shape[0] // n_parts
        part_sizes = [size] * n_parts
    if n_parts is None:
        n_parts = len(part_sizes)

    L_list = np.empty(n_parts, dtype=np.float32)
    delta_list = np.empty(n_parts, dtype=np.float32)
    start = 0
    for i, size in enumerate(part_sizes):
        L_list[i] = compute_robust_linear_normed_L(
            A[start:start+size], b[start:start+size], r_x, r_y, regcoef_x, regcoef_y)
        delta_list[i] = compute_robust_linear_normed_delta(
            A, A[start:start+size], r_x, r_y, n_parts)
        start += size
    return np.max(L_list), np.max(delta_list), min(regcoef_x, regcoef_y)


def gen_matrices_decentralized(num_matrices: int, l: int, d: int, mean: float, std: float,
                               noise: float, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    num_matrices: int
        Number of generated matrices

    l: int
        Number of rows in each matrix

    d: int
        Dimension

    mean: float
        Mean of random normal distribution to generate matrix entries

    std: float
        Standard deviation of normal distribution to generate matrix entries

    noise: float
        Amplitude of random noise added to each matrix held by an agent

    Returns
    -------
    A: np.ndarray, shape (num_matrices * l, d)
        Stacked array of matrices

    b: np.ndarray, shape (num_summands * l,)
        Stacked array of vectors
    """

    np.random.seed(seed)
    A_one = mean + std * np.random.randn(l, d)
    A = np.tile(A_one.T, num_matrices).T
    A[l:] += noise * np.random.randn(l * (num_matrices - 1), d)

    b_one = mean + std * np.random.randn(l)
    b = np.tile(b_one, num_matrices)
    b[l:] += noise * np.random.randn(l * (num_matrices - 1))

    return A, b


def line_adj_mat(n: int):
    """
    Adjacency matrix of a line graph over n nodes

    Parameters
    ----------
    n: int
        Number of nodes
    """

    mat = np.zeros((n, n), dtype=np.int32)
    ids = np.arange(n)
    mat[ids[:-1], ids[1:]] = 1
    mat[ids[1:], ids[:-1]] = 1
    return mat


def ring_adj_mat(n: int):
    """
    Adjacency matrix of a ring graph over n nodes

    Parameters
    ----------
    n: int
        Number of nodes
    """

    mat = line_adj_mat(n)
    mat[0, n - 1] = 1
    mat[n - 1, 0] = 1
    return mat


def grid_adj_mat(n: int, m: int):
    """
    Adjacency matrix of a rectangle grid graph over n x m nodes

    Parameters
    ----------
    n: int
        Vertical size of grid

    m: int
        Horizontal size of grid

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.grid_2d_graph(n, m)
    return np.array(nx.linalg.graphmatrix.adjacency_matrix(graph).todense()).astype(np.float32)


def star_adj_mat(n: int):
    """
    Adjacency matrix of a start graph over n nodes (1 center and n - 1 leaves)

    Parameters
    ----------
    n: int
        Number of vertices

    Returns
    -------
    mat: np.ndarray
    """

    graph = nx.generators.star_graph(n - 1)
    return np.array(nx.linalg.graphmatrix.adjacency_matrix(graph).todense()).astype(np.float32)


def metropolis_weights(adj_mat: np.ndarray):
    """
    Computes Metropolis weights for a graph with a given adjacency matrix

    Parameters
    ----------
    adj_mat: np.ndarray
        Adjacency matrix
    """

    weights = adj_mat / (1 + np.maximum(
        adj_mat.sum(1, keepdims=True), adj_mat.sum(0, keepdims=True)))
    ids = np.arange(adj_mat.shape[0])
    weights[ids, ids] = 1 - np.sum(weights, axis=0)
    return weights


def sliding_comm_per_iter(runner: DecentralizedSaddleSlidingRunner):
    return 2 * (runner.con_iters_grad + runner.con_iters_pt)


def run_experiment(n_one: int, d: int, mat_mean: float, mat_std: float, noise: float,
                   num_nodes: int, mix_mat: np.ndarray, regcoef_x: float, regcoef_y: float,
                   r_x: float, r_y: float, eps: float, num_iter_solution: int,
                   max_time_solution: int, tolerance_solution: float, comm_budget_experiment: int,
                   seed: int = 0):

    A, b = gen_matrices_decentralized(
        num_matrices=num_nodes,
        l=n_one,
        d=d,
        mean=mat_mean,
        std=mat_std,
        noise=noise,
        seed=seed
    )
    oracles = [create_robust_linear_oracle(A[i:i+n_one], b[i:i+n_one], regcoef_x, regcoef_y,
                                           normed=True) for i in range(0, n_one * num_nodes, n_one)]

    L, delta, mu = compute_robust_linear_normed_L_delta_mu(A, b, None, num_nodes, r_x, r_y,
                                                           regcoef_x, regcoef_y)
    print('L = {:.3f}, delta = {:.3f}, mu = {:.3f}'.format(L, delta, mu))

    z_0 = ArrayPair.zeros(d)

    oracle_mean = OracleLinearComb(oracles, [1 / num_nodes] * num_nodes)
    print('Solving with extragradient...')
    z_true = solve_with_extragradient(
        oracle_mean, 1. / L, r_x, r_y, z_0, tolerance=tolerance_solution,
        num_iter=num_iter_solution, max_time=max_time_solution).z_star

    print('Running decentralized extragradient...')
    extragrad = DecentralizedExtragradientGTRunner(oracles, L, mu, mu, mix_mat,
                                                   LoggerDecentralized(z_true))
    extragrad.compute_method_params()
    extragrad.create_method(z_0)
    extragrad.run(max_iter=comm_budget_experiment // 2)
    extragrad.logger.comm_budget = comm_budget_experiment

    print('Running decentralized sliding...')
    sliding = DecentralizedSaddleSlidingRunner(oracles, L, mu, delta, mix_mat, r_x, r_y, eps,
                                               LoggerDecentralized(z_true))
    sliding.compute_method_params()
    sliding.create_method(z_0)

    print('H_0 = {}, H_1 = {}, T_subproblem = {}'
          .format(sliding.con_iters_grad, sliding.con_iters_pt, sliding.method.inner_iterations))
    sliding.run(max_iter=comm_budget_experiment // sliding_comm_per_iter(sliding))
    sliding.logger.comm_per_iter = sliding_comm_per_iter(sliding.method)
    sliding.logger.comm_budget = comm_budget_experiment
    print('Done')

    return extragrad, sliding, z_true


def run_extragrad_con(n_one: int, d: int, mat_mean: float, mat_std: float, noise: float,
                      num_nodes: int, mix_mat: np.ndarray, regcoef_x: float, regcoef_y: float,
                      r_x: float, r_y: float, eps: float,
                      comm_budget_experiment: int, z_true: ArrayPair, seed: int = 0):

    A, b = gen_matrices_decentralized(
        num_matrices=num_nodes,
        l=n_one,
        d=d,
        mean=mat_mean,
        std=mat_std,
        noise=noise,
        seed=seed
    )

    oracles = [create_robust_linear_oracle(A[i:i + n_one], b[i:i + n_one], regcoef_x, regcoef_y,
                                           normed=True) for i in range(0, n_one * num_nodes, n_one)]

    L, _, mu = compute_robust_linear_normed_L_delta_mu(A, b, None, num_nodes, r_x, r_y,
                                                       regcoef_x, regcoef_y)

    z_0 = ArrayPair.zeros(d)

    print('Running decentralized extragradient-con...')
    runner = DecentralizedExtragradientConRunner(oracles, L, mu, mix_mat, r_x, r_y, eps,
                                                 LoggerDecentralized(z_true))
    runner.compute_method_params()
    runner.create_method(z_0)
    print('T_consensus = {}'.format(runner.method.con_iters))

    runner.logger.comm_per_iter = 2 * runner.method.con_iters
    runner.run(max_iter=comm_budget_experiment // runner.logger.comm_per_iter)
    runner.logger.comm_budget_experiment = comm_budget_experiment
    print('Done')

    return runner


def solve_with_extragradient_real_data(
        A: np.ndarray, b: np.ndarray, regcoef_x: float, regcoef_y, r_x: float, r_y: float,
        num_iter: int, max_time: float, tolerance: float) -> ArrayPair:

    L = compute_robust_linear_normed_L(A, b, r_x, r_y, regcoef_x, regcoef_y)
    z_0 = ArrayPair.zeros(A.shape[1])
    oracle = create_robust_linear_oracle(A, b, regcoef_x, regcoef_y, normed=True)
    print('Solving with extragradient...')
    print('L = {:.3f}'.format(L))
    z_true = solve_with_extragradient(
        oracle, 1. / L, r_x, r_y, z_0, tolerance, num_iter, max_time).z_star
    print()
    return z_true


def run_experiment_real_data(
        A: np.ndarray, b: np.ndarray,
        num_nodes: int, mix_mat: np.ndarray, regcoef_x: float, regcoef_y: float,
        r_x: float, r_y: float, eps: float, comm_budget_experiment: int, z_true: ArrayPair):

    oracles = []
    part_sizes = np.empty(num_nodes, dtype=np.int32)
    part_sizes[:] = A.shape[0] // num_nodes
    part_sizes[:A.shape[0] - part_sizes.sum()] += 1
    start = 0
    for part_size in part_sizes:
        A_small = A[start: start + part_size]
        b_small = b[start: start + part_size]
        oracles.append(create_robust_linear_oracle(
            A_small, b_small, regcoef_x, regcoef_y, normed=True))
        start += part_size

    L, delta, mu = compute_robust_linear_normed_L_delta_mu(
        A, b, part_sizes, None, r_x, r_y, regcoef_x, regcoef_y)
    print('L = {:.3f}, delta = {:.3f}, mu = {:.3f}'.format(L, delta, mu))

    z_0 = ArrayPair.zeros(A.shape[1])

    print('Running decentralized extragradient...')
    extragrad = DecentralizedExtragradientGTRunner(oracles, L, mu, mu, mix_mat,
                                                   LoggerDecentralized(z_true))
    extragrad.compute_method_params()
    extragrad.create_method(z_0)
    extragrad.run(max_iter=comm_budget_experiment // 2)
    extragrad.logger.comm_budget = comm_budget_experiment
    print()

    print('Running decentralized extragradient-con...')
    extragrad_con = DecentralizedExtragradientConRunner(oracles, L, mu, mix_mat, r_x, r_y, eps,
                                                        LoggerDecentralized(z_true))
    extragrad_con.compute_method_params()
    extragrad_con.create_method(z_0)
    print('T_consensus = {}'.format(extragrad_con.method.con_iters))
    print()

    extragrad_con.logger.comm_per_iter = 2 * extragrad_con.method.con_iters
    extragrad_con.run(max_iter=comm_budget_experiment // extragrad_con.logger.comm_per_iter)
    extragrad_con.logger.comm_budget_experiment = comm_budget_experiment

    print('Running decentralized sliding...')
    sliding = DecentralizedSaddleSlidingRunner(oracles, L, mu, delta, mix_mat, r_x, r_y, eps,
                                               LoggerDecentralized(z_true))
    sliding.compute_method_params()
    sliding.create_method(z_0)

    print('H_0 = {}, H_1 = {}, T_subproblem = {}'
          .format(sliding.con_iters_grad, sliding.con_iters_pt, sliding.method.inner_iterations))
    sliding.run(max_iter=comm_budget_experiment // sliding_comm_per_iter(sliding))
    sliding.logger.comm_per_iter = sliding_comm_per_iter(sliding.method)
    sliding.logger.comm_budget = comm_budget_experiment
    print('Done')
    print()

    return extragrad, extragrad_con, sliding
