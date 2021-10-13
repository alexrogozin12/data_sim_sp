import numpy as np


def gen_mix_mat(n: int) -> np.ndarray:
    """
    Metropolis weights for ring graph over n nodes
    """

    mat = np.zeros((n, n))
    ids = np.arange(n)
    mat[ids, ids] = 1 / 3
    mat[ids[:-1], ids[1:]] = 1 / 3
    mat[ids[1:], ids[:-1]] = 1 / 3
    mat[0, n - 1] = 1 / 3
    mat[n - 1, 0] = 1 / 3
    return mat


def compute_lam_2(mat):
    eigs = np.sort(np.linalg.eigvals(mat))
    return max(np.abs(eigs[0]), np.abs(eigs[-2]))
