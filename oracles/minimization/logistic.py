import numpy as np
from scipy.special import expit
from .base import BaseSmoothOracle


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Logistic regression oracle with L2 regularization.

    Parameters
    ----------
    matvec_Ax: Callable
        Multiplication by feature matrix A.

    matvec_ATx: Callable
        Multiplication by A.T.

    matmat_ATsA: Callable
        Computes A.T diag(s) A, where diag(s) is a diagonal matrix with values of s on diagonal.

    b: np.ndarray
        Vector of labels.

    regcoef: float
        Regularization coefficient.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = self.b.shape[0]
        degree1 = np.zeros(m)
        degree2 = self.matvec_Ax(x)
        degree2 = np.multiply(-self.b, degree2)
        summ = np.sum(np.logaddexp(degree1, degree2))

        return summ / m + self.regcoef / 2 * np.dot(x, x)

    def grad(self, x):
        m = self.b.shape[0]
        degrees = -np.multiply(self.b, self.matvec_Ax(x))
        self.sigmas = expit(degrees)
        return -1 / m * self.matvec_ATx(np.multiply(self.sigmas, self.b)) + self.regcoef * x

    def hess(self, x):
        m = self.b.shape[0]
        n = x.size
        degrees = -np.multiply(self.b, self.matvec_Ax(x))
        sigmas = expit(degrees)
        diagonal = np.multiply(self.b ** 2, sigmas)
        diagonal = np.multiply(diagonal, 1 - sigmas)
        return np.array(1 / m * self.matmat_ATsA(diagonal) + self.regcoef * np.eye(n))

    def hess_mat_prod(self, x, S):
        m = self.b.shape[0]
        n = x.size
        diagonal = np.multiply(self.b ** 2, self.sigmas)
        diagonal = np.multiply(diagonal, 1 - self.sigmas)
        AS = self.matvec_Ax(S)
        if isinstance(AS, np.ndarray):
            res = np.multiply(diagonal.reshape(diagonal.shape[0], 1), AS)
            return np.array(1 / m * self.matvec_ATx(res)) + self.regcoef * S
        else:
            res = AS.multiply(diagonal.reshape(diagonal.shape[0], 1))
            return 1 / m * self.matvec_ATx(res) + self.regcoef * S


def create_log_reg_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().transpose() * x

    def matmat_ATsA(s, mat=A):
        if isinstance(mat, np.ndarray):
            return mat.T.dot(np.multiply(mat, s.reshape(len(s), 1)))
        A = mat.tocsr()
        sA = A.multiply(s.reshape(len(s), 1))
        return A.transpose() * sA

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
