import numpy as np
import scipy


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x, fval, dnum = x.astype(np.float64), func(x), np.zeros_like(x)

    grad = []
    n = x.size
    for i in range(n):
        dnum = np.zeros(n)
        dnum[i] = 1
        der = (func(x + eps * dnum) - func(x)) / eps
        grad.append(der)

    return np.array(grad)


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    from itertools import combinations_with_replacement
    x, fval, dnum = x.astype(np.float64), func(x).astype(np.float64), \
                    np.zeros((x.size, x.size), dtype=np.float64)
    n = x.size
    hess = np.zeros((n, n))

    for i, j in combinations_with_replacement(range(x.size), 2):
        dnum_i = np.zeros(x.size)
        dnum_i[i] = eps
        dnum_j = np.zeros(x.size)
        dnum_j[j] = eps
        hess[i][j] = (func(x + dnum_i + dnum_j) - func(x + dnum_i) - func(x + dnum_j) + fval) / eps**2
        hess[j][i] = hess[i][j]
    return np.array(hess)
