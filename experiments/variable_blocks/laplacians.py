import numpy as np
import scipy.linalg as sla
import random


def check_laplacian(mat: np.ndarray):
    return np.all(mat == mat.T) and np.all(mat.sum(axis=0) == 0)


def clique_with_two_similar_edges(k):
    lapl = np.zeros((k + 2, k + 2))
    lapl -= np.ones((k + 2, k + 2))
    np.fill_diagonal(lapl, k - 1)
    lapl[3:, 0:2] = 0
    lapl[0:2 , 3:] = 0
    lapl[0, 0] = 1
    lapl[1, 1] = 1
    lapl[2, 2] = k + 1
    lapl[0, 1] = 0
    lapl[1, 0] = 0

    return lapl


# big laplacian
# для того, чтобы раздуть матрицу, нужно знать те индексы, где есть наши вершины
# small = k+2 x k+2
def big_laplacian(start, size_clique, num_of_cliques):
    total_size = size_clique * num_of_cliques
    W = np.zeros((total_size, total_size))
    if start % size_clique != 0:
        raise ValueError('Error! Start point is wrong - need a division by size of clique')
    else:
        W[start:start + size_clique, start:start + size_clique] = clique_with_two_similar_edges(size_clique)[2:, 2:]
        W[start, (start + size_clique) % total_size] = -1
        W[start, start - size_clique] = -1
        
        W[(start + size_clique) % total_size, (start + size_clique) % total_size] = 1
        W[start - size_clique, start - size_clique] = 1
        
        W[(start + size_clique) % total_size, start] = -1
        W[start - size_clique, start] = -1
        return W


def special_supermatrix(n, k):
    WW = np.zeros((k * n**2, k * n**2))
    for i in range(n):
        temp = big_laplacian(i * k, k, n)
        basis = np.zeros((n, n))
        basis[i, i] = 1
        temp = np.kron(temp, basis)
        WW += temp
    return WW


def creation(n, k, matrix_dimension=1, seed=None):
    # n - number of cliques = number of coordinates in our case
    # k - size of clique
    random.seed(seed)
    data = list()
    for i in range(n * k):
        A = np.zeros((matrix_dimension, n))
        b = np.zeros((matrix_dimension))
        if i % k == 0:
            index = i // k
            for j in range(matrix_dimension):
                A[j, index - 1] = random.randint(-5, 5)    
                A[j, index] = random.randint(-5, 5)
                A[j, (index + 1) % n] = random.randint(-5, 5)
        else:
            for j in range(matrix_dimension):
                A[j, index] = random.randint(-5, 5) 
        for j in range(matrix_dimension):
            b[j] = random.randint(-5, 5)
        data.append([A, b])
        
    return data


def usual_laplacian(n, k):
    L = np.zeros((n * k, n * k))
    for i in range(n):
        L[i * k:(i + 1) * k, i * k:(i + 1) * k] = clique_with_two_similar_edges(k)[2:, 2:]
        L[i * k, (i * k + k) % (n * k)] = -1
        L[i * k, i * k - k] = -1
        L[(i * k + k) % (n * k), i * k] = -1
        L[i * k - k, i * k] = -1
    return L
