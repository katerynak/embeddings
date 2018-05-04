"""Fastmap.
"""

from __future__ import print_function
import numpy as np


def find_pivot_points(D):
    """Find two points (a, b) far away, with a heuristic.
    """
    idx_r = np.random.randint(D.shape[0])
    idx_a = D[idx_r, :].argmax()
    idx_b = D[idx_a, :].argmax()
    # print(D[idx_r, idx_a], D[idx_r, idx_b], D[idx_a, idx_b])
    return idx_a, idx_b


def projection(D, idx_a, idx_b):
    """Compute the projection of each obejct in 
    """
    size = D.shape[0]
    Yj = np.zeros(size)
    for i in range(size):
        Yj[i] = (D[i, idx_a]**2 + D[idx_a, idx_b]**2 - D[i, idx_b]**2) / (2.0 * D[idx_a, idx_b])

    return Yj

def update_residual_distance(D, Y):
    D2 = D * D
    size = D.shape[0]
    for u in range(size):
        for v in range(size):
            D2[u, v] = np.abs(D2[u, v] - ((Y[u, :] - Y[v, :]) ** 2).sum())

    return D2


if __name__ == '__main__':
    from scipy.spatial import distance_matrix
    
    np.random.seed(0)
    N = 1000
    d = 40
    X = np.random.uniform(size=(N, d))
    D = distance_matrix(X, X)

    k = 15
    Y = np.zeros([len(X), k])

    for i in range(k):
        print(i)
        # while True:
        idx_a, idx_b = find_pivot_points(D)
        Y[:, i] = projection(D, idx_a, idx_b)
        D2 = update_residual_distance(D, Y)
        # if D2[idx_a, idx_b] >= 0.0:
        #     break
              
        D = np.sqrt(D2)

    DY = distance_matrix(Y, Y)
    print(np.corrcoef(D.flatten(), DY.flatten()))
