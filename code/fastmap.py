"""Fastmap.
"""

from __future__ import print_function
import numpy as np


def find_pivot_points(D):
    """Find two points (a, b) far away, with a heuristic, from the
    distance matrix D.
    """
    idx_r = np.random.randint(D.shape[0])
    idx_a = D[idx_r, :].argmax()
    idx_b = D[idx_a, :].argmax()
    # print(D[idx_r, idx_a], D[idx_r, idx_b], D[idx_a, idx_b])
    return idx_a, idx_b



def projection(D, idx_a, idx_b):
    """Compute the projection of each obejct.
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
            # D2[u, v] = D2[u, v] - ((Y[u, :] - Y[v, :]) ** 2).sum()  # this is the original one
            D2[u, v] = np.abs(D2[u, v] - ((Y[u, :] - Y[v, :]) ** 2).sum())  # rigged

    return D2


def recursive_distance2(X, idx, distance, Y):
    tmp1 = distance(X, X[idx])
    tmp1 *= tmp1
    tmp2 = (Y - Y[idx])
    tmp2 *= tmp2
    return tmp1.squeeze() - tmp2.sum(1)


def find_pivot_points_from_X_fast(X, distance, Y):
    """Find two points (a, b) far away, with a heuristic, from the objects
    X given the distance function.
    """
    idx_r = np.random.randint(len(X))
    idx_a = recursive_distance2(X, idx_r, distance, Y).argmax()
    idx_b = recursive_distance2(X, idx_a, distance, Y).argmax()
    return idx_a, idx_b


def find_pivot_points_scalable(X, distance, Y, k, permutation=True, c=2.0):
    """Find two points (a, b) far away, with a heuristic, from the objects
    X given the distance function, assuming objects as clustered in k
    clusters and subsampling X accordingly.
    """
    size = int(max(1, np.ceil(c * k * np.log(k))))
    if permutation:
        idx = np.random.permutation(len(X))[:size]
    else:
        idx = range(size)

    tmp_a, tmp_b = find_pivot_points_from_X_fast(X[idx], distance, Y[idx])
    return idx[tmp_a], idx[tmp_b]


def projection_from_X(X, distance, idx_a, idx_b, Y):
    tmp1 = recursive_distance2(X, idx_a, distance, Y)
    tmp2 = tmp1[idx_b]
    tmp3 = recursive_distance2(X, idx_b, distance, Y)
    Yj = (tmp1 + tmp2 - tmp3) / (2.0 * np.sqrt(tmp2))
    return Yj


def distance_euclidean(A, B):
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B))


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

if __name__ == '__main__':
    from scipy.spatial import distance_matrix
    
    np.random.seed(1)
    N = 100000
    d = 20
    X = np.random.uniform(size=(N, d))
    k = 14

    # alpha = np.random.uniform(low=0.0, high=(2.0 * np.pi), size=N)
    # beta = np.random.uniform(low=0.0, high=(2.0 * np.pi), size=N)
    # X = np.array([sph2cart(alpha[i], beta[i], 1.0) for i in range(N)])
    # k = 2

    if N <= 1000:
        D_original = distance_matrix(X, X)
        Y = np.zeros([len(X), k])
        D = D_original.copy()
        for i in range(k):
            print(i)
            idx_a, idx_b = find_pivot_points(D)
            Y[:, i] = projection(D, idx_a, idx_b)
            D2 = update_residual_distance(D, Y)
            D = np.sqrt(D2)

        DY = distance_matrix(Y, Y)
        print(np.corrcoef(D_original.flatten(), DY.flatten()))


    Y = np.zeros([len(X), k])
    n_clusters = 10
    for i in range(k):
        print(i)
        idx_a, idx_b = find_pivot_points_from_X_fast(X, distance_euclidean, Y)
        Y[:, i] = projection_from_X(X, distance_euclidean, idx_a, idx_b, Y)

    idx = np.random.permutation(len(X))[:1000]
    D_sub = distance_matrix(X[idx], X[idx])
    DY_sub = distance_matrix(Y[idx], Y[idx])
    print(np.corrcoef(D_sub.flatten(), DY_sub.flatten()))
    
