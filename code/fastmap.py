"""Fastmap algorithm.
"""

from __future__ import print_function
import numpy as np


def find_pivot_points(D):
    """Find two points (a, b) far away, with a heuristic, from the
    distance matrix D. Textbook version.
    """
    idx_r = np.random.randint(D.shape[0])
    idx_a = D[idx_r, :].argmax()
    idx_b = D[idx_a, :].argmax()
    # print(D[idx_r, idx_a], D[idx_r, idx_b], D[idx_a, idx_b])
    return idx_a, idx_b



def projection(D, idx_a, idx_b):
    """Compute the j-th projection of each object given the (updated)
    distance matrix D. Textbook version.
    """
    size = D.shape[0]
    Yj = np.zeros(size)
    for i in range(size):
        Yj[i] = (D[i, idx_a]**2 + D[idx_a, idx_b]**2 - D[i, idx_b]**2) / (2.0 * D[idx_a, idx_b])

    return Yj


def update_residual_distance(D, Y):
    """Update distance matrix. Textbook version.
    """
    D2 = D * D
    size = D.shape[0]
    for u in range(size):
        for v in range(size):
            D2[u, v] = D2[u, v] - ((Y[u, :] - Y[v, :]) ** 2).sum()  # this is the original one
            # if D2[u, v] < 0.0:
            #     print("D2[%s, %s] = %s" % (u, v, D2[u, v]))

            # D2[u, v] = np.abs(D2[u, v] - ((Y[u, :] - Y[v, :]) ** 2).sum())  # rigged

    return D2


def fastmap_textbook(D, k):
    """Fastmap algorithm. Textbook version.
    """
    print("THIS IMPLEMENTATION HAS A BUG AND DIFFERS FROM THE FAST ONE!")
    raise Exception
    Y = np.zeros([D.shape[0], k])
    for i in range(k):
        print("Dimension: %s" % i)
        idx_a, idx_b = find_pivot_points(D)
        Y[:, i] = projection(D, idx_a, idx_b)
        D2 = update_residual_distance(D, Y)
        D = np.sqrt(D2)

    return Y


def recursive_distance2(X, idx, distance, Y):
    """Compute the squared recursive distance between an iterable of
    objects and a single object with index (idx), given the original
    distance and the (partial) projection Y. Necessary for
    Fastmap. This is a pretty fast implementation.
    """
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
    """Compute projections of objects X, given a distance function, two
    indices of pivot points (idx_a, idx_b), and their partial
    projection Y.

    """
    tmp1 = recursive_distance2(X, idx_a, distance, Y)
    tmp2 = tmp1[idx_b]
    tmp3 = recursive_distance2(X, idx_b, distance, Y)
    Yj = (tmp1 + tmp2 - tmp3) / (2.0 * np.sqrt(tmp2))
    return Yj


def fastmap(X, distance, k, subsample=False, n_clusters=10, verbose=False):
    """Fastmap algorithm. This is a pretty fast implementation.
    """
    Y = np.zeros([len(X), k])
    for i in range(k):
        if verbose:
            print("Dimension %s" % i)

        if subsample:
            idx_a, idx_b = find_pivot_points_scalable(X, distance, Y, n_clusters)
        else:
            idx_a, idx_b = find_pivot_points_from_X_fast(X, distance, Y)

        Y[:, i] = projection_from_X(X, distance, idx_a, idx_b, Y)

    return Y
