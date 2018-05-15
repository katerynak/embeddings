"""Simple tests for Euclidean embeddings.
"""

import numpy as np
from fastmap import fastmap, fastmap_textbook
from scipy.spatial import distance_matrix
from lmds import compute_lmds
    

def distance_euclidean(A, B):
    """Wrapper of the euclidean distance between two vectors, iterables of
    vectors, etc.
    """
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B))


def sph2cart(az, el, r):
    """Spherical to Cartesian conversion, just for testing.
    """
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


if __name__ == '__main__':
    np.random.seed(1)
    N = 10000
    d = 20
    X = np.random.uniform(size=(N, d))
    k = 14

    # alpha = np.random.uniform(low=0.0, high=(2.0 * np.pi), size=N)
    # beta = np.random.uniform(low=0.0, high=(2.0 * np.pi), size=N)
    # X = np.array([sph2cart(alpha[i], beta[i], 1.0) for i in range(N)])
    # k = 2

    if N <= 1000:
        D_original = distance_euclidean(X, X)
        D = D_original.copy()
        Y = fastmap_textbook(D, k)
        DY = distance_matrix(Y, Y)
        print(np.corrcoef(D_original.flatten(), DY.flatten()))


    n_clusters = 10
    subsample = False
    n_clusters = 10
    Y = fastmap(X, distance_euclidean, k, subsample, n_clusters)

    print("Estimating the correlation between original distances and embedded distances.")
    idx = np.random.permutation(len(X))[:1000]
    D_sub = distance_matrix(X[idx], X[idx])
    DY_sub = distance_matrix(Y[idx], Y[idx])
    print("Correlation: %s" % (np.corrcoef(D_sub.flatten(), DY_sub.flatten())[0, 1]))
    
    lmds_embeddings = np.array(compute_lmds(X, nl=50, k=k,
                                            distance=distance_euclidean))
    D_lmds_sub = distance_matrix(lmds_embeddings[idx], lmds_embeddings[idx])
    print("Correlation: %s" % (np.corrcoef(D_sub.flatten(), D_lmds_sub.flatten())[0, 1]))
