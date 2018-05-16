"""Simple tests for Euclidean embeddings.
"""

import numpy as np
from fastmap import fastmap, fastmap_textbook
from scipy.spatial import distance_matrix
from lmds import compute_lmds, compute_lmds2
from eval_metrics import stress
from dissimilarity import compute_dissimilarity
    

def euclidean_distance(A, B):
    """Wrapper of the euclidean distance between two vectors, or array and
    vector, or two arrays.
    """
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B), p=2)


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

    print("Estimating the quality of embedded distances vs. original distances.")

    print("Fastmap:")
    subsample = False
    n_clusters = 10
    Y = fastmap(X, euclidean_distance, k, subsample, n_clusters)

    idx = np.random.permutation(len(X))[:1000]
    D_sub = distance_matrix(X[idx], X[idx])
    DY_sub = distance_matrix(Y[idx], Y[idx])
    print("  Correlation: %s" % (np.corrcoef(D_sub.flatten(), DY_sub.flatten())[0, 1]))
    print("  Stress : %s" % (stress(D_sub.flatten(), DY_sub.flatten())))

    print("lMDS:")
    lmds_embedding = np.array(compute_lmds2(X, nl=100, k=k,
                                             distance=euclidean_distance,
                                             landmark_policy='sff'))
    D_lmds_sub = distance_matrix(lmds_embedding[idx], lmds_embedding[idx])
    print("  Correlation: %s" % (np.corrcoef(D_sub.flatten(), D_lmds_sub.flatten())[0, 1]))
    print("  Stress : %s" % (stress(D_sub.flatten(), D_lmds_sub.flatten())))

    print("Dissimilarity Representation:")
    dissimilarity_embedding, prototype_idx = compute_dissimilarity(X, num_prototypes=40,
                                                                   distance=euclidean_distance,
                                                                   verbose=False)
    D_dissimilarity_sub = distance_matrix(dissimilarity_embedding[idx], dissimilarity_embedding[idx])
    print("  Correlation: %s" % (np.corrcoef(D_sub.flatten(), D_dissimilarity_sub.flatten())[0, 1]))
    print("  Stress : %s" % (stress(D_sub.flatten(), D_dissimilarity_sub.flatten())))
