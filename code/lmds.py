"""
LMDS algorithm implementation
references:
      -https://www.sciencedirect.com/science/article/pii/S0031320308005049#sec2
      -https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7214117
      -http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/dimreduc_landmarks.pdf

1. choose n landmax points
2. compute distance-matrix Dl of landmax points, using the original distance
3. apply MDS to landmax points to obtain euclidean embedding of landmarks of size k
4. compute U: the eigenvectors matrix and E: the diagonal eigenvalue matrix
5. compute E ^ (-1/2) * transpose (U)
6. compute median column means of Dl: mu
7. calculate squared distances of each point from n landmark points d_i
8.1 project points on eigenvectors to obtain embedding coordinates, use formula:
       y_i = 0.5 * pseudoinv_transpose(M) * ( mu - d_i )
where pseudoinv_transpose(M) is defined as : E ^ (-1/2) * transpose (U)
8.2 possible alternative: use only a subset of eigenvectors having the biggest
eigenvalues, dimention of embedded space in this case is n' < n

to compute standard MDS scikit learn function will be used :
http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

to compute eigenvalues/eigenvectors:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html

to compute distances of each dataset point from landmarks already implemented dissimilarity
embedding can be used:
https://github.com/emanuele/dissimilarity/blob/master/dissimilarity.py
"""

import numpy as np
import sklearn.metrics, sklearn.manifold
import dissimilarity
import distances as dist
from scipy import linalg as LA


def choose_landmarks_random(n, dataset):
    """
    returns n random points from dataset

    Parameters
        ----------

        n : number of datapoints to choose
        dataset : vector of values to choose from

    Return
        ------
        idx : array of int
            an array of n indices of the n selected datapoints.

    """
    return np.random.choice(dataset, n, replace=False)


# def landmark_dist_matrix(landmarks, distance):
#     """
#     computes matrix of distances between landmarks, given the distance

#     Parameters
#         ----------
#         landmarks: points from dataset
#         distance: original distance function

#     Return
#         ------
#         matrix containing pairwise distances, ndarray
#     """
#     n = len(landmarks)
#     dist_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(n):
#             dist_matrix[i,j] = distance(landmarks[i], landmarks[j])

#     return dist_matrix


def landmark_mds(Dl, k):
    """computes euclidean muldidimentional scaling of landmarks points
    transforms matrix of pairwise distances into matrix of euclidean
    embeddings of size k
    """
    mds = sklearn.manifold.MDS(dissimilarity='precomputed',
                               n_components=k, n_jobs=-1,
                               metric=False)  # Why?
    return mds.fit_transform(Dl)


def centering_matrix(n):
    return np.identity(n) - (1./n)* np.ones((n, n))


def eig(distance_matrix, k):
    """function computes eigenvector/eigenvalue decomposition of the
    mean-centered "inner-product" matrix B and returns k largest
    eigenvalues and corresponding eigenvectors, if among k largest
    eigenvalues there are negative values, function discard them and
    returns only positive eigenvalues and corresponding eigenvectors
    """
    H = centering_matrix(distance_matrix.shape[0])
    B = -0.5*(np.matmul(np.matmul(H,distance_matrix), H))
    E, U = np.linalg.eigh(B)
    U = np.transpose(U)
    sorted_idx = np.argsort(E)[::-1][:k]
    posE = sum(x > 0 for x in E[sorted_idx])
    sorted_idx = sorted_idx[:posE]
    return E[sorted_idx],U[sorted_idx]


def compute_M_sharp(E, U):
    """computes M sharp from eigenvalues and eigenvectors of
    landmarks_euclidean_matrix
    M sharp : E ^ (-1/2) * transpose (U), where
    U is the eigenvectors matrix and E is the diagonal eigenvalue matrix
    """
    return np.matmul((np.diag(np.reciprocal(np.power(E,0.5)))), U)


def columns_mean(Dl):
    """
    function computes mean values of each column of Dl
    """
    return np.mean(Dl, axis=0)


def compute_final_embedding(d_i, M_sharp, mu):
    """
    computes final embedding of object d_i
    """
    y_i = 0.5 * np.matmul(M_sharp , (mu - d_i))
    return y_i.tolist()


def compute_lmds(dataset, distance=None, nl=10, k=4):
    """Given a dataset, computes the lMDS Euclidean embedding of size k
    using nl landmarks. The dataset must be an array.

    """
    # if distance is None:
    #     distance = dist.original_distance

    distances, landmarks_idx = dissimilarity.compute_dissimilarity(dataset,
                                                                   distance=distance,
                                                                   prototype_policy='random',
                                                                   verbose=True,
                                                                   num_prototypes=nl)
    # distances = np.power(distances, 2)
    distances *= distances
    # landmarks = [tracks[i] for i in landmarks_idx]
    landmarks = dataset[landmarks_idx]

    # Dl = landmark_dist_matrix(landmarks, distance=distance)
    Dl = distance(landmarks, landmarks)
    # Dl = np.power(Dl, 2)
    Dl *= Dl

    E, U = eig(Dl, k)
    M_sharp = compute_M_sharp(E, U)
    mu = columns_mean(Dl)

    embeddings = []
    for d_i in distances:
        embeddings.append(compute_final_embedding(d_i, M_sharp, mu))

    return embeddings


def compute_lmds2(dataset, distance=None, nl=10, k=4, landmark_policy='random'):
    """Given a dataset, computes the lMDS Euclidean embedding of size k
    using nl landmarks. The dataset must be an array.
    """
    d, landmarks_idx = dissimilarity.compute_dissimilarity(dataset,
                                                           distance=distance,
                                                           prototype_policy=landmark_policy,
                                                           verbose=True,
                                                           num_prototypes=nl)
    # Use squared distances
    d *= d
    D = distance(dataset[landmarks_idx], dataset[landmarks_idx])
    D *= D

    # Compute cMDS on landmarks and get top k eigenvalues/eigenvectors
    n = D.shape[0]
    H = np.identity(n) - (1.0 / n) * np.ones((n, n))
    B = -0.5 * H.dot(D).dot(H)
    Lambda, U = np.linalg.eigh(B)  # eigh() because B (and D) must be symmetric
    U = U.T
    idx = Lambda.argsort()[::-1]
    Lambda_plus = Lambda[idx][:k]
    U_plus = U[idx][:k]

    # Compute M_sharp and the embedding:
    M_sharp = np.diag(1.0 / np.sqrt(Lambda_plus)).dot(U_plus)
    Y = (0.5 * M_sharp.dot((D.mean(0) - d).T)).T

    return Y
