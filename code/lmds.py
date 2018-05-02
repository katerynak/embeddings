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

to compute eugenvalues/eugenvectors:
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

def landmark_dist_matrix(landmarks, distance=""):

    """
    computes matrix of distances between landmarks, given the distance

    Parameters
        ----------
        landmarks: points from dataset
        distance: original distance function

    Return
        ------
        matrix containing pairwise distances, ndarray
    """
    n = len(landmarks)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i,j] = distance(landmarks[i], landmarks[j])

    return dist_matrix


def landmark_mds(Dl, k):
    """
    computes euclidean muldidimentional scaling of landmarks points
    transforms matrix of pairwise distances into matrix of euclidean embeddings of size k
    """
    mds = sklearn.manifold.MDS(dissimilarity = 'precomputed', n_components=k, n_jobs=-1)
    return mds.fit_transform(Dl)

def compute_M_sharp(landmarks_euclidean_matrix):
    """
    computes M sharp from eigenvalues and eigenvectors of landmarks_euclidean_matrix
    M sharp : E ^ (-1/2) * transpose (U), where
    U is the eigenvectors matrix and E is the diagonal eigenvalue matrix
    """

    #E, U = np.linalg.eig(landmarks_euclidean_matrix)
    E, U = LA.eig(landmarks_euclidean_matrix)
    return np.reciprocal(np.sqrt(np.diag(E))) * np.transpose(U)

def landmarks_original_mean(Dl):
    """
    function computes mean values of each column of Dl
    """
    return np.mean(Dl, axis=0)

def compute_final_embedding(d_i, M_sharp, mu):
    """
    computes final embedding of object d_i
    """
    y_i = 0.5 * M_sharp * (mu - d_i)
    return y_i


def compute_landmarks_fastmap(tracks):
    n = 10 #number of landmark points
    k = 10 #dimention of final embedding
    distances, landmarks_idx = dissimilarity.compute_dissimilarity(tracks,
                                                               verbose=True, num_prototypes=n)
    landmarks = [tracks[i] for i in landmarks_idx]
    Dl = landmark_dist_matrix(landmarks, distance=dist.original_distance)
    landmarks_euclidean = landmark_mds(Dl,k)
    M_sharp = compute_M_sharp(landmarks_euclidean)
    mu = landmarks_original_mean(Dl)
    embeddings = []
    for d_i in distances:
        embeddings.append(compute_final_embedding(d_i, M_sharp, mu))

if __name__ == '__main__':
    import load
    tracks = load.load()
    fastmap_embeddings = compute_landmarks_fastmap(tracks)

