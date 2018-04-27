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