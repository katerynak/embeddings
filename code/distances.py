import numpy as np
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.distances import bundles_distances_mdf
from load import flatt
from scipy.spatial import distance_matrix
try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except:
    joblib_available = False



# def original_distance(a,b):
#     return bundles_distances_mam([a], [b])

def original_distance(a,b):
    return bundles_distances_mam(a, b)

# def euclidean_distance(a,b):
#     return np.linalg.norm(np.asarray(a)-np.asarray(b))

def euclidean_distance(A, B):
    """Wrapper of the euclidean distance between two vectors, or array and
    vector, or two arrays.
    """
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B), p=2)


def mdf(a,b):
    return bundles_distances_mdf(a,b)

def mdf1(a,b):
    d1 = np.linalg.norm(flatt([a])[0]-flatt([b])[0])
    d2 = np.linalg.norm(flatt([a[::-1]])[0]-flatt([b])[0])
    return min(d1,d2)

def mdf2(a,b):
    sum_dist = 0
    sum_dist_flip = 0
    for (p1, p2) in zip([a][0], [b][0]):
        sum_dist += np.linalg.norm(p1-p2)
    for (p1, p2) in zip([a[::-1]][0],[b][0]):
        sum_dist_flip += np.linalg.norm(p1-p2)
    return (min(sum_dist, sum_dist_flip))/len([a][0])


def parallel_distance_computation(distance, A, B, n_jobs=-1, verbose=True):
    """Computes the distance matrix between all objects in A and all
    objects in B in parallel over all cores.
    """
    if joblib_available and n_jobs != 1:
        if n_jobs is None or n_jobs == -1:
            n_jobs = cpu_count()

        if verbose:
            print("Parallel computation of the dissimilarity matrix: %s cpus." % n_jobs)

        if n_jobs > 1:
            tmp = np.linspace(0, len(A), 2 * n_jobs + 1).astype(np.int)
        else:  # corner case: joblib detected 1 cpu only.
            tmp = (0, len(A))

        chunks = zip(tmp[:-1], tmp[1:])
        dissimilarity_matrix = np.vstack(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(distance)(A[start:stop], B) for start, stop in chunks))
    else:
        dissimilarity_matrix = distance(A, B)

    if verbose:
        print("Done.")

    return dissimilarity_matrix
