import numpy as np
from joblib import Parallel, delayed
import multiprocessing

#function to execute in parallel for reference set selection
def compute_A_i(i, sizeD, sizeA):
    return dataset[np.random.permutation(sizeD)[: sizeA[i]]]

#function to execute in parallel for lipschitz_embedding calculation for object obj
def compute_dataset_embedded(data_dim, i, object, R, distance_function):
    print("Computing embedding of object {0}/{1}".format(i, data_dim))
    return compute_distance_from_reference_sets(object,
                                                                  R,
                                                                  distance_function)

def compute_reference_sets(dataset, k, sizeA=None):
    """Compute k reference sets for a given dataset. Optionally, specify
    the size of the reference sets.
    """
    sizeD = len(dataset)
    if sizeA is None:
        sizeA = [(np.random.randint(sizeD) + 1) for i in range(k)]
        
    R = []
    for i in range(k):
        A_i = dataset[np.random.permutation(sizeD)[: sizeA[i]]]
        R.append(A_i)

    return R


def compute_reference_sets_linial1994(dataset, k):
    sizeD = len(dataset)
    R = []
    for i in range(k):
        A_i = dataset[np.random.permutation(sizeD)[: 2**(i+1)]]
        R.append(A_i)

    return R


def compute_distance_from_reference_set(object, A, distance_function):
    return np.min([distance_function(object, x) for x in A])


def compute_distance_from_reference_sets(object, R, distance_function):
    return np.array([compute_distance_from_reference_set(object,
                                                         A,
                                                         distance_function) for A in R])


def lipschitz_embedding(dataset, distance_function, k=None,
                        linial1994=True, sizeA=None):
    """Compute the Lipschitz embedding of a given dataset of objects,
    given its distance_function.

    Optional parameters: the target dimension k and whether to choose
    the sizes of the reference sets following the theorem of Linial et
    al. (1994), in order to have explicit bounds on the embedding.

    """
    if k is None:
        k = int(np.floor(np.log2(len(dataset)))) ** 2
        print("k = %s" % k)

    if linial1994:
        R = compute_reference_sets_linial1994(dataset, k)
    else:
        R = compute_reference_sets(dataset, k, sizeA)

    num_cores = multiprocessing.cpu_count()
    print("computation with {0} cores".format(num_cores))

    #for i, object in enumerate(dataset):
    # dataset_embedded[i, :] = compute_distance_from_reference_sets(object,
    #                                                               R,
    #                                                               distance_function)


    # dataset_embedded = np.zeros([len(dataset), k])
    data_dim = len(dataset)
    dataset_embedded = np.asarray(Parallel(n_jobs=num_cores)(delayed(compute_dataset_embedded)(data_dim,i,object,R,distance_function) for i,object
                               in enumerate(dataset)))

    if linial1994:
        dataset_embedded = dataset_embedded / (k ** (1.0 / 2.0))
        
    return dataset_embedded, R


def euclidean(a, b):
    return np.linalg.norm(a - b)


if __name__ == '__main__':
    np.random.seed(0)
    dataset = np.array([np.random.uniform(size=5) for i in range(100)], dtype=np.object)
    distance_function = euclidean

    dataset_embedded, R = lipschitz_embedding(dataset, distance_function)

