import numpy as np
from dipy.tracking.streamline import set_number_of_points
from scipy.spatial.distance import pdist
from euclidean_embeddings.evaluation_metrics import stress, correlation, distortion
import os
import time
from euclidean_embeddings.distances import euclidean_distance, parallel_distance_computation
from functools import partial
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam

def resampling_eval(s, original_dist_matrixes, idxs, eval_seeds):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of resampling embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :return:
    """

    results_dir = "../eval_results/resampling/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    max_len = np.max([len(i) for i in s])

    sampling_n = np.geomspace(12, max_len, num=10)
    sampling_n = sampling_n.astype(int)
    for n in sampling_n:
        start_time = time.time()
        embeddings = set_number_of_points(s, nb_points=n)
        comp_time = time.time() - start_time

        for (idx, original_dist, eval_seed) in zip(idxs, original_dist_matrixes, eval_seeds):
            embedded_dist = pdist((np.asarray(embeddings[idx], dtype=object).reshape(len(idx), -1)), 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)

            #write results to file
            resFileName = "resampling_pts_{0}__eval_seed_{1}__eval_streamlines_{2}__emb_streamlines{3}".format(n, eval_seed, len(idx), len(s))
            with open(results_dir + resFileName, 'w') as f:
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('resampling_points\t' + str(n) + '\n')


def dissimilarity_eval(s, original_dist_matrixes, idxs, eval_seeds):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of dissimilarity embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :return:
    """

    from euclidean_embeddings.dissimilarity import compute_dissimilarity

    results_dir = "../eval_results/dissimilarity/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    num_prototypes = np.linspace(10, 30, 3)

    for n in num_prototypes:
        start_time = time.time()
        embeddings, _ = compute_dissimilarity(s, verbose=True, k=n, distance=distance)
        comp_time = time.time() - start_time

        for (idx, original_dist, eval_seed) in zip(idxs, original_dist_matrixes, eval_seeds):
            embedded_dist = pdist(embeddings[idx], 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)

            # write results to file
            resFileName = "num_prototypes_{0}__eval_seed_{1}__n_streamlines_{2}".format(n, eval_seed, len(idx))
            with open(results_dir + resFileName, 'w') as f:
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('n_prototypes\t' + str(n) + '\n')


def lipshitz_eval(s, original_dist_matrixes, idxs, eval_seeds):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of lipschitz embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """

    import lipschitz

    results_dir = "../eval_results/lipschitz/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    k = [2, 4, 8, 16]
    sizeA = [[2, 4], [2, 2, 4, 4], [2, 2, 4, 4, 8, 8, 8, 8],
             [2, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16]]

    for (n, A) in zip(k, sizeA):
        start_time = time.time()
        embeddings, _ = lipschitz.lipschitz_embedding(s, distance, linial1994=False, k=n, sizeA=A)
        comp_time = time.time() - start_time

        for (idx, original_dist, eval_seed) in zip(idxs, original_dist_matrixes, eval_seeds):
            embedded_dist = pdist(embeddings[idx], 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)

            resFileName = "n_reference_objects_{0}__eval_seed_{1}__n_streamlines_{2}".format(n, eval_seed, len(idx))
            with open(results_dir + resFileName, 'w') as f:
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('n_reference_objects\t' + str(n) + '\n')
                f.write('object_sizes\t' + ' '.join([str(x) for x in A]) + '\n')


def lmds_eval(s, original_dist_matrixes, idxs, eval_seeds):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of lmds embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """
    from euclidean_embeddings.lmds import compute_lmds

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    results_dir = "../eval_results/lmds/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    k = [2, 4, 8, 10, 12, 20, 30, 40]
    n_landmarks = [30, 50, 100, 150]

    for n in k:
        for l in n_landmarks:
            start_time = time.time()
            embeddings = compute_lmds(s, distance=distance, nl=l, k=n)
            comp_time = time.time() - start_time

            for (idx, original_dist, eval_seed) in zip(idxs, original_dist_matrixes, eval_seeds):
                embedded_dist = pdist(embeddings[idx], 'euclidean')
                emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
                emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
                emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)

                resFileName = "n_landmarks__{0}__embedding_size_{1}__eval_seed_{2}__n_streamlines_{3}".format(l, n,
                                                                                                              eval_seed, len(idx))
                with open(results_dir + resFileName, 'w') as f:
                    f.write('stress\t' + str(emb_stress) + '\n')
                    f.write('correlation\t' + str(emb_correlation) + '\n')
                    f.write('distortion\t' + str(emb_distortion) + '\n')
                    f.write('n_streamlines\t' + str(len(s)) + '\n')
                    f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                    f.write('exec_time\t' + str(comp_time) + '\n')
                    f.write('n_landmarks\t' + str(l) + '\n')
                    f.write('embedding_size\t' + str(n) + '\n')


def fastmap_eval(s, original_dist_matrixes, idxs, eval_seeds):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of fastmap embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """
    from euclidean_embeddings.fastmap import compute_fastmap

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    results_dir = "../eval_results/fastmap/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    k = [2, 4, 8, 10, 12, 20, 30, 40]

    for n in k:
        start_time = time.time()
        embeddings = compute_fastmap(s, distance, n)
        comp_time = time.time() - start_time

        for (idx, original_dist, eval_seed) in zip(idxs, original_dist_matrixes, eval_seeds):
            embedded_dist = pdist(embeddings[idx], 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)

            resFileName = "embedding_size_{0}__eval_seed_{1}__n_streamlines_{2}".format(n,eval_seed,len(idx))
            with open(results_dir + resFileName, 'w') as f:
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('embedding_size\t' + str(n) + '\n')


def load(filename="data/sub-100307/sub-100307_var-FNAL_tract.trk"):
    print('Loading %s' % filename)
    data = nib.streamlines.load(filename)
    s = data.streamlines
    print("%s streamlines" % len(s))
    return s

if __name__ == '__main__':

    sl = load()
    s = np.array(sl, dtype=np.object)

    #comment following lines if all data is used

    # s_size = 100
    # idx = np.random.permutation(s.shape[0])[:s_size]
    # s = s[idx]
    # sl = sl[idx]

    #precompute original distance matrixes for stress evaluation
    stress_samples = [100, 1000]
    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)
    original_dist_matrixes = []
    idxs = []

    #try different seeds if needed
    seeds = [1,2,3,4]
    data_seeds = []
    for seed in seeds:
        np.random.seed(seed)
        for n in stress_samples:
            idxs.append(np.random.choice(len(s), n))
            distances = distance(s[idxs[-1]], s[idxs[-1]])
            original_dist_matrixes.append(distances[np.triu_indices(len(distances), 1)])
            data_seeds.append(seed)

    #resampling_eval(sl, original_dist_matrixes, idxs, data_seeds)
    dissimilarity_eval(s, original_dist_matrixes, idxs, data_seeds)
    #lmds_eval(s, original_dist_matrixes, idxs, data_seeds)
    #lipshitz_eval(s, original_dist_matrixes, idxs, data_seeds)
    #fastmap_eval(s, original_dist_matrixes, idxs, data_seeds)