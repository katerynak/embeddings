import numpy as np
from dipy.tracking.streamline import set_number_of_points
from scipy.spatial.distance import pdist
from euclidean_embeddings.evaluation_metrics import stress, correlation, distortion, stress_normalized, relative_distance_error
import os
import time
from argparse import ArgumentParser
from euclidean_embeddings.distances import euclidean_distance, parallel_distance_computation
from functools import partial
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
import os.path
import sys

sys.path.insert(0, './euclidean_embeddings')


def resampling_eval(s, original_dist_matrixes, idxs, exec_number, track):
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

    sampling_n = [2, 4, 8, 10, 12, 20, 30, 40]
    for n in sampling_n:

        resFileName = "resampling_pts_{0}__exec_num_{1}__eval_streamlines_{2}__emb_streamlines{3}__track_{4}".format(n,exec_number,
                                                                                                                     len(idxs[0]),
                                                                                                                     len(s),
                                                                                                                     track)
        if os.path.isfile(results_dir + resFileName):
            continue

        start_time = time.time()
        embeddings = set_number_of_points(s, nb_points=n)
        comp_time = time.time() - start_time

        for (idx, original_dist) in zip(idxs, original_dist_matrixes):

            embedded_dist = pdist((np.asarray(embeddings[idx], dtype=object).reshape(len(idx), -1)), 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_stress_norm = stress_normalized(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_rel_dist = relative_distance_error(dist_embedd=embedded_dist, dist_original=original_dist)

            #write results to file

            with open(results_dir + resFileName, 'w') as f:
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('stress_norm\t' + str(emb_stress_norm) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('resampling_points\t' + str(n) + '\n')


def dissimilarity_eval(s, original_dist_matrixes, idxs, exec_number, track):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of dissimilarity embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :return:
    """

    from dissimilarity import compute_dissimilarity

    results_dir = "../eval_results/dissimilarity/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    num_prototypes = [2, 4, 8, 10, 12, 20, 30, 40]

    for n in num_prototypes:

        resFileName = "num_prototypes_{0}__exec_num_{1}__n_streamlines_{2}__track_{3}".format(n, exec_number,
                                                                                              len(idxs[0]),
                                                                                              track)

        if os.path.isfile(results_dir + resFileName):
            continue

        start_time = time.time()
        embeddings, _ = compute_dissimilarity(s, verbose=True, k=n, distance=distance)
        comp_time = time.time() - start_time

        for (idx, original_dist) in zip(idxs, original_dist_matrixes):

            embedded_dist = pdist(embeddings[idx], 'euclidean')

            emb_stress_norm = stress_normalized(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_rel_dist = relative_distance_error(dist_embedd=embedded_dist, dist_original=original_dist)

            # write results to file
            with open(results_dir + resFileName, 'w') as f:
                f.write('rel_dist\t' + str(emb_rel_dist) + '\n')
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('stress_norm\t' + str(emb_stress_norm) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('n_prototypes\t' + str(n) + '\n')


def lipschitz_eval(s, original_dist_matrixes, idxs, exec_number, track):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of lipschitz embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """

    from lipschitz import compute_lipschitz

    results_dir = "../eval_results/lipschitz/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    k = [2, 4, 8, 10, 12, 20, 30, 40]

    for n in k:

        resFileName = "n_reference_objects_{0}__exec_num_{1}__n_streamlines_{2}__track_{3}".format(n, exec_number,
                                                                                                   len(idxs[0]),
                                                                                                   track)

        if os.path.isfile(results_dir + resFileName):
            continue

        start_time = time.time()
        embeddings, _ = compute_lipschitz(s, distance, linial1994=False, k=n)
        comp_time = time.time() - start_time

        for (idx, original_dist) in zip(idxs, original_dist_matrixes):

            embedded_dist = pdist(embeddings[idx], 'euclidean')
            emb_stress_norm = stress_normalized(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_rel_dist = relative_distance_error(dist_embedd=embedded_dist, dist_original=original_dist)

            with open(results_dir + resFileName, 'w') as f:
                f.write('rel_dist\t' + str(emb_rel_dist) + '\n')
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('stress_norm\t' + str(emb_stress_norm) + '\n')
                f.write('correlation\t' + str(emb_correlation) + '\n')
                f.write('distortion\t' + str(emb_distortion) + '\n')
                f.write('n_streamlines\t' + str(len(s)) + '\n')
                f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                f.write('exec_time\t' + str(comp_time) + '\n')
                f.write('n_reference_objects\t' + str(n) + '\n')


def lmds_eval(s, original_dist_matrixes, idxs, exec_number, track):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of lmds embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """
    from lmds import compute_lmds

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    results_dir = "../eval_results/lmds/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    k = [2, 4, 8, 10, 12, 20, 30, 40]
    n_landmarks = [30, 50, 100, 150]

    for n in k:
        for l in n_landmarks:

            resFileName = "n_landmarks__{0}__embedding_size_{1}__exec_num_{2}__n_streamlines_{3}__track_{4}".format(l,
                                                                                                                    n,
                                                                                                                    exec_number,
                                                                                                                    len(idxs[0]),
                                                                                                                    track)

            if os.path.isfile(results_dir + resFileName):
                continue

            start_time = time.time()
            embeddings = compute_lmds(s, distance=distance, nl=l, k=n, )
            comp_time = time.time() - start_time

            for (idx, original_dist) in zip(idxs, original_dist_matrixes):
                embedded_dist = pdist(embeddings[idx], 'euclidean')
                emb_stress_norm = stress_normalized(dist_embedd=embedded_dist, dist_original=original_dist)
                emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
                emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
                emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)
                emb_rel_dist = relative_distance_error(dist_embedd=embedded_dist, dist_original=original_dist)

                with open(results_dir + resFileName, 'w') as f:
                    f.write('rel_dist\t' + str(emb_rel_dist) + '\n')
                    f.write('stress\t' + str(emb_stress) + '\n')
                    f.write('stress_norm\t' + str(emb_stress_norm) + '\n')
                    f.write('correlation\t' + str(emb_correlation) + '\n')
                    f.write('distortion\t' + str(emb_distortion) + '\n')
                    f.write('n_streamlines\t' + str(len(s)) + '\n')
                    f.write('eval_streamlines\t' + str(len(idx)) + '\n')
                    f.write('exec_time\t' + str(comp_time) + '\n')
                    f.write('n_landmarks\t' + str(l) + '\n')
                    f.write('required_emb_size\t' + str(n) + '\n')
                    f.write('embedding_size\t' + str(len(embeddings[0])) + '\n')


def fastmap_eval(s, original_dist_matrixes, idxs, exec_number, track):
    """
    function evaluates stress, correlation distance, distortion and embedding computation time
     given array of fastmap embedding of streamlines s
    :param s: input data
    :param original_dist_matrixes: array of matrixes of stress_samples dimensions
    :param idx: array of indexes of data used for calculation of original_dist_matrixes
    :param eval_seeds: list of seeds used for data sampling
    :return:
    """
    from fastmap import compute_fastmap

    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    results_dir = "../eval_results/fastmap/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    k = [2, 4, 8, 10, 12, 20, 30, 40]

    for n in k:

        resFileName = "embedding_size_{0}__exec_num_{1}__n_streamlines_{2}__track_{3}".format(n, exec_number,
                                                                                              len(idxs[0]),
                                                                                              track)

        if os.path.isfile(results_dir + resFileName):
            continue

        start_time = time.time()
        embeddings = compute_fastmap(s, distance, n)
        comp_time = time.time() - start_time

        for (idx, original_dist) in zip(idxs, original_dist_matrixes):

            embedded_dist = pdist(embeddings[idx], 'euclidean')
            emb_stress = stress(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_stress_norm = stress_normalized(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_correlation = correlation(embedded_dist.flatten(), original_dist.flatten())
            emb_distortion = distortion(dist_embedd=embedded_dist, dist_original=original_dist)
            emb_rel_dist = relative_distance_error(dist_embedd=embedded_dist, dist_original=original_dist)

            with open(results_dir + resFileName, 'w') as f:
                f.write('rel_dist\t' + str(emb_rel_dist) + '\n')
                f.write('stress\t' + str(emb_stress) + '\n')
                f.write('stress_norm\t' + str(emb_stress_norm) + '\n')
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

    parser = ArgumentParser()
    parser.add_argument("embedding")
    parser.add_argument("trk_num")
    args = parser.parse_args()

    embedding = args.embedding
    track = args.trk_num

    print("{0} embedding".format(embedding))

    #filename = "data/fna-ifof/deterministic_tracking_dipy_FNAL/sub-{0}/sub-{0}_var-FNAL_tract.trk".format(track)
    filename = "data/sub-{0}/sub-{0}_var-FNAL_tract.trk".format(track)
    sl = load(filename)
    s = np.array(sl, dtype=np.object)

    #comment following lines if all data is used

    # s_size = 1000
    # idx = np.random.permutation(s.shape[0])[:s_size]
    # s = s[idx]
    # sl = sl[idx]

    #precompute original distance matrixes for stress evaluation

    eval_samples = [10000]
    print("initial distance matrix computation on subsample of dataset")
    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)

    idxs = [np.random.choice(len(s), eval_samples)]

    distances = distance(s[idxs[-1]], s[idxs[-1]])
    original_dist_matrixes = [(distances[np.triu_indices(len(distances), 1)])]

    # cut-off distances bigger then the mean sample distance
    # f = lambda x, m: x if x < m else -1
    # vfunc = np.vectorize(f)
    # original_dist_matrixes[0] = vfunc(original_dist_matrixes[0], original_dist_matrixes[0].mean())

    iter = 100

    if embedding=="lipschitz":
        for i in range(1, iter):
            print("lipschitz iteration {0}".format(i))
            lipschitz_eval(s, original_dist_matrixes, idxs, i, track)

    if embedding=="lmds":
        for i in range(1, iter):
            print("lmds iteration {0}".format(i))
            lmds_eval(s, original_dist_matrixes, idxs, i,  track)

    if embedding=="fastmap":
        for i in range(1, iter):
            print("fastmap iteration {0}".format(i))
            fastmap_eval(s, original_dist_matrixes, idxs, i, track)

    if embedding=="dissimilarity":
        for i in range(1, iter):
            print("dissimilarity iteration {0}".format(i))
            dissimilarity_eval(s, original_dist_matrixes, idxs, i, track)

    if embedding=="resampling":
        for i in range(1, iter):
            print("resampling iteration {0}".format(i))
            resampling_eval(sl, original_dist_matrixes, idxs, i, track)
