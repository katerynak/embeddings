#!/usr/bin/python3

import load
import numpy as np
import matplotlib.pyplot as plt
import utils
from preferences import *
import dissimilarity
import eval_metrics as em
import lipschitz
import distances as dist
import lmds
import random
import os
import time
import fastmap


if __name__=="__main__":
    s = load.load()
    if 'RESAMPLING_EMBEDDING' in globals():
        sampling_n = np.geomspace(12, utils.max_len(s), num=10)
        sampling_n = [int(i) for i in sampling_n]
        # sampling_n = [12,24]
        stress_samples = 10
        stress = []
        for n in sampling_n:
            embeddings = utils.resampling_embedding(s, n)
            stress.append(utils.eval_stress(s,embeddings,num_rows_2_sample=stress_samples))
        print(sampling_n)
        print(stress)
        plt.xlabel('n. resampling points')
        plt.ylabel('stress')
        plt.plot(sampling_n, stress, 'ro-')
        plt.show()

    if 'DISSIMILARITY_EMBEDDING' in globals():
        sterss_samples = 10
        num_prototypes = np.linspace(10,30,3)
        stress = []
        inv_corr = []
        distortion = []
        for n in num_prototypes:
            embeddings, prototype_idx = dissimilarity.compute_dissimilarity(s,
                                                                    verbose=True, num_prototypes=n)
            original_distances, embedded_distances = utils.eval_distances(s, embeddings,
                                                                    num_rows_2_sample=sterss_samples)
            if 'STRESS' in globals():
                stress.append(em.stress(dist_original=original_distances,
                                        dist_embedd=embedded_distances))
            #print("dissimilarity stress: ", stress)
            if 'PEARSON_CORRELATION' in globals():
                inv_corr.append(em.correlation_distance(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
                                                       dist_embedd=np.ndarray.flatten(np.asarray(embedded_distances))))
            if 'DISTORTION' in globals():
                distortion.append(em.distorsion(dist_original=original_distances,
                                        dist_embedd=embedded_distances))
        print(num_prototypes)
        print("Stress: ", stress)
        print("Correlation distance: ", inv_corr)
        print("Distiortion: ", distortion)
        plt.figure(1)

        plt.subplot(211)
        plt.xlabel('n. prototypes')
        plt.ylabel('stress')
        plt.plot(num_prototypes, stress, 'ro-')

        if 'PEARSON_CORRELATION' in globals():
            plt.subplot(212)
            plt.xlabel('n. prototypes')
            plt.ylabel('correlation distance')
            plt.plot(num_prototypes, inv_corr, 'ro-')

        if 'DISTORTION' in globals():
            plt.subplot(211)
            plt.xlabel('n. prototypes')
            plt.ylabel('distortion')
            plt.plot(num_prototypes, distortion, 'ro-')

        plt.show()

    if 'LIPSCHITZ_EMBEDDING' in globals():
        stress_samples = 100
        results_dir = "../eval_results/lipshitz"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # num_prototypes = np.linspace(10, 30, 3)
        k = [2,4,8,16]
        sizeA = [[2,4],[2,2,4,4],[2,2,4,4,8,8,8,8],
                 [2,2,4,4,8,8,8,8,16,16,16,16,16,16,16,16]]
        # k = [8]
        # sizeA = [[2,2,4,4,8,8,8,8]]
        stress = []
        inv_corr = []
        distortion = []
        for data_seed in range(4):
            s_idx = list(range(0, len(s)))
            data_seed = 0
            random.seed(data_seed)
            random.shuffle(s_idx)
            """
            s_size : number of streamlines
            """
            s_size = 50000
            s = s[s_idx][:s_size]
            for (n, A) in zip(k, sizeA):
                for stress_seed in range(4):

                    #embeddings, R = lipschitz.lipschitz_embedding(s, dist.original_distance)

                    start_time = time.time()
                    embeddings, R = lipschitz.lipschitz_embedding(s, dist.original_distance, linial1994=False, k=n, sizeA=A)
                    comp_time = time.time() - start_time
                    original_distances, embedded_distances = utils.eval_distances(s, embeddings,
                                                                                  num_rows_2_sample=stress_samples)
                    if 'STRESS' in globals():
                        stress.append(em.stress(dist_original=original_distances,
                                                dist_embedd=embedded_distances))
                    # print("dissimilarity stress: ", stress)
                    if 'PEARSON_CORRELATION' in globals():
                        inv_corr.append(em.correlation_distance(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
                                                               dist_embedd=np.ndarray.flatten(np.asarray(embedded_distances))))
                    if 'DISTORTION' in globals():
                        distortion.append(em.distorsion(dist_original=original_distances,
                                                        dist_embedd=embedded_distances))

                    resFileName = results_dir + "/" + "k_" + str(n) + \
                                  "__data_seed_" + str(data_seed) + \
                                  "__eval_seed_" + str(stress_seed) + \
                                  "__streamlines_" + str(len(embeddings))
                    with open(resFileName, 'w') as f:
                        f.write('stress\t' + str(stress[-1]) + '\n')

                        # TODO: call this variable correlation_distance
                        f.write('inverse_correlation\t' + str(inv_corr[-1]) + '\n')

                        f.write('distortion\t' + str(distortion[-1]) + '\n')
                        f.write('n_streamlines\t' + str(len(embeddings)) + '\n')
                        f.write('exec_time\t' + str(comp_time) + '\n')
                        f.write('eval_samples\t' + str(stress_samples) + '\n')
                        f.write('n_reference_objects\t' + str(n) + '\n')
                        f.write('object sizes\t' + ' '.join([str(x) for x in A]) + '\n')
        print("Stress: ", stress)
        print("Correlation distance: ", inv_corr)
        print("Distiortion: ", distortion)
        plt.figure(1)

        plt.subplot(211)
        plt.xlabel('n. prototypes')
        plt.ylabel('stress')
        #plt.plot(num_prototypes, stress, 'ro-')

        if 'PEARSON_CORRELATION' in globals():
            plt.subplot(212)
            plt.xlabel('n. prototypes')
            plt.ylabel('correlation distance')
            plt.plot(k, inv_corr, 'ro-')

        if 'DISTORTION' in globals():
            plt.subplot(211)
            plt.xlabel('n. prototypes')
            plt.ylabel('distortion')
            plt.plot(k, distortion, 'ro-')

        plt.show()

    if 'LMDS_EMBEDDING' in globals():
        results_dir = "../eval_results/lmds"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        """
        change stress_samples value to get more accurate evaluation results
        """
        stress_samples = 1000

        """
        pay attention to k: : size of euclidean embedding may be smaller in practice 
                              if first k autovalues contain negative values
        """
        k = [2,4,8,10,12]
        n_landmarks = [12,15,20,25,30,40]
        #stress_seed = 4
        stress = []
        inv_corr = []
        distortion = []
        for data_seed in range(4):
            s_idx = list(range(0, len(s)))
            random.seed(data_seed)
            random.shuffle(s_idx)
            """
            s_size : number of streamlines
            """
            s_size = 50000
            s = s[s_idx][:s_size]
            for n in k:
                for l in n_landmarks:
                    for stress_seed in range(4):
                        start_time = time.time()
                        embeddings = lmds.compute_lmds(s, distance=dist.original_distance, nl=l, k=n)
                        comp_time = time.time()-start_time

                        original_distances, embedded_distances = utils.eval_distances(s, embeddings,
                                                                                      num_rows_2_sample=stress_samples,
                                                                                      seed=stress_seed)
                        if 'STRESS' in globals():
                            stress.append(em.stress(dist_original=original_distances,
                                                    dist_embedd=embedded_distances))
                        # print("dissimilarity stress: ", stress)
                        if 'PEARSON_CORRELATION' in globals():
                            inv_corr.append(
                                em.correlation_distance(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
                                                       dist_embedd=np.ndarray.flatten(np.asarray(embedded_distances))))
                        if 'DISTORTION' in globals():
                            distortion.append(em.distorsion(dist_original=original_distances,
                                                            dist_embedd=embedded_distances))

                        """
                        file name has no importance for plotting and gathering statistics on data as long as
                        file names are different
                        """
                        resFileName = results_dir + "/" + "k_" + str(n) + \
                                      "__n_landmarks_" + str(l) + \
                                      "__data_seed_" + str(data_seed) + \
                                      "__eval_seed_" + str(stress_seed) + \
                                      "__streamlines_" + str(len(embeddings))
                        with open(resFileName,'w') as f:
                            f.write('stress\t' + str(stress[-1]) + '\n')

                            #TODO: call this variable correlation_distance
                            f.write('inverse_correlation\t' + str(inv_corr[-1]) + '\n')

                            f.write('distortion\t' + str(distortion[-1]) + '\n')
                            f.write('n_streamlines\t' + str(len(embeddings))+ '\n')
                            f.write('exec_time\t' + str(comp_time) + '\n')
                            f.write('eval_samples\t' + str(stress_samples) + '\n')
                            f.write('k\t'+ str(n)+'\n')
                            f.write('n_landmarks\t' + str(l)+'\n')
        # print("Stress: ", stress)
        # print("Correlation distance: ", inv_corr)
        # print("Distiortion: ", distortion)
        # plt.figure(1)
        #
        # plt.subplot(211)
        # plt.xlabel('n. prototypes')
        # plt.ylabel('stress')
        # # plt.plot(num_prototypes, stress, 'ro-')
        #
        # if 'PEARSON_CORRELATION' in globals():
        #     plt.subplot(212)
        #     plt.xlabel('n. prototypes')
        #     plt.ylabel('correlation distance')
        #     plt.plot(k, inv_corr, 'ro-')
        #
        # if 'STRESS' in globals():
        #     plt.subplot(211)
        #     plt.xlabel('n. prototypes')
        #     plt.ylabel('stress')
        #     plt.plot(k, stress, 'ro-')
        #
        # plt.show()

    if 'FASTMAP_EMBEDDING' in globals():
        results_dir = "../eval_results/fastmap"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        """
        change stress_samples value to get more accurate evaluation results
        """
        stress_samples = 1000

        stress = []
        inv_corr = []
        distortion = []

        stress_seed = 0
        data_seed = 0

        s_idx = list(range(0, len(s)))
        random.seed(data_seed)
        random.shuffle(s_idx)
        """
        s_size : number of streamlines
        """
        s_size = 50000
        s = s[s_idx][:s_size]

        k = 14
        n_clusters = 10
        subsample = False
        n_clusters = 10

        start_time = time.time()
        embeddings = fastmap.fastmap(s, dist.original_distance, k, subsample, n_clusters)
        comp_time = time.time() - start_time

        original_distances, embedded_distances = utils.eval_distances(s, embeddings,
                                                                      num_rows_2_sample=stress_samples,
                                                                      seed=stress_seed)

        if 'STRESS' in globals():
            stress.append(em.stress(dist_original=original_distances,
                                    dist_embedd=embedded_distances))
        # print("dissimilarity stress: ", stress)
        if 'PEARSON_CORRELATION' in globals():
            inv_corr.append(
                em.correlation_distance(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
                                        dist_embedd=np.ndarray.flatten(np.asarray(embedded_distances))))
        if 'DISTORTION' in globals():
            distortion.append(em.distorsion(dist_original=original_distances,
                                            dist_embedd=embedded_distances))

        print("Stress: ", stress)
        print("Correlation distance: ", inv_corr)
        print("Distiortion: ", distortion)
