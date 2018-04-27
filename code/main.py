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
                inv_corr.append(em.inverse_correlation(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
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
        sterss_samples = 10
        # num_prototypes = np.linspace(10, 30, 3)
        #k = [2,4]
        #sizeA = [[2,4],[2,2,4,4]]
        k = [8]
        sizeA = [[2,2,4,4,8,8,8,8]]
        stress = []
        inv_corr = []
        distortion = []
        for (n, A) in zip(k, sizeA):
            embeddings, R = lipschitz.lipschitz_embedding(s, dist.original_distance)
            #embeddings, R = lipschitz.lipschitz_embedding(s, dist.original_distance, linial1994=False, k=n, sizeA=A)
            original_distances, embedded_distances = utils.eval_distances(s, embeddings,
                                                                          num_rows_2_sample=sterss_samples)
            if 'STRESS' in globals():
                stress.append(em.stress(dist_original=original_distances,
                                        dist_embedd=embedded_distances))
            # print("dissimilarity stress: ", stress)
            if 'PEARSON_CORRELATION' in globals():
                inv_corr.append(em.inverse_correlation(dist_original=np.ndarray.flatten(np.asarray(original_distances)),
                                                       dist_embedd=np.ndarray.flatten(np.asarray(embedded_distances))))
            if 'DISTORTION' in globals():
                distortion.append(em.distorsion(dist_original=original_distances,
                                                dist_embedd=embedded_distances))
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