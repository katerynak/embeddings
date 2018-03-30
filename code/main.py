#!/usr/bin/python3

import load
import numpy as np
import matplotlib.pyplot as plt
import utils
from preferences import *
import dissimilarity

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
        num_prototypes = np.linspace(10,60,6)
        stress = []
        for n in num_prototypes:
            embeddings, prototype_idx = dissimilarity.compute_dissimilarity(s,
                                                                    verbose=True, num_prototypes=n)
            stress.append(utils.eval_stress(s,embeddings,num_rows_2_sample=sterss_samples))
            #print("dissimilarity stress: ", stress)
        print(num_prototypes)
        print(stress)
        plt.xlabel('n. prototypes')
        plt.ylabel('stress')
        plt.plot(num_prototypes, stress, 'ro-')
        plt.show()