#!/usr/bin/python3

import load
import numpy as np
import matplotlib.pyplot as plt
import utils
import distances as d
from numpy.random import uniform, randint
from dissimilarity import compute_dissimilarity

def prova_resampling():
    s = load.load()
    sampling_n = np.geomspace(12, utils.max_len(s), num=10)
    sampling_n = [int(i) for i in sampling_n]
    # sampling_n = [12,24]
    stress_samples = 10
    e_dist = []
    for n in sampling_n:
        embedding = load.load_structured_embedding(s, n)
        print("number of resamplings: ", n)
        print("original streamline: ", s[2])
        print("embedded streamline: ", embedding[2])
        e_dist.append(d.original_distance([embedding[2]], [embedding[4]])[0])
    print(sampling_n)
    print(e_dist)
    plt.xlabel('n. resampling points')
    plt.ylabel('distance of 2 fixed points')
    plt.plot(sampling_n, e_dist, 'ro-')
    plt.show()
    return 0

def prova_dissimilarity():
    s = load.load()

    dissimilarity_matrix, prototype_idx = compute_dissimilarity(s,
                                                                verbose=True)

    return 0


if __name__=="__main__":
    prova_dissimilarity()


