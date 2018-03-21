#!/usr/bin/python3

import load
import eval_metrics as em
import numpy as np
import distances as d

if __name__=="__main__":
    s, embedding = load.load()
    num_rows_2_sample = 10
    np.random.seed(0)
    original_distances = []
    embedded_distances = []
    idx = np.random.choice(len(s),num_rows_2_sample)
    idx2 = np.random.choice(len(s),num_rows_2_sample)
    for (a, ae) in zip(s[idx], embedding[idx]):
        for (b, be) in zip(s[idx2], embedding[idx2]):
            original_distances.append(d.original_distance([a], [b]))
            embedded_distances.append(d.euclidean_distance(ae, be))
    stress = em.stress(original_distances, embedded_distances)
    print("Stress of resampling embedding and euclidean distance: ", stress)