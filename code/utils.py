import load
import eval_metrics as em
import numpy as np
import distances as d
from itertools import combinations

def max_len(streamlines):
    max = 0
    for a in streamlines:
        if len(a) > max:
            max = len(a)
    return max

def eval_stress(streamlines, nb_points, num_rows_2_sample):
    embedding = load.load_embedding(streamlines, nb_points)
    np.random.seed(0)
    original_distances = []
    embedded_distances = []
    idx = np.random.choice(len(streamlines), num_rows_2_sample)
    #iterazion over all possible combinations of randomly selected indexes
    idx_comb = combinations(idx,2)
    for (idx0, idx1) in idx_comb:
        original_distances.append(d.original_distance([streamlines[idx0]], [streamlines[idx1]]))
        embedded_distances.append(d.euclidean_distance(embedding[idx0], embedding[idx1]))
    stress = em.stress(dist_embedd=embedded_distances, dist_original=original_distances)
    return stress