import load
import eval_metrics as em
import numpy as np
import distances as d
from itertools import combinations
from load import flatt

def max_len(streamlines):
    max = 0
    for a in streamlines:
        if len(a) > max:
            max = len(a)
    return max


def resampling_embedding(streamlines, nb_points):
    embedding = load.load_structured_embedding(streamlines, nb_points)
    return embedding


def eval_stress(streamlines, embeddings, num_rows_2_sample, original_distance=d.original_distance,
                embedded_distance=d.euclidean_distance):
    np.random.seed(0)
    original_distances = []
    embedded_distances = []
    idx = np.random.choice(len(streamlines), num_rows_2_sample)
    #iterazion over all possible combinations of randomly selected indexes
    idx_comb = combinations(idx,2)
    #print ("streamline normale: ", flatt([embedding[0]])[0])
    #print ("streamline reverse: ", flatt(flatt([embedding[0][::-1]])))
    for (idx0, idx1) in idx_comb:
        original_distances.append(original_distance(streamlines[idx0], streamlines[idx1]))
        embedded_distances.append(embedded_distance(embeddings[idx0], embeddings[idx1]))
        #embedded_distances.append(d.original_distance(embedding[idx0], embedding[idx1]))
        #embedded_distances.append(d.mdf2(embedding[idx0], embedding[idx1],nb_points))
        #TODO: cambiare mdf2 per determinare dinamicamente la dimensione dell'array
    stress = em.stress(dist_embedd=embedded_distances, dist_original=original_distances)
    return stress

