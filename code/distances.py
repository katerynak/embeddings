import numpy as np
from dipy.tracking.distances import bundles_distances_mam

def original_distance(a,b):
    return bundles_distances_mam(a,b)

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)