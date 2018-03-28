import numpy as np
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.distances import bundles_distances_mdf
from load import flatt

def original_distance(a,b):
    return bundles_distances_mam(a,b)

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def mdf(a,b):
    return bundles_distances_mdf(a,b)

def mdf1(a,b):
    d1 = np.linalg.norm(flatt(a[0])-flatt(b[0]))
    d2 = np.linalg.norm(flatt(a[0][::-1])-flatt(b[0]))
    return min(d1,d2)