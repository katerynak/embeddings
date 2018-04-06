import numpy as np
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points

def load():
    filename = "data/sub-100307/sub-100307_var-FNAL_tract.trk"
    data = nib.streamlines.load(filename)
    s = data.streamlines
    return s

def load_embedding(streamlines, nb_points):
    ss = set_number_of_points(streamlines, nb_points=nb_points)
    embedding = np.array([strm.flatten() for strm in ss])
    return embedding

def load_structured_embedding(streamlines, nb_points):
    embedding = set_number_of_points(streamlines, nb_points=nb_points)
    return embedding

def flatt(a):
    b = np.array([strm.flatten() for strm  in a])
    return b