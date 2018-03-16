import numpy as np
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mam

if __name__=="__main__":
    filename = "data/sub-100307/sub-100307_var-FNAL_tract.trk"
    nb_points = 12
    
    data = nib.streamlines.load(filename)
    s = data.streamlines
    ss = set_number_of_points(s, nb_points=nb_points)
    embedding = np.array([strm.flatten() for strm in ss])
    d_s_01 = bundles_distances_mam([s[0]], [s[1]])
    d_v_01 = np.linalg.norm(embedding[0]-embedding[1])
    print(d_s_01)
    print(d_v_01)
    
