import numpy as np
from scipy.stats.stats import pearsonr

def stress(dist_embedd, dist_original):
    s_diff = 0
    s_original = 0
    for (de, do) in zip(dist_embedd, dist_original):
        s_diff+=pow(de-do,2)
        s_original+=pow(do,2)
    return float(s_diff/s_original)

def inverse_correlation(dist_embedd, dist_original):
    return 1 - abs(np.corrcoef(dist_embedd, dist_original)[1,0])
