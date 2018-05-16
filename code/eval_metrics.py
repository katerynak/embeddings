import numpy as np
from scipy.stats.stats import pearsonr

def stress_def1(dist_embedd, dist_original):
    s_diff = 0
    s_original = 0
    for (de, do) in zip(dist_embedd, dist_original):
        s_diff+=pow(de-do,2)
        s_original+=pow(do,2)
    return float(s_diff/s_original)

def stress2(dist_embedd, dist_original):
    s_diff = 0
    norm_orig = 0
    norm_embedd = 0

    for de in dist_embedd:
        norm_embedd += pow(de,2)
    for do in dist_original:
        norm_orig += pow(do, 2)

    for (de, do) in zip(dist_embedd, dist_original):
        s_diff += pow(float(de/norm_embedd) - float(do/norm_orig), 2)
    return float (s_diff)


def stress(dist_embedd, dist_original):
    tmp = dist_embedd / (dist_embedd * dist_embedd).sum() - dist_original / (dist_original * dist_original).sum()
    return (tmp * tmp).sum()


# def inverse_correlation(dist_embedd, dist_original):
#     return 1 - abs(np.corrcoef(dist_embedd, dist_original)[1,0])

def correlation_distance(dist_embedd, dist_original):
    return 1 - abs(np.corrcoef(dist_embedd, dist_original)[1,0])

#   function returns values c1 and c2 such that
#   (1/c1)*do(o1, o2) <= de(f(o1), f(o2)) <= c2 * do(o1,o2)
#   where do is the original distance, de is the embedded distance,
#   f(o1) is the embeddig of object o1
def distorsion(dist_embedd, dist_original):
    c1 = 1
    c2 = 1

    for (de, do) in zip(dist_embedd, dist_original):
        f = float(de/do)
        if (f>1): #de > do, update c2
            if (f>c2):
                c2 = f
        else:     #de < do, update c1
            if ((1/f)>c1):
                c1 = (1/f)

    return c1,c2

