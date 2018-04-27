
# LMDS algorithm implementation
# references:
#       -https://www.sciencedirect.com/science/article/pii/S0031320308005049#sec2
#       -https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7214117


# 1. choose n landmax points
# 2. compute distance-matrix Dl of landmax points, using the original distance
# 3. apply MDS to landmax points to obtain euclidean embedding of landmarks
# 4. compute U: the eigenvectors matrix and E: the diagonal eigenvalue matrix
# 5. compute E ^ (-1/2) * transpose (U)
# 6. compute median column means of Dl: mu
# 7. calculate squared distances of each point from n landmark points d_i
# 8. project points on eigenvectors to obtain embedding coordinates, use formula:
#        y_i = 0.5 * pseudoinv_transpose(M) * ( mu - d_i )
# where pseudoinv_transpose(M) is defined as : E ^ (-1/2) * transpose (U)

#to compute standard MDS scikit learn function will be used :
#http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

#to compute eugenvalues/eugenvectors:
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html