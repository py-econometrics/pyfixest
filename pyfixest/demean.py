import numpy as np
from numba import njit, prange



@njit(parallel = True, cache = False, fastmath = False)
def demean(cx, flist, weights, tol = 1e-08, maxiter = 2000):

    '''
    Demean a Matrix cx by fixed effects in flist.
    The fixed effects are weighted by weights. Convervence tolerance
    is set to 1e-08 for the sum of absolute differences.
    Args:
        cx: Matrix to be demeaned
        flist: Matrix of fixed effects
        weights: Weights for fixed effects
        tol: Convergence tolerance. 1e-08 by default.
    Returns
        res: Demeaned matrix of dimension cx.shape
    '''
    N = cx.shape[0]
    fixef_vars = flist.shape[1]
    K = cx.shape[1]

    res = np.zeros((N,K))

    if np.sum(weights) != N:
        # save some computations when weights are all 1

        for k in prange(K):

            cxk = cx[:,k]
            oldxk = cxk - 1

            converged = False
            #while np.sum(np.abs(cxk - oldxk)) >= tol:
            for _ in range(maxiter):

                if converged:
                    break

                oldxk = cxk.copy()

                for i in range(fixef_vars):
                    weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    uvals = unique2(fmat) # unique2(fmat)
                    for j in uvals:
                        selector = fmat == j
                        cxkj = cxk[selector]
                        wj = weights[selector]
                        w = np.zeros(1)
                        wx = np.zeros(1)
                        for l in range(len(cxkj)):
                            w += wj[l]
                            wx += wj[l] * cxkj[l]
                        weighted_ave[selector] = wx / w

                    cxk = cxk - weighted_ave

                if np.sum(np.abs(cxk - oldxk)) < tol:
                    converged = True
                    break

            res[:,k] = cxk

    else:

        for k in prange(K):

            cxk = cx[:,k]#.copy()
            oldxk = cx[:,k] - 1

            converged = False
            #while np.sum(np.abs(cxk - oldxk)) >= tol:
            for _ in range(maxiter):

                if converged:
                    break

                oldxk = cxk.copy()
                for i in range(fixef_vars):
                    weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    uvals = unique2(fmat) # unique2(fmat)
                    for j in uvals:
                        selector = fmat == j
                        cxkj = cxk[selector]
                        w = 1.0 # np.zeros(1)
                        wx = np.zeros(1)
                        for l in range(len(cxkj)):
                            w += 1.0
                            wx += cxkj[l]
                        weighted_ave[selector] = wx / w

                    cxk -= weighted_ave

                if np.sum(np.abs(cxk - oldxk)) < tol:
                    converged = True
                    break

            res[:,k] = cxk

    return res



@njit
def unique2(x):
    '''
    Returns the unique values of a numpy array as a list
    Args:
        A numpy array.
    Returns:
        A list with the unique values of the numpy array.
    '''
    unique_values = set()
    res = []
    for i in range(len(x)):
        if x[i] not in unique_values:
            unique_values.add(x[i])
            res.append(x[i])

    return res
