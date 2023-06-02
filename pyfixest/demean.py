import numpy as np
from numba import njit, prange


@njit(parallel = True)
def demean(cx, flist, weights, tol = 1e-08):

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

            while np.sum(np.abs(cxk - oldxk)) >= tol:

                #if np.sqrt(np.sum((cxk - oldxk) ** 2)) >= 1e-10:
                #    break

                oldxk = cxk
                for i in range(fixef_vars):
                    weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    for j in np.unique(fmat):
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

            res[:,k] = cxk

    else:

        for k in prange(K):

            cxk = cx[:,k]
            oldxk = cxk - 1

            while np.sum(np.abs(cxk - oldxk)) >= tol:

                #if np.sqrt(np.sum((cxk - oldxk) ** 2)) >= 1e-10:
                #    break

                oldxk = cxk
                for i in range(fixef_vars):
                    weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    for j in np.unique(fmat):
                        selector = fmat == j
                        cxkj = cxk[selector]
                        #wj = weights[selector]
                        w = np.zeros(1)
                        wx = np.zeros(1)
                        for l in range(len(cxkj)):
                            w += 1.0
                            wx += cxkj[l]
                        weighted_ave[selector] = wx / w

                    cxk = cxk - weighted_ave

            res[:,k] = cxk

    return res
