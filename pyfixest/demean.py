import numpy as np
from numba import njit, prange


@njit(parallel = False)
def demean(cx, flist, weights, tol = 1e-08):

    # Check dimensions
    #if x.ndim != 2:
    #    raise ValueError("x needs to be a np.array of dimension 2.")
    #if flist.ndim != 2:
    #    raise ValueError("flist needs to be a np.array of dimension 2.")
    #if weights.ndim != 2 and weights.shape[1] != 1:
    #    raise ValueError("weights needs to be a two dimension array with only one column.")

    #cx = x

    N = cx.shape[0]
    fixef_vars = flist.shape[1]
    K = cx.shape[1]

    res = np.zeros((N,K))

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
                    for l in range(len(wj)):
                        w += wj[l]
                        wx += wj[l] * cxkj[l]
                    weighted_ave[selector] = wx / w

                cxk = cxk - weighted_ave

        res[:,k] = cxk

    return res
