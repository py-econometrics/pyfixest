import numpy as np
from numba import jit, njit

@njit
def demean(x, fixef_vars, tol = 1e-10):

    '''
    demean variables by MAP algorithm

    Args:
        x (np.array): Variable(s) to demean
        fixef_vars (np.array): fixef_vars to demean by

    Returns:
        A np.array of the same dimension as x, with demeaned x.

    Examples:

        N = 100000
        fixef_vars = np.random.choice([0, 1, 2, 3, 4, 5, 6], N)
        fixef_vars.shape = (fixef_vars.shape[0], 1)
        x = np.random.normal(0, 1, N)
        demean(x, g)

    Notes:

        The algorithm follows the vignette of the lfe R package by Simen Gaure.
        The R example code therin is

        demean <- function(x, flist) {
            cx <- x
            oldx <- x - 1
            while(sqrt(sum((cx - oldx) ^ 2)) >= 1e-10) {
                oldx <- cx
                for(i in 1:length(flist)){
                cx <- cx - ave(cx, flist[[i]])
                }
            }
            return(cx)
        }

    '''


    def ave(x, f):

        '''
        helper function: compute group-wise averages
        '''

        N = x.shape[0]
        res = np.empty(N)
        res[:] = np.nan
        unique_f = np.unique(f[~np.isnan(f)])
        for i, ival in enumerate(unique_f):
            idx = np.where(np.equal(f, i))
            res[idx] = np.mean(x[idx])

        return res

    if len(fixef_vars.shape) == 1:
        fixef_vars.reshape(N, 1)

    k_fixef = fixef_vars.shape[1]

    #isna_fixef = np.all(np.isnan(fixef_vars), axis = 1)
    isna_fixef = np.sum(np.isnan(fixef_vars), axis = 1)

    cx = x
    cx[isna_fixef] = np.nan
    oldx = x - 1

    while np.sqrt(np.power(np.sum(cx - oldx), 2)) >= tol:
        oldx = cx
        for f in range(0, k_fixef-1):
            cx = cx - ave(x, fixef_vars[:,f])

    return cx


# N = 100000
# fixef_vars = np.random.choice([0, 1, 2, 3, 4, 5, 6], N*2).astype(float)
# fixef_vars = fixef_vars.reshape(N, 2)
# fixef_vars[1, 1] = np.nan
# #fixef_vars.shape = (fixef_vars.shape[0], 1)
# x = np.random.normal(0, 1, N)
# x[0] = np.nan
# res = demean(x, fixef_vars)
# res
# len(res)
