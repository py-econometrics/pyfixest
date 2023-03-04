import numpy as np

from numba import njit, prange

@njit
def ave(x, f):

    '''
    helper function: compute group-wise averages
    '''

    group_means = np.bincount(f, x) / np.bincount(f)

    return group_means[f]

@njit
def demean_jit(x, f, res, tol):

    '''
    demean variables by MAP algorithm
    Args:
        x (np.array of dimension 2): Variable(s) to demean
        f (np.array of dimension 2): fixed effects to demean by
        res (np.arry of dimension 2): empty matrix, needed as
            numba does not allow for matrix creation within a
            compiled function
        tol (float): precision for the iterative procedure
    Returns:
        A np.array of the same dimension as x, with demeaned x.
    Examples:
        N = 100000
        fixef_vars = np.random.choice([0, 1, 2, 3, 4, 5, 6], N)
        fixef_vars.shape = (fixef_vars.shape[0], 1)
        x = np.random.normal(0, 1, N).reshape(N, 1)
        demean(x, g)
    '''

    _, k_x = x.shape
    _, k_f = f.shape

    for i in range(k_x):

        cx = x[:,i].flatten()
        na_index = np.isnan(cx)
        cx[na_index] = 0
        oldx = cx - 1

        while np.sqrt(np.power(np.nansum(cx - oldx), 2)) >= tol:
            oldx = cx
            for j in range(k_f):
                cx += - ave(cx, f[:,j])

        cx[na_index] = np.nan
        res[:,i] = cx

    return res


def demean(x, f, tol=1e-6):

    '''
    demeaning algo that allows for missing values in x,
    but not in f
    Args:
        x (np.array of dimension 2): Variable(s) to demean
        f (np.array of dimension 2): fixef_vars to demean by
        tol (float): precision for the iterative procedure. f
                     default is 1e-06, which is fixest's default
    Returns:
        A np.array of the same dimension as x, with demeaned x.
    tba: dropping of duplixates
    '''

    N, k_x = x.shape
    _, k_f = f.shape
    f = f.astype(int)

    res = np.zeros((N, k_x))

    res = demean_jit(x, f, res, tol)

    return res
