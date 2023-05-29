import numpy as np
from numba import njit, prange


@njit(parallel = False)
def demean(x, flist, weights, tol = 1e-06):

    # Check dimensions
    #if x.ndim != 2:
    #    raise ValueError("x needs to be a np.array of dimension 2.")
    #if flist.ndim != 2:
    #    raise ValueError("flist needs to be a np.array of dimension 2.")
    #if weights.ndim != 2 and weights.shape[1] != 1:
    #    raise ValueError("weights needs to be a two dimension array with only one column.")

    cx = x
    res = cx.copy()

    N = cx.shape[0]
    fixef_vars = flist.shape[1]
    K = x.shape[1]

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

        cx[:,k] = cxk

    return cx


#@njit
def subset_matrix(A, k):

    N = A.shape[0]
    column_vector = np.zeros((N, 1), dtype=A.dtype)
    for i in range(N):
        column_vector[i, 0] = A[i, k]
    return column_vector

#@njit
def select_elements(v, selector):
    N = v.shape[0]
    k = np.sum(selector)
    selected_elements = np.empty((k,1), dtype=v.dtype)
    j = 0
    for i in range(N):
        if selector[i,0]:
            selected_elements[j,0] = v[i,0]
            j += 1
    return selected_elements

@njit
def fill_matrix_columns(A, v, k):

    '''
    Fill a matrix column with a vector
    Args:
        A: matrix
        v: vector
        k: column index
    Example:
        fill_matrix_columns(cx, cxk, k)
    '''

    N = A.shape[0]
    for i in range(N):
        A[i,k] = v[i]

    return A
