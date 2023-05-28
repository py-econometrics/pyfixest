import numpy as np
from numba import njit, prange

@njit(parallel = True)
def demean(x, flist, weights, tol = 1e-08, maxiter = 2000):

    # Check dimensions
    #if x.ndim != 2:
    #    raise ValueError("x needs to be a np.array of dimension 2.")
    #if flist.ndim != 2:
    #    raise ValueError("flist needs to be a np.array of dimension 2.")
    #if weights.ndim != 2 and weights.shape[1] != 1:
    #    raise ValueError("weights needs to be a two dimension array with only one column.")

    #x = x.transpose()
    #flist = flist.transpose()
    cx = x

    N = cx.shape[0]
    fixef_vars = flist.shape[1]
    K = x.shape[1]
    weights = weights.flatten()

    #res = np.zeros((N, K))

    for k in prange(K):

        cxk = cx[:,k]#subset_matrix(cx, k)
        oldxk = cxk - 1


        for _ in range(maxiter):

            #if np.sqrt(np.sum((cxk - oldxk) ** 2)) < tol:
            if np.sum(np.abs(cxk - oldxk)) < tol:
                break

            oldxk = cxk
            for i in range(fixef_vars):

                weighted_ave = np.zeros(N)
                fmat = flist[:,i]

                for j in np.unique(fmat):
                    selector = fmat == j
                    cxkj = cxk[selector]#select_elements(cxk, selector)
                    wj = weights[selector]
                    w = np.sum(wj)
                    wx = np.sum(wj * cxkj)
                    weighted_ave[selector] = wx / w # np.repeat(wx / w, len(wj))

                cxk -= weighted_ave

            cx[:,k] = cxk#fill_matrix_columns(cx, cxk, k)

    return cx


@njit
def subset_matrix(A, k):

    N = A.shape[0]
    column_vector = np.zeros(N)
    for i in range(N):
        column_vector[i] = A[i,k]
    return column_vector

@njit
def select_elements(v, selector):
    N = v.shape[0]
    k = np.sum(selector)
    selected_elements = np.zeros(k)
    j = 0
    for i in range(N):
        if selector[i]:
            selected_elements[j] = v[i]
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
