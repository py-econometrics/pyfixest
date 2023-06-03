import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from formulaic import model_matrix



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

            cxk = cx[:,k].copy()
            oldxk = cxk - 1

            converged = False
            #while np.sum(np.abs(cxk - oldxk)) >= tol:
            for _ in range(maxiter):

                #if converged:
                #    break

                oldxk = cxk

                for i in range(fixef_vars):
                    #weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    for i in range(fixef_vars):
                        weighted_ave = np.zeros(N)
                        fmat = flist[:,i]
                        weighted_ave = ave(cxk, fmat, weights)

                    #uvals = unique2(fmat) # unique2(fmat)
                    #for j in uvals:
                        #selector = fmat == j
                        #cxkj = cxk[selector]
                        #wj = weights[selector]
                        #w = np.sum(wj)
                        #wx = np.sum(wj * cxkj)
                        #w = np.zeros(1)
                        #wx = np.zeros(1)
                        #for l in range(len(cxkj)):
                        #    w += wj[l]
                        #    wx += wj[l] * cxkj[l]
                        #weighted_ave[selector] = wx / w

                    cxk = cxk - weighted_ave

                if np.sum(np.abs(cxk - oldxk)) < tol:
                    converged = True
                    break

            res[:,k] = cxk

    else:

        for k in prange(K):

            cxk = cx[:,k].copy()
            oldxk = cx[:,k] - 1

            converged = False
            #while np.sum(np.abs(cxk - oldxk)) >= tol:
            for _ in range(maxiter):

                #if converged:
                #    break

                oldxk = cxk.copy()
                for i in range(fixef_vars):
                    weighted_ave = np.zeros(N)
                    fmat = flist[:,i]
                    weighted_ave = ave(cxk, fmat, weights)
                    #uvals = unique2(fmat) # unique2(fmat)
                    #for j in uvals:
                        #selector = fmat == j
                        #cxkj = cxk[selector]
                        #w = 1.0 # np.zeros(1)
                        #wx = np.zeros(1)
                        #for l in range(len(cxkj)):
                        #    w += 1.0
                        #    wx += cxkj[l]
                        #weighted_ave[selector] = #np.sum(cxkj) / len(cxkj)

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

@njit
def ave(x, f, w = None):


    N = len(x)
    wx = np.bincount(f, w * x )
    w = np.bincount(f, w)

    # drop zeros
    #wx = wxw[wxw != 0]
    #w = w[w != 0]

    wxw = wx / w
    wxw_long = np.zeros(N)
    for j in range(len(wxw)):
        selector = f == j
        wxw_long[selector] = wxw[j]

    return wxw_long

@njit
def ave2(x, f, w):

    N =  len(x)
    weighted_ave = np.zeros(N)
    uvals = unique2(f)

    for j in uvals:
        selector = f == j
        cxkj = x[selector]
        wj = w[selector]
        wsum = np.zeros(1)
        wx = np.zeros(1)
        for l in range(len(cxkj)):
            wsum += wj[l]
            wx += wj[l] * cxkj[l]
        weighted_ave[selector] = wx / wsum

    return weighted_ave


def getfe(uhat, fe_fml, data):

  '''
  Get fixed effects estimates after running a regression on demeaned data.
    Args:
        uhat: Residuals from a regression on demeaned data.
        fe_fml: A one sided formula with the fixed effects.
        data: A pandas dataframe with the fixed effects
    Returns:
        alpha: A numpy array with the fixed effects estimates.
    Example:
        get_fe(uhat, "~ firm + year", data)
  '''

  # check if uhat is a numpy array
  if not isinstance(uhat, np.ndarray):
    raise ValueError("uhat must be a numpy array")
  if not isinstance(fe_fml, str):
    raise ValueError("fe_fml must be a string")
  if not isinstance(data, pd.DataFrame):
    raise ValueError("data must be a pandas dataframe")

  if not fe_fml.startswith("~") or len(fe_fml.split("~")) > 1:
      raise ValueError("fe_fml must be a one sided formula")


  D = model_matrix(fe_fml, data = data, output = "sparse")
  DD = D.transpose().dot(D)
  Du = D.transpose().dot(uhat)
  alpha = sp.linalg.spsolve(DD, Du)

  return alpha
