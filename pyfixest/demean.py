import numpy as np
import scipy.sparse as sp
from numba import njit, prange, types, typed, float64, int64
from formulaic import model_matrix


# @njit(float64[:](float64[:], int64[:], float64[:]))
@njit
def _ave3(x, f, w):
    N = len(x)

    # wx_dict = {}
    # w_dict = {}

    wx_dict = typed.Dict.empty(key_type=types.int64, value_type=types.float64)
    w_dict = typed.Dict.empty(key_type=types.int64, value_type=types.float64)

    # Compute weighted sums using a dictionary
    for i in range(N):
        j = f[i]
        if j in wx_dict:
            wx_dict[j] += w[i] * x[i]
        else:
            wx_dict[j] = w[i] * x[i]

        if j in w_dict:
            w_dict[j] += w[i]
        else:
            w_dict[j] = w[i]

    wxw_vec = np.zeros_like(f, dtype=x.dtype)

    for i in range(N):
        j = f[i]
        wxw_vec[i] = wx_dict[j] / w_dict[j]

    return wxw_vec


# @njit(float64[:,:](float64[:,:], int64[:,:], float64[:,:], float64, int64))
@njit(parallel=True)
def demean(cx, fmat, weights, tol=1e-08, maxiter=2000):
    """
    Demean a Matrix cx by fixed effects in fmat.
    The fixed effects are weighted by weights. Convervence tolerance
    is set to 1e-08 for the sum of absolute differences.
    Args:
        x: Matrix to be demeaned
        fmat: Matrix of fixed effects
        weights: Weights for fixed effects
        tol: Convergence tolerance. 1e-08 by default.
    Returns
        res: Demeaned matrix of dimension cx.shape
    """

    # cx = x.copy()

    fixef_vars = fmat.shape[1]
    K = cx.shape[1]

    res = np.zeros_like(cx)

    for k in prange(K):
        cxk = cx[:, k].copy()
        oldxk = cxk - 1

        # initiate
        weighted_ave = np.empty_like(cxk)
        fvec = np.empty_like(cxk)

        for _ in range(maxiter):
            for i in range(fixef_vars):
                fvec = fmat[:, i]
                weighted_ave[:] = _ave3(cxk, fvec, weights)
                cxk -= weighted_ave

            if (np.abs(cxk - oldxk)).max() < tol:
                break

            oldxk = cxk.copy()

        res[:, k] = cxk

    return res


@njit
def _ave2(x, f, w):
    N = len(x)
    weighted_ave = np.zeros(N)
    uvals = _unique2(f)

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


@njit
def _unique2(x):
    """
    Returns the unique values of a numpy array as a list
    Args:
        A numpy array.
    Returns:
        A list with the unique values of the numpy array.
    """
    unique_values = set()
    res = []
    for i in range(len(x)):
        if x[i] not in unique_values:
            unique_values.add(x[i])
            res.append(x[i])

    return res


@njit
def _ave(x, f, w):
    N = len(x)
    wx = np.bincount(f, w * x)
    w = np.bincount(f, w)

    # drop zeros
    # wx = wxw[wxw != 0]
    # w = w[w != 0]

    wxw = wx / w
    wxw_long = np.zeros(N)
    for j in range(len(wxw)):
        selector = f == j
        wxw_long[selector] = wxw[j]

    return wxw_long


def getfe(uhat, fe_fml, data):
    """
    Get fixed effects estimates after running a regression on demeaned data.
      Args:
          uhat: Residuals from a regression on demeaned data.
          fe_fml: A one sided formula with the fixed effects.
          data: A pandas dataframe with the fixed effects
      Returns:
          alpha: A numpy array with the fixed effects estimates.
      Example:
          get_fe(uhat, "~ firm + year", data)
    """

    # check if uhat is a numpy array
    if not isinstance(uhat, np.ndarray):
        raise ValueError("uhat must be a numpy array")
    if not isinstance(fe_fml, str):
        raise ValueError("fe_fml must be a string")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas dataframe")

    if not fe_fml.startswith("~") or len(fe_fml.split("~")) > 1:
        raise ValueError("fe_fml must be a one sided formula")

    D = model_matrix(fe_fml, data=data, output="sparse")
    DD = D.transpose().dot(D)
    Du = D.transpose().dot(uhat)
    alpha = sp.linalg.spsolve(DD, Du)

    return alpha
