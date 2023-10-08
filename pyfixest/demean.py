import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import pyhdfe
from numba import jit, njit, prange

def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: Optional[pd.DataFrame],
    weights: Optional[np.ndarray],
    lookup_demeaned_data: Dict[str, Any],
    na_index_str: str,
    drop_singletons: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Demeans a single regression model.

    If the model has fixed effects, the fixed effects are demeaned using the PyHDFE package.
    Prior to demeaning, the function checks if some of the variables have already been demeaned and uses values
    from the cache `lookup_demeaned_data` if possible. If the model has no fixed effects, the function does not demean the data.

    Args:
        Y (pd.DataFrame): A DataFrame of the dependent variable.
        X (pd.DataFrame): A DataFrame of the covariates.
        fe (pd.DataFrame or None): A DataFrame of the fixed effects. None if no fixed effects specified.
        weights (np.ndarray or None): A numpy array of weights. None if no weights.
        lookup_demeaned_data (Dict[str, Any]): A dictionary with keys for each fixed effects combination and
            potentially values of demeaned data frames. The function checks this dictionary to see if some of
            the variables have already been demeaned.
        na_index_str (str): A string with indices of dropped columns. Used for caching of demeaned variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: A tuple of the following elements:
            - Yd (pd.DataFrame): A DataFrame of the demeaned dependent variable.
            - Xd (pd.DataFrame): A DataFrame of the demeaned covariates.
            - Id (pd.DataFrame or None): A DataFrame of the demeaned Instruments. None if no IV.
    """

    YX = pd.concat([Y, X], axis=1)

    yx_names = YX.columns
    YX = YX.to_numpy()

    if fe is not None:
        # check if looked dict has data for na_index
        if lookup_demeaned_data.get(na_index_str) is not None:
            # get data out of lookup table: list of [algo, data]
            algorithm, YX_demeaned_old = lookup_demeaned_data.get(na_index_str)

            # get not yet demeaned covariates
            var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

            # if some variables still need to be demeaned
            if var_diff_names:
                # var_diff_names = var_diff_names

                yx_names_list = list(yx_names)
                var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                # var_diff_index = list(yx_names).index(var_diff_names)
                var_diff = YX[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                weights = np.ones(YX.shape[0])
                YX_demean_new = demean(var_diff, fe.to_numpy(), weights)
                YX_demeaned = pd.DataFrame(YX_demean_new)

                YX_demeaned = np.concatenate([YX_demeaned_old, YX_demean_new], axis=1)
                YX_demeaned = pd.DataFrame(YX_demeaned)

                # check if var_diff_names is a list
                if isinstance(var_diff_names, str):
                    var_diff_names = [var_diff_names]

                YX_demeaned.columns = list(YX_demeaned_old.columns) + var_diff_names

            else:
                # all variables already demeaned
                YX_demeaned = YX_demeaned_old[yx_names]

        else:
            # not data demeaned yet for NA combination
            algorithm = pyhdfe.create(
                ids=fe,
                residualize_method="map",
                drop_singletons=drop_singletons,
                # weights=weights
            )

            if (
                drop_singletons == True
                and algorithm.singletons != 0
                and algorithm.singletons is not None
            ):
                print(
                    algorithm.singletons,
                    "columns are dropped due to singleton fixed effects.",
                )
                dropped_singleton_indices = np.where(algorithm._singleton_indices)[
                    0
                ].tolist()
                na_index += dropped_singleton_indices

            weights = np.ones(YX.shape[0])

            YX_demeaned = demean(cx = YX, flist = fe.to_numpy(), weights = weights)
            YX_demeaned = pd.DataFrame(YX_demeaned)
            YX_demeaned.columns = yx_names

        lookup_demeaned_data[na_index_str] = [algorithm, YX_demeaned]

    else:
        # nothing to demean here
        pass

        YX_demeaned = pd.DataFrame(YX)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd



@njit(parallel = False, cache = False, fastmath = False)
def demean(cx, flist, weights, tol = 1e-10, maxiter = 2000):

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

    # loop over all variables to demean, in parallel
    for k in prange(K):

        cxk = cx[:,k]#.copy()
        oldxk = cxk - 1

        converged = False
        for _ in range(maxiter):

            for i in range(fixef_vars):
                fmat = flist[:,i]
                weighted_ave = _ave3(cxk, fmat, weights)
                cxk -= weighted_ave

            if np.sum(np.abs(cxk - oldxk)) < tol:
                converged = True
                break

            # update
            oldxk = cxk.copy()



        res[:,k] = cxk

    return res

@njit
def _ave3(x, f, w):

    N = len(x)

    wx_dict = {}
    w_dict = {}

    # Compute weighted sums using a dictionary
    for i in prange(N):
        j = f[i]
        if j in wx_dict:
            wx_dict[j] += w[i] * x[i]
        else:
            wx_dict[j] = w[i] * x[i]

        if j in w_dict:
            w_dict[j] += w[i]
        else:
            w_dict[j] = w[i]

    # Convert the dictionaries to arrays
    wx = np.zeros_like(f, dtype=x.dtype)
    w = np.zeros_like(f, dtype=w.dtype)

    for i in range(N):
        j = f[i]
        wx[i] = wx_dict[j]
        w[i] = w_dict[j]

    # Compute the average
    wxw_long = wx / w

    return wxw_long