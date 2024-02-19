from typing import Any, Optional

import numba as nb
import numpy as np
import pandas as pd


def demean_model(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    fe: Optional[pd.DataFrame],
    weights: Optional[np.ndarray],
    lookup_demeaned_data: dict[str, Any],
    na_index_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Demean a regression model.

    Demeans a single regression model via the alternating projections algorithm
    (see `demean` function). Prior to demeaning, the function checks if some of
    the variables have already been demeaned and uses values from the cache
    `lookup_demeaned_data` if possible. If the model has no fixed effects, the
    function does not demean the data.

    Parameters
    ----------
    Y : pandas.DataFrame
        A DataFrame of the dependent variable.
    X : pandas.DataFrame
        A DataFrame of the covariates.
    fe : pandas.DataFrame or None
        A DataFrame of the fixed effects. None if no fixed effects specified.
    weights : numpy.ndarray or None
        A numpy array of weights. None if no weights.
    lookup_demeaned_data : dict[str, Any]
        A dictionary with keys for each fixed effects combination and potentially
        values of demeaned data frames. The function checks this dictionary to
        see if some of the variables have already been demeaned.
    na_index_str : str
        A string with indices of dropped columns. Used for caching of demeaned
        variables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]
        A tuple of the following elements:
        - Yd : pd.DataFrame
            A DataFrame of the demeaned dependent variable.
        - Xd : pd.DataFrame
            A DataFrame of the demeaned covariates.
        - Id : pd.DataFrame or None
            A DataFrame of the demeaned Instruments. None if no IV.
    """
    YX = pd.concat([Y, X], axis=1)

    yx_names = YX.columns
    YX = YX.to_numpy()

    if YX.dtype != np.dtype("float64"):
        YX = YX.astype(np.float64)

    if weights is not None and weights.ndim > 1:
        weights = weights.flatten()

    if fe is not None:
        fe = fe.to_numpy()
        # check if looked dict has data for na_index
        if lookup_demeaned_data.get(na_index_str) is not None:
            # get data out of lookup table: list of [algo, data]
            _, YX_demeaned_old = lookup_demeaned_data.get(na_index_str)

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

                YX_demean_new, success = demean(var_diff, fe, weights)
                if success is False:
                    raise ValueError("Demeaning failed after 100_000 iterations.")

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
            YX_demeaned, success = demean(x=YX, flist=fe, weights=weights)
            if success is False:
                raise ValueError("Demeaning failed after 100_000 iterations.")

            YX_demeaned = pd.DataFrame(YX_demeaned)
            YX_demeaned.columns = yx_names

        lookup_demeaned_data[na_index_str] = [None, YX_demeaned]

    else:
        # nothing to demean here
        pass

        YX_demeaned = pd.DataFrame(YX)
        YX_demeaned.columns = yx_names

    # get demeaned Y, X (if no fixef, equal to Y, X, I)
    Yd = YX_demeaned[Y.columns]
    Xd = YX_demeaned[X.columns]

    return Yd, Xd


@nb.njit
def _sad_converged(a, b, tol):
    for i in range(a.size):
        if np.abs(a[i] - b[i]) >= tol:
            return False
    return True


@nb.njit(locals=dict(id=nb.uint32))
def _subtract_weighted_group_mean(
    x,
    sample_weights,
    group_ids,
    group_weights,
    _group_weighted_sums,
):
    _group_weighted_sums[:] = 0

    for i in range(x.size):
        id = group_ids[i]
        _group_weighted_sums[id] += sample_weights[i] * x[i]

    for i in range(x.size):
        id = group_ids[i]
        x[i] -= _group_weighted_sums[id] / group_weights[id]


@nb.njit
def _calc_group_weights(sample_weights, group_ids, n_groups):
    n_samples, n_factors = group_ids.shape
    dtype = sample_weights.dtype
    group_weights = np.zeros((n_factors, n_groups), dtype=dtype).T

    for j in range(n_factors):
        for i in range(n_samples):
            id = group_ids[i, j]
            group_weights[id, j] += sample_weights[i]

    return group_weights


@nb.njit(parallel=True)
def demean(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[np.ndarray, bool]:
    """
    Demean an array.

    Workhorse for demeaning an input array `x` based on the specified fixed
    effects and weights via the alternating projections algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n_samples, n_features). Needs to be of type float.
    flist : numpy.ndarray
        Array of shape (n_samples, n_factors) specifying the fixed effects.
        Needs to already be converted to integers.
    weights : numpy.ndarray
        Array of shape (n_samples,) specifying the weights.
    tol : float, optional
        Tolerance criterion for convergence. Defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 100_000.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        A tuple containing the demeaned array of shape (n_samples, n_features)
        and a boolean indicating whether the algorithm converged successfully.
    """
    n_samples, n_features = x.shape
    n_factors = flist.shape[1]

    if x.flags.f_contiguous:
        res = np.empty((n_features, n_samples), dtype=x.dtype).T
    else:
        res = np.empty((n_samples, n_features), dtype=x.dtype)

    n_threads = nb.get_num_threads()

    n_groups = flist.max() + 1
    group_weights = _calc_group_weights(weights, flist, n_groups)
    _group_weighted_sums = np.empty((n_threads, n_groups), dtype=x.dtype)

    x_curr = np.empty((n_threads, n_samples), dtype=x.dtype)
    x_prev = np.empty((n_threads, n_samples), dtype=x.dtype)

    not_converged = 0
    for k in nb.prange(n_features):
        tid = nb.get_thread_id()

        xk_curr = x_curr[tid, :]
        xk_prev = x_prev[tid, :]
        for i in range(n_samples):
            xk_curr[i] = x[i, k]
            xk_prev[i] = x[i, k] - 1.0

        for _ in range(maxiter):
            for j in range(n_factors):
                _subtract_weighted_group_mean(
                    xk_curr,
                    weights,
                    flist[:, j],
                    group_weights[:, j],
                    _group_weighted_sums[tid, :],
                )
            if _sad_converged(xk_curr, xk_prev, tol):
                break

            xk_prev[:] = xk_curr[:]
        else:
            not_converged += 1

        res[:, k] = xk_curr[:]

    success = not not_converged
    return (res, success)
