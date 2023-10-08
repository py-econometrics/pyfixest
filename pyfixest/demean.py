import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import pyhdfe
import numba as nb 


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
                YX_demean_new, success = demean(var_diff, fe.to_numpy(), weights)
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
                    "observations are dropped due to singleton fixed effects.",
                )
                dropped_singleton_indices = np.where(algorithm._singleton_indices)[
                    0
                ].tolist()
                na_index += dropped_singleton_indices

                YX = np.delete(YX, dropped_singleton_indices, axis=0)

            weights = np.ones(YX.shape[0])

            YX_demeaned, success = demean(cx = YX, flist = fe.to_numpy(), weights = weights)
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


@nb.njit
def _sad_converged(a, b, tol):
    for i in range(0, a.size, 4):
        tol -= np.abs(a[i] - b[i])
        if tol < 0:
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
    tol: float = 1e-10,
    maxiter: int = 2_000,
) -> tuple[np.ndarray, bool]:
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
