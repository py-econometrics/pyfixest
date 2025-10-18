from typing import Optional, Union

import numba as nb
import numpy as np
import pandas as pd

from pyfixest.errors import NanInClusterVarError
from pyfixest.utils.dev_utils import _narwhals_to_pandas


def _compute_bread(
    _is_iv: bool,
    _tXZ: np.ndarray,
    _tZZinv: np.ndarray,
    _tZX: np.ndarray,
    _hessian: np.ndarray,
):
    return np.linalg.inv(_tXZ @ _tZZinv @ _tZX) if _is_iv else np.linalg.inv(_hessian)


def _get_cluster_df(data: pd.DataFrame, clustervar: list[str]):
    if not data.empty:
        data_pandas = _narwhals_to_pandas(data)
        cluster_df = data_pandas[clustervar].copy()
    else:
        raise AttributeError(
            """The input data set needs to be stored in the model object if
            you call `vcov()` post estimation with a novel cluster variable.
            Please set the function argument `store_data=True` when calling
            the regression.
            """
        )

    return cluster_df


def _check_cluster_df(cluster_df: pd.DataFrame, data: pd.DataFrame):
    if np.any(cluster_df.isna().any()):
        raise NanInClusterVarError(
            "CRV inference not supported with missing values in the cluster variable."
            "Please drop missing values before running the regression."
        )

    N = data.shape[0]
    if cluster_df.shape[0] != N:
        raise ValueError(
            "The cluster variable must have the same length as the data set."
        )


def _count_G_for_ssc_correction(
    cluster_df: pd.DataFrame, ssc_dict: dict[str, Union[str, bool]]
):
    G = []
    for col in cluster_df.columns:
        G.append(cluster_df[col].nunique())

    if ssc_dict["G_df"] == "min":
        G = [min(G)] * 3

    return G


def _get_vcov_type(
    vcov: Union[str, dict[str, str], None], fval: str
) -> Union[str, dict[str, str]]:
    """
    Pass the specified vcov type.

    Passes the specified vcov type. If no vcov type specified, always defaults
    to "iid" inference, regardless of whether fixed effects are included in the model.

    Parameters
    ----------
    vcov : Union[str, dict[str, str], None]
        The specified vcov type.
    fval : str
        The specified fixed effects. (i.e. "X1+X2")

    Returns
    -------
    str
        vcov_type (str) : The specified vcov type, or "iid" by default.
    """
    return vcov if vcov is not None else "iid"


@nb.njit(parallel=False)
def _hac_meat_loop(
    scores: np.ndarray, weights: np.ndarray, time_periods: int, k: int, lag: int
):
    """
    Compute the HAC meat matrix. Used for both time series and DK HAC.

    Parameters
    ----------
    scores: np.ndarray
        The scores matrix.
    weights: np.ndarray
        The weights matrix.
    time_periods: int
        The number of time periods.
    k: int
        The number of regressors.
    lag: int
        The number of lag for the HAC estimator.

    Returns
    -------
    meat: np.ndarray
        The HAC meat matrix.
    """
    meat = np.zeros((k, k))
    gamma_buffer = np.zeros((k, k))

    # Vectorized computation for all lag values
    for lag_value in range(lag + 1):
        gamma_buffer.fill(0.0)
        weight = weights[lag_value]

        scores_current = scores[lag_value:time_periods]
        scores_lagged = scores[: time_periods - lag_value]

        gamma_buffer[:, :] = scores_current.T @ scores_lagged
        meat += weight * (gamma_buffer + gamma_buffer.T)

    return meat


@nb.njit(parallel=False)
def _get_bartlett_weights(lag: int):
    # Pre-compute bartlett kernel weights more efficiently
    weights = np.empty(lag + 1)
    lag_plus_one = lag + 1
    for j in range(lag + 1):
        weights[j] = 1.0 - j / lag_plus_one
    weights[0] = 0.5  # Halve first weight

    return weights


@nb.njit(parallel=False)
def _nw_meat_time(scores: np.ndarray, time_arr: np.ndarray, lag: int):
    if time_arr is None:
        ordered_scores = scores
    else:
        order = np.argsort(time_arr)
        ordered_scores = scores[order]

    time_periods, k = ordered_scores.shape
    weights = _get_bartlett_weights(lag=lag)

    return _hac_meat_loop(
        scores=ordered_scores, weights=weights, time_periods=time_periods, k=k, lag=lag
    )


def _get_panel_idx(
    panel_arr: np.ndarray, time_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get indices for each unit. I.e. the first value ("starts") and how many
    observations ("counts") each unit has.

    Parameters
    ----------
    panel_arr : ndarray, shape (N*T,)
        Panel ID variable.
    time_arr : ndarray, shape (N*T,)
        Time ID variable.

    Returns
    -------
      order  : indices that sort by (panel, time)
      units  : unique panel ids in sorted order
      starts : start index of each unit slice in the sorted arrays
      counts : length of each unit slice
      panel_arr_sorted : panel variable in sorted order
      time_arr_sorted : time variable in sorted order
    """
    order = np.lexsort((time_arr, panel_arr))  # sort by panel, then time
    p_sorted = panel_arr[order]
    units, starts, counts = np.unique(p_sorted, return_index=True, return_counts=True)
    panel_arr_sorted = panel_arr[order]
    time_arr_sorted = time_arr[order]
    duplicate_mask = (np.diff(panel_arr_sorted) == 0) & (np.diff(time_arr_sorted) == 0)
    if np.any(duplicate_mask):
        raise ValueError(
            "There are duplicate time periods for the same panel id. This is not supported for HAC SEs."
        )
    return order, units, starts, counts, panel_arr_sorted, time_arr_sorted


@nb.njit(parallel=False)
def _nw_meat_panel(
    scores: np.ndarray,
    time_arr: np.ndarray,
    panel_arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    lag: Optional[int] = None,
):
    """
    Compute the panel Newey-West (HAC) covariance estimator.

    Parameters
    ----------
    scores: np.ndarray
        The scores matrix.
    time_arr : ndarray, shape (N*T,)
        The time variable for clustering.
    panel_arr : ndarray, shape (N*T,)
        The panel variable for clustering.
    starts : np.ndarray
        The start index of each unit slice in the sorted arrays.
    counts : np.ndarray
        The length of each unit slice.
    lag : int
        Maximum lag for autocovariance. If not provided, defaults to floor(N**0.25), where
        N is the number of time periods.

    Returns
    -------
    vcov_nw : ndarray, shape (k, k)
        HAC Newey-West covariance matrix.
    """
    if lag is None:
        lag = int(np.floor(len(np.unique(time_arr)) ** 0.25))

    weights = _get_bartlett_weights(lag=lag)

    k = scores.shape[1]

    meat_nw_panel = np.zeros((k, k))
    gamma_l = np.zeros((k, k))
    gamma_l_sum = np.zeros((k, k))

    for start, count in zip(starts, counts):
        end = start + count
        gamma0 = np.zeros((k, k))
        for t in range(start, end):
            score_t = scores[t, :]
            gamma0 += np.outer(score_t, score_t)

        gamma_l_sum.fill(0.0)
        Lmax = min(lag, count - 1)
        for lag_value in range(1, Lmax + 1):
            gamma_l.fill(0.0)
            for t in range(lag_value, count):
                curr_t = start + t
                prev_t = start + t - lag_value
                score_curr = scores[curr_t, :]
                score_prev = scores[prev_t, :]
                gamma_l += np.outer(score_curr, score_prev)
            gamma_l_sum += weights[lag_value] * (gamma_l + gamma_l.T)

        meat_nw_panel += gamma0 + gamma_l_sum

    return meat_nw_panel


@nb.njit(parallel=False)
def _dk_meat_panel(
    scores: np.ndarray,
    time_arr: np.ndarray,
    idx: np.ndarray,
    lag: Optional[int] = None,
):
    """Compute Driscoll-Kraay HAC meat matrix.

    Parameters
    ----------
    scores: np.ndarray
        The time-aggregated scores. Is assumed to be sorted by time.
    time_arr: np.ndarray, optional
        The time variable for clustering. Assume that there are no duplicate time periods.
        Is assumed to be sorted by time.
    idx: np.ndarray, optional
        The indices of the unique time periods.
    lag: int, optional
        The number of lag for the HAC estimator. Defaults to floor (# of time periods)^(1/4).
    """
    # Set lag if not provided
    if lag is None:
        lag = int(np.floor(np.unique(time_arr).shape[0] ** 0.25))

    scores_time = np.zeros((len(idx), scores.shape[1]))
    for t in range(len(idx) - 1):
        scores_time[t, :] = scores[idx[t] : idx[t + 1], :].sum(axis=0)
    scores_time[-1, :] = scores[idx[-1] :, :].sum(axis=0)

    time_periods, k = scores_time.shape

    weights = _get_bartlett_weights(lag=lag)

    return _hac_meat_loop(
        scores=scores_time, weights=weights, time_periods=time_periods, k=k, lag=lag
    )


def _prepare_twoway_clustering(clustervar: list, cluster_df: pd.DataFrame):
    cluster_one = clustervar[0]
    cluster_two = clustervar[1]
    cluster_df_one_str = cluster_df[cluster_one].astype(str)
    cluster_df_two_str = cluster_df[cluster_two].astype(str)
    cluster_df.loc[:, "cluster_intersection"] = cluster_df_one_str.str.cat(
        cluster_df_two_str, sep="-"
    )

    return cluster_df


# CODE from Styfen Schaer (@styfenschaer)
@nb.njit(parallel=False)
def bucket_argsort(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sorts the input array using the bucket sort algorithm.

    Parameters
    ----------
    arr : array_like
        An array_like object that needs to be sorted.

    Returns
    -------
    array_like
        A sorted copy of the input array.

    Raises
    ------
    ValueError
        If the input is not an array_like object.

    Notes
    -----
    The bucket sort algorithm works by distributing the elements of an array
    into a number of buckets. Each bucket is then sorted individually, either
    using a different sorting algorithm, or by recursively applying the bucket
    sorting algorithm.
    """
    counts = np.zeros(arr.max() + 1, dtype=np.uint32)
    for i in range(arr.size):
        counts[arr[i]] += 1

    locs = np.empty(counts.size + 1, dtype=np.uint32)
    locs[0] = 0
    pos = np.empty(counts.size, dtype=np.uint32)
    for i in range(counts.size):
        locs[i + 1] = locs[i] + counts[i]
        pos[i] = locs[i]

    args = np.empty(arr.size, dtype=np.uint32)
    for i in range(arr.size):
        e = arr[i]
        args[pos[e]] = i
        pos[e] += 1

    return args, locs


# CODE from Styfen Schaer (@styfenschaer)
@nb.njit(parallel=False)
def _crv1_meat_loop(
    scores: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    k = scores.shape[1]
    dtype = scores.dtype
    meat = np.zeros((k, k), dtype=dtype)

    g_indices, g_locs = bucket_argsort(cluster_col)

    score_g = np.empty((k, 1), dtype=dtype)
    meat_i = np.empty((k, k), dtype=dtype)

    for i in range(clustid.size):
        g = clustid[i]
        start = g_locs[g]
        end = g_locs[g + 1]
        g_index = g_indices[start:end]
        score_g = scores[g_index, :].sum(axis=0)
        np.outer(score_g, score_g, out=meat_i)
        meat += meat_i

    return meat
