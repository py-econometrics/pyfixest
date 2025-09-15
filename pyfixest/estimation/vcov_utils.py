from typing import Optional, Union

import numba as nb
import numpy as np
import pandas as pd
from numba import types as nb_types

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

    if ssc_dict["cluster_df"] == "min":
        G = [min(G)] * 3

    return G


def _get_vcov_type(
    vcov: Union[str, dict[str, str], None], fval: str
) -> Union[str, dict[str, str]]:
    """
    Pass the specified vcov type.

    Passes the specified vcov type. If no vcov type specified, sets the default
    vcov type as iid if no fixed effect is included in the model, and CRV1
    clustered by the first fixed effect if a fixed effect is included in the model.

    Parameters
    ----------
    vcov : Union[str, dict[str, str], None]
        The specified vcov type.
    fval : str
        The specified fixed effects. (i.e. "X1+X2")

    Returns
    -------
    str
        vcov_type (str) : The specified vcov type.
    """
    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"  # type: ignore
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}  # type: ignore
    else:
        vcov_type = vcov  # type: ignore

    return vcov_type  # type: ignore


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

    # Vectorized computation for all lag values
    for lag_value in range(lag + 1):
        weight = weights[lag_value]

        # Vectorized computation: scores[t] @ scores[t-lag_value].T for all t
        # This replaces the nested loops with matrix operations
        scores_current = scores[lag_value:time_periods]  # Shape: (time_periods-lag_value, k)
        scores_lagged = scores[:time_periods-lag_value]  # Shape: (time_periods-lag_value, k)

        # Compute gamma_lag = sum_t scores[t] @ scores[t-lag_value].T
        gamma_lag = scores_current.T @ scores_lagged

        # Add weighted contribution to meat matrix
        meat += weight * (gamma_lag + gamma_lag.T)

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
def _binary_search_time(time_arr: np.ndarray, start: int, end: int, target_time: int) -> int:
    """Binary search for exact time match. Returns -1 if not found."""
    left, right = start, end - 1

    while left <= right:
        mid = (left + right) // 2
        mid_time = time_arr[mid]

        if mid_time == target_time:
            return mid
        elif mid_time < target_time:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found


@nb.njit(parallel=True)
def _nw_meat_panel(
    scores: np.ndarray,
    time_arr: np.ndarray,
    panel_arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    lag: Optional[int] = None,
    balanced: Optional[bool] = None,
):
    """
    Compute the panel Newey-West (HAC) covariance estimator using C++ algorithm structure.

    Parameters
    ----------
    scores : ndarray, shape (N, k)
        Pre-computed scores matrix (X * u_hat), already sorted by panel and time.
    time_arr : ndarray, shape (N,)
        The time variable for clustering, already sorted by panel and time.
    panel_arr : ndarray, shape (N,)
        The panel variable for clustering, already sorted by panel and time.
    starts : np.ndarray
        The start index of each unit slice in the sorted arrays.
    counts : np.ndarray
        The length of each unit slice.
    lag : int, optional
        Maximum lag for autocovariance. If not provided, defaults to floor(N**0.25), where
        N is the number of time periods.

    Returns
    -------
    vcov_nw : ndarray, shape (k, k)
        HAC Newey-West covariance matrix.
    """
    N, k = scores.shape

    if lag is None:
        lag = int(np.floor(len(np.unique(time_arr)) ** 0.25))

    weights = _get_bartlett_weights(lag=lag)
    meat = np.zeros((k, k))

    # Pre-compute all k1, k2 pairs for parallel processing
    k_sq = k * k

    # LAG 0: Exactly like C++ - parallel over K×K matrix elements
    # l == 0 => easy: parallel over all matrix elements
    for index in nb.prange(k_sq):
        k1 = index // k
        k2 = index % k

        if k1 > k2:
            continue

        # Compute sum_i scores[i, k1] * scores[i, k2] for lag 0
        tmp = 0.0
        for i in range(N):
            tmp += scores[i, k1] * scores[i, k2]

        meat[k1, k2] = weights[0] * tmp
        if k1 != k2:
            meat[k2, k1] = weights[0] * tmp

    # LAG > 0: Check if panel is balanced for optimization
    unique_times = np.unique(time_arr)
    T = len(unique_times)

    # Auto-detect balanced panels if not specified
    if balanced is None:
        # Check if all panels have the same count and data is consecutive
        n_panels = len(starts)
        if n_panels > 0:
            first_count = counts[0]
            balanced = np.all(counts == first_count) and (first_count == T)
        else:
            balanced = False

    if balanced:
        # C++ balanced panel optimization: much simpler and faster
        G = len(starts)  # number of panels

        for l in range(1, lag + 1):
            # For balanced panels, we can use simple indexing
            if l >= T:
                continue

            # Number of observations per time period in balanced case
            obs_per_t = G
            start_curr = l * obs_per_t  # Start of current period observations
            start_lag = 0  # Start of lagged period observations

            n_valid_periods = T - l
            nmax = n_valid_periods * obs_per_t

            # Parallel over K×K elements for balanced case
            for index in nb.prange(k_sq):
                k1 = index // k
                k2 = index % k

                # Simple vectorized computation for balanced panels
                tmp = 0.0
                for i in range(nmax):
                    tmp += scores[start_curr + i, k1] * scores[start_lag + i, k2]

                meat[k1, k2] += weights[l] * tmp
    else:
        # Unbalanced panel case: use the complex matching logic
        # Pre-compute time period start/end indices (more efficient than storing arrays)
        time_starts = np.zeros(T, dtype=np.int32)
        time_ends = np.zeros(T, dtype=np.int32)

        for t in range(T):
            mask = (time_arr == unique_times[t])
            indices = np.where(mask)[0]
            time_starts[t] = indices[0] if len(indices) > 0 else 0
            time_ends[t] = indices[-1] + 1 if len(indices) > 0 else 0

        for l in range(1, lag + 1):
            # Estimate maximum possible matches to pre-allocate arrays
            max_matches = 0
            for t_idx in range(l, T):
                curr_count = time_ends[t_idx] - time_starts[t_idx]
                lag_count = time_ends[t_idx - l] - time_starts[t_idx - l]
                max_matches += min(curr_count, lag_count)

            # Pre-allocate arrays instead of using lists
            curr_obs_buffer = np.empty(max_matches, dtype=np.int32)
            lag_obs_buffer = np.empty(max_matches, dtype=np.int32)
            match_count = 0

            for t_idx in range(l, T):
                curr_start = time_starts[t_idx]
                curr_end = time_ends[t_idx]
                lag_start = time_starts[t_idx - l]
                lag_end = time_ends[t_idx - l]

                # Skip if no observations in either period
                if curr_start >= curr_end or lag_start >= lag_end:
                    continue

                # Two-pointer matching using direct array access
                i_curr = curr_start
                i_lag = lag_start

                while i_curr < curr_end and i_lag < lag_end:
                    if panel_arr[i_curr] == panel_arr[i_lag]:
                        # Match found! Store in pre-allocated buffers
                        curr_obs_buffer[match_count] = i_curr
                        lag_obs_buffer[match_count] = i_lag
                        match_count += 1
                        i_curr += 1
                        i_lag += 1
                    elif panel_arr[i_curr] < panel_arr[i_lag]:
                        i_curr += 1
                    else:
                        i_lag += 1

            # Use only the filled portion of buffers
            if match_count > 0:
                curr_obs = curr_obs_buffer[:match_count]
                lag_obs = lag_obs_buffer[:match_count]

                # Now parallel over K×K elements with pre-computed matches
                for index in nb.prange(k_sq):
                    k1 = index // k
                    k2 = index % k

                    # Vectorized computation over all matches for this (k1, k2) pair
                    tmp = np.sum(scores[curr_obs, k1] * scores[lag_obs, k2])
                    meat[k1, k2] += weights[l] * tmp

    # Final step: Add transpose (like C++ finishing step)
    for k1 in nb.prange(k):
        for k2 in range(k):
            meat[k1, k2] += meat[k2, k1]

    return meat


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
