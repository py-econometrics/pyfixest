from typing import Optional, Tuple, Union

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
def _nw_meat(scores: np.ndarray, time_arr: np.ndarray, lag: Optional[int] = None):
    """
    Compute Newey-West HAC meat matrix.

    Parameters
    ----------
    scores: np.ndarray
        The scores matrix.
    time_arr: np.ndarray, optional
        The time variable for clustering.
    lag: int, optional
        The number of lag for the HAC estimator. Defaults to floor (# of time periods)^(1/4).
    """
    order = np.argsort(time_arr)
    ordered_scores = scores[order]

    time_periods, k = ordered_scores.shape

    # resolve lag
    if lag is None:
        raise ValueError(
            "We still have not implemented the default Newey-West HAC lag. Please provide a lag value via the `vcov_kwargs`."
        )

    # bartlett kernel weights
    weights = np.array([1 - j / (lag + 1) for j in range(lag + 1)])
    weights[0] = 0.5  # Halve first weight

    meat = np.zeros((k, k))

    # this implementation follows the same that fixest does in R
    for lag_value in range(lag + 1):
        weight = weights[lag_value]
        gamma_lag = np.zeros((k, k))

        for t in range(lag_value, time_periods):
            gamma_lag += np.outer(
                ordered_scores[t, :], ordered_scores[t - lag_value, :]
            )

        meat += weight * (gamma_lag + gamma_lag.T)

    return meat


def _get_panel_idx(
    panel_id: np.ndarray, time_id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get indices for each unit. I.e. the first value ("starts") and how many
    observations ("counts") each unit has.

    Parameters
    ----------
    panel_id : ndarray, shape (N*T,)
        Panel ID variable.
    time_id : ndarray, shape (N*T,)
        Time ID variable.

    Returns
    -------
      order  : indices that sort by (panel, time)
      units  : unique panel ids in sorted order
      starts : start index of each unit slice in the sorted arrays
      counts : length of each unit slice
    """
    order = np.lexsort((time_id, panel_id))  # sort by panel, then time
    p_sorted = panel_id[order]
    units, starts, counts = np.unique(p_sorted, return_index=True, return_counts=True)
    return order, units, starts, counts


def _nw_meat_panel(
    X: np.ndarray,
    u_hat: np.ndarray,
    time_id: np.ndarray,
    panel_id: np.ndarray,
    lag: Optional[int] = None,
):
    """
    Computes the panel Newey-West (HAC) covariance estimator.

    Parameters
    ----------
    X : ndarray, shape (N*T, k)
        Stacked regressor matrix, where each block of T rows corresponds to one panel unit.
    u_hat : ndarray, shape (N*T,)
        Residuals from the panel regression.
    time_id : ndarray, shape (N*T,)
        Time ID variable.
    panel_id : ndarray, shape (N*T,)
        Panel ID variable.
    lag : int
        Maximum lag for autocovariance. If not provided, defaults to floor(N**0.25), where
        N is the number of time periods.

    Returns
    -------
    vcov_nw : ndarray, shape (k, k)
        HAC Newey-West covariance matrix.
    """
    if lag is None:
        lag = int(np.floor(len(np.unique(time_id)) ** 0.25))

    # order the data by (panel, time)
    order, units, starts, counts = _get_panel_idx(panel_id, time_id)

    X_sorted = X[order]
    u_sorted = u_hat[order]

    k = X.shape[1]

    meat_nw_panel = np.zeros((k, k))

    for start, count in zip(starts, counts):
        end = start + count
        gamma0 = np.zeros((k, k))
        for t in range(start, end):
            xi = X_sorted[t, :]
            gamma0 += np.outer(xi, xi) * u_sorted[t] ** 2

        gamma_l_sum = np.zeros((k, k))
        Lmax = min(lag, count - 1)
        for l in range(1, Lmax + 1):
            w = 1 - l / (lag + 1)
            gamma_l = np.zeros((k, k))
            for t in range(l, count):
                curr_t = start + t
                prev_t = start + t - l
                xi1 = X_sorted[curr_t, :] * u_sorted[curr_t]
                xi2 = X_sorted[prev_t, :] * u_sorted[prev_t]
                gamma_l += np.outer(xi1, xi2)
            gamma_l_sum += w * (gamma_l + gamma_l.T)

        meat_nw_panel += gamma0 + gamma_l_sum

    return meat_nw_panel


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
