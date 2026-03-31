import numba as nb
import numpy as np
import pandas as pd

from pyfixest.core.nw import (
    dk_meat_panel as _dk_meat_panel_rs,
)
from pyfixest.core.nw import (
    nw_meat_panel as _nw_meat_panel_rs,
)
from pyfixest.core.nw import (
    nw_meat_time as _nw_meat_time_rs,
)
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
    cluster_df: pd.DataFrame, ssc_dict: dict[str, str | bool]
):
    G = []
    for col in cluster_df.columns:
        G.append(cluster_df[col].nunique())

    if ssc_dict["G_df"] == "min":
        G = [min(G)] * 3

    return G


def _get_vcov_type(
    vcov: str | dict[str, str] | None,
) -> str | dict[str, str]:
    """
    Pass the specified vcov type.

    Passes the specified vcov type. If no vcov type specified, always defaults
    to "iid" inference, regardless of whether fixed effects are included in the model.

    Parameters
    ----------
    vcov : Union[str, dict[str, str], None]
        The specified vcov type.

    Returns
    -------
    str
        vcov_type (str) : The specified vcov type, or "iid" by default.
    """
    return vcov if vcov is not None else "iid"


def _nw_meat_time(scores: np.ndarray, time_arr: np.ndarray, lag: int):
    """Compute time-series Newey-West HAC meat matrix (Rust backend)."""
    return np.asarray(
        _nw_meat_time_rs(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(time_arr, dtype=np.float64),
            lag,
        )
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


def _nw_meat_panel(
    scores: np.ndarray,
    time_arr: np.ndarray,
    panel_arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    lag: int | None = None,
):
    """Compute the panel Newey-West (HAC) meat matrix (Rust backend)."""
    if lag is None:
        lag = int(np.floor(len(np.unique(time_arr)) ** 0.25))
    return np.asarray(
        _nw_meat_panel_rs(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(starts, dtype=np.uint64),
            np.ascontiguousarray(counts, dtype=np.uint64),
            lag,
        )
    )


def _dk_meat_panel(
    scores: np.ndarray,
    time_arr: np.ndarray,
    idx: np.ndarray,
    lag: int | None = None,
):
    """Compute Driscoll-Kraay HAC meat matrix (Rust backend)."""
    if lag is None:
        lag = int(np.floor(np.unique(time_arr).shape[0] ** 0.25))
    return np.asarray(
        _dk_meat_panel_rs(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(idx, dtype=np.uint64),
            lag,
        )
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


@nb.njit(parallel=False)
def _crv1_vcov_loop(
    X: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
    q: float,
    u_hat: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, k = X.shape

    A = np.zeros((k, k))
    B = np.zeros((k, k))
    g_indices, g_locs = bucket_argsort(cluster_col)

    eps = 1e-7

    for g in clustid:
        start = g_locs[g]
        end = g_locs[g + 1]
        g_index = g_indices[start:end]

        Xg = X[g_index, :]
        ug = u_hat[g_index]

        ng = g_index.size
        for i in range(ng):
            Xgi = Xg[i, :]
            psi_i = q - 1.0 * (ug[i] <= eps)
            for j in range(ng):
                Xgj = Xg[j, :]
                psi_j = q - 1.0 * (ug[j] <= eps)
                A += np.outer(Xgi, Xgj) * psi_i * psi_j

            mask_i = (np.abs(ug[i]) < delta) * 1.0
            B += np.outer(Xgi, Xgi) * mask_i

    B /= 2 * delta

    return A, B
