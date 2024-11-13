from typing import Union

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
    _Z: np.ndarray,
    weighted_uhat: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    k = _Z.shape[1]
    dtype = _Z.dtype
    meat = np.zeros((k, k), dtype=dtype)

    g_indices, g_locs = bucket_argsort(cluster_col)

    score_g = np.empty((k, 1), dtype=dtype)
    meat_i = np.empty((k, k), dtype=dtype)

    for i in range(clustid.size):
        g = clustid[i]
        start = g_locs[g]
        end = g_locs[g + 1]
        g_index = g_indices[start:end]

        Zg = _Z[g_index]
        ug = weighted_uhat[g_index]

        np.dot(Zg.T, ug, out=score_g)
        np.outer(score_g, score_g, out=meat_i)
        meat += meat_i

    return meat
