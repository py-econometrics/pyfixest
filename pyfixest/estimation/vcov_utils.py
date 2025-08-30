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
def _nw_meat(scores, time_var=None, lags=None, data=None):
    """
    Compute Newey-West HAC meat matrix.

    Parameters
    ----------
    scores: np.ndarray
        The scores matrix.
    time_var: np.ndarray, optional
        The time variable for clustering. Default is None.
    lags: int, optional
        The number of lags for the HAC estimator. Defaults to floor (# of time periods)^(1/4).
    data: pd.DataFrame, optional
        The original data set. Default is None.
    """
    # Determine whether data is time ordered or not - if not provided, data is assumed to be time ordered
    if time_var is None:
        ordered_scores = scores
        n_time = len(ordered_scores)
    else:
        time_data = data[
            time_var
        ].to_numpy()  # need to ensure that this is datetime format - might have to write an internal converter for pf.
        order = np.argsort(time_data)
        ordered_scores = scores[order]
        n_time = len(np.unique(time_data))

    # resolve lags
    if lags is None:
        # these are the fixest default lags for HAC
        # see https://lrberge.github.io/fixest/reference/vcov_hac.html
        lags = int(np.floor(n_time ** (1 / 4)))

    # bartlett kernel weights
    weights = np.array([1 - j / (lags + 1) for j in range(lags + 1)])
    weights[0] = 0.5  # Halve first weight

    n, k = ordered_scores.shape
    meat = np.zeros((k, k))

    for j in range(lags + 1):
        if j == 0:
            gamma_j = ordered_scores.T @ ordered_scores

            meat += weights[j] * gamma_j
        else:
            gamma_j = ordered_scores[j:].T @ ordered_scores[:-j]
            meat += weights[j] * (gamma_j + gamma_j.T)

    meat = meat / n

    return meat



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
