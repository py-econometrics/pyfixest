import numpy as np
import pandas as pd

from pyfixest.core.conley import (
    conley_meat as _conley_meat_rs,
)
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


def _normalize_conley_coordinates(
    lon_arr: np.ndarray, lat_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize Conley coordinates to longitude [-180, 180) and latitude [-90, 90]."""
    lon_arr = np.asarray(lon_arr, dtype=np.float64).flatten()
    lat_arr = np.asarray(lat_arr, dtype=np.float64).flatten()

    if not np.isfinite(lon_arr).all() or not np.isfinite(lat_arr).all():
        raise ValueError(
            "Conley inference is not supported with missing values or non-finite coordinate values."
        )

    lon_min = lon_arr.min(initial=0.0)
    lon_max = lon_arr.max(initial=0.0)
    lon_in_standard = lon_min >= -180 and lon_max <= 180
    lon_in_positive = lon_min >= 0 and lon_max <= 360
    lon_in_negative = lon_min >= -360 and lon_max <= 0
    if not (lon_in_standard or lon_in_positive or lon_in_negative):
        raise ValueError(
            "The longitude variable must be in [-180, 180], [0, 360], or [-360, 0]."
        )

    lat_min = lat_arr.min(initial=0.0)
    lat_max = lat_arr.max(initial=0.0)
    lat_in_standard = lat_min >= -90 and lat_max <= 90
    lat_in_positive = lat_min >= 0 and lat_max <= 180
    lat_in_negative = lat_min >= -180 and lat_max <= 0
    if not (lat_in_standard or lat_in_positive or lat_in_negative):
        raise ValueError(
            "The latitude variable must be in [-90, 90], [0, 180], or [-180, 0]."
        )

    if not lat_in_standard:
        lat_arr = lat_arr - 90 if lat_in_positive else lat_arr + 90

    lon_arr = np.mod(lon_arr + 180, 360) - 180
    return lon_arr, lat_arr


def _aggregate_conley_scores(
    scores: np.ndarray, lon_arr: np.ndarray, lat_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum scores over exact coordinate cells."""
    coordinates = np.column_stack((lat_arr, lon_arr))
    unique_coordinates, inverse = np.unique(coordinates, axis=0, return_inverse=True)

    if unique_coordinates.shape[0] == scores.shape[0]:
        return scores, lon_arr, lat_arr

    scores_agg = np.zeros(
        (unique_coordinates.shape[0], scores.shape[1]), dtype=np.float64
    )
    np.add.at(scores_agg, inverse, scores)
    return scores_agg, unique_coordinates[:, 1], unique_coordinates[:, 0]


def _conley_meat(
    scores: np.ndarray,
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
    cutoff: float,
    distance: str = "triangular",
    aggregate: bool = True,
) -> np.ndarray:
    """Compute Conley spatial HAC meat matrix (Rust backend)."""
    distance_code = {"spherical": 1, "triangular": 2}[distance]

    scores = np.ascontiguousarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("Scores must be a two-dimensional array.")
    # Python owns coordinate normalization because exact-coordinate aggregation
    # depends on canonical coordinate cells before entering the Rust kernel.
    lon_arr, lat_arr = _normalize_conley_coordinates(lon_arr, lat_arr)

    if scores.shape[0] != lon_arr.shape[0] or scores.shape[0] != lat_arr.shape[0]:
        raise ValueError(
            "Scores, longitude, and latitude arrays must have the same number of observations."
        )

    if aggregate:
        scores, lon_arr, lat_arr = _aggregate_conley_scores(scores, lon_arr, lat_arr)

    return np.asarray(
        _conley_meat_rs(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(lon_arr, dtype=np.float64),
            np.ascontiguousarray(lat_arr, dtype=np.float64),
            distance_code,
            cutoff,
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
