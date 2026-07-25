from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all
from pyfixest.core.nw import (
    dk_meat_panel as _dk_meat_panel_rs,
)
from pyfixest.core.nw import (
    nw_meat_panel as _nw_meat_panel_rs,
)
from pyfixest.core.nw import (
    nw_meat_time as _nw_meat_time_rs,
)
from pyfixest.errors import NanInClusterVarError, VcovTypeNotSupportedError
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import get_ssc


@dataclass(frozen=True)
class VcovSpec:
    """One user-facing `vcov` option and what the dispatcher needs for it.

    Attributes
    ----------
    detail : str
        The user-facing name, stored on the model as `_vcov_type_detail`.
    family : str
        The coarse estimator family, stored as `_vcov_type`. Several details
        share a family: `HC1`, `HC2` and `HC3` are all `"hetero"`.
    takes_cluster : bool
        Whether the option is written as `{detail: clustervar}` rather than as
        a bare string, and therefore clusters.
    method_name : str | None
        Name of the `Feols` method returning the unscaled vcov. `None` for
        clustered specs, whose per-cluster loop runs in `run_crv_loop`.
    ssc_family : str | None
        The `vcov_type` handed to `get_ssc`. Defaults to `family`; only `nid`
        differs, borrowing the `"hetero"` correction.
    ssc_G : Callable[[Any], int] | None
        Returns the `G` that `get_ssc` needs, given the fitted model.
    prepare : Callable[[Any, dict | None], None] | None
        Runs before the ssc and vcov computation to unpack `vcov_kwargs` onto
        the model. Only the HAC estimators need it.
    check_kwargs : Callable[[dict | None], None] | None
        Validates `vcov_kwargs` at the API boundary, without a fitted model.
    check_model : Callable[[bool, bool], None] | None
        Validates the option against `(has_fixef, is_iv)` once the model is known.
    """

    detail: str
    family: str
    takes_cluster: bool = False
    method_name: str | None = None
    ssc_family: str | None = None
    ssc_G: Callable[[Any], int] | None = None
    prepare: Callable[[Any, dict | None], None] | None = None
    check_kwargs: Callable[[dict | None], None] | None = None
    check_model: Callable[[bool, bool], None] | None = None

    @property
    def ssc_vcov_type(self) -> str:
        "The `vcov_type` to hand to `get_ssc`."
        return self.ssc_family if self.ssc_family is not None else self.family


def _G_one(model: Any) -> int:
    return 1


def _G_nobs(model: Any) -> int:
    # fixest:::vcov_hetero_internal: adj = ifelse(ssc$cluster.adj, n/(n - 1), 1)
    return model._N


def _G_time_periods(model: Any) -> int:
    "Count the distinct time periods T used by the HAC estimator."
    return int(np.unique(model._data[model._time_id]).shape[0])


def _prepare_hac(model: Any, vcov_kwargs: dict | None) -> None:
    "Unpack the HAC keyword arguments onto the model."
    kw = vcov_kwargs or {}
    model._lag = kw.get("lag")
    model._time_id = kw.get("time_id")
    model._panel_id = kw.get("panel_id")


def _require_time_id(vcov_kwargs: dict | None) -> None:
    if not vcov_kwargs or "time_id" not in vcov_kwargs:
        raise ValueError("Missing required 'time_id' for NW/DK vcov")


def _reject_fixef_and_iv(has_fixef: bool, is_iv: bool) -> None:
    if has_fixef:
        raise VcovTypeNotSupportedError(
            "HC2 and HC3 inference types are not supported for regressions with fixed effects."
        )
    if is_iv:
        raise VcovTypeNotSupportedError(
            "HC2 and HC3 inference types are not supported for IV regressions."
        )


VCOV_REGISTRY: dict[str, VcovSpec] = {
    "iid": VcovSpec("iid", "iid", method_name="_vcov_iid", ssc_G=_G_one),
    "hetero": VcovSpec("hetero", "hetero", method_name="_vcov_hetero", ssc_G=_G_nobs),
    "HC1": VcovSpec("HC1", "hetero", method_name="_vcov_hetero", ssc_G=_G_nobs),
    "HC2": VcovSpec(
        "HC2",
        "hetero",
        method_name="_vcov_hetero",
        ssc_G=_G_nobs,
        check_model=_reject_fixef_and_iv,
    ),
    "HC3": VcovSpec(
        "HC3",
        "hetero",
        method_name="_vcov_hetero",
        ssc_G=_G_nobs,
        check_model=_reject_fixef_and_iv,
    ),
    "NW": VcovSpec(
        "NW",
        "HAC",
        method_name="_vcov_hac",
        ssc_G=_G_time_periods,
        prepare=_prepare_hac,
        check_kwargs=_require_time_id,
    ),
    "DK": VcovSpec(
        "DK",
        "HAC",
        method_name="_vcov_hac",
        ssc_G=_G_time_periods,
        prepare=_prepare_hac,
        check_kwargs=_require_time_id,
    ),
    "nid": VcovSpec(
        "nid", "nid", method_name="_vcov_nid", ssc_family="hetero", ssc_G=_G_nobs
    ),
    "CRV1": VcovSpec("CRV1", "CRV", takes_cluster=True),
    "CRV3": VcovSpec("CRV3", "CRV", takes_cluster=True),
}

VCOV_STRING_OPTIONS: tuple[str, ...] = tuple(
    k for k, spec in VCOV_REGISTRY.items() if not spec.takes_cluster
)
VCOV_CLUSTER_OPTIONS: tuple[str, ...] = tuple(
    k for k, spec in VCOV_REGISTRY.items() if spec.takes_cluster
)


@dataclass
class ClusterPrep:
    "Precomputed cluster state shared across the CRV per-cluster loop."

    cluster_df: pd.DataFrame
    cluster_arr_int: np.ndarray  # (N, n_cluster_cols), int-factorized
    G: list[int]  # cluster counts per column, post ssc_dict["G_df"] adjustment
    k_fe_nested: int
    n_fe_fully_nested: int


def prepare_cluster_state(
    *,
    data: DataFrameType,
    clustervar: list[str],
    ssc_dict: dict,
    fixef: str | None,
    fe: pd.DataFrame | np.ndarray | None,
    k_fe: np.ndarray | pd.Series,
) -> ClusterPrep:
    "Build cluster_df, int-factorized cluster array, G, and nested-FE counts."
    cluster_df = _get_cluster_df(data=data, clustervar=clustervar)
    _check_cluster_df(cluster_df=cluster_df, data=data)

    if cluster_df.shape[1] > 1:
        cluster_df = _prepare_twoway_clustering(
            clustervar=clustervar, cluster_df=cluster_df
        )

    G = _count_G_for_ssc_correction(cluster_df=cluster_df, ssc_dict=ssc_dict)

    cluster_arr_int = np.column_stack(
        [pd.factorize(cluster_df[col])[0] for col in cluster_df.columns]
    )

    k_fe_nested = 0
    n_fe_fully_nested = 0
    if fixef is not None and ssc_dict["k_fixef"] == "nonnested":
        if fe is None:
            raise ValueError("`fe` must not be None when `fixef` is specified.")
        k_fe_nested_flag, n_fe_fully_nested = count_fixef_fully_nested_all(
            all_fixef_array=np.array(fixef.replace("^", "_").split("+"), dtype=str),
            cluster_colnames=np.array(cluster_df.columns, dtype=str),
            cluster_data=cluster_arr_int.astype(np.uintp),
            fe_data=fe.to_numpy().astype(np.uintp)
            if isinstance(fe, pd.DataFrame)
            else fe.astype(np.uintp),
        )
        k_fe_nested = np.sum(k_fe[k_fe_nested_flag]) if n_fe_fully_nested > 0 else 0

    return ClusterPrep(
        cluster_df=cluster_df,
        cluster_arr_int=cluster_arr_int,
        G=G,
        k_fe_nested=k_fe_nested,
        n_fe_fully_nested=n_fe_fully_nested,
    )


def run_crv_loop(
    *,
    prep: ClusterPrep,
    k: int,
    make_ssc_kwargs: Callable[..., dict],
    cluster_vcov: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    "Accumulate per-cluster CRV vcov, ssc weights, df_k, and df_t."
    vcov_sign_list = [1, 1, -1]
    n_clusters = prep.cluster_df.shape[1]

    vcov = np.zeros((k, k))
    ssc_arr: np.ndarray | None = None
    df_t_full = np.zeros(n_clusters)
    df_k = 0

    for x in range(n_clusters):
        cluster_col = prep.cluster_arr_int[:, x]
        clustid = np.unique(cluster_col)

        ssc, df_k, df_t = get_ssc(
            **make_ssc_kwargs(
                vcov_type="CRV",
                G=prep.G[x],
                vcov_sign=vcov_sign_list[x],
                k_fe_nested=prep.k_fe_nested,
                n_fe_fully_nested=prep.n_fe_fully_nested,
            )
        )
        ssc_arr = np.array([ssc]) if ssc_arr is None else np.append(ssc_arr, ssc)
        df_t_full[x] = df_t
        vcov += ssc_arr[x] * cluster_vcov(clustid, cluster_col)

    assert ssc_arr is not None  # n_clusters >= 1 in the CRV branch
    return vcov, ssc_arr, df_k, int(np.min(df_t_full))


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
