from __future__ import annotations

from typing import Literal

import numpy as np

from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.estimation.internals.literals import WeightsTypeOptions
from pyfixest.estimation.internals.vcov_utils import (
    _dk_meat_panel,
    _get_panel_idx,
    _nw_meat_panel,
    _nw_meat_time,
)

HeteroVcovTypeOptions = Literal["hetero", "HC1", "HC2", "HC3"]
HacVcovTypeOptions = Literal["NW", "DK"]


def _sandwich(
    meat: np.ndarray,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Assemble bread @ meat @ bread, with the IV projection of the meat."
    projected_meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX if is_iv else meat
    return bread @ projected_meat @ bread


def vcov_iid_ols(residuals: np.ndarray, bread: np.ndarray, N: int) -> np.ndarray:
    "IID Variance-Covariance Matrix for OLS."
    sigma2 = np.sum(residuals.flatten() ** 2) / (N - 1)
    return bread * sigma2


def vcov_iid_glm(bread: np.ndarray) -> np.ndarray:
    "IID Variance-Covariance Matrix for GLMs."
    return bread


def vcov_hetero(
    scores: np.ndarray,
    X: np.ndarray,
    tZX: np.ndarray,
    weights: np.ndarray,
    weights_type: WeightsTypeOptions | None,
    vcov_type_detail: HeteroVcovTypeOptions,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
) -> np.ndarray:
    "Unscaled heteroskedasticity-robust vcov (HC1/HC2/HC3)."
    if vcov_type_detail in ["hetero", "HC1"]:
        transformed_scores = scores
    elif vcov_type_detail in ["HC2", "HC3"]:
        leverage = np.sum(X * (X @ np.linalg.inv(tZX)), axis=1)
        if weights_type == "fweights":
            leverage = leverage / weights.flatten()
        transformed_scores = (
            scores / np.sqrt(1 - leverage)[:, None]
            if vcov_type_detail == "HC2"
            else scores / (1 - leverage)[:, None]
        )
    else:
        raise ValueError(
            f"vcov_type_detail must be one of {HeteroVcovTypeOptions}, got {vcov_type_detail}."
        )

    # for fweights, need to divide by sqrt(weights)
    if weights_type == "fweights":
        transformed_scores = transformed_scores / np.sqrt(weights)

    Omega = transformed_scores.T @ transformed_scores

    meat = tXZ @ tZZinv @ Omega @ tZZinv @ tZX if is_iv else Omega
    vcov = bread @ meat @ bread

    return vcov


def vcov_hac(
    scores: np.ndarray,
    time_arr: np.ndarray,
    panel_arr: np.ndarray | None,
    lag: int | None,
    vcov_type_detail: HacVcovTypeOptions,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Unscaled HAC vcov: Newey-West (time or panel) or Driscoll-Kraay."
    if vcov_type_detail == "NW":
        if panel_arr is None:
            if lag is None:
                raise ValueError(
                    "We have not yet implemented the default Newey-West HAC lag. Please provide a lag value via the `vcov_kwargs`."
                )
            if len(np.unique(time_arr)) != len(time_arr):
                raise ValueError(
                    "There are duplicate time periods in the data. This is not supported for HAC SEs."
                )
            hac_meat = _nw_meat_time(scores=scores, time_arr=time_arr, lag=lag)
        else:
            order, _, starts, counts, panel_arr_sorted, time_arr_sorted = (
                _get_panel_idx(panel_arr=panel_arr, time_arr=time_arr)
            )

            hac_meat = _nw_meat_panel(
                scores=scores[order],
                time_arr=time_arr_sorted,
                panel_arr=panel_arr_sorted,
                starts=starts,
                counts=counts,
                lag=lag,
            )
    elif vcov_type_detail == "DK":
        if panel_arr is None:
            raise ValueError("Missing required 'panel_id' for DK vcov")

        order, _, starts, _, time_arr_sorted, _ = _get_panel_idx(
            # hack: sort first by time, than panel
            # we need the data sorted by time, but sort by
            # panel too to check for duplicate time periods
            # per panel
            panel_arr=time_arr,
            time_arr=panel_arr,
        )
        hac_meat = _dk_meat_panel(
            scores=scores[order], time_arr=time_arr_sorted, idx=starts, lag=lag
        )
    else:
        raise ValueError("vcov_type_detail must be one of 'NW' or 'DK'.")

    return _sandwich(
        meat=hac_meat,
        bread=bread,
        is_iv=is_iv,
        tXZ=tXZ,
        tZZinv=tZZinv,
        tZX=tZX,
    )


def vcov_crv1(
    scores: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Unscaled CRV1 cluster-robust vcov."
    meat = crv1_meat_loop(
        scores=scores.astype(np.float64),
        clustid=clustid.astype(np.uintp),
        cluster_col=cluster_col.astype(np.uintp),
    )

    return _sandwich(
        meat=meat,
        bread=bread,
        is_iv=is_iv,
        tXZ=tXZ,
        tZZinv=tZZinv,
        tZX=tZX,
    )


def _jackknife_vcov(beta_jack: np.ndarray, beta_center: np.ndarray) -> np.ndarray:
    "Aggregate leave-one-cluster-out coefficients into a CRV3 vcov."
    vcov_mat = np.zeros((beta_jack.shape[1], beta_jack.shape[1]))
    for ixg in range(beta_jack.shape[0]):
        beta_centered = beta_jack[ixg, :] - beta_center
        vcov_mat += np.outer(beta_centered, beta_centered)

    return vcov_mat


def vcov_crv3_fast(
    X: np.ndarray,
    Y: np.ndarray,
    beta_hat: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    "Unscaled CRV3 vcov via the closed-form cluster jackknife (no fixed effects)."
    beta_jack = np.zeros((len(clustid), X.shape[1]))

    tXX = X.T @ X
    tXy = X.T @ Y

    for ixg, g in enumerate(clustid):
        Xg = X[np.equal(g, cluster_col)]
        Yg = Y[np.equal(g, cluster_col)]
        tXgXg = Xg.T @ Xg
        beta_jack[ixg, :] = (np.linalg.pinv(tXX - tXgXg) @ (tXy - Xg.T @ Yg)).flatten()

    return _jackknife_vcov(beta_jack=beta_jack, beta_center=beta_hat)
