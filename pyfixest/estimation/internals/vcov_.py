"""Pure, array-based vcov estimators.

Each function computes one (unscaled) variance-covariance matrix from numpy
arrays — no DataFrames, no model objects. Small-sample corrections (ssc) are
applied by the caller (``Feols.vcov``), exactly as before. Model classes
delegate their ``_vcov_*`` methods here.
"""

from __future__ import annotations

import numpy as np

from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.estimation.internals.vcov_utils import (
    _dk_meat_panel,
    _get_panel_idx,
    _nw_meat_panel,
    _nw_meat_time,
)


def _sandwich(
    meat: np.ndarray,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Assemble bread @ meat @ bread, with the IV projection of the meat."
    meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX if is_iv else meat
    return bread @ meat @ bread


def vcov_iid(residuals_wls: np.ndarray, bread: np.ndarray, N: int) -> np.ndarray:
    """Unscaled iid vcov: bread * sigma^2.

    ``residuals_wls`` are solve-scale residuals (sqrt(w)-scaled for WLS), so
    sigma^2 is the weighted mean squared error.
    """
    sigma2 = np.sum(residuals_wls.flatten() ** 2) / (N - 1)
    return bread * sigma2


def vcov_hetero(
    scores: np.ndarray,
    X_wls: np.ndarray,
    tZX: np.ndarray,
    weights: np.ndarray,
    weights_type: str,
    vcov_type_detail: str,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
) -> np.ndarray:
    """Unscaled heteroskedasticity-robust vcov (HC1/HC2/HC3).

    ``X_wls`` is the sqrt(w)-scaled design matrix of the final solve;
    ``weights`` are the weights of that solve (user weights for OLS/IV,
    IRLS weights for GLMs).
    """
    if vcov_type_detail in ["hetero", "HC1"]:
        transformed_scores = scores
    elif vcov_type_detail in ["HC2", "HC3"]:
        leverage = np.sum(X_wls * (X_wls @ np.linalg.inv(tZX)), axis=1)
        if weights_type == "fweights":
            leverage = leverage / weights.flatten()
        transformed_scores = (
            scores / np.sqrt(1 - leverage)[:, None]
            if vcov_type_detail == "HC2"
            else scores / (1 - leverage)[:, None]
        )

    # for fweights, need to divide by sqrt(weights)
    if weights_type == "fweights":
        transformed_scores = transformed_scores / np.sqrt(weights)

    Omega = transformed_scores.T @ transformed_scores
    return _sandwich(Omega, bread, is_iv, tXZ, tZZinv, tZX)


def vcov_hac(
    scores: np.ndarray,
    time_arr: np.ndarray,
    panel_arr: np.ndarray | None,
    lag: int | None,
    vcov_type_detail: str,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Unscaled HAC vcov: Newey-West (time or panel) or Driscoll-Kraay."
    if vcov_type_detail == "NW":
        # Newey-West
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
            # order the data by (panel, time)
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
        # Driscoll-Kraay

        order, _, starts, counts, time_arr_sorted, panel_arr_sorted = _get_panel_idx(
            # hack: sort first by time, than panel
            # we need the data sorted by time, but sort by
            # panel too to check for duplicate time periods
            # per panel
            panel_arr=time_arr,
            time_arr=panel_arr,
        )
        scores_sorted = scores[order]
        hac_meat = _dk_meat_panel(
            scores=scores_sorted, time_arr=time_arr_sorted, idx=starts, lag=lag
        )

    return _sandwich(hac_meat, bread, is_iv, tXZ, tZZinv, tZX)


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
    return _sandwich(meat, bread, is_iv, tXZ, tZZinv, tZX)


def _jackknife_vcov(beta_jack: np.ndarray, beta_center: np.ndarray) -> np.ndarray:
    "Aggregate leave-one-cluster-out coefficients into a CRV3 vcov."
    k = beta_jack.shape[1]
    vcov_mat = np.zeros((k, k))
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
    """Unscaled CRV3 vcov via the closed-form cluster jackknife (no fixed effects).

    ``X``/``Y`` are the arrays of the final solve (sqrt(w)-scaled for WLS).
    """
    k = X.shape[1]
    beta_jack = np.zeros((len(clustid), k))

    # inverse hessian precomputed?
    tXX = np.transpose(X) @ X
    tXy = np.transpose(X) @ Y

    # compute leave-one-out regression coefficients (aka clusterjacks')
    for ixg, g in enumerate(clustid):
        Xg = X[np.equal(g, cluster_col)]
        Yg = Y[np.equal(g, cluster_col)]
        tXgXg = np.transpose(Xg) @ Xg
        # jackknife regression coefficient
        beta_jack[ixg, :] = (
            np.linalg.pinv(tXX - tXgXg) @ (tXy - np.transpose(Xg) @ Yg)
        ).flatten()

    # optional: beta_bar in MNW (2022)
    # center = "estimate"
    # if center == 'estimate':
    #    beta_center = beta_hat
    # else:
    #    beta_center = np.mean(beta_jack, axis = 0)
    return _jackknife_vcov(beta_jack=beta_jack, beta_center=beta_hat)
