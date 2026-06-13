from __future__ import annotations

import numpy as np


def _sandwich(
    meat: np.ndarray,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
    tZX: np.ndarray,
) -> np.ndarray:
    "Assemble bread @ meat @ bread, with the IV projection of the meat."
    pass


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
    weights_type: str,
    vcov_type_detail: str,
    bread: np.ndarray,
    is_iv: bool,
    tXZ: np.ndarray,
    tZZinv: np.ndarray,
) -> np.ndarray:
    "Unscaled heteroskedasticity-robust vcov (HC1/HC2/HC3)."
    pass


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
    pass


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
    pass


def _jackknife_vcov(beta_jack: np.ndarray, beta_center: np.ndarray) -> np.ndarray:
    "Aggregate leave-one-cluster-out coefficients into a CRV3 vcov."
    pass


def vcov_crv3_fast(
    X: np.ndarray,
    Y: np.ndarray,
    beta_hat: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    "Unscaled CRV3 vcov via the closed-form cluster jackknife (no fixed effects)."
    pass
