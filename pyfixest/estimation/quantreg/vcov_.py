from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

from pyfixest.core.crv1 import crv1_vcov_qreg_loop
from pyfixest.estimation.internals.literals import QuantregMethodOptions
from pyfixest.estimation.quantreg.frisch_newton_ip import QuantregSolution
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth


def _invert_covariance_component(matrix: np.ndarray, *, component: str) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Quantile regression could not compute the {component} variance-covariance matrix. "
            f"A {matrix.shape[0]} by {matrix.shape[1]} matrix could not be inverted. "
            "Check for redundant regressors or insufficient variation at the fitted quantile."
        ) from exc


def vcov_iid_qreg(
    X: np.ndarray, Y: np.ndarray, u_hat: np.ndarray, q: float, N: int
) -> np.ndarray:
    "Implement the kernel-based sandwich estimator from Powell (1991)."
    h = get_hall_sheather_bandwidth(q=q, N=N)
    # interquartile range of u_hat - this is what both quantreg and statsmodels use
    # (all three logical lines below in fact)
    rq = np.quantile(np.abs(u_hat), 0.75) - np.quantile(np.abs(u_hat), 0.25)
    sigma = np.std(Y)
    hk = np.minimum(sigma, rq / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

    f = 1 / (2 * N * hk) * np.sum(np.abs(u_hat) < hk)

    D = X.T @ X
    Dinv = _invert_covariance_component(D, component="iid")

    return 1 / (f**2) * q * (1 - q) * Dinv


def vcov_hetero_qreg(
    X: np.ndarray, Y: np.ndarray, u_hat: np.ndarray, q: float, N: int
) -> np.ndarray:
    """
    Implement R quantreg's heteroskedasticity-robust kernel sandwich.

    This follows ``summary.rq(se="ker")`` from R's quantreg package: a
    Hall-Sheather probability bandwidth, a Gaussian density estimate for each
    residual, and the Powell (1991) sandwich covariance estimator.
    """
    h = get_hall_sheather_bandwidth(q=q, N=N)
    while (q - h < 0) or (q + h > 1):
        h /= 2

    residual_iqr = np.quantile(u_hat, 0.75) - np.quantile(u_hat, 0.25)
    residual_scale = min(np.std(u_hat, ddof=1), residual_iqr / 1.34)
    residual_bandwidth = residual_scale * (norm.ppf(q + h) - norm.ppf(q - h))
    if not np.isfinite(residual_bandwidth) or residual_bandwidth <= 0:
        raise ValueError(
            "The kernel residual bandwidth must be finite and strictly positive."
        )

    density = norm.pdf(u_hat / residual_bandwidth) / residual_bandwidth
    hessian = X.T @ (density[:, np.newaxis] * X)
    bread = _invert_covariance_component(hessian, component="heteroskedasticity-robust")
    meat = q * (1 - q) * X.T @ X

    return bread @ meat @ bread


def vcov_nid_qreg(
    X: np.ndarray,
    Y: np.ndarray,
    beta_hat: np.ndarray,
    q: float,
    N: int,
    method: QuantregMethodOptions,
    fit: Callable[..., QuantregSolution],
) -> np.ndarray:
    """
    Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth.

    Note: the estimator is actually heteroskedasticity robust, despite its name.
    'nid' stands for 'non-iid'.
    """
    h = get_hall_sheather_bandwidth(q=q, N=N)
    if q - h < 0 or q + h > 1:
        raise ValueError(
            "vcov='nid', the quantreg() default, needs quantile plus or minus its "
            f"Hall-Sheather bandwidth inside [0, 1], but quantile={q:g} with "
            f"bandwidth {h:.3g} gives "
            f"[{q - h:.3g}, {q + h:.3g}] for {N} observations. Use vcov='iid' or "
            "vcov='hetero', a quantile closer to 0.5, or more observations."
        )
    beta_init = beta_hat if method == "pfn" else None

    beta_hat_plus = fit(X=X, Y=Y, q=q + h, beta_init=beta_init).beta
    beta_hat_minus = fit(X=X, Y=Y, q=q - h, beta_init=beta_init).beta

    # eps: small tolerance parameter to avoid division by zero
    # when di = 0; set to sqrt of machine epsilon in quantreg
    eps = np.finfo(float).eps ** 0.5
    # equation (2)
    di = X @ (beta_hat_plus - beta_hat_minus)
    # equation (3)
    Fplus = np.maximum(0, (2 * h) / (di - eps))

    # general Huber structure, see page 74 in Koenker.
    J = X.T @ X
    XFplus = X * np.sqrt(Fplus[:, np.newaxis])
    H = XFplus.T @ XFplus
    Hinv = _invert_covariance_component(H, component="nid")

    return q * (1 - q) * Hinv @ J @ Hinv


def vcov_crv1_qreg(
    X: np.ndarray,
    u_hat: np.ndarray,
    q: float,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    """
    Implement cluster robust variance estimator for quantile regression following
    Parente and Santos Silva, 2016.
    """
    N, _ = X.shape

    kappa = np.median(np.abs(u_hat - np.median(u_hat)))
    h_G = get_hall_sheather_bandwidth(q=q, N=N)
    delta = kappa * (norm.ppf(q + h_G) - norm.ppf(q - h_G))

    A, B = crv1_vcov_qreg_loop(
        X, clustid.astype(np.uintp), cluster_col.astype(np.uintp), q, u_hat, delta
    )
    B_inv = _invert_covariance_component(B, component="cluster-robust")
    vcov = B_inv @ A @ B_inv

    return vcov
