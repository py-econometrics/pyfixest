from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm

from pyfixest.core import crv1_vcov_qreg_loop
from pyfixest.estimation.internals.literals import QuantregMethodOptions
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth


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

    # uniform kernel
    f = 1 / (2 * N * hk) * np.sum(np.abs(u_hat) < hk)

    D = X.T @ X
    Dinv = np.linalg.inv(D)

    return 1 / (f**2) * q * (1 - q) * Dinv


def vcov_hetero_qreg(
    X: np.ndarray, Y: np.ndarray, u_hat: np.ndarray, q: float, N: int
) -> np.ndarray:
    "Implement the kernel-based sandwich estimator from Powell (1991) for heteroskedasticity robust inference."
    h = get_hall_sheather_bandwidth(q=q, N=N)
    # interquartile range of u_hat
    rq = np.quantile(np.abs(u_hat), 0.75) - np.quantile(np.abs(u_hat), 0.25)
    sigma = np.std(Y)
    hk = np.minimum(sigma, rq / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

    # uniform kernel
    f = 1 / (2 * N * hk) * np.sum(np.abs(u_hat) < hk)

    D = X.T @ X
    C = f * D
    Cinv = np.linalg.inv(C)

    return q * (1 - q) * Cinv @ D @ Cinv


def vcov_nid_qreg(
    X: np.ndarray,
    Y: np.ndarray,
    beta_hat: np.ndarray,
    q: float,
    N: int,
    method: QuantregMethodOptions,
    fit: Callable[..., tuple[Any, ...]],
) -> np.ndarray:
    """
    Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth.

    Note: the estimator is actually heteroskedasticity robust, despite its name.
    'nid' stands for 'non-iid'.
    """
    h = get_hall_sheather_bandwidth(q=q, N=N)
    beta_init = beta_hat if method == "pfn" else None

    beta_hat_plus = fit(X=X, Y=Y, q=q + h, beta_init=beta_init)[0]
    beta_hat_minus = fit(X=X, Y=Y, q=q - h, beta_init=beta_init)[0]

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
    Hinv = np.linalg.inv(H)

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

    # kappa: median absolute deviation of the a-th quantile regression residuals
    kappa = np.median(np.abs(u_hat - np.median(u_hat)))
    h_G = get_hall_sheather_bandwidth(q=q, N=N)
    delta = kappa * (norm.ppf(q + h_G) - norm.ppf(q - h_G))

    A, B = crv1_vcov_qreg_loop(
        X, clustid.astype(np.uintp), cluster_col.astype(np.uintp), q, u_hat, delta
    )
    B_inv = np.linalg.inv(B)
    vcov = B_inv @ A @ B_inv

    return vcov
