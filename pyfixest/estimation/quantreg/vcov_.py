from __future__ import annotations

import numpy as np
from scipy.stats import norm

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

