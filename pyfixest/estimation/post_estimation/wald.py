"""Shared utilities for Wald tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f

from pyfixest.estimation.internals.literals import WaldDistributionOptions
from pyfixest.estimation.internals.model_state import WaldTest


def _normalize_q(q: float | np.ndarray | None, n_restrictions: int) -> np.ndarray:
    """Normalize the right-hand side of a Wald restriction."""
    if q is None:
        return np.zeros(n_restrictions)

    q_array = np.asarray(q)
    if q_array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("q must be a numeric scalar or array.")
    q_array = q_array.astype(float, copy=False)

    if q_array.ndim == 0:
        return np.full(n_restrictions, float(q_array))
    if q_array.ndim != 1:
        raise ValueError("q must be a one-dimensional array or a scalar.")
    if q_array.shape[0] != n_restrictions:
        raise ValueError("q must have the same number of rows as R.")
    return q_array


def _wald_statistic(
    beta_hat: np.ndarray,
    vcov: np.ndarray,
    R: np.ndarray,
    q: float | np.ndarray | None = None,
) -> tuple[float, int]:
    """Compute a Wald quadratic form and its numerator degrees of freedom."""
    beta_hat = np.asarray(beta_hat, dtype=float)
    vcov = np.asarray(vcov, dtype=float)
    R = np.asarray(R, dtype=float)

    if R.ndim == 1:
        R = R.reshape((1, len(R)))

    if R.ndim != 2:
        raise ValueError("R must be a one- or two-dimensional array.")

    if R.shape[1] != beta_hat.shape[0]:
        raise ValueError(
            "The number of columns of R must be equal to the number of coefficients."
        )

    if R.shape[0] == 0 or np.linalg.matrix_rank(R) != R.shape[0]:
        raise ValueError("R must have full row rank.")

    q_array = _normalize_q(q, R.shape[0])

    bread = R @ beta_hat - q_array
    meat = np.linalg.pinv(R @ vcov @ R.T)
    wald_statistic = float(bread.T @ meat @ bread)
    return wald_statistic, R.shape[0]


def wald_test(
    *,
    beta_hat: np.ndarray,
    vcov: np.ndarray,
    R: np.ndarray,
    q: float | np.ndarray | None,
    df2: int | float,
    distribution: WaldDistributionOptions,
    vcov_type: str,
) -> WaldTest:
    """Test the linear hypothesis R @ beta = q.

    Parameters
    ----------
    beta_hat : np.ndarray
        Estimated coefficients, shape (n_coefficients,).
    vcov : np.ndarray
        Covariance estimate of `beta_hat`, shape (n_coefficients,
        n_coefficients).
    R : np.ndarray
        Restriction matrix of full row rank, shape (n_restrictions,
        n_coefficients).
    q : float or np.ndarray or None
        Right-hand side of the restriction. `None` is a vector of zeros.
    df2 : int or float
        Denominator degrees of freedom of the F distribution.
    distribution : {"F", "chi2"}
        Reference distribution used for the p-value.
    vcov_type : str
        Name of the covariance estimator, recorded on the result.

    Returns
    -------
    WaldTest
        The statistic of `distribution`, its p-value, both scalings of the
        quadratic form, and the degrees of freedom.
    """
    W, df1 = _wald_statistic(beta_hat=beta_hat, vcov=vcov, R=R, q=q)
    f_statistic = W / df1

    if distribution == "F":
        stat = f_statistic
        pvalue = 1 - f.cdf(f_statistic, dfn=df1, dfd=df2)
    else:
        stat = W
        pvalue = chi2.sf(W, df1)

    return WaldTest(
        stat=float(stat),
        pvalue=float(pvalue),
        df1=df1,
        df2=df2,
        distribution=distribution,
        vcov_type=vcov_type,
        wald_statistic=W,
        f_statistic=f_statistic,
    )
