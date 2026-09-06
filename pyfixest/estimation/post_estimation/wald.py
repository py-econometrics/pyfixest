"""Shared utilities for Wald tests."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2, f

if TYPE_CHECKING:
    from pyfixest.estimation.models.feols_ import Feols


def wald_test(
    model: Feols,
    R: np.ndarray | None = None,
    q: float | np.ndarray | None = None,
    distribution: str = "F",
) -> pd.Series:
    """Conduct a Wald test for a fitted model."""
    k_fe = np.sum(np.asarray(model._k_fe.values)) if model._has_fixef else 0

    R = np.eye(model._k) if R is None else np.atleast_2d(np.asarray(R, dtype=float))

    W, model._dfn = _wald_statistic(
        beta_hat=model._beta_hat,
        vcov=model._vcov,
        R=R,
        q=q,
    )

    if model._is_clustered:
        model._dfd = np.min(np.array(model._G)) - 1
    else:
        model._dfd = model._N - model._k - k_fe

    model._wald_statistic = W

    # The F distribution is only used for the joint test that all
    # coefficients are zero (R identity, q zero).
    if distribution == "F" and (
        not np.array_equal(R, np.eye(model._k)) or (q is not None and np.any(q))
    ):
        warnings.warn(
            "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector."
        )
        distribution = "chi2"

    if distribution == "F":
        model._f_statistic = W / model._dfn
        model._p_value = 1 - f.cdf(
            model._f_statistic,
            dfn=model._dfn,
            dfd=model._dfd,
        )
        return pd.Series({"statistic": model._f_statistic, "pvalue": model._p_value})
    if distribution == "chi2":
        model._f_statistic = W / model._dfn
        model._p_value = chi2.sf(model._wald_statistic, model._dfn)
        return pd.Series({"statistic": model._wald_statistic, "pvalue": model._p_value})

    raise ValueError("Distribution must be F or chi2")


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
