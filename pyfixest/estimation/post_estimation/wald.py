"""Shared utilities for Wald tests."""

from __future__ import annotations

import numpy as np


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

    if q is None:
        q_array = np.zeros(R.shape[0])
    else:
        try:
            q_array = np.asarray(q, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("q must be a numeric scalar or array.") from exc

        if q_array.ndim == 0:
            q_array = np.full(R.shape[0], float(q_array))
        elif q_array.ndim != 1:
            raise ValueError("q must be a one-dimensional array or a scalar.")
        elif q_array.shape[0] != R.shape[0]:
            raise ValueError("q must have the same number of rows as R.")

    bread = R @ beta_hat - q_array
    meat = np.linalg.pinv(R @ vcov @ R.T)
    wald_statistic = float(bread.T @ meat @ bread)
    return wald_statistic, R.shape[0]
