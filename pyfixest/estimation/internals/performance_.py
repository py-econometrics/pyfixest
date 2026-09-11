from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceMeasures:
    """Goodness-of-fit measures; the within variants are NaN without fixed effects."""

    rmse: float
    r2: float
    adj_r2: float
    r2_within: float
    adj_r2_within: float


def performance_measures(
    *,
    Y: np.ndarray,
    Y_within: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray | None,
    N: int | float,
    k: int,
    k_fe: int,
    has_intercept: bool,
    has_fixef: bool,
) -> PerformanceMeasures:
    """Compute R² measures from response-scale arrays and observation weights.

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable, shape (N, 1).
    Y_within : np.ndarray
        Dependent variable residualized on the fixed effects, shape (N, 1),
        in the units of ``Y``. Ignored when ``has_fixef`` is False.
    residuals : np.ndarray
        Residuals in the units of ``Y``, shape (N,).
    weights : np.ndarray or None
        User-scale observation weights, shape (N,) or (N, 1). ``None``
        applies no weights.
    N : int or float
        Effective number of observations.
    k : int
        Number of estimated coefficients.
    k_fe : int
        Number of fixed-effect coefficients. Ignored when ``has_fixef`` is False.
    has_intercept, has_fixef : bool
        Whether the model has an intercept and fixed effects.
    """
    if weights is None:
        ssu = np.sum(residuals**2)
        ssy = np.sum((Y - np.mean(Y)) ** 2)
    else:
        w = weights.reshape((-1, 1))
        ssu = np.sum(w.flatten() * residuals**2)
        ssy = np.sum(w * (Y - np.average(Y, weights=w)) ** 2)

    if has_fixef:
        adj_factor = (N - has_intercept) / (N - k - k_fe)
    else:
        adj_factor = (N - has_intercept) / (N - k)

    r2_within = adj_r2_within = np.nan
    if has_fixef:
        ssy_within = np.sum(Y_within**2) if weights is None else np.sum(w * Y_within**2)
        adj_factor_within = (N - k_fe) / (N - k - k_fe)
        r2_within = 1 - (ssu / ssy_within)
        adj_r2_within = 1 - (ssu / ssy_within) * adj_factor_within

    return PerformanceMeasures(
        rmse=np.sqrt(ssu / N),
        r2=1 - (ssu / ssy),
        adj_r2=1 - (ssu / ssy) * adj_factor,
        r2_within=r2_within,
        adj_r2_within=adj_r2_within,
    )
