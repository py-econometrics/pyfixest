from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


@dataclass(frozen=True, slots=True, kw_only=True)
class FitStatistics:
    """Goodness-of-fit measures of a fitted model.

    Every field is a float, and ``NaN`` marks a measure the estimator does not
    define, as R ``fixest::fitstat()`` returns ``NA``. ``FitStatistics()`` is
    the all-``NaN`` value carried by IV and quantile regression fits.

    Parameters
    ----------
    rmse : float
        Root mean squared error. Defined for ``feols()`` and Gaussian
        ``feglm()`` fits.
    r2 : float
        R-squared. Defined for ``feols()`` and Gaussian ``feglm()`` fits.
    adj_r2 : float
        Adjusted R-squared. Defined for ``feols()`` and Gaussian ``feglm()``
        fits.
    r2_within : float
        R-squared on the dependent variable residualized on the fixed effects.
        Defined for ``feols()`` and Gaussian ``feglm()`` fits with fixed
        effects.
    adj_r2_within : float
        Adjusted within R-squared. Defined for ``feols()`` and Gaussian
        ``feglm()`` fits with fixed effects.
    deviance : float
        Deviance of the fitted model. Defined for ``feglm()`` and ``fepois()``
        fits.
    loglik : float
        Log-likelihood. Defined for ``fepois()`` fits.
    loglik_null : float
        Log-likelihood of the intercept-only model. Defined for unweighted
        ``fepois()`` fits.
    pseudo_r2 : float
        McFadden pseudo R-squared, ``1 - loglik / loglik_null``. Defined for
        unweighted ``fepois()`` fits.
    pearson_chi2 : float
        Pearson chi-squared statistic. Defined for ``fepois()`` fits.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.fitstat.r2, fit.fitstat.r2_within
    ```
    """

    rmse: float = np.nan
    r2: float = np.nan
    adj_r2: float = np.nan
    r2_within: float = np.nan
    adj_r2_within: float = np.nan
    deviance: float = np.nan
    loglik: float = np.nan
    loglik_null: float = np.nan
    pseudo_r2: float = np.nan
    pearson_chi2: float = np.nan


def linear_fit_statistics(
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
    deviance: float = np.nan,
) -> FitStatistics:
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
        Number of observations.
    k : int
        Number of estimated coefficients.
    k_fe : int
        Number of fixed-effect coefficients. Ignored when ``has_fixef`` is False.
    has_intercept, has_fixef : bool
        Whether the model has an intercept and fixed effects.
    deviance : float
        Deviance to carry alongside the linear measures; ``NaN`` for ``feols()``.

    Returns
    -------
    FitStatistics
        The linear measures; the within variants are ``NaN`` without fixed
        effects.
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

    return FitStatistics(
        rmse=np.sqrt(ssu / N),
        r2=1 - (ssu / ssy),
        adj_r2=1 - (ssu / ssy) * adj_factor,
        r2_within=r2_within,
        adj_r2_within=adj_r2_within,
        deviance=deviance,
    )


def poisson_fit_statistics(
    *,
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
    deviance: float,
) -> FitStatistics:
    """Compute the Poisson likelihood measures from the fitted means.

    The null log-likelihood and the McFadden (1974) pseudo R-squared are
    ``NaN`` for weighted fits, where fixest's values are not replicated yet.

    Parameters
    ----------
    y : NDArray[np.float64]
        Dependent variable, shape (N,).
    mu : NDArray[np.float64]
        Fitted means of the final IRLS iteration, shape (N,).
    weights : NDArray[np.float64] or None
        User-scale observation weights, shape (N,). ``None`` applies no
        weights.
    deviance : float
        Deviance of the fitted model.

    Returns
    -------
    FitStatistics
        The Poisson measures; the linear measures are ``NaN``.
    """

    def _weighted_sum(values: np.ndarray) -> float:
        if weights is None:
            return float(np.sum(values))
        return float(np.sum(weights * values))

    loglik = _weighted_sum(y * np.log(mu) - mu - gammaln(y + 1))

    loglik_null = pseudo_r2 = np.nan
    if weights is None:
        mu_null = np.full_like(y, np.mean(y), dtype=float)
        loglik_null = _weighted_sum(y * np.log(mu_null) - mu_null - gammaln(y + 1))
        pseudo_r2 = 1 - (loglik / loglik_null)

    pearson_chi2 = _weighted_sum((y - mu) ** 2 / mu)

    return FitStatistics(
        deviance=deviance,
        loglik=loglik,
        loglik_null=loglik_null,
        pseudo_r2=pseudo_r2,
        pearson_chi2=pearson_chi2,
    )
