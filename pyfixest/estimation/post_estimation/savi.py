"""
Safe anytime-valid inference (SAVI) formulas for linear models.

This module implements the asymptotic t/g-mixture e-value and confidence
sequence formulas from:

    Lindon, Michael; Ham, Dae Woong; Tingley, Martin; and Bojinov, Iavor
    (2026). "Anytime-Valid Inference in Linear Models with Applications to
    Regression-Adjusted Causal Inference." Journal of the American Statistical
    Association. https://doi.org/10.1080/01621459.2026.2692052

The public pyfixest model methods expose these formulas as SAVI e-values,
sequential p-values, and confidence sequences. They use an asymptotic
time-uniform approximation. Users must prespecify the tested restriction and
mixture precision, and ensure that the sequential sampling units satisfy the
assumptions of the chosen covariance estimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pyfixest.errors import EmptyVcovError
from pyfixest.estimation.post_estimation.wald import _wald_statistic
from pyfixest.utils.dev_utils import _select_coefnames_and_indices

if TYPE_CHECKING:
    from pyfixest.estimation.models._result_accessor_mixin import ResultAccessorMixin

SAVI_REFERENCE = (
    "Lindon, Michael; Ham, Dae Woong; Tingley, Martin; and Bojinov, Iavor "
    '(2026). "Anytime-Valid Inference in Linear Models with Applications to '
    'Regression-Adjusted Causal Inference." Journal of the American '
    "Statistical Association. https://doi.org/10.1080/01621459.2026.2692052"
)

_SAVI_SUPPORTED_VCOV_TYPES = ("iid", "hetero")


def _validate_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha


def _savi_e_value(
    f_statistic: float | np.ndarray,
    dfn: float,
    dfd: float,
    nobs: float,
    mixture_precision: float,
) -> np.ndarray:
    """Compute asymptotic SAVI e-values from F statistics.

    The formula is the g-prior/asymptotic t expression from Lindon, Ham,
    Tingley, and Bojinov (2026) for a restriction with numerator dimension
    `dfn` and denominator degrees of freedom `dfd`. It is an asymptotic
    approximation, not a finite-sample guarantee. For coefficient-wise t tests,
    pass `t_statistic ** 2` as `f_statistic` and `dfn=1`.
    """
    f_statistic = np.asarray(f_statistic, dtype=float)
    g_ratio = mixture_precision / (mixture_precision + nobs)
    stat_ratio = dfn / dfd * f_statistic
    log_e_value = dfn / 2 * np.log(g_ratio) + (dfd + dfn) / 2 * (
        np.log1p(stat_ratio) - np.log1p(g_ratio * stat_ratio)
    )
    return np.exp(log_e_value)


def _savi_confidence_radius(
    alpha: float,
    mixture_precision: float,
    nobs: float,
    dfd: float,
) -> float:
    """Compute the coefficient-wise SAVI confidence-sequence radius."""
    g_ratio = mixture_precision / (mixture_precision + nobs)
    boundary = (alpha**2 * g_ratio) ** (1 / (dfd + 1))
    denominator = boundary - g_ratio
    if denominator <= 0:
        return np.inf
    return float(dfd * (1 - boundary) / denominator)


def optimal_mixture_precision(
    nobs: float, number_of_coefficients: float, alpha: float
) -> float:
    """Compute the mixture precision that minimizes SAVI sequence width.

    The returned precision minimizes the width of the coefficient-wise
    `1 - alpha` SAVI confidence sequence at sample size `nobs`. It matches
    `avlm::optimal_g`.

    Parameters
    ----------
    nobs : float
        Target number of observations.
    number_of_coefficients : float
        Number of estimated coefficients, including the intercept.
    alpha : float
        Significance level between zero and one.

    Returns
    -------
    float
        Mixture precision optimized for the target sample size.

    Examples
    --------
    ```{python}
    from pyfixest.estimation.post_estimation.savi import optimal_mixture_precision

    optimal_mixture_precision(
        nobs=100,
        number_of_coefficients=3,
        alpha=0.05,
    )
    ```
    """
    nobs = _validate_positive_float(nobs, "nobs")
    number_of_coefficients = _validate_positive_float(
        number_of_coefficients, "number_of_coefficients"
    )
    if nobs <= number_of_coefficients:
        raise ValueError("nobs must be greater than number_of_coefficients.")
    alpha = _validate_alpha(alpha)

    dfd = nobs - number_of_coefficients
    t_max = alpha ** (2 / dfd)
    upper_bound = nobs * t_max / (1 - t_max)
    if upper_bound <= 1:
        raise ValueError(
            "No finite optimal mixture precision exists for these inputs because "
            "the optimization upper bound is not greater than 1."
        )

    result = minimize_scalar(
        lambda mixture_precision: _savi_confidence_radius(
            alpha=alpha,
            mixture_precision=mixture_precision,
            nobs=nobs,
            dfd=dfd,
        ),
        bounds=(1.0, upper_bound),
        method="bounded",
    )
    return float(result.x)


def _validate_savi_model(model: ResultAccessorMixin) -> None:
    """Reject fitted-model configurations not supported by SAVI."""
    if model._method != "feols" or model._is_iv:
        raise NotImplementedError(
            "SAVI inference is currently supported only for feols models."
        )
    if model._has_weights:
        raise NotImplementedError(
            "SAVI inference does not currently support weighted feols models."
        )
    if model._has_fixef:
        raise NotImplementedError(
            "SAVI inference does not currently support feols models with fixed effects."
        )
    if model._vcov_type not in _SAVI_SUPPORTED_VCOV_TYPES:
        raise NotImplementedError(
            f"SAVI inference does not support vcov type {model._vcov_type!r}. "
            "Supported types are iid and HC."
        )
    if len(model._vcov) == 0:
        raise EmptyVcovError()


def _coefficient_evalues(
    model: ResultAccessorMixin, mixture_precision: float
) -> pd.Series:
    """Compute coefficient-wise e-values for a validated model."""
    values = _savi_e_value(
        model._tstat**2,
        dfn=1,
        dfd=model._df_t,
        nobs=model._N,
        mixture_precision=mixture_precision,
    )
    return pd.Series(values, index=model._coefnames, name="e_value")


def _evalue(
    model: ResultAccessorMixin,
    R: np.ndarray | None = None,
    q: float | np.ndarray | None = None,
    mixture_precision: float = 1.0,
) -> pd.Series | float:
    """Compute coefficient-wise or joint SAVI e-values."""
    _validate_savi_model(model)
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")

    if R is None:
        return _coefficient_evalues(model, mixture_precision)

    wald_statistic, dfn = _wald_statistic(
        beta_hat=model._beta_hat,
        vcov=model._vcov,
        R=R,
        q=q,
    )
    return float(
        _savi_e_value(
            wald_statistic / dfn,
            dfn=dfn,
            dfd=model._df_t,
            nobs=model._N,
            mixture_precision=mixture_precision,
        )
    )


def _sequential_pvalue(
    model: ResultAccessorMixin,
    R: np.ndarray | None = None,
    q: float | np.ndarray | None = None,
    mixture_precision: float = 1.0,
) -> pd.Series | float:
    """Compute coefficient-wise or joint SAVI sequential p-values."""
    if R is None:
        return _pvalue(model=model, mixture_precision=mixture_precision)

    e_value = _evalue(
        model=model,
        R=R,
        q=q,
        mixture_precision=mixture_precision,
    )
    assert isinstance(e_value, float)
    return float(min(1.0, 1.0 / e_value))


def _pvalue(
    model: ResultAccessorMixin,
    mixture_precision: float = 1.0,
) -> pd.Series:
    """Compute coefficient-wise SAVI sequential p-values."""
    _validate_savi_model(model)
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")
    e_values = _coefficient_evalues(model, mixture_precision)
    values = np.minimum(1.0, 1.0 / e_values.to_numpy())
    return pd.Series(values, index=e_values.index, name="Pr(>|t|)")


def _tidy(
    model: ResultAccessorMixin,
    alpha: float = 0.05,
    mixture_precision: float = 1.0,
) -> pd.DataFrame:
    """Build a tidy table with SAVI coefficient inference."""
    _validate_savi_model(model)
    alpha = _validate_alpha(alpha)
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")

    critical_value = np.sqrt(
        _savi_confidence_radius(
            alpha=alpha,
            mixture_precision=mixture_precision,
            nobs=model._N,
            dfd=model._df_t,
        )
    )
    z_se = critical_value * model._se
    conf_int = np.array([model._beta_hat - z_se, model._beta_hat + z_se])
    e_values = _coefficient_evalues(model, mixture_precision)

    ub, lb = 1 - alpha / 2, alpha / 2
    data = {
        "Coefficient": model._coefnames,
        "Estimate": model._beta_hat,
        "Std. Error": model._se,
        "t value": model._tstat,
        "e_value": e_values.to_numpy(),
        f"{lb * 100:.1f}%": conf_int[:1].flatten(),
        f"{ub * 100:.1f}%": conf_int[1:2].flatten(),
    }
    if (
        getattr(model, "_sample_split_var", None) is not None
        and (sample := getattr(model, "_sample_split_value", None)) is not None
    ):
        data["Sample"] = sample
    return pd.DataFrame(data).set_index("Coefficient")


def _confint(
    model: ResultAccessorMixin,
    alpha: float = 0.05,
    mixture_precision: float = 1.0,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
) -> pd.DataFrame:
    """Compute coefficient-wise SAVI confidence sequences."""
    _validate_savi_model(model)
    alpha = _validate_alpha(alpha)
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")

    coefnames, coef_indices = _select_coefnames_and_indices(
        model._coefnames, keep, drop, exact_match
    )
    critical_value = np.sqrt(
        _savi_confidence_radius(
            alpha=alpha,
            mixture_precision=mixture_precision,
            nobs=model._N,
            dfd=model._df_t,
        )
    )
    standard_errors = model._se[coef_indices]
    estimates = model._beta_hat[coef_indices]

    df = pd.DataFrame(
        {
            f"{alpha / 2 * 100:.1f}%": estimates - critical_value * standard_errors,
            f"{(1 - alpha / 2) * 100:.1f}%": estimates
            + critical_value * standard_errors,
        },
        index=coefnames,
    )
    return df
