"""
Safe anytime-valid inference (SAVI) formulas for linear models.

Observation-level (iid and heteroskedasticity-robust) inference implements the
asymptotic t/g-mixture e-value and confidence sequence formulas from:

    Lindon, Michael; Ham, Dae Woong; Tingley, Martin; and Bojinov, Iavor
    (2026). "Anytime-Valid Inference in Linear Models with Applications to
    Regression-Adjusted Causal Inference." Journal of the American Statistical
    Association. https://doi.org/10.1080/01621459.2026.2692052

One-way cluster-robust inference implements the cluster-robust e-process and
confidence sequence, equations (ECR) and (kappa), from:

    Lindon, Michael (2026). "Anytime-Valid Inference in Linear Models with
    Clustered Outcomes: Exact Multivariate Tests and Cluster-Robust Confidence
    Sequences." arXiv:2610.25613. https://arxiv.org/abs/2610.25613

The two settings use different finite-sample conventions. The clustered
process counts clusters rather than rows, uses nu = G - 1 whatever the number
of coefficients, raises its ratio to the power G / 2 for every restriction
dimension, and uses the CR1 covariance with only the factor G / (G - 1).

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
from pyfixest.utils.dev_utils import _select_coefnames_and_indices

if TYPE_CHECKING:
    from pyfixest.estimation.models.feols_ import Feols

SAVI_REFERENCE = (
    "Lindon, Michael; Ham, Dae Woong; Tingley, Martin; and Bojinov, Iavor "
    '(2026). "Anytime-Valid Inference in Linear Models with Applications to '
    'Regression-Adjusted Causal Inference." Journal of the American '
    "Statistical Association. https://doi.org/10.1080/01621459.2026.2692052"
)

_SAVI_SUPPORTED_VCOV_TYPES = ("iid", "hetero", "CRV")


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
    `dfn` and denominator degrees of freedom `dfd`.
    It For coefficient-wise t tests, pass `t_statistic ** 2` as `f_statistic`
    and `dfn=1`.
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
    return float(np.sqrt(dfd * (1 - boundary) / denominator))


def _savi_cluster_e_value(
    wald_statistic: float | np.ndarray,
    dfn: int,
    n_clusters: int,
    mixture_precision: float,
) -> np.ndarray:
    """Compute cluster-robust SAVI e-values from Wald statistics.

    Equation (ECR) of Lindon (2026, arXiv:2610.25613): the one-sample
    Hotelling e-value with Hotelling's T^2 replaced by the CR1 Wald statistic
    `wald_statistic` of a restriction with `dfn` rows. Unlike the
    observation-level formula, nu = G - 1 and the exponent is G / 2 for every
    `dfn`. The process is one while G <= `dfn`.
    """
    wald_statistic = np.asarray(wald_statistic, dtype=float)
    if n_clusters <= dfn:
        return np.ones_like(wald_statistic)
    g_ratio = mixture_precision / (mixture_precision + n_clusters)
    stat_ratio = wald_statistic / (n_clusters - 1)
    log_e_value = dfn / 2 * np.log(g_ratio) + n_clusters / 2 * (
        np.log1p(stat_ratio) - np.log1p(g_ratio * stat_ratio)
    )
    return np.exp(log_e_value)


def _savi_cluster_confidence_radius(
    alpha: float,
    mixture_precision: float,
    n_clusters: int,
) -> float:
    """Compute the coefficient-wise cluster-robust confidence-sequence radius.

    The square root of kappa_G(phi, 1, alpha), equation (kappa) of Lindon
    (2026, arXiv:2610.25613), with nu = G - 1.
    """
    if n_clusters <= 1:
        return np.inf
    g_ratio = mixture_precision / (mixture_precision + n_clusters)
    boundary = (alpha**2 * g_ratio) ** (1 / n_clusters)
    denominator = boundary - g_ratio
    if denominator <= 0:
        return np.inf
    return float(np.sqrt((n_clusters - 1) * (1 - boundary) / denominator))


def _cluster_robust_se(model: Feols) -> np.ndarray:
    """Compute CR1 standard errors with only the G / (G - 1) factor.

    Liang and Zeger (1986) with the finite-sample factor G / (G - 1), equation
    (CR1) of Lindon (2026, arXiv:2610.25613). The fitted covariance carries
    pyfixest's small-sample factor, by default also (N - 1) / (N - k), which
    is divided out so that the result does not depend on `ssc`.
    """
    covariance = model.variance_covariance
    n_clusters = covariance.G[0]
    unadjusted_vcov = covariance.vcov / covariance.ssc[0]
    return np.sqrt(np.diag(unadjusted_vcov) * n_clusters / (n_clusters - 1))


def optimal_mixture_precision(
    nobs: float, number_of_coefficients: float, alpha: float
) -> float:
    """Compute the mixture precision that minimizes SAVI sequence width
    at a specified sample size `nobs`.

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
    import pyfixest as pf

    pf.optimal_mixture_precision(
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


def _validate_savi_model(*, model: Feols, method: str) -> None:
    """Reject fitted-model configurations not supported by SAVI."""
    model._require_capability(capability="anytime_valid_inference", method=method)
    if model.options.has_weights:
        raise NotImplementedError(
            "SAVI inference does not currently support weighted feols models."
        )
    if model.model.has_fixef:
        raise NotImplementedError(
            "SAVI inference does not currently support feols models with fixed effects."
        )
    if not hasattr(model, "variance_covariance"):
        raise EmptyVcovError()
    spec = model.variance_covariance.spec
    if spec.vcov_type not in _SAVI_SUPPORTED_VCOV_TYPES:
        raise NotImplementedError(
            f"SAVI inference does not support vcov type {spec.vcov_type!r}. "
            "Supported types are iid, hetero, HC1, HC2, HC3, and one-way CRV1."
        )
    if spec.vcov_type == "CRV":
        if len(spec.clustervar) != 1:
            raise NotImplementedError(
                "SAVI inference does not support multiway clustering; "
                "cluster by exactly one variable."
            )
        if spec.vcov_type_detail != "CRV1":
            raise NotImplementedError(
                f"SAVI inference does not support {spec.vcov_type_detail}; "
                "use CRV1 for clustered inference."
            )


def _evalue(
    model: Feols,
    mixture_precision: float = 1.0,
) -> pd.Series:
    """Compute coefficient-wise SAVI e-values for a validated model."""
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")
    covariance = model.variance_covariance
    if covariance.spec.vcov_type == "CRV":
        values = _savi_cluster_e_value(
            (model._beta_hat / _cluster_robust_se(model)) ** 2,
            dfn=1,
            n_clusters=covariance.G[0],
            mixture_precision=mixture_precision,
        )
    else:
        values = _savi_e_value(
            model.coeftable.tstat**2,
            dfn=1,
            dfd=covariance.df_t,
            nobs=model.sample_info.n_obs,
            mixture_precision=mixture_precision,
        )
    return pd.Series(values, index=model._coefnames, name="e_value")


def _pvalue_savi(
    model: Feols,
    mixture_precision: float = 1.0,
) -> pd.Series:
    """Compute coefficient-wise SAVI sequential p-values for a validated model."""
    e_values = _evalue(model=model, mixture_precision=mixture_precision)
    values = np.minimum(1.0, 1.0 / e_values.to_numpy())
    return pd.Series(values, index=e_values.index, name="Pr(>|t|)")


def _confint(
    model: Feols,
    alpha: float = 0.05,
    mixture_precision: float = 1.0,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
) -> pd.DataFrame:
    """Compute coefficient-wise SAVI confidence sequences for a validated model."""
    alpha = _validate_alpha(alpha)
    mixture_precision = _validate_positive_float(mixture_precision, "mixture_precision")

    coefnames, coef_indices = _select_coefnames_and_indices(
        model._coefnames, keep, drop, exact_match
    )
    covariance = model.variance_covariance
    if covariance.spec.vcov_type == "CRV":
        critical_value = _savi_cluster_confidence_radius(
            alpha=alpha,
            mixture_precision=mixture_precision,
            n_clusters=covariance.G[0],
        )
        standard_errors = _cluster_robust_se(model)[coef_indices]
    else:
        critical_value = _savi_confidence_radius(
            alpha=alpha,
            mixture_precision=mixture_precision,
            nobs=model.sample_info.n_obs,
            dfd=covariance.df_t,
        )
        standard_errors = model.coeftable.se[coef_indices]
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
