"""
Safe anytime-valid inference (SAVI) formulas for linear models.

Observation-level (iid and heteroskedasticity-robust) inference implements the
asymptotic t/g-mixture e-value and confidence sequence formulas from:

    Lindon, Michael; Ham, Dae Woong; Tingley, Martin; and Bojinov, Iavor
    (2026). "Anytime-Valid Inference in Linear Models with Applications to
    Regression-Adjusted Causal Inference." Journal of the American Statistical
    Association. https://doi.org/10.1080/01621459.2026.2692052

One-way cluster-robust inference implements the cluster-robust e-process
(17) and confidence sequence (22) from:

    Lindon, Michael (2026). "Anytime-Valid Inference in Linear Models with
    Clustered Outcomes: Exact Multivariate Tests and Cluster-Robust Confidence
    Sequences." arXiv:2610.25613. https://arxiv.org/abs/2610.25613

The two settings use different finite-sample conventions (Lindon 2026, Remark
8). The clustered process counts clusters rather than rows, uses nu = G - 1
whatever the number of coefficients, raises its ratio to the power G / 2
whatever the restriction dimension, and uses the CR1 covariance with only the
factor G / (G - 1). For the single-coefficient restrictions computed here,
(17) and (22) are the observation-level formulas evaluated at n = G and
nu = G - 1, so both settings share `_savi_e_value` and
`_savi_confidence_radius`; a joint clustered test would instead need the
exponent G / 2 of (17) in place of (nu + q) / 2.

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
    `dfn` and denominator degrees of freedom `dfd`. For coefficient-wise t
    tests, pass `t_statistic ** 2` as `f_statistic` and `dfn=1`; with
    `nobs=G` and `dfd=G - 1` this is the cluster-robust e-process (17) of
    Lindon (2026) for one coefficient.
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


def _cluster_robust_se(model: Feols) -> np.ndarray:
    """Compute CR1 standard errors with only the G / (G - 1) factor.

    Liang and Zeger (1986) with the finite-sample factor G / (G - 1), equation
    (16) of Lindon (2026, arXiv:2610.25613). The fitted covariance carries
    pyfixest's small-sample factor, by default also (N - 1) / (N - k), which
    is divided out so that the result does not depend on `ssc`.
    """
    covariance = model.variance_covariance
    n_clusters = covariance.G[0]
    unadjusted_vcov = covariance.vcov / covariance.ssc[0]
    return np.sqrt(np.diag(unadjusted_vcov) * n_clusters / (n_clusters - 1))


def _savi_sample(model: Feols) -> tuple[np.ndarray, int | float, int | float]:
    """Return the standard errors, sample size, and degrees of freedom SAVI uses.

    Observation-level fits use the fitted standard errors, the number of rows,
    and the t degrees of freedom of the covariance. Clustered fits use the CR1
    standard errors, the number of clusters, and G - 1 (Lindon 2026).
    """
    covariance = model.variance_covariance
    if covariance.spec.vcov_type == "CRV":
        n_clusters = covariance.G[0]
        return _cluster_robust_se(model), n_clusters, n_clusters - 1
    return model.coeftable.se, model.sample_info.n_obs, covariance.df_t


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

    Notes
    -----
    For a one-way `CRV1` fit the sequential units are clusters, with
    `G - 1` degrees of freedom whatever the number of coefficients. Pass the
    target number of clusters as `nobs` and `number_of_coefficients=1`.

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
    standard_errors, nobs, dfd = _savi_sample(model)
    values = _savi_e_value(
        (model._beta_hat / standard_errors) ** 2,
        dfn=1,
        dfd=dfd,
        nobs=nobs,
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
    standard_errors, nobs, dfd = _savi_sample(model)
    critical_value = _savi_confidence_radius(
        alpha=alpha,
        mixture_precision=mixture_precision,
        nobs=nobs,
        dfd=dfd,
    )
    standard_errors = standard_errors[coef_indices]
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
