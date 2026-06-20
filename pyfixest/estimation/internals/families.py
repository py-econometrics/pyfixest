from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t


@dataclass(frozen=True, slots=True)
class InferenceDist:
    """Reference distribution used to compute Wald p-values and CI critical values.

    `pvalue(tstat, df_t)` returns two-sided p-values.
    `crit_val(alpha, df_t)` returns |q(alpha/2)| for confidence intervals.
    The `df_t` argument is ignored for the normal distribution.
    """

    pvalue: Callable[[np.ndarray, int], np.ndarray]
    crit_val: Callable[[float, int], float]


T_DIST = InferenceDist(
    pvalue=lambda tstat, df_t: 2 * (1 - t.cdf(np.abs(tstat), df_t)),
    crit_val=lambda alpha, df_t: float(np.abs(t.ppf(alpha / 2, df_t))),
)

NORMAL_DIST = InferenceDist(
    pvalue=lambda tstat, df_t: 2 * (1 - norm.cdf(np.abs(tstat))),
    crit_val=lambda alpha, df_t: float(np.abs(norm.ppf(alpha / 2))),
)


@dataclass(frozen=True, slots=True)
class GlmFamily:
    """Family-specific operations for a fixed-effects GLM.

    Bundles the link function, inverse link, derivative of the link, variance function,
    deviance, initial mu, and dependent-variable validation into a single object.
    """

    name: str
    link: Callable[[np.ndarray], np.ndarray]
    inv_link: Callable[[np.ndarray], np.ndarray]
    gprime: Callable[[np.ndarray], np.ndarray]
    variance: Callable[[np.ndarray], np.ndarray]
    deviance: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]
    mu_start: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    check_y: Callable[[np.ndarray], None]
    inference_dist: InferenceDist = NORMAL_DIST


def _check_y_binary(Y: np.ndarray) -> None:
    Y_unique = np.unique(Y)
    if len(Y_unique) != 2:
        raise ValueError("The dependent variable must have two unique values.")
    if np.any(~np.isin(Y_unique, [0, 1])):
        raise ValueError("The dependent variable must be binary (0 or 1).")


def _check_y_noop(Y: np.ndarray) -> None:
    return None


def _mu_start_binary(Y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    y = Y.flatten().astype(float)
    w = weights.flatten() if weights is not None else np.ones_like(y)
    return (w * y + 0.5) / (w + 1.0)


def _mu_start_mean(Y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    return np.full_like(Y.flatten(), np.mean(Y), dtype=float)


def _flatten_weights(weights: np.ndarray | None) -> np.ndarray | float:
    return weights.flatten() if weights is not None else 1.0


def _logit_deviance(y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None) -> float:
    w = _flatten_weights(weights)
    return float(-2 * np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))


def _probit_deviance(
    y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None
) -> float:
    w = _flatten_weights(weights)
    ll_fitted = np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    # divide by zero warnings because of the log(0) terms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_saturated = np.sum(
            w
            * (
                np.where(y == 0, 0, y * np.log(y))
                + np.where(y == 1, 0, (1 - y) * np.log(1 - y))
            )
        )
    return float(-2.0 * (ll_fitted - ll_saturated))


def _gaussian_deviance(
    y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None
) -> float:
    w = _flatten_weights(weights)
    return float(np.sum(w * (y - mu) ** 2))


def _pois_deviance(y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None) -> float:
    "Poisson deviance, optionally weighted. Defined as 2·(LL_saturated - LL_fitted)."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = _flatten_weights(weights)
        y_flat = y.flatten()
        mu_flat = mu.flatten()
        return float(
            2
            * np.sum(
                w
                * (
                    np.where(y_flat == 0, 0, y_flat * np.log(y_flat / mu_flat))
                    - (y_flat - mu_flat)
                )
            )
        )


def _check_y_nonneg(Y: np.ndarray) -> None:
    if np.any(Y < 0):
        raise ValueError("The dependent variable must be weakly positive.")


def _mu_start_pois(Y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    y = Y.flatten().astype(float)
    return (y + y.mean()) / 2


LOGIT = GlmFamily(
    name="logit",
    link=lambda mu: np.log(mu / (1 - mu)),
    inv_link=lambda eta: np.exp(eta) / (1 + np.exp(eta)),
    gprime=lambda mu: 1 / (mu * (1 - mu)),
    variance=lambda mu: mu * (1 - mu),
    deviance=_logit_deviance,
    mu_start=_mu_start_binary,
    check_y=_check_y_binary,
)

PROBIT = GlmFamily(
    name="probit",
    link=norm.ppf,
    inv_link=norm.cdf,
    gprime=lambda mu: 1 / norm.pdf(norm.ppf(mu)),
    variance=lambda mu: mu * (1 - mu),
    deviance=_probit_deviance,
    mu_start=_mu_start_binary,
    check_y=_check_y_binary,
)

GAUSSIAN = GlmFamily(
    name="gaussian",
    link=lambda mu: mu,
    inv_link=lambda eta: eta,
    gprime=lambda mu: np.ones_like(mu),
    variance=lambda mu: np.ones_like(mu),
    deviance=_gaussian_deviance,
    mu_start=_mu_start_mean,
    check_y=_check_y_noop,
    inference_dist=T_DIST,
)

POISSON = GlmFamily(
    name="poisson",
    link=np.log,
    inv_link=np.exp,
    gprime=lambda mu: 1 / mu,
    variance=lambda mu: mu,
    deviance=_pois_deviance,
    mu_start=_mu_start_pois,
    check_y=_check_y_nonneg,
)


FAMILY_REGISTRY: dict[str, GlmFamily] = {
    "logit": LOGIT,
    "probit": PROBIT,
    "gaussian": GAUSSIAN,
    "poisson": POISSON,
}
