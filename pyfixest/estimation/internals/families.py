from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class GlmFamily:
    """Family-specific operations for a fixed-effects GLM.

    Bundles the link, inverse link, derivative of the link, variance function,
    deviance, initial mu, and dependent-variable validation into a single value
    object. Replaces the per-subclass abstract-method pattern previously spread
    across ``Felogit`` / ``Feprobit`` / ``Fegaussian``.
    """

    name: str
    link: Callable[[np.ndarray], np.ndarray]
    inv_link: Callable[[np.ndarray], np.ndarray]
    gprime: Callable[[np.ndarray], np.ndarray]
    variance: Callable[[np.ndarray], np.ndarray]
    deviance: Callable[[np.ndarray, np.ndarray], float]
    mu_start: Callable[[np.ndarray], np.ndarray]
    check_y: Callable[[np.ndarray], None]


def _check_y_binary(Y: np.ndarray) -> None:
    Y_unique = np.unique(Y)
    if len(Y_unique) != 2:
        raise ValueError("The dependent variable must have two unique values.")
    if np.any(~np.isin(Y_unique, [0, 1])):
        raise ValueError("The dependent variable must be binary (0 or 1).")


def _check_y_noop(Y: np.ndarray) -> None:
    return None


def _mu_start_binary(Y: np.ndarray) -> np.ndarray:
    return np.full_like(Y.flatten(), 0.5, dtype=float)


def _mu_start_mean(Y: np.ndarray) -> np.ndarray:
    return np.full_like(Y.flatten(), float(np.mean(Y)), dtype=float)


def _logit_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    return float(-2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


def _probit_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    ll_fitted = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    # divide-by-zero warnings from log(0) in the saturated likelihood
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_saturated = np.sum(
            np.where(y == 0, 0, y * np.log(y))
            + np.where(y == 1, 0, (1 - y) * np.log(1 - y))
        )
    return float(-2.0 * (ll_fitted - ll_saturated))


def _gaussian_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sum((y - mu) ** 2))


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
)
