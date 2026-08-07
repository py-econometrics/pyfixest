from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from pyfixest.demeaners import AnyDemeaner
from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals.demean_ import DemeanCache
from pyfixest.estimation.internals.families import GlmFamily
from pyfixest.estimation.internals.fit_ import fit_iv, fit_ols
from pyfixest.estimation.internals.fit_glm_ import fit_glm_irls
from pyfixest.estimation.internals.literals import (
    BootstrapWeightDistribution,
    SolverOptions,
    _validate_literal_argument,
)

BootstrapEstimator = Literal["ols", "iv", "glm"]


@dataclass(frozen=True, slots=True)
class WeightingBootstrapInput:
    """Arrays and estimator settings used by every bootstrap draw.

    Attributes
    ----------
    Y : np.ndarray
        Untransformed outcome, shape (N, 1).
    X : np.ndarray
        Untransformed second-stage design, shape (N, k).
    beta_hat : np.ndarray
        Full-sample coefficient estimates, shape (k,).
    user_weights : np.ndarray
        Original user estimation weights, shape (N,).
    coefnames : tuple[str, ...]
        Coefficient names, length k.
    estimator : BootstrapEstimator
        Estimator used for each draw.
    solver : SolverOptions
        Linear solver used by the original model.
    Z : np.ndarray | None
        Untransformed instrument matrix, shape (N, k_z).
    fe : np.ndarray | None
        Fixed-effect identifiers, shape (N, n_fe).
    demeaner : AnyDemeaner | None
        Demeaner used by the original model.
    family : GlmFamily | None
        GLM family for nonlinear draws.
    """

    Y: np.ndarray
    X: np.ndarray
    beta_hat: np.ndarray
    user_weights: np.ndarray
    coefnames: tuple[str, ...]
    estimator: BootstrapEstimator
    solver: SolverOptions
    Z: np.ndarray | None = None
    fe: np.ndarray | None = None
    demeaner: AnyDemeaner | None = None
    family: GlmFamily | None = None
    collin_tol: float = 1e-10
    maxiter: int = 25
    tol: float = 1e-8
    fixef_tol: float = 1e-8


@dataclass(frozen=True, slots=True)
class WeightingBootstrapResult:
    """Weighting-bootstrap inference and successful coefficient draws.

    Attributes
    ----------
    inference : pd.DataFrame
        Method-specific inference indexed by coefficient name.
    draws : np.ndarray
        Retained coefficient draws, shape (reps, k).
    failed_draws : int
        Number of attempted draws discarded before `reps` fits were retained.
    attempts : int
        Total number of attempted draws.
    """

    inference: pd.DataFrame
    draws: np.ndarray
    failed_draws: int
    attempts: int


def _validate_weighting_bootstrap_inputs(
    *,
    reps: int,
    weight_distribution: BootstrapWeightDistribution,
    alpha: float,
    ci_level: float,
) -> None:
    """Validate arguments shared by single- and multiple-model calls."""
    if isinstance(reps, bool) or not isinstance(reps, (int, np.integer)) or reps < 2:
        raise ValueError("`reps` must be an integer greater than or equal to 2.")
    _validate_literal_argument(weight_distribution, BootstrapWeightDistribution)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("`alpha` must be a finite positive number.")
    if (
        isinstance(ci_level, bool)
        or not isinstance(ci_level, Real)
        or not np.isfinite(ci_level)
        or not 0 < ci_level < 1
    ):
        raise ValueError("`ci_level` must be finite and strictly between 0 and 1.")


def _factorize_bootstrap_cluster(cluster: np.ndarray, n_obs: int) -> np.ndarray:
    """Validate and factorize one cluster variable in a row-order-stable way."""
    cluster = np.asarray(cluster).reshape(-1)
    if cluster.size != n_obs:
        raise ValueError("The cluster variable must have one value per estimation row.")
    if pd.isna(cluster).any():
        raise ValueError("The cluster variable must not contain missing values.")

    try:
        codes, levels = pd.factorize(cluster, sort=True)
    except TypeError as exc:
        raise ValueError(
            "The cluster variable must contain mutually comparable values."
        ) from exc
    if len(levels) < 2:
        raise ValueError("The weighting bootstrap requires at least two clusters.")
    return codes


def _draw_bootstrap_weights(
    *,
    rng: np.random.Generator,
    weight_distribution: BootstrapWeightDistribution,
    n_units: int,
    alpha: float,
) -> np.ndarray:
    """Draw unit weights for a multinomial or Rubin Bayesian bootstrap.

    The Bayesian implementation follows [Rubin
    (1981)](https://doi.org/10.1214/aos/1176345338): normalized independent
    Gamma draws are Dirichlet posterior probabilities. We scale them to have
    mean one because a common weight scale does not change the fitted model.
    """
    if weight_distribution == "multinomial":
        probabilities = np.full(n_units, 1 / n_units)
        return rng.multinomial(n_units, probabilities).astype(float)
    return rng.dirichlet(np.full(n_units, alpha)) * n_units


def _run_weighting_bootstrap(
    *,
    inputs: WeightingBootstrapInput,
    reps: int,
    weight_distribution: BootstrapWeightDistribution,
    alpha: float,
    ci_level: float,
    seed: int | None,
    cluster_codes: np.ndarray | None,
    max_attempts: int | None = None,
) -> WeightingBootstrapResult:
    """Generate, refit, and summarize weighting-bootstrap coefficient draws.

    Dirichlet draws implement the posterior construction from [Rubin
    (1981)](https://doi.org/10.1214/aos/1176345338). Multinomial draws implement
    a pairs bootstrap on fixed rowwise model matrices.
    """
    n_obs, k = inputs.X.shape
    _validate_bootstrap_state(inputs=inputs, n_obs=n_obs, k=k)
    inverse = np.arange(n_obs) if cluster_codes is None else cluster_codes
    if cluster_codes is not None and (
        cluster_codes.size != n_obs
        or cluster_codes.min() < 0
        or np.unique(cluster_codes).size < 2
    ):
        raise ValueError("Stored cluster codes are invalid or not row-aligned.")
    n_units = n_obs if cluster_codes is None else int(cluster_codes.max()) + 1
    attempt_limit = 2 * reps if max_attempts is None else max_attempts
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, k))
    n_success = 0
    attempts = 0

    while n_success < reps and attempts < attempt_limit:
        attempts += 1
        unit_weights = _draw_bootstrap_weights(
            rng=rng,
            weight_distribution=weight_distribution,
            n_units=n_units,
            alpha=alpha,
        )
        combined_weights = inputs.user_weights * unit_weights[inverse]
        try:
            draws[n_success] = _fit_bootstrap_draw(
                inputs=inputs, combined_weights=combined_weights
            )
        except (NonConvergenceError, np.linalg.LinAlgError, RuntimeError, ValueError):
            continue
        n_success += 1

    failed_draws = attempts - n_success
    if n_success < reps:
        raise NonConvergenceError(
            "Weighting bootstrap retained "
            f"{n_success} of {reps} requested draws after {attempts} attempts; "
            f"{failed_draws} draws failed convergence, finite-value, rank, or "
            "identification checks."
        )
    if failed_draws:
        warnings.warn(
            f"Weighting bootstrap discarded {failed_draws} of {attempts} attempted "
            f"draws. Inference is conditional on the {reps} successful draws.",
            UserWarning,
            stacklevel=2,
        )
    inference = _summarize_weighting_bootstrap(
        draws=draws,
        beta_hat=inputs.beta_hat,
        coefnames=inputs.coefnames,
        weight_distribution=weight_distribution,
        ci_level=ci_level,
    )
    return WeightingBootstrapResult(
        inference=inference,
        draws=draws,
        failed_draws=failed_draws,
        attempts=attempts,
    )


def _summarize_weighting_bootstrap(
    *,
    draws: np.ndarray,
    beta_hat: np.ndarray,
    coefnames: tuple[str, ...],
    weight_distribution: BootstrapWeightDistribution,
    ci_level: float,
) -> pd.DataFrame:
    """Summarize posterior draws or multinomial bootstrap replicates."""
    q_lower = (1 - ci_level) / 2
    ci_lower, ci_upper = np.quantile(draws, [q_lower, 1 - q_lower], axis=0)
    if weight_distribution == "dirichlet":
        tail_probability = 2 * np.minimum(
            np.mean(draws <= 0, axis=0), np.mean(draws >= 0, axis=0)
        )
        summary: dict[str, np.ndarray | str] = {
            "Original estimate": beta_hat,
            "Posterior mean": draws.mean(axis=0),
            "Posterior SD": draws.std(axis=0, ddof=1),
            "CI lower": ci_lower,
            "CI upper": ci_upper,
            "Posterior tail probability": np.minimum(tail_probability, 1),
            "interval": "equal-tail credible",
        }
    else:
        exceedances = np.sum(np.abs(draws - beta_hat) >= np.abs(beta_hat), axis=0)
        summary = {
            "Estimate": beta_hat,
            "CI lower": ci_lower,
            "CI upper": ci_upper,
            "Bootstrap SE": draws.std(axis=0, ddof=1),
            "P-value": (1 + exceedances) / (len(draws) + 1),
            "interval": "percentile confidence",
        }

    inference = pd.DataFrame(summary, index=coefnames)
    inference.index.name = "Coefficient"
    return inference


def _validate_bootstrap_state(
    *, inputs: WeightingBootstrapInput, n_obs: int, k: int
) -> None:
    if n_obs < 1 or k < 1:
        raise ValueError("Weighting bootstrap requires a nonempty design matrix.")
    if inputs.Y.reshape(-1).size != n_obs or inputs.user_weights.size != n_obs:
        raise ValueError("Stored weighting-bootstrap arrays are not row-aligned.")
    if inputs.fe is not None and inputs.fe.shape[0] != n_obs:
        raise ValueError("Stored fixed effects are not row-aligned.")
    if inputs.Z is not None and inputs.Z.shape[0] != n_obs:
        raise ValueError("Stored instruments are not row-aligned.")
    if inputs.beta_hat.size != k or len(inputs.coefnames) != k:
        raise ValueError("Stored coefficients are not column-aligned with the design.")
    if inputs.estimator == "iv" and inputs.Z is None:
        raise ValueError("IV weighting bootstrap requires stored instruments.")
    if inputs.estimator == "glm" and inputs.family is None:
        raise ValueError("GLM weighting bootstrap requires a stored family.")
    if inputs.fe is not None and inputs.demeaner is None:
        raise ValueError("Fixed-effect weighting bootstrap requires a demeaner.")
    _check_finite(inputs.Y, inputs.X, inputs.user_weights)
    if np.any(inputs.user_weights <= 0):
        raise ValueError("Original user estimation weights must be strictly positive.")


def _fit_bootstrap_draw(
    *, inputs: WeightingBootstrapInput, combined_weights: np.ndarray
) -> np.ndarray:
    active = combined_weights > 0
    weights = combined_weights[active]
    if not active.any() or not np.isfinite(weights).all():
        raise ValueError("The draw has no finite positive weights.")

    if inputs.estimator == "glm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return _fit_glm_draw(inputs=inputs, active=active, weights=weights)

    Y, X, Z = _prepare_linear_arrays(inputs=inputs, active=active, weights=weights)
    sqrt_weights = np.sqrt(weights)
    Yw = Y * sqrt_weights[:, None]
    Xw = X * sqrt_weights[:, None]
    _check_full_column_rank(Xw, "the regressor matrix")

    if inputs.estimator == "ols":
        beta = fit_ols(X=Xw, Y=Yw, solver=inputs.solver).beta
    else:
        assert Z is not None
        Zw = Z * sqrt_weights[:, None]
        _check_full_column_rank(Zw, "the instrument matrix")
        if np.linalg.matrix_rank(Zw.T @ Xw) < Xw.shape[1]:
            raise ValueError("The draw does not identify every IV coefficient.")
        beta = fit_iv(X=Xw, Z=Zw, Y=Yw, solver=inputs.solver).beta

    _check_finite(beta)
    return beta


def _prepare_linear_arrays(
    *,
    inputs: WeightingBootstrapInput,
    active: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    Y = inputs.Y[active].reshape(-1, 1)
    X = inputs.X[active]
    Z = inputs.Z[active] if inputs.Z is not None else None
    if inputs.fe is None:
        return Y, X, Z

    assert inputs.demeaner is not None
    arrays = [Y, X]
    if Z is not None:
        arrays.append(Z)
    demeaned = DemeanCache().demean_array(
        x=np.column_stack(arrays),
        flist=inputs.fe[active],
        weights=weights,
        na_index=frozenset(),
        demeaner=inputs.demeaner,
    )
    y_end = 1
    x_end = y_end + X.shape[1]
    Zd = demeaned[:, x_end:] if Z is not None else None
    return demeaned[:, :y_end], demeaned[:, y_end:x_end], Zd


def _fit_glm_draw(
    *, inputs: WeightingBootstrapInput, active: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    family = inputs.family
    assert family is not None
    Y = inputs.Y[active].reshape(-1)
    X = inputs.X[active]
    fe = inputs.fe[active] if inputs.fe is not None else None
    if family.name in ("logit", "probit") and np.unique(Y).size != 2:
        raise ValueError("The binary outcome has one class in this draw.")
    if family.name == "poisson" and not np.any(Y > 0):
        raise ValueError("The Poisson outcome is zero for every active row.")
    _check_finite(Y, X, weights)

    cache = DemeanCache()

    def _demean(
        v: np.ndarray, design: np.ndarray, irls_weights: np.ndarray, tol: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if fe is None:
            return v, design
        assert inputs.demeaner is not None
        residualized = cache.demean_array(
            x=np.column_stack([v, design]),
            flist=fe,
            weights=irls_weights,
            na_index=frozenset(),
            demeaner=inputs.demeaner.with_tol(tol),
        )
        return residualized[:, 0], residualized[:, 1:]

    fit = fit_glm_irls(
        X=X,
        Y=Y,
        family=family,
        demean=_demean,
        coefnames=list(inputs.coefnames),
        collin_tol=inputs.collin_tol,
        accelerate=False,
        weights=weights,
        solver=inputs.solver,
        maxiter=inputs.maxiter,
        tol=inputs.tol,
        fixef_tol=inputs.fixef_tol,
    )
    if not fit.converged or tuple(fit.coefnames) != inputs.coefnames:
        raise ValueError("The GLM draw did not retain the identified coefficient set.")
    _check_finite(fit.beta, fit.eta, fit.mu)
    _check_full_column_rank(
        fit.sqrt_W.reshape(-1, 1) * fit.X_tilde, "the final GLM working design"
    )
    if family.name in ("logit", "probit") and np.any((fit.mu <= 0) | (fit.mu >= 1)):
        raise ValueError("The binary GLM draw reached a separated boundary solution.")
    if family.name == "poisson" and np.any(fit.mu <= 0):
        raise ValueError("The Poisson draw reached an invalid boundary solution.")
    return fit.beta


def _check_full_column_rank(matrix: np.ndarray, name: str) -> None:
    _check_finite(matrix)
    if (
        matrix.shape[0] < matrix.shape[1]
        or np.linalg.matrix_rank(matrix) < matrix.shape[1]
    ):
        raise ValueError(f"The draw is rank deficient in {name}.")


def _check_finite(*arrays: np.ndarray) -> None:
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("The draw contains non-finite numerical values.")
