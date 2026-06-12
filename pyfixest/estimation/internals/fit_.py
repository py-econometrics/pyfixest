"""Pure, array-based estimation core.

Pure functions on numpy arrays returning small frozen dataclasses — no
DataFrames, no self-mutation. Model classes delegate their math here
(`Feols.get_fit` -> `fit_ols`, `Feiv.get_fit` -> `fit_iv`).

Inputs are expected to be fully prepared: demeaned, multicollinearity-pruned,
and WLS-transformed (i.e. already multiplied by sqrt-weights where relevant).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pyfixest.core.collinear import find_collinear_variables
from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals.literals import SolverOptions
from pyfixest.estimation.internals.solvers import solve_ols


@dataclass(frozen=True, slots=True)
class OlsFit:
    """Result of a (weighted) least-squares fit.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    residuals : np.ndarray
        Response-scale residuals Y - X @ beta, shape (N,).
    residuals_wls : np.ndarray
        Solve-scale residuals sqrt(w)(Y - X @ beta), shape (N,). Equal to
        ``residuals`` when weights are all one.
    scores : np.ndarray
        Score matrix X_wls * residuals_wls (= w * X * residuals), shape (N, k).
    hessian : np.ndarray
        Hessian X'WX, shape (k, k).
    tZX : np.ndarray
        Z'WX (= X'WX for OLS), shape (k, k).
    tZy : np.ndarray
        Z'WY (= X'WY for OLS), shape (k, 1).
    X_wls, Y_wls : np.ndarray
        sqrt(w)-scaled design matrix / dependent variable of the solve.
    """

    beta: np.ndarray
    residuals: np.ndarray
    residuals_wls: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    tZX: np.ndarray
    tZy: np.ndarray
    X_wls: np.ndarray
    Y_wls: np.ndarray


@dataclass(frozen=True, slots=True)
class IvFit:
    """Result of a 2SLS fit.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    residuals : np.ndarray
        Response-scale second-stage residuals Y - X @ beta, shape (N,).
    residuals_wls : np.ndarray
        Solve-scale residuals sqrt(w)(Y - X @ beta), shape (N,).
    scores : np.ndarray
        Score matrix Z_wls * residuals_wls, shape (N, k_z).
    hessian : np.ndarray
        Z'WZ, shape (k_z, k_z).
    tZX : np.ndarray
        Z'WX, shape (k_z, k).
    tXZ : np.ndarray
        X'WZ, shape (k, k_z).
    tZy : np.ndarray
        Z'WY, shape (k_z, 1).
    tZZinv : np.ndarray
        (Z'WZ)^{-1}, shape (k_z, k_z).
    bread : np.ndarray
        Bread matrix of the sandwich estimator, shape (k, k).
    X_wls, Z_wls, Y_wls : np.ndarray
        sqrt(w)-scaled design matrix / instruments / dependent variable.
    """

    beta: np.ndarray
    residuals: np.ndarray
    residuals_wls: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    tZX: np.ndarray
    tXZ: np.ndarray
    tZy: np.ndarray
    tZZinv: np.ndarray
    bread: np.ndarray
    X_wls: np.ndarray
    Z_wls: np.ndarray
    Y_wls: np.ndarray


def fit_ols(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    solver: SolverOptions = "np.linalg.solve",
) -> OlsFit:
    """Fit a (weighted) least-squares model on demeaned arrays.

    Weighting happens here — inputs are *not* pre-multiplied by
    sqrt(weights).

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (N, k). Demeaned, collinearity-pruned,
        unweighted.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned, unweighted.
    weights : np.ndarray
        Weights, shape (N, 1).
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    w_sqrt = np.sqrt(weights)
    X_wls = w_sqrt * X
    Y_wls = w_sqrt * Y

    tZX = X_wls.T @ X_wls
    tZy = X_wls.T @ Y_wls
    beta = solve_ols(tZX, tZy, solver)
    residuals_wls = Y_wls.flatten() - (X_wls @ beta).flatten()
    residuals = Y.flatten() - (X @ beta).flatten()
    scores = X_wls * residuals_wls[:, None]
    hessian = tZX.copy()
    return OlsFit(
        beta=beta,
        residuals=residuals,
        residuals_wls=residuals_wls,
        scores=scores,
        hessian=hessian,
        tZX=tZX,
        tZy=tZy,
        X_wls=X_wls,
        Y_wls=Y_wls,
    )


def fit_iv(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    solver: SolverOptions = "np.linalg.solve",
) -> IvFit:
    """Fit a (weighted) 2SLS model on demeaned arrays.

    Weighting happens here — inputs are *not* pre-multiplied by
    sqrt(weights).

    Parameters
    ----------
    X : np.ndarray
        Design matrix (incl. endogenous regressors), shape (N, k).
        Demeaned, collinearity-pruned, unweighted.
    Z : np.ndarray
        Instrument matrix, shape (N, k_z). Demeaned, pruned, unweighted.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned, unweighted.
    weights : np.ndarray
        Weights, shape (N, 1).
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    w_sqrt = np.sqrt(weights)
    X_wls = w_sqrt * X
    Z_wls = w_sqrt * Z
    Y_wls = w_sqrt * Y

    tZX = Z_wls.T @ X_wls
    tXZ = X_wls.T @ Z_wls
    tZy = Z_wls.T @ Y_wls
    tZZ = Z_wls.T @ Z_wls
    tZZinv = np.linalg.inv(tZZ)

    H = tXZ @ tZZinv
    A = H @ tZX
    B = H @ tZy
    beta = solve_ols(A, B, solver)

    residuals_wls = Y_wls.flatten() - (X_wls @ beta).flatten()
    residuals = Y.flatten() - (X @ beta).flatten()
    scores = Z_wls * residuals_wls[:, None]
    hessian = tZZ

    D = np.linalg.inv(tXZ @ tZZinv @ tZX)
    bread = H.T @ D @ H

    return IvFit(
        beta=beta,
        residuals=residuals,
        residuals_wls=residuals_wls,
        scores=scores,
        hessian=hessian,
        tZX=tZX,
        tXZ=tXZ,
        tZy=tZy,
        tZZinv=tZZinv,
        bread=bread,
        X_wls=X_wls,
        Z_wls=Z_wls,
        Y_wls=Y_wls,
    )


def _drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Check for multicollinearity in the design matrices X and Z.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    names : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : numpy.ndarray
        Logical array, where True indicates that the variable is collinear.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = np.ascontiguousarray(X.T @ X, dtype=np.float64)
    id_excl, n_excl, all_removed = find_collinear_variables(tXX, collin_tol)

    collin_vars = []
    collin_index = []

    if all_removed:
        raise ValueError(
            """
            All variables are collinear. Maybe your model specification introduces multicollinearity? If not, please reach out to the package authors!.
            """
        )

    names_array = np.array(names)
    if n_excl > 0:
        collin_vars = names_array[id_excl].tolist()
        if len(collin_vars) > 5:
            indent = "    "
            formatted_collinear_vars = (
                f"\n{indent}" + f"\n{indent}".join(collin_vars[:5]) + f"\n{indent}..."
            )
        else:
            formatted_collinear_vars = str(collin_vars)

        warnings.warn(
            f"""
            {len(collin_vars)} variables dropped due to multicollinearity.
            The following variables are dropped: {formatted_collinear_vars}.
            """
        )

        X = np.delete(X, id_excl, axis=1)
        if X.ndim == 2 and X.shape[1] == 0:
            raise ValueError(
                """
                All variables are collinear. Please check your model specification.
                """
            )

        names_array = np.delete(names_array, id_excl)
        collin_index = id_excl.tolist()

    return X, list(names_array), collin_vars, collin_index


# ------------------------------------------------------------------
# IWLS cores (Fepois / Feglm)
# ------------------------------------------------------------------


def poisson_deviance(
    Y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None
) -> np.float64:
    """
    Deviance is defined as twice the difference in log likelihood between the
    saturated model and the model being fit.

    deviance = 2 * (log_likelihood_saturated - log_likelihood_fitted)

    See [1] chapter 5.6 for more details.
    [1] Dobson, Annette J., and Adrian G. Barnett. An introduction to generalized linear models. Chapman and Hall/CRC, 2018.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if weights is None:
            weights = np.ones_like(Y)
        weights = weights.flatten()
        Y_flat = Y.flatten()
        mu_flat = mu.flatten()
        deviance = np.float64(
            2
            * np.sum(
                weights
                * (
                    np.where(Y_flat == 0, 0, Y_flat * np.log(Y_flat / mu_flat))
                    - (Y_flat - mu_flat)
                )
            )
        )
    return deviance


def _relative_deviance_change(
    deviance: np.ndarray, deviance_old: np.ndarray
) -> np.ndarray:
    "Compute relative change in deviance for convergence check."
    return np.abs(deviance - deviance_old) / (0.1 + np.abs(deviance_old))


@dataclass(frozen=True, slots=True)
class PoissonIwlsFit:
    """Result of the Poisson IWLS loop (ppmlhdfe algorithm).

    ``WX``/``WZ`` are the weighted, demeaned design/working dependent
    variable from the final iteration; ``X`` is the original (non-demeaned)
    design matrix with collinear columns dropped.
    """

    beta: np.ndarray  # (k, 1)
    eta: np.ndarray
    mu: np.ndarray
    irls_weights: np.ndarray  # combined user * IRLS weights, (N, 1)
    WX: np.ndarray
    WZ: np.ndarray
    residuals_working: np.ndarray  # (N, 1)
    hessian: np.ndarray  # X'WX from the final iteration
    convergence: bool
    coefnames: list[str]
    collin_vars: list[str]
    collin_index: list[int]
    X: np.ndarray


def fit_iwls_poisson(
    Y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray,
    coefnames: list[str],
    collin_tol: float,
    solver: SolverOptions,
    tol: float,
    maxiter: int,
    demean: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
) -> PoissonIwlsFit:
    """Fit a Poisson regression via IWLS (ppmlhdfe algorithm) on arrays.

    Parameters
    ----------
    Y, X : np.ndarray
        Dependent variable (N, 1) and design matrix (N, k).
    weights : np.ndarray
        User weights, (N, 1).
    offset : np.ndarray
        Offset, (N, 1) (zeros if none).
    demean : callable or None
        ``demean(x, weights_flat) -> x_resid`` residualizing by the fixed
        effects; ``None`` when the model has no fixed effects.
    """
    N = Y.shape[0]

    stop_iterating = False
    crit = 1
    convergence = False
    collin_vars: list[str] = []
    collin_index: list[int] = []

    for i in range(maxiter):
        if stop_iterating:
            convergence = True
            break
        if i == maxiter:  # kept from legacy loop (unreachable: i < maxiter)
            raise NonConvergenceError(
                f"""
                The IRLS algorithm did not converge with {maxiter}
                iterations. Try to increase the maximum number of iterations.
                """
            )

        if i == 0:
            _mean = np.mean(Y)
            mu = (Y + _mean) / 2
            eta = np.log(mu)
            Z = eta - offset + Y / mu - 1
            reg_Z = Z.copy()
            last = poisson_deviance(Y, mu)
        else:
            # update w and Z
            Z = eta - offset + Y / mu - 1  # eq (8)
            reg_Z = Z.copy()  # eq (9)

        # Step 1: weighted demeaning
        ZX = np.concatenate([reg_Z, X], axis=1)

        combined_weights = weights * mu

        if demean is None:
            ZX_resid = ZX
        else:
            ZX_resid = demean(ZX, combined_weights.flatten())

        Z_resid = ZX_resid[:, 0].reshape((N, 1))  # z_resid
        X_resid = ZX_resid[:, 1:]  # x_resid

        if i == 0:
            # Check multicollinearity
            # We do this here after the first demeaning to also catch
            # collinearity with fixed effects
            X_resid, coefnames, collin_vars, collin_index = (
                _drop_multicollinear_variables(
                    X_resid,
                    coefnames,
                    collin_tol,
                )
            )
            if collin_index:
                # Drop covariates collinear with fixed effects
                X = X[:, ~np.array(collin_index)]

        WX = np.sqrt(combined_weights) * X_resid
        WZ = np.sqrt(combined_weights) * Z_resid

        XWX = WX.transpose() @ WX
        XWZ = WX.transpose() @ WZ

        delta_new = solve_ols(XWX, XWZ, solver).reshape(
            (-1, 1)
        )  # eq (10), delta_new -> reg_z
        resid = Z_resid - X_resid @ delta_new

        # more updating
        eta = Z - resid + offset
        mu = np.exp(eta)

        # same criterion as fixest
        # https://github.com/lrberge/fixest/blob/6b852fa277b947cea0bad8630986225ddb2d6f1b/R/ESTIMATION_FUNS.R#L2746
        deviance = poisson_deviance(Y, mu)
        crit = _relative_deviance_change(deviance, last)
        last = deviance

        stop_iterating = crit < tol

    return PoissonIwlsFit(
        beta=delta_new,
        eta=eta,
        mu=mu,
        irls_weights=combined_weights,
        WX=WX,
        WZ=WZ,
        residuals_working=resid,
        hessian=XWX,
        convergence=convergence,
        coefnames=coefnames,
        collin_vars=collin_vars,
        collin_index=collin_index,
        X=X,
    )


class GlmFamily(Protocol):
    """The family hooks the GLM IWLS loop needs (satisfied by ``Feglm``)."""

    def _get_mu(self, eta: np.ndarray) -> np.ndarray: ...

    def _get_link(self, mu: np.ndarray) -> np.ndarray: ...

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray: ...

    def _get_gprime(self, mu: np.ndarray) -> np.ndarray: ...

    def _get_V(self, mu: np.ndarray) -> np.ndarray: ...


def _check_glm_convergence(
    rel_deviance_change: float,
    tol: float,
    r: int,
    maxiter: int,
    model: str,
) -> bool:
    if model == "feglm-gaussian":
        converged = True
    else:
        converged = rel_deviance_change < tol
        if r == maxiter:  # kept from legacy loop (unreachable: r < maxiter)
            raise NonConvergenceError(
                f"""
                The IRLS algorithm did not converge with {maxiter}
                iterations. Try to increase the maximum number of iterations.
                """
            )

    return converged


def _glm_step_halving(
    y: np.ndarray,
    family: GlmFamily,
    tol: float,
    eta: np.ndarray,
    eta_new: np.ndarray,
    mu_new: np.ndarray,
    deviance: np.ndarray,
    deviance_new: np.ndarray,
    step_halving_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply step-halving if deviance did not decrease.

    Returns updated (eta_new, mu_new, deviance_new).
    """
    if deviance_new < deviance:
        return eta_new, mu_new, deviance_new

    alpha = 1.0
    while alpha > step_halving_tol:
        alpha /= 2.0
        eta_try = eta + alpha * (eta_new - eta)
        mu_try = family._get_mu(eta=eta_try)
        deviance_try = family._get_deviance(y, mu_try)
        if deviance_try < deviance:
            return eta_try, mu_try, deviance_try

    # Step-halving exhausted - check if change is within tolerance
    if _relative_deviance_change(deviance_new, deviance) < tol:
        return eta_new, mu_new, deviance_new

    raise RuntimeError(
        f"Step-halving failed. Deviance: {deviance_new:.6f} vs {deviance:.6f}"
    )


@dataclass(frozen=True, slots=True)
class GlmIwlsFit:
    """Result of the generic GLM IWLS loop.

    ``z_tilde``/``X_tilde``/``sqrt_W`` come from the final executed
    iteration; ``X`` is the original (non-demeaned) design matrix with
    collinear columns dropped.
    """

    beta: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    irls_weights: np.ndarray  # W from the final iteration
    sqrt_W: np.ndarray
    z_tilde: np.ndarray
    X_tilde: np.ndarray
    deviance: np.ndarray
    convergence: bool
    coefnames: list[str]
    collin_vars: list[str]
    collin_index: list[int]
    X: np.ndarray


def fit_iwls_glm(
    Y: np.ndarray,
    X: np.ndarray,
    family: GlmFamily,
    method: str,
    coefnames: list[str],
    collin_tol: float,
    solver: SolverOptions,
    tol: float,
    maxiter: int,
    accelerate: bool,
    fixef_tol: float,
    residualize: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]
    ],
) -> GlmIwlsFit:
    """Fit a GLM via IWLS on arrays.

    The implementation follows ideas developed in
    - Bergé (2018): https://ideas.repec.org/p/luc/wpaper/18-13.html
    - Correia, Guimaraes, Zylkin (2019): https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691
    - Stamann (2018): https://arxiv.org/pdf/1707.01815

    Parameters
    ----------
    family : GlmFamily
        The object providing the family hooks (link, inverse link,
        deviance, g', V) — in practice the ``Feglm`` model itself.
    method : str
        The model method string (e.g. "feglm-logit"); controls the
        starting values and the Gaussian one-step special case.
    accelerate : bool
        Whether to use ppmlhdfe-style warm-start acceleration (resolved by
        the caller; requires fixed effects).
    fixef_tol : float
        Starting inner (demeaning) tolerance.
    residualize : callable
        ``residualize(v, X, weights_flat, tol) -> (v_tilde, X_tilde)``
        residualizing by the fixed effects (identity when there are none).
    """
    _mean = np.mean(Y)
    if method in ("feglm-logit", "feglm-probit"):
        mu = np.full_like(Y.flatten(), 0.5, dtype=float)
    else:
        mu = np.full_like(Y.flatten(), _mean, dtype=float)

    eta = family._get_link(mu)
    deviance = family._get_deviance(Y.flatten(), mu)
    deviance_old = deviance + 1.0

    # Warm-start (for ppmlhdfe accelerations)
    z_prev = None
    z_tilde_prev = None
    X_tilde_prev = None
    inner_tol = fixef_tol

    convergence = False
    collin_vars: list[str] = []
    collin_index: list[int] = []

    for r in range(maxiter):
        if r > 0:
            rel_deviance_change = _relative_deviance_change(deviance, deviance_old)
            converged = _check_glm_convergence(
                rel_deviance_change=rel_deviance_change,
                tol=tol,
                r=r,
                maxiter=maxiter,
                model=method,
            )
            if converged:
                convergence = True
                break

            # Adaptive tolerance as in ppmlhdfe
            if accelerate and rel_deviance_change < 10 * inner_tol:
                inner_tol = inner_tol / 10

        gprime = family._get_gprime(mu=mu)
        W = 1 / (gprime**2 * family._get_V(mu=mu))
        sqrt_W = np.sqrt(W)

        z = eta + (Y.flatten() - mu) * gprime

        if accelerate and r > 0:
            z_input = z_tilde_prev + (z - z_prev)
            X_input = X_tilde_prev
        else:
            z_input = z
            X_input = X

        z_tilde, X_tilde = residualize(z_input, X_input, W.flatten(), inner_tol)

        if r == 0:
            # Check multicollinearity
            # We do this here after the first demeaning to also catch
            # collinearity with fixed effects
            X_tilde, coefnames, collin_vars, collin_index = (
                _drop_multicollinear_variables(
                    X_tilde,
                    coefnames,
                    collin_tol,
                )
            )
            if collin_index:
                # Drop covariates collinear with fixed effects
                X = X[:, ~np.array(collin_index)]

        WX = sqrt_W.flatten()[:, None] * X_tilde
        WZ = sqrt_W.flatten() * z_tilde

        tXX = WX.T @ WX
        tXz = WX.T @ WZ
        beta_new = solve_ols(tXX, tXz, solver)

        # Residual from demeaned regression (not weighted)
        e_new = z_tilde - X_tilde @ beta_new
        eta_new = z - e_new

        mu_new = family._get_mu(eta=eta_new)
        deviance_new = family._get_deviance(Y.flatten(), mu_new)

        # Step-halving if deviance did not decrease
        eta_new, mu_new, deviance_new = _glm_step_halving(
            y=Y.flatten(),
            family=family,
            tol=tol,
            eta=eta,
            eta_new=eta_new,
            mu_new=mu_new,
            deviance=deviance,
            deviance_new=deviance_new,
        )

        z_prev = z
        z_tilde_prev = z_tilde
        X_tilde_prev = X_tilde

        deviance_old = deviance
        eta = eta_new
        mu = mu_new
        deviance = deviance_new

        z_tilde_final = z_tilde
        X_tilde_final = X_tilde
        sqrt_W_final = sqrt_W
        beta_final = beta_new

    return GlmIwlsFit(
        beta=beta_final,
        eta=eta,
        mu=mu,
        irls_weights=W,
        sqrt_W=sqrt_W_final,
        z_tilde=z_tilde_final,
        X_tilde=X_tilde_final,
        deviance=deviance,
        convergence=convergence,
        coefnames=coefnames,
        collin_vars=collin_vars,
        collin_index=collin_index,
        X=X,
    )
