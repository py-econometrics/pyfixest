from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.families import GlmFamily
from pyfixest.estimation.internals.literals import SolverOptions
from pyfixest.estimation.internals.solvers import solve_ols

DemeanFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float],
    tuple[np.ndarray, np.ndarray],
]


@dataclass(frozen=True, slots=True)
class GlmFit:
    """Result of a GLM IRLS fit on prepared arrays.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    eta : np.ndarray
        Final linear predictor (link scale), shape (N,).
    mu : np.ndarray
        Final fitted mean (response scale), shape (N,).
    W : np.ndarray
        Final IRLS weights, shape (N,).
    sqrt_W : np.ndarray
        Square root of W, shape (N,).
    z_tilde : np.ndarray
        Final demeaned working response, shape (N,).
    X_tilde : np.ndarray
        Final demeaned design matrix, shape (N, k).
    X : np.ndarray
        The (un-demeaned) design matrix with collinear columns dropped, shape (N, k).
    deviance : float
        Final deviance.
    converged : bool
        Whether the IRLS loop converged within ``maxiter`` iterations.
    n_iter : int
        Number of completed iterations.
    coefnames : list[str]
        Coefficient names after the collinearity drop.
    collin_vars : list[str]
        Names of variables dropped due to collinearity.
    collin_index : list[bool]
        Boolean mask over the input X's columns: True marks a dropped column.
        Empty when no columns were dropped.
    """

    beta: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    sqrt_W: np.ndarray
    z_tilde: np.ndarray
    X_tilde: np.ndarray
    X: np.ndarray
    deviance: float
    converged: bool
    n_iter: int
    coefnames: list[str]
    collin_vars: list[str]
    collin_index: list[bool]


def _rel_dev_change(deviance: float, deviance_old: float) -> float:
    return float(np.abs(deviance - deviance_old) / (0.1 + np.abs(deviance_old)))


def _check_convergence(
    rel_dev_change: float,
    tol: float,
    r: int,
    maxiter: int,
    is_gaussian: bool,
) -> bool:
    if is_gaussian:
        return True
    converged = rel_dev_change < tol
    if r == maxiter:
        raise NonConvergenceError(
            f"""
            The IRLS algorithm did not converge with {maxiter}
            iterations. Try to increase the maximum number of iterations.
            """
        )
    return converged


def _step_halving(
    family: GlmFamily,
    y_flat: np.ndarray,
    eta: np.ndarray,
    eta_new: np.ndarray,
    mu_new: np.ndarray,
    deviance: float,
    deviance_new: float,
    tol: float,
    weights: np.ndarray | None,
    step_halving_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    if deviance_new < deviance:
        return eta_new, mu_new, deviance_new

    alpha = 1.0
    while alpha > step_halving_tol:
        alpha /= 2.0
        eta_try = eta + alpha * (eta_new - eta)
        mu_try = family.inv_link(eta_try)
        deviance_try = family.deviance(y_flat, mu_try, weights)
        if deviance_try < deviance:
            return eta_try, mu_try, deviance_try

    if _rel_dev_change(deviance_new, deviance) < tol:
        return eta_new, mu_new, deviance_new

    raise RuntimeError(
        f"Step-halving failed. Deviance: {deviance_new:.6f} vs {deviance:.6f}"
    )


def fit_glm_irls(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    family: GlmFamily,
    demean: DemeanFn,
    coefnames: list[str],
    collin_tol: float,
    accelerate: bool,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    solver: SolverOptions = "np.linalg.solve",
    maxiter: int = 25,
    tol: float = 1e-8,
    fixef_tol: float = 1e-8,
) -> GlmFit:
    """Fit a fixed-effects GLM via iterated weighted least squares.

    The implementation follows ideas developed in
    - Bergé (2018): https://ideas.repec.org/p/luc/wpaper/18-13.html
    - Correia, Guimaraes, Zylkin (2019): https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691
    - Stamann (2018): https://arxiv.org/pdf/1707.01815

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (N, k). Un-demeaned.
    Y : np.ndarray
        Dependent variable, shape (N,) or (N, 1).
    family : GlmFamily
        Family providing link / inverse link / gprime / variance / deviance /
        initial mu.
    demean : Callable
        ``demean(v, X, weights, tol) -> (v_tilde, X_tilde)``. The caller is
        responsible for capturing the fixed-effects list and any caching state.
        For no fixed effects, pass an identity function.
    coefnames : list[str]
        Names of the columns of X. Used by the collinearity drop.
    collin_tol : float
        Tolerance for the collinearity drop.
    accelerate : bool
        Enable ppmlhdfe-style warm-start acceleration.
    offset : np.ndarray, optional
        Additive offset on the link scale, shape (N,) or (N, 1). `None`
        (default) is equivalent to a zero offset.
    weights : np.ndarray, optional
        User-supplied regression weights, shape (N,) or (N, 1). `None`
        (default) is equivalent to unit weights.
    solver, maxiter, tol, fixef_tol : see Feglm docs.
    """
    Y_flat = Y.flatten()
    N = Y_flat.shape[0]
    offset_flat = offset.flatten() if offset is not None else np.zeros(N)
    weights_flat = weights.flatten() if weights is not None else np.ones(N)

    mu = family.mu_start(Y)
    eta = family.link(mu)
    deviance = family.deviance(Y_flat, mu, weights)
    deviance_old = deviance + 1.0

    z_prev: np.ndarray | None = None
    z_tilde_prev: np.ndarray | None = None
    X_tilde_prev: np.ndarray | None = None
    inner_tol = fixef_tol
    X_eff = X

    collin_vars: list[str] = []
    collin_index: list[bool] = []
    converged = False

    # Buffers populated each iteration; used after the loop.
    beta_final: np.ndarray
    z_tilde_final: np.ndarray
    X_tilde_final: np.ndarray
    W_final: np.ndarray
    sqrt_W_final: np.ndarray
    r = 0

    for r in range(maxiter):
        if r > 0:
            rel_dev_change = _rel_dev_change(deviance, deviance_old)
            converged = _check_convergence(
                rel_dev_change=rel_dev_change,
                tol=tol,
                r=r,
                maxiter=maxiter,
                is_gaussian=family.name == "gaussian",
            )
            if converged:
                break

            if accelerate and rel_dev_change < 10 * inner_tol:
                inner_tol = inner_tol / 10

        gprime = family.gprime(mu)
        W = weights_flat / (gprime**2 * family.variance(mu))
        sqrt_W = np.sqrt(W)

        z = (eta - offset_flat) + (Y_flat - mu) * gprime

        if accelerate and r > 0:
            assert z_tilde_prev is not None and z_prev is not None
            assert X_tilde_prev is not None
            z_input = z_tilde_prev + (z - z_prev)
            X_input = X_tilde_prev
        else:
            z_input = z
            X_input = X_eff

        z_tilde, X_tilde = demean(z_input, X_input, W.flatten(), inner_tol)

        if r == 0:
            X_tilde, coefnames, collin_vars, collin_index = (
                drop_multicollinear_variables(X_tilde, coefnames, collin_tol)
            )
            if collin_index:
                X_eff = X_eff[:, ~np.array(collin_index)]

        WX = sqrt_W.flatten()[:, None] * X_tilde
        WZ = sqrt_W.flatten() * z_tilde

        beta = solve_ols(WX.T @ WX, WX.T @ WZ, solver)

        e_new = z_tilde - X_tilde @ beta
        eta_new = (z - e_new) + offset_flat

        mu_new = family.inv_link(eta_new)
        deviance_new = family.deviance(Y_flat, mu_new, weights)

        if r > 0:
            eta_new, mu_new, deviance_new = _step_halving(
                family=family,
                y_flat=Y_flat,
                eta=eta,
                eta_new=eta_new,
                mu_new=mu_new,
                deviance=deviance,
                deviance_new=deviance_new,
                tol=tol,
                weights=weights,
            )

        z_prev = z
        z_tilde_prev = z_tilde
        X_tilde_prev = X_tilde

        deviance_old = deviance
        eta = eta_new
        mu = mu_new
        deviance = deviance_new

        beta_final = beta
        z_tilde_final = z_tilde
        X_tilde_final = X_tilde
        W_final = W
        sqrt_W_final = sqrt_W

    return GlmFit(
        beta=beta_final,
        eta=eta,
        mu=mu,
        W=W_final,
        sqrt_W=sqrt_W_final,
        z_tilde=z_tilde_final,
        X_tilde=X_tilde_final,
        X=X_eff,
        deviance=deviance,
        converged=converged,
        n_iter=r,
        coefnames=coefnames,
        collin_vars=collin_vars,
        collin_index=collin_index,
    )
