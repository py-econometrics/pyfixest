from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.literals import SolverOptions
from pyfixest.estimation.internals.solvers import solve_ols

DemeanFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class PoisFit:
    """Result of a Poisson IRLS fit on prepared arrays.

    Mirrors ``GlmFit`` but specialised for Poisson regression: the working
    variable carries an additive offset, the IRLS weights are user_weights * mu,
    and there is no step-halving.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    eta : np.ndarray
        Final linear predictor (link scale), shape (N, 1).
    mu : np.ndarray
        Final fitted mean (response scale), shape (N, 1).
    W : np.ndarray
        Final IRLS weights, ``user_weights * mu``, shape (N, 1).
    sqrt_W : np.ndarray
        ``sqrt(W)``, shape (N, 1).
    z_tilde : np.ndarray
        Final demeaned working response, shape (N, 1).
    X_tilde : np.ndarray
        Final demeaned design matrix, shape (N, k).
    X : np.ndarray
        Un-demeaned design matrix with collinear columns dropped, shape (N, k).
    deviance : float
        Final in-loop (unweighted) deviance.
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


def pois_deviance(
    Y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Poisson deviance.

    Defined as twice the difference in log likelihood between the saturated
    model and the model being fit (Dobson & Barnett, ch. 5.6).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if weights is None:
            weights = np.ones_like(Y)
        w = weights.flatten()
        Y_flat = Y.flatten()
        mu_flat = mu.flatten()
        return float(
            np.float64(
                2
                * np.sum(
                    w
                    * (
                        np.where(Y_flat == 0, 0, Y_flat * np.log(Y_flat / mu_flat))
                        - (Y_flat - mu_flat)
                    )
                )
            )
        )


def fit_pois_irls(
    X: np.ndarray,
    Y: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray,
    *,
    demean: DemeanFn,
    coefnames: list[str],
    collin_tol: float,
    solver: SolverOptions = "np.linalg.solve",
    maxiter: int = 25,
    tol: float = 1e-8,
) -> PoisFit:
    """Fit a fixed-effects Poisson model via Iterated Weighted Least Squares.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (N, k). Un-demeaned.
    Y : np.ndarray
        Dependent variable, shape (N, 1).
    offset : np.ndarray
        Additive offset on the link scale, shape (N, 1). Pass an array of
        zeros if no offset is desired.
    weights : np.ndarray
        User-supplied observation weights, shape (N, 1).
    demean : Callable
        ``demean(joint, w) -> joint_resid``. The caller is responsible for
        capturing the fixed-effects list, na-index, and demean cache. If no
        fixed effects are present, pass an identity function.
    coefnames : list[str]
        Names of the columns of X. Used by the collinearity drop.
    collin_tol : float
        Tolerance for the collinearity drop.
    solver, maxiter, tol : see Fepois docs.
    """
    N = Y.shape[0]
    X_eff = X
    collin_vars: list[str] = []
    collin_index: list[bool] = []
    converged = False
    stop_iterating = False

    Y_arr = Y if Y.ndim == 2 else Y.reshape((N, 1))
    last: float = 0.0
    crit: float = 1.0

    # Buffers populated each iteration; used after the loop.
    beta_final: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    Z_resid: np.ndarray
    X_resid: np.ndarray
    combined_weights: np.ndarray
    i = 0

    for i in range(maxiter):
        if stop_iterating:
            converged = True
            break

        if i == 0:
            _mean = np.mean(Y_arr)
            mu = (Y_arr + _mean) / 2
            eta = np.log(mu)
            Z = eta - offset + Y_arr / mu - 1
            last = pois_deviance(Y_arr, mu)
        else:
            Z = eta - offset + Y_arr / mu - 1

        reg_Z = Z.copy()
        ZX = np.concatenate([reg_Z, X_eff], axis=1)
        combined_weights = weights * mu

        ZX_resid = demean(ZX, combined_weights.flatten())

        Z_resid = ZX_resid[:, 0].reshape((N, 1))
        X_resid = ZX_resid[:, 1:]

        if i == 0:
            X_resid, coefnames, collin_vars, collin_index = (
                drop_multicollinear_variables(X_resid, coefnames, collin_tol)
            )
            if collin_index:
                X_eff = X_eff[:, ~np.array(collin_index)]

        sqrt_cw = np.sqrt(combined_weights)
        WX = sqrt_cw * X_resid
        WZ = sqrt_cw * Z_resid

        XWX = WX.T @ WX
        XWZ = WX.T @ WZ

        delta_new = solve_ols(XWX, XWZ, solver).reshape((-1, 1))
        resid = Z_resid - X_resid @ delta_new

        eta = Z - resid + offset
        mu = np.exp(eta)

        deviance = pois_deviance(Y_arr, mu)
        crit = float(np.abs(deviance - last) / (0.1 + np.abs(last)))
        last = deviance

        stop_iterating = crit < tol
        beta_final = delta_new

    sqrt_W_final = np.sqrt(combined_weights)
    return PoisFit(
        beta=beta_final.flatten(),
        eta=eta,
        mu=mu,
        W=combined_weights,
        sqrt_W=sqrt_W_final,
        z_tilde=Z_resid,
        X_tilde=X_resid,
        X=X_eff,
        deviance=last,
        converged=converged,
        n_iter=i,
        coefnames=coefnames,
        collin_vars=collin_vars,
        collin_index=collin_index,
    )
