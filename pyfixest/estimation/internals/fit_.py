from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfixest.estimation.internals.literals import SolverOptions
from pyfixest.estimation.internals.model_state import (
    IvSandwichComponents,
    SandwichComponents,
)
from pyfixest.estimation.internals.solvers import solve_ols


@dataclass(frozen=True, slots=True)
class OlsFit:
    """Result of a (weighted) least-squares fit.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    residuals : np.ndarray
        Residuals Y - X @ beta, shape (N,). Always on the scale of the
        supplied Y; weights never rescale them.
    sandwich : SandwichComponents
        Weighted scores W X * residuals, the Hessian X' W X, and its inverse.
    """

    beta: np.ndarray
    residuals: np.ndarray
    sandwich: SandwichComponents


@dataclass(frozen=True, slots=True)
class IvFit:
    """Result of a (weighted) 2SLS fit.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    residuals : np.ndarray
        Second-stage residuals Y - X @ beta, shape (N,). Always on the scale
        of the supplied Y; weights never rescale them.
    sandwich : IvSandwichComponents
        Weighted instrument scores W Z * residuals, the 2SLS Hessian, its
        inverse, and the projection X' W Z (Z' W Z)^{-1}.
    """

    beta: np.ndarray
    residuals: np.ndarray
    sandwich: IvSandwichComponents


def fit_ols(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    solver: SolverOptions = "np.linalg.solve",
) -> OlsFit:
    """Fit OLS/WLS while keeping inputs and residuals in response scale.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (N, k). Demeaned but not WLS-transformed.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned but not WLS-transformed.
    weights : np.ndarray or None
        Non-negative observation weights, shape (N,) or (N, 1). ``None``
        applies no weights. Otherwise, the square-root transform is local to
        this function.
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    if weights is None:
        X_solver = X
        Y_solver = Y
        weight_values = None
    else:
        weight_values = weights.reshape(-1)
        sqrt_weights = np.sqrt(weight_values)[:, None]
        X_solver = X * sqrt_weights
        Y_solver = Y * sqrt_weights

    hessian = X_solver.T @ X_solver
    tXy = X_solver.T @ Y_solver
    beta = solve_ols(hessian, tXy, solver)
    residuals = Y.flatten() - (X @ beta).flatten()
    if weight_values is None:
        scores = X * residuals[:, None]
    else:
        scores = X * (weight_values * residuals)[:, None]
    return OlsFit(
        beta=beta,
        residuals=residuals,
        sandwich=SandwichComponents.from_hessian(scores=scores, hessian=hessian),
    )


def fit_iv(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    solver: SolverOptions = "np.linalg.solve",
) -> IvFit:
    """Fit 2SLS while keeping inputs and residuals in response scale.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (incl. endogenous regressors), shape (N, k).
        Demeaned but not WLS-transformed.
    Z : np.ndarray
        Full instrument matrix, including exogenous regressors that instrument
        themselves, shape (N, k_z). Demeaned but not WLS-transformed.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned but not WLS-transformed.
    weights : np.ndarray or None
        Non-negative observation weights, shape (N,) or (N, 1). ``None``
        applies no weights. Otherwise, the square-root transform is local to
        this function.
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    if weights is None:
        X_solver = X
        Z_solver = Z
        Y_solver = Y
        weight_values = None
    else:
        weight_values = weights.reshape(-1)
        sqrt_weights = np.sqrt(weight_values)[:, None]
        X_solver = X * sqrt_weights
        Z_solver = Z * sqrt_weights
        Y_solver = Y * sqrt_weights

    tZX = Z_solver.T @ X_solver
    tXZ = X_solver.T @ Z_solver
    tZy = Z_solver.T @ Y_solver
    tZZ = Z_solver.T @ Z_solver
    tZZinv = np.linalg.inv(tZZ)

    # 2SLS normal equations: X'WZ (Z'WZ)^-1 Z'WX beta = X'WZ (Z'WZ)^-1 Z'Wy.
    projection = tXZ @ tZZinv
    hessian = projection @ tZX
    beta = solve_ols(hessian, projection @ tZy, solver)

    residuals = Y.flatten() - (X @ beta).flatten()
    if weight_values is None:
        scores = Z * residuals[:, None]
    else:
        scores = Z * (weight_values * residuals)[:, None]

    return IvFit(
        beta=beta,
        residuals=residuals,
        sandwich=IvSandwichComponents.from_projection(
            scores=scores, hessian=hessian, projection=projection
        ),
    )
