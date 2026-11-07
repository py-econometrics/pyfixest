from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.estimation.internals.literals import SolverOptions
from pyfixest.estimation.internals.model_state import SandwichComponents
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
    sandwich : SandwichComponents
        Weighted scores W X_hat * residuals of the first-stage projection
        X_hat = Z (Z' W Z)^{-1} Z' W X, the 2SLS Hessian X_hat' W X_hat, and
        its inverse.
    """

    beta: np.ndarray
    residuals: np.ndarray
    sandwich: SandwichComponents


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
        sandwich=SandwichComponents(
            scores=scores, hessian=hessian, bread=np.linalg.inv(hessian)
        ),
    )


def project_onto_design(
    *,
    response: NDArray[np.float64],
    design: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64] | None,
    solver: SolverOptions,
) -> NDArray[np.float64]:
    """Project response onto design, returning fitted values on response scale.

    Inputs are unweighted; when provided, ``sqrt_weights`` is applied for the
    projection only.
    """
    response_solver = response
    design_solver = design
    if sqrt_weights is not None:
        response_solver = response * sqrt_weights
        design_solver = design * sqrt_weights

    coefficients = solve_ols(
        tZX=design_solver.T @ design_solver,
        tZY=design_solver.T @ response_solver,
        solver=solver,
    ).reshape(design.shape[1], response.shape[1])

    return design @ coefficients


def fit_iv(
    X: NDArray[np.float64],
    X_hat: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    solver: SolverOptions = "np.linalg.solve",
) -> IvFit:
    """Fit 2SLS while keeping inputs and residuals in response scale.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (incl. endogenous regressors), shape (N, k).
        Demeaned but not WLS-transformed.
    X_hat : np.ndarray
        First-stage fitted values of X, shape (N, k), computed with the same
        observation weights. Demeaned but not WLS-transformed, with columns
        in the same order as X.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned but not WLS-transformed.
    weights : np.ndarray or None
        Non-negative observation weights, shape (N,) or (N, 1). ``None``
        applies no weights. Otherwise, the square-root transform is local to
        this function.
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    weight_values = None if weights is None else weights.reshape(-1)
    sqrt_weights = None if weight_values is None else np.sqrt(weight_values)[:, None]

    X_hat_solver = X_hat if sqrt_weights is None else X_hat * sqrt_weights
    hessian = X_hat_solver.T @ X_hat_solver

    Y_solver = Y if sqrt_weights is None else Y * sqrt_weights
    beta = solve_ols(hessian, X_hat_solver.T @ Y_solver, solver)
    residuals = Y.flatten() - (X @ beta).flatten()
    if weight_values is None:
        scores = X_hat * residuals[:, None]
    else:
        scores = X_hat * (weight_values * residuals)[:, None]

    return IvFit(
        beta=beta,
        residuals=residuals,
        sandwich=SandwichComponents(
            scores=scores, hessian=hessian, bread=np.linalg.inv(hessian)
        ),
    )
