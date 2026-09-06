from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        Residuals Y - X @ beta, shape (N,). Always on the scale of the
        supplied Y; weights never rescale them.
    scores : np.ndarray
        Weighted score matrix W X * residuals, shape (N, k).
    hessian : np.ndarray
        Weighted Hessian X' W X, shape (k, k).
    tZX : np.ndarray
        Z'X (= X' W X for OLS), shape (k, k).
    tZy : np.ndarray
        Z'Y (= X' W Y for OLS), shape (k, 1).
    """

    beta: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    tZX: np.ndarray
    tZy: np.ndarray


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
    scores : np.ndarray
        Weighted score matrix W Z * residuals, shape (N, k_z).
    hessian : np.ndarray
        Weighted instrument cross-product Z' W Z, shape (k_z, k_z).
    tZX : np.ndarray
        Weighted cross-product Z' W X, shape (k_z, k).
    tXZ : np.ndarray
        Weighted cross-product X' W Z, shape (k, k_z).
    tZy : np.ndarray
        Weighted cross-product Z' W Y, shape (k_z, 1).
    tZZinv : np.ndarray
        (Z' W Z)^{-1}, shape (k_z, k_z).
    """

    beta: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    tZX: np.ndarray
    tXZ: np.ndarray
    tZy: np.ndarray
    tZZinv: np.ndarray


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
        selects the unweighted path without creating unit weights or transformed
        copies. Otherwise, the square-root transform is local to this function.
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

    tZX = X_solver.T @ X_solver
    tZy = X_solver.T @ Y_solver
    beta = solve_ols(tZX, tZy, solver)
    residuals = Y.flatten() - (X @ beta).flatten()
    if weight_values is None:
        scores = X * residuals[:, None]
    else:
        scores = X * (weight_values * residuals)[:, None]
    hessian = tZX.copy()
    return OlsFit(
        beta=beta,
        residuals=residuals,
        scores=scores,
        hessian=hessian,
        tZX=tZX,
        tZy=tZy,
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
        selects the unweighted path without creating unit weights or transformed
        copies. Otherwise, the square-root transform is local to this function.
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

    H = tXZ @ tZZinv
    A = H @ tZX
    B = H @ tZy
    beta = solve_ols(A, B, solver)

    residuals = Y.flatten() - (X @ beta).flatten()
    if weight_values is None:
        scores = Z * residuals[:, None]
    else:
        scores = Z * (weight_values * residuals)[:, None]
    hessian = tZZ

    return IvFit(
        beta=beta,
        residuals=residuals,
        scores=scores,
        hessian=hessian,
        tZX=tZX,
        tXZ=tXZ,
        tZy=tZy,
        tZZinv=tZZinv,
    )
