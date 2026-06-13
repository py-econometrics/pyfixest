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
        Residuals Y - X @ beta, shape (N,).
    scores : np.ndarray
        Score matrix X * residuals, shape (N, k).
    hessian : np.ndarray
        Hessian X'X, shape (k, k).
    tZX : np.ndarray
        Z'X (= X'X for OLS), shape (k, k).
    tZy : np.ndarray
        Z'Y (= X'Y for OLS), shape (k, 1).
    """

    beta: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    tZX: np.ndarray
    tZy: np.ndarray


@dataclass(frozen=True, slots=True)
class IvFit:
    """Result of a 2SLS fit.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient estimates, shape (k,).
    residuals : np.ndarray
        Second-stage residuals Y - X @ beta, shape (N,).
    scores : np.ndarray
        Score matrix Z * residuals, shape (N, k_z).
    hessian : np.ndarray
        Z'Z, shape (k_z, k_z).
    tZX : np.ndarray
        Z'X, shape (k_z, k).
    tXZ : np.ndarray
        X'Z, shape (k, k_z).
    tZy : np.ndarray
        Z'Y, shape (k_z, 1).
    tZZinv : np.ndarray
        (Z'Z)^{-1}, shape (k_z, k_z).
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
    solver: SolverOptions = "np.linalg.solve",
) -> OlsFit:
    """Fit a least-squares model on prepared arrays.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (N, k). Demeaned and WLS-transformed.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned and WLS-transformed.
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    tZX = X.T @ X
    tZy = X.T @ Y
    beta = solve_ols(tZX, tZy, solver)
    residuals = Y.flatten() - (X @ beta).flatten()
    scores = X * residuals[:, None]
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
    solver: SolverOptions = "np.linalg.solve",
) -> IvFit:
    """Fit a 2SLS model on prepared arrays.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (incl. endogenous regressors), shape (N, k).
        Demeaned and WLS-transformed.
    Z : np.ndarray
        Instrument matrix, shape (N, k_z). Demeaned and WLS-transformed.
    Y : np.ndarray
        Dependent variable, shape (N, 1). Demeaned and WLS-transformed.
    solver : SolverOptions
        Solver passed through to ``solve_ols``.
    """
    tZX = Z.T @ X
    tXZ = X.T @ Z
    tZy = Z.T @ Y
    tZZ = Z.T @ Z
    tZZinv = np.linalg.inv(tZZ)

    H = tXZ @ tZZinv
    A = H @ tZX
    B = H @ tZy
    beta = solve_ols(A, B, solver)

    residuals = Y.flatten() - (X @ beta).flatten()
    scores = Z * residuals[:, None]
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
