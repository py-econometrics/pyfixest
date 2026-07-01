"""
Optimized Frisch-Newton Interior Point Solver for Quantile Regression.

This module provides an optimized implementation that:
1. Uses Cholesky factorization reuse (factor once per iteration, solve twice)
2. Avoids scipy sparse overhead for moderate problem sizes
3. Uses NumPy's optimized BLAS routines for dense operations

Key insight: For quantile regression, the normal matrix M = A @ D @ A.T is (k x k)
where k = number of coefficients, which is typically small (10-100). The overhead
of sparse matrix operations outweighs benefits for most practical problem sizes.
"""

from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _duality_gap(x: np.ndarray, z: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """Compute duality gap x'z + s'w."""
    return float(x @ z + s @ w)


def _bound(v: np.ndarray, dv: np.ndarray, backoff: float) -> float:
    """Compute max step maintaining v + alpha*dv > 0."""
    mask = dv < 0
    if not mask.any():
        return 1.0
    alpha_max = (-v[mask] / dv[mask]).min()
    return min(backoff * alpha_max, 1.0)


def _step_length(primal: tuple, slack: tuple, backoff: float) -> float:
    """Compute step length for primal and slack variables."""
    x, dx = primal
    s, ds = slack
    return min(_bound(x, dx, backoff), _bound(s, ds, backoff))


def cold_start(A: np.ndarray, c: np.ndarray, q: float) -> tuple:
    """Initialize interior point variables."""
    n = A.shape[1]
    x = np.full(n, 1.0 - q)
    s = np.full(n, q)

    d_plus = np.maximum(c, 0.0)
    d_minus = np.maximum(-c, 0.0)

    U = x @ d_plus + s @ d_minus
    mu0 = max(1.0, U / n)
    alpha = (n * mu0 - U) / (np.sum(1 / x) + np.sum(1 / s))

    eps = 1e-8
    z = np.maximum(d_plus, eps) + alpha / x
    w = np.maximum(d_minus, eps) + alpha / s

    # Solve (A @ A.T) @ y = A @ (c - z + w)
    rhs = A @ (c - z + w)
    AAt = A @ A.T
    y = np.linalg.solve(AAt, rhs)

    return x, s, z, w, y


def frisch_newton_optimized(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
    backoff: float = 0.9995,
    beta_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized Frisch-Newton interior point solver.

    Key optimizations:
    1. Cholesky factorization computed once per iteration, reused for both solves
    2. Pre-allocated work arrays to avoid memory allocation in loop
    3. Uses BLAS-optimized dense matrix operations

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix (k, N) - MUST be dense numpy array
    b : np.ndarray
        RHS of equality constraint (k,)
    c : np.ndarray
        Objective coefficients (N,)
    u : np.ndarray
        Upper bounds (N,)
    q : float
        Quantile level
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    backoff : float
        Step length safety factor

    Returns
    -------
    tuple
        (beta, converged, iterations, x, s, z, w, y)
    """
    m, n = A.shape
    c = np.asarray(c).ravel()
    b = np.asarray(b).ravel()
    u = np.asarray(u).ravel()

    # Initialize
    x, s, z, w, y = cold_start(A, c, q)

    # Pre-allocate all work arrays
    r1_tilde = np.empty(n)
    r2_tilde = np.empty(m)
    Qinv = np.empty(n)
    work_m = np.empty(m)
    work_n = np.empty(n)

    dx_aff = np.empty(n)
    ds_aff = np.empty(n)
    dz_aff = np.empty(n)
    dw_aff = np.empty(n)
    dy_aff = np.empty(m)

    dx_cor = np.empty(n)
    ds_cor = np.empty(n)
    dz_cor = np.empty(n)
    dw_cor = np.empty(n)
    dy_cor = np.empty(m)

    x_pred = np.empty(n)
    s_pred = np.empty(n)
    z_pred = np.empty(n)
    w_pred = np.empty(n)
    y_pred = np.empty(m)

    # Validate
    for val in [x, z, s, w]:
        if np.any(val <= 0):
            raise ValueError("Initial values must be positive")

    mu_curr = _duality_gap(x, z, s, w)
    has_converged = False
    _it = 0

    for _it in range(max_iter):
        if mu_curr < tol:
            has_converged = True
            break

        # Residuals
        r1_tilde[:] = c - A.T @ y
        r2_tilde[:] = b - A @ x

        # Diagonal scaling
        Qinv[:] = 1.0 / (z / x + w / s)

        # Form normal matrix M = A @ diag(Qinv) @ A.T
        # Optimized: M[i,j] = sum_l A[i,l] * Qinv[l] * A[j,l]
        # Using: M = A @ (Qinv[:, None] * A.T) with BLAS
        M = A @ (Qinv[:, np.newaxis] * A.T)

        # Cholesky factorization (reused for both solves)
        try:
            cho_M = cho_factor(M, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # Regularize if nearly singular
            M += 1e-10 * np.eye(m)
            cho_M = cho_factor(M, lower=True, check_finite=False)

        # Affine direction
        work_n[:] = Qinv * r1_tilde
        work_m[:] = r2_tilde + A @ work_n
        dy_aff[:] = cho_solve(cho_M, work_m, check_finite=False)

        dx_aff[:] = Qinv * (A.T @ dy_aff - r1_tilde)
        ds_aff[:] = -dx_aff
        dz_aff[:] = -z - (z / x) * dx_aff
        dw_aff[:] = -w - (w / s) * ds_aff

        # Step lengths
        alpha_p_aff = _step_length((x, dx_aff), (s, ds_aff), backoff)
        alpha_d_aff = _step_length((z, dz_aff), (w, dw_aff), backoff)

        # Predictor step
        x_pred[:] = x + alpha_p_aff * dx_aff
        s_pred[:] = s + alpha_p_aff * ds_aff
        y_pred[:] = y + alpha_d_aff * dy_aff
        z_pred[:] = z + alpha_d_aff * dz_aff
        w_pred[:] = w + alpha_d_aff * dw_aff

        # Centering parameter
        mu_aff = _duality_gap(x_pred, z_pred, s_pred, w_pred)
        sigma = (mu_aff / mu_curr) ** 2
        mu_targ = sigma * mu_curr / n

        # Corrector direction (REUSE Cholesky factorization)
        r1_hat = mu_targ * (1/s - 1/x) + (dx_aff * dz_aff)/x - (ds_aff * dw_aff)/s
        work_m[:] = A @ (Qinv * r1_hat)
        dy_cor[:] = cho_solve(cho_M, work_m, check_finite=False)

        dx_cor[:] = Qinv * (A.T @ dy_cor - r1_hat)
        ds_cor[:] = -dx_cor
        dz_cor[:] = -(z / x) * dx_cor + (mu_targ - dx_aff * dz_aff) / x
        dw_cor[:] = -(w / s) * ds_cor + (mu_targ - ds_aff * dw_aff) / s

        # Final step lengths
        alpha_p_cor = _step_length((x_pred, dx_cor), (s_pred, ds_cor), backoff)
        alpha_d_cor = _step_length((z_pred, dz_cor), (w_pred, dw_cor), backoff)

        # Update
        x[:] = x_pred + alpha_p_cor * dx_cor
        s[:] = s_pred + alpha_p_cor * ds_cor
        y[:] = y_pred + alpha_d_cor * dy_cor
        z[:] = z_pred + alpha_d_cor * dz_cor
        w[:] = w_pred + alpha_d_cor * dw_cor

        mu_curr = _duality_gap(x, z, s, w)

    return -y, has_converged, _it, x, s, z, w, y


# Backward compatible wrapper
def frisch_newton_solver(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
    chol: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    backoff: float = 0.9995,
    beta_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper for backward compatibility (chol and P are ignored)."""
    return frisch_newton_optimized(A, b, c, u, q, tol, max_iter, backoff, beta_init)
