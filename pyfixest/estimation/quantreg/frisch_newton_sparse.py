"""
Sparse-aware Frisch-Newton Interior Point Solver for Quantile Regression.

This module provides a sparse matrix implementation of the Frisch-Newton
interior point algorithm for solving quantile regression problems. The
implementation automatically detects whether the input matrix is sparse
and uses optimized code paths accordingly.

Algorithm Reference:
    Koenker, R., & Ng, P. (2005). "A Frisch-Newton Algorithm for Sparse
    Quantile Regression." Acta Mathematica Applicatae Sinica, 21(2), 225-236.

Performance Notes:
    - For dense matrices, performance is similar to the original implementation
    - For sparse matrices (>50% zeros), significant speedups are achieved:
      * Matrix-vector products: O(nnz) instead of O(k*N)
      * Normal matrix formation: O(nnz*k) instead of O(k^2*N)
    - The normal matrix M = A @ diag(Qinv) @ A.T is (k x k) and typically dense,
      so the linear solve remains O(k^3) regardless of input sparsity

Author: PyFixest Contributors
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, issparse


# =============================================================================
# Helper Functions
# =============================================================================


def _duality_gap(x: np.ndarray, z: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the duality gap for the interior point method.

    The duality gap measures how far the current solution is from optimality.
    The algorithm converges when this value falls below the tolerance.

    Parameters
    ----------
    x : np.ndarray
        Primal variable (N,)
    z : np.ndarray
        Dual variable for lower bound (N,)
    s : np.ndarray
        Slack variable s = u - x (N,)
    w : np.ndarray
        Dual variable for upper bound (N,)

    Returns
    -------
    float
        The duality gap: x'z + s'w
    """
    return float(x @ z + s @ w)


def _bound(v: np.ndarray, dv: np.ndarray, backoff: float) -> float:
    """
    Compute maximum step length to maintain positivity of v + alpha * dv.

    Parameters
    ----------
    v : np.ndarray
        Current values (must be positive)
    dv : np.ndarray
        Step direction
    backoff : float
        Safety factor (typically 0.9995) to stay strictly inside the boundary

    Returns
    -------
    float
        Maximum allowable step length
    """
    mask = dv < 0
    if not mask.any():
        return 1.0
    alpha_max = (-v[mask] / dv[mask]).min()
    return min(backoff * alpha_max, 1.0)


def _step_length(
    primal: tuple[np.ndarray, np.ndarray],
    slack: tuple[np.ndarray, np.ndarray],
    backoff: float,
) -> float:
    """
    Compute step length for primal variables maintaining positivity.

    Parameters
    ----------
    primal : tuple
        (x, dx) - current value and direction for primal variable
    slack : tuple
        (s, ds) - current value and direction for slack variable
    backoff : float
        Safety factor for step length

    Returns
    -------
    float
        Maximum step length maintaining x > 0 and s > 0
    """
    x, dx = primal
    s, ds = slack
    return min(_bound(x, dx, backoff), _bound(s, ds, backoff))


# =============================================================================
# Normal Matrix Formation (Key for Sparse Efficiency)
# =============================================================================


def _form_normal_matrix_sparse(A_csr: csr_matrix, Qinv: np.ndarray) -> np.ndarray:
    """
    Efficiently compute M = A @ diag(Qinv) @ A.T for sparse A.

    This is the key operation that benefits from sparsity. Instead of forming
    the full (k, N) @ (N, N) @ (N, k) product, we:
    1. Scale each column j of A by sqrt(Qinv[j])
    2. Compute A_scaled @ A_scaled.T

    Complexity: O(nnz * k) instead of O(k^2 * N)

    Parameters
    ----------
    A_csr : csr_matrix
        Constraint matrix in CSR format, shape (k, N)
    Qinv : np.ndarray
        Diagonal scaling factors, shape (N,)

    Returns
    -------
    np.ndarray
        Normal matrix M, shape (k, k), returned as dense array
    """
    sqrt_Qinv = np.sqrt(Qinv)
    # Multiply each column of A by corresponding sqrt(Qinv) element
    # For CSR matrix, this scales elements by their column index
    A_scaled = A_csr.multiply(sqrt_Qinv)
    # M = A_scaled @ A_scaled.T is (k, k) - convert to dense
    M = (A_scaled @ A_scaled.T).toarray()
    return M


def _form_normal_matrix_dense(A: np.ndarray, Qinv: np.ndarray) -> np.ndarray:
    """
    Compute M = A @ diag(Qinv) @ A.T for dense A.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix, shape (k, N)
    Qinv : np.ndarray
        Diagonal scaling factors, shape (N,)

    Returns
    -------
    np.ndarray
        Normal matrix M, shape (k, k)
    """
    # Equivalent to A @ np.diag(Qinv) @ A.T but memory efficient
    return A @ (Qinv[:, np.newaxis] * A.T)


# =============================================================================
# Cold Start Initialization
# =============================================================================


def cold_start(
    A: Union[np.ndarray, csr_matrix],
    c: np.ndarray,
    q: float,
    is_sparse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize the interior point method with a cold start.

    Computes initial values for primal and dual variables that satisfy
    the positivity constraints and provide a reasonable starting point.

    Parameters
    ----------
    A : array or sparse matrix
        Constraint matrix, shape (k, N)
    c : np.ndarray
        Objective coefficients (negative of Y), shape (N,)
    q : float
        Quantile level in (0, 1)
    is_sparse : bool
        Whether A is a sparse matrix

    Returns
    -------
    tuple
        (x, s, z, w, y) - initial values for all variables
    """
    n = A.shape[1]

    # Initialize primal variables
    x = np.full(n, 1.0 - q)
    s = np.full(n, q)

    # Initialize dual variables based on objective
    d_plus = np.maximum(c, 0.0)
    d_minus = np.maximum(-c, 0.0)

    U = x @ d_plus + s @ d_minus
    mu0 = max(1.0, U / n)
    alpha = (n * mu0 - U) / (np.sum(1 / x) + np.sum(1 / s))

    eps = 1e-8
    z = np.maximum(d_plus, eps) + alpha / x
    w = np.maximum(d_minus, eps) + alpha / s

    # Solve for y from normal equations: (A @ A.T) @ y = A @ (c - z + w)
    rhs_vec = c - z + w

    if is_sparse:
        rhs = np.asarray(A @ rhs_vec).ravel()
        AAt = (A @ A.T).toarray()
    else:
        rhs = A @ rhs_vec
        AAt = A @ A.T

    y = np.linalg.solve(AAt, rhs)

    return x, s, z, w, y


# =============================================================================
# Main Solver
# =============================================================================


def frisch_newton_solver_sparse(
    A: Union[np.ndarray, csr_matrix],
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
    backoff: float = 0.9995,
    beta_init: Optional[np.ndarray] = None,
) -> tuple[
    np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Sparse-aware Frisch-Newton interior point solver for quantile regression.

    Solves the linear program:
        min_x  c^T x
        s.t.   A x = b
               0 <= x <= u

    The quantile regression problem is formulated as an LP where:
        - A = X.T (transposed design matrix), shape (k, N)
        - b = (1 - q) * X.T @ 1
        - c = -Y (negative response)
        - u = 1 (box constraints)

    The dual variable y gives the regression coefficients (returned as -y).

    Parameters
    ----------
    A : array or sparse matrix
        Constraint matrix, shape (k, N). Can be dense numpy array or
        scipy sparse matrix (CSR format recommended).
    b : np.ndarray
        Right-hand side of equality constraints, shape (k,)
    c : np.ndarray
        Objective coefficients, shape (N,)
    u : np.ndarray
        Upper bounds on x, shape (N,)
    q : float
        Quantile level in (0, 1)
    tol : float
        Convergence tolerance on duality gap
    max_iter : int
        Maximum number of iterations
    backoff : float, optional
        Step length safety factor, default 0.9995
    beta_init : np.ndarray, optional
        Initial coefficient estimate (not currently used)

    Returns
    -------
    tuple
        - beta_hat : np.ndarray - Estimated coefficients, shape (k,)
        - has_converged : bool - Whether algorithm converged
        - iterations : int - Number of iterations performed
        - x_final : np.ndarray - Final primal variable
        - s_final : np.ndarray - Final slack variable
        - z_final : np.ndarray - Final dual variable (lower bound)
        - w_final : np.ndarray - Final dual variable (upper bound)
        - y_final : np.ndarray - Final dual variable (equality)

    Notes
    -----
    The algorithm uses a predictor-corrector scheme:
    1. Compute affine-scaling direction (predictor)
    2. Compute centering-corrector direction
    3. Combine directions and take step

    For sparse matrices, the key optimizations are:
    - Sparse matrix-vector products for A @ x and A.T @ y
    - Efficient normal matrix formation M = A @ diag(Qinv) @ A.T
    - Cholesky factorization reuse for predictor and corrector solves
    """
    m, n = A.shape
    c = np.asarray(c).ravel()
    b = np.asarray(b).ravel()
    u = np.asarray(u).ravel()

    # Detect sparsity and prepare matrix operations
    is_sparse = issparse(A)

    if is_sparse:
        # Ensure CSR format for efficient row operations (A @ x)
        A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
        # For A.T @ y, we compute as (A.T).tocsr() @ y
        A_T_csr = A_csr.T.tocsr()

        def matvec_A(x):
            """Compute A @ x for sparse A."""
            return np.asarray(A_csr @ x).ravel()

        def matvec_AT(y):
            """Compute A.T @ y for sparse A."""
            return np.asarray(A_T_csr @ y).ravel()

        def form_M(Qinv):
            """Form normal matrix M = A @ diag(Qinv) @ A.T."""
            return _form_normal_matrix_sparse(A_csr, Qinv)

    else:
        A = np.asarray(A)

        def matvec_A(x):
            """Compute A @ x for dense A."""
            return A @ x

        def matvec_AT(y):
            """Compute A.T @ y for dense A."""
            return A.T @ y

        def form_M(Qinv):
            """Form normal matrix M = A @ diag(Qinv) @ A.T."""
            return _form_normal_matrix_dense(A, Qinv)

    # Cold start initialization
    x, s, z, w, y = cold_start(
        A_csr if is_sparse else A,
        c,
        q,
        is_sparse,
    )

    # Pre-allocate work arrays (all dense vectors of size n or m)
    r1_tilde = np.empty(n)
    r2_tilde = np.empty(m)
    Qinv = np.empty(n)
    work_m = np.empty(m)

    # Predictor direction vectors
    dx_aff = np.empty(n)
    ds_aff = np.empty(n)
    dz_aff = np.empty(n)
    dw_aff = np.empty(n)
    dy_aff = np.empty(m)

    # Corrector direction vectors
    dx_cor = np.empty(n)
    ds_cor = np.empty(n)
    dz_cor = np.empty(n)
    dw_cor = np.empty(n)
    dy_cor = np.empty(m)

    # Predicted values
    x_pred = np.empty(n)
    s_pred = np.empty(n)
    z_pred = np.empty(n)
    w_pred = np.empty(n)
    y_pred = np.empty(m)

    # Validate starting values
    for name, val in [("x", x), ("z", z), ("s", s), ("w", w)]:
        if np.any(val <= 0):
            raise ValueError(f"Initial {name} has non-positive entries.")

    mu_curr = _duality_gap(x, z, s, w)
    has_converged = False
    _it = 0

    # Main iteration loop
    for _it in range(max_iter):
        # Check convergence
        if mu_curr < tol:
            has_converged = True
            break

        # Compute residuals (equation 7 in Koenker & Ng)
        r1_tilde[:] = c - matvec_AT(y)
        r2_tilde[:] = b - matvec_A(x)

        # Compute diagonal scaling Q^{-1} = (Z/X + W/S)^{-1}
        Qinv[:] = 1.0 / (z / x + w / s)

        # Form normal matrix M = A @ diag(Qinv) @ A.T
        # This is the key computation that benefits from sparsity
        M = form_M(Qinv)

        # Cholesky factorization of M (reused for predictor and corrector)
        try:
            lu_M = cho_factor(M, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # Fall back to regularized solve if Cholesky fails
            M += 1e-10 * np.eye(m)
            lu_M = cho_factor(M, lower=True, check_finite=False)

        # =====================================================================
        # Affine-Scaling Predictor Direction (equation 8)
        # =====================================================================
        work_m[:] = r2_tilde + matvec_A(Qinv * r1_tilde)
        dy_aff[:] = cho_solve(lu_M, work_m, check_finite=False)

        dx_aff[:] = Qinv * (matvec_AT(dy_aff) - r1_tilde)
        ds_aff[:] = -dx_aff
        dz_aff[:] = -z - (z / x) * dx_aff
        dw_aff[:] = -w - (w / s) * ds_aff

        # Step lengths for affine direction (equation 9)
        alpha_p_aff = _step_length((x, dx_aff), (s, ds_aff), backoff)
        alpha_d_aff = _step_length((z, dz_aff), (w, dw_aff), backoff)

        # Predictor step
        x_pred[:] = x + alpha_p_aff * dx_aff
        s_pred[:] = s + alpha_p_aff * ds_aff
        y_pred[:] = y + alpha_d_aff * dy_aff
        z_pred[:] = z + alpha_d_aff * dz_aff
        w_pred[:] = w + alpha_d_aff * dw_aff

        # Compute centering parameter (equation 10)
        mu_aff = _duality_gap(x_pred, z_pred, s_pred, w_pred)
        sigma = (mu_aff / mu_curr) ** 2
        mu_targ = sigma * mu_curr / n

        # =====================================================================
        # Centering-Corrector Direction
        # =====================================================================
        r1_hat = (
            mu_targ * (1 / s - 1 / x)
            + (dx_aff * dz_aff) / x
            - (ds_aff * dw_aff) / s
        )

        work_m[:] = matvec_A(Qinv * r1_hat)
        # Reuse Cholesky factorization from predictor step
        dy_cor[:] = cho_solve(lu_M, work_m, check_finite=False)

        dx_cor[:] = Qinv * (matvec_AT(dy_cor) - r1_hat)
        ds_cor[:] = -dx_cor
        dz_cor[:] = -(z / x) * dx_cor + (mu_targ - dx_aff * dz_aff) / x
        dw_cor[:] = -(w / s) * ds_cor + (mu_targ - ds_aff * dw_aff) / s

        # Final step lengths (equation 12)
        alpha_p_cor = _step_length((x_pred, dx_cor), (s_pred, ds_cor), backoff)
        alpha_d_cor = _step_length((z_pred, dz_cor), (w_pred, dw_cor), backoff)

        # =====================================================================
        # Update all variables
        # =====================================================================
        x[:] = x_pred + alpha_p_cor * dx_cor
        s[:] = s_pred + alpha_p_cor * ds_cor
        y[:] = y_pred + alpha_d_cor * dy_cor
        z[:] = z_pred + alpha_d_cor * dz_cor
        w[:] = w_pred + alpha_d_cor * dw_cor

        # Update duality gap
        mu_curr = _duality_gap(x, z, s, w)

    # Return -y as the coefficient estimates (dual formulation)
    return -y, has_converged, _it, x, s, z, w, y


# =============================================================================
# Convenience wrapper matching original API
# =============================================================================


def frisch_newton_solver(
    A: Union[np.ndarray, csr_matrix],
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
) -> tuple[
    np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Wrapper for backward compatibility with original API.

    The chol and P parameters are ignored in this implementation as we
    recompute the factorization each iteration (required because the
    diagonal scaling Qinv changes).

    See frisch_newton_solver_sparse for full documentation.
    """
    return frisch_newton_solver_sparse(
        A=A,
        b=b,
        c=c,
        u=u,
        q=q,
        tol=tol,
        max_iter=max_iter,
        backoff=backoff,
        beta_init=beta_init,
    )
