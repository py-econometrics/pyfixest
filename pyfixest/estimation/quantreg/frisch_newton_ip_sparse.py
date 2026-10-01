import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _duality_gap(x, z, s, w):
    return x @ z + s @ w


def _bound(v: np.ndarray, dv: np.ndarray, backoff: float):
    mask = dv < 0
    if not mask.any():
        return 1.0
    alpha_max = (-v[mask] / dv[mask]).min()
    return min(backoff * alpha_max, 1.0)


def _step_length(a: tuple, b: tuple, backoff: float) -> float:
    x, dx = a
    s, ds = b
    return min(_bound(x, dx, backoff), _bound(s, ds, backoff))


def _normal_matrix(A: sp.csr_matrix, D: np.ndarray) -> sp.csc_matrix:
    """Form A diag(D) A^T without densifying A."""
    return (A.multiply(D)) @ A.T


def _solve_spd(M: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve M y = rhs for a sparse symmetric positive definite M.

    ``splu`` is an LU rather than a Cholesky, which costs roughly a factor of two over a
    dedicated sparse Cholesky, but it is in scipy and so adds no dependency. Sparsity is
    preserved through COLAMD ordering.
    """
    return splu(sp.csc_matrix(M)).solve(rhs)


def cold_start_sparse(
    A: sp.csr_matrix, c: np.ndarray, q: float
) -> tuple[np.ndarray, ...]:
    """Initiate the sparse Frisch-Newton solver with a cold start.

    Mirrors ``cold_start`` in ``frisch_newton_ip.py``, but the initial dual solve uses a
    sparse factorisation of A A^T instead of the pre-computed dense Cholesky factors.
    """
    n = A.shape[1]
    x = np.full(n, 1.0 - q)

    s = np.full_like(x, q)
    d_plus = np.maximum(c, 0.0)
    d_minus = np.maximum(-c, 0.0)
    U = x @ d_plus + s @ d_minus
    mu0 = max(1, U / n)
    alpha = (n * mu0 - U) / (np.sum(1 / x) + np.sum(1 / s))
    eps = 1e-8

    z = np.maximum(d_plus, eps) + alpha / x
    w = np.maximum(d_minus, eps) + alpha / s

    rhs = A @ (c - z + w)
    y = _solve_spd(_normal_matrix(A, np.ones(n)), rhs)
    return x, s, z, w, y


def frisch_newton_solver_sparse(
    A: sp.spmatrix,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
    backoff: float = 0.9995,
    beta_init: np.ndarray | None = None,
) -> tuple[
    np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Solve
        min_x  c^T x
        s.t.   A x = b,
               0 <= x <= u
    via the Frisch Newton interior point solver of Koenker and Ng ("A Frisch Newton
    Algorithm for Sparse Quantile Regression"), keeping the design sparse throughout.

    This follows the same predictor-corrector steps as ``frisch_newton_solver`` and returns
    the same tuple. The difference is that A stays a scipy sparse matrix, so the normal
    matrix A diag(Q^-1) A^T is formed and factorised sparsely. With categorical regressors
    the design is mostly zeros, and the dense version spends its time multiplying by them.
    """
    A = sp.csr_matrix(A)
    _m, n = A.shape
    c = np.asarray(c).ravel()
    b = np.asarray(b).ravel()
    u = np.asarray(u).ravel()

    x, s, z, w, y = cold_start_sparse(A=A, c=c, q=q)

    AT = sp.csr_matrix(A.T)

    for val in (x, z, s, w):
        if np.any(val < 0):
            raise ValueError(
                f"Initial value {val} has negative entries, which is not allowed."
            )

    mu_curr = _duality_gap(x=x, z=z, s=s, w=w)
    has_converged = False
    _it = 0

    for _it in range(max_iter):
        if mu_curr < tol:
            has_converged = True
            break

        # Residuals: equ. (7)
        r1_tilde = c - AT @ y
        r2_tilde = b - A @ x

        # Affine-scaling predictor direction (eq. (8))
        Qinv = 1.0 / (z / x + w / s)
        M = _normal_matrix(A, Qinv)
        lu = splu(sp.csc_matrix(M))

        work_m = r2_tilde + A @ (Qinv * r1_tilde)
        dy_aff = lu.solve(work_m)

        dx_aff = Qinv * (AT @ dy_aff - r1_tilde)
        ds_aff = -dx_aff
        dz_aff = -z - (z / x) * dx_aff
        dw_aff = -w - (w / s) * ds_aff

        # Step lengths (eq. (9))
        alpha_p_aff = _step_length(a=(x, dx_aff), b=(s, ds_aff), backoff=backoff)
        alpha_d_aff = _step_length(a=(z, dz_aff), b=(w, dw_aff), backoff=backoff)

        # Centering parameter (eq. (10))
        x_pred = x + alpha_p_aff * dx_aff
        s_pred = s + alpha_p_aff * ds_aff
        y_pred = y + alpha_d_aff * dy_aff
        z_pred = z + alpha_d_aff * dz_aff
        w_pred = w + alpha_d_aff * dw_aff

        mu_aff = _duality_gap(x=x_pred, z=z_pred, s=s_pred, w=w_pred)
        sigma = (mu_aff / mu_curr) ** 2
        mu_targ = sigma * mu_curr / n

        # Corrector direction (reuses the factorisation of M)
        r1_hat = (
            mu_targ * (1 / s - 1 / x) + (dx_aff * dz_aff) / x - (ds_aff * dw_aff) / s
        )
        work_m = A @ (Qinv * r1_hat)
        dy_cor = lu.solve(work_m)
        dx_cor = Qinv * (AT @ dy_cor - r1_hat)
        ds_cor = -dx_cor
        dz_cor = -(z / x) * dx_cor + (mu_targ - dx_aff * dz_aff) / x
        dw_cor = -(w / s) * ds_cor + (mu_targ - ds_aff * dw_aff) / s

        # Final step lengths (eq. (12))
        alpha_p_cor = _step_length(
            a=(x_pred, dx_cor), b=(s_pred, ds_cor), backoff=backoff
        )
        alpha_d_cor = _step_length(
            a=(z_pred, dz_cor), b=(w_pred, dw_cor), backoff=backoff
        )

        x = x_pred + alpha_p_cor * dx_cor
        s = s_pred + alpha_p_cor * ds_cor
        y = y_pred + alpha_d_cor * dy_cor
        z = z_pred + alpha_d_cor * dz_cor
        w = w_pred + alpha_d_cor * dw_cor

        mu_curr = _duality_gap(x=x, z=z, s=s, w=w)

    return -y, has_converged, _it, x, s, z, w, y
