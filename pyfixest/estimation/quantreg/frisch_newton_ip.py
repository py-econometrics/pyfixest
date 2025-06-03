from typing import Optional

import numpy as np
from scipy.linalg import lapack, solve_triangular


# @nb.njit
def _duality_gap(x, z, s, w):
    return x @ z + s @ w


# @nb.njit
def _bound(v: np.ndarray, dv: np.ndarray, backoff: float):
    mask = dv < 0
    if not mask.any():
        return 1.0
    alpha_max = (-v[mask] / dv[mask]).min()
    return min(backoff * alpha_max, 1.0)


# @nb.njit
def _step_length(a: tuple, b: tuple, backoff: float) -> float:
    x, dx = a
    s, ds = b
    return min(_bound(x, dx, backoff), _bound(s, ds, backoff))


def _solve_ADAt(
    rhs: np.ndarray, D: np.ndarray, chol: np.ndarray, P: np.ndarray
) -> np.ndarray:
    u = solve_triangular(chol, rhs, lower=True)

    sqrtD = np.sqrt(D)
    W = P * sqrtD[np.newaxis, :]
    K = W @ W.T

    u_buf = u.copy()
    lapack.dposv(K, u_buf, lower=1, overwrite_a=True, overwrite_b=True)
    y = solve_triangular(chol.T, u_buf, lower=False, check_finite=False)

    # S = np.linalg.cholesky(K)
    # z = solve_triangular(S, u, lower=True)
    # z = solve_triangular(S.T, z, lower=False)
    # y = solve_triangular(chol.T, z, lower=False)
    return y


# @nb.njit
def cold_start(
    A: np.ndarray, c: np.ndarray, q: float, chol: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, ...]:
    "Initiatiate Frisch-Newton solver with a cold start."
    n = A.shape[1]
    x = np.full(n, 1.0 - q)

    # iniate all other variables
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
    y = _solve_ADAt(rhs, D=np.ones(n), chol=chol, P=P)
    # y = np.linalg.solve(A @ A.T, rhs)
    return x, s, z, w, y


# @nb.njit
def warm_start(A, c, beta, q, chol, P, eps=1e-8):
    "Initiate Frisch-Newton solver with a warm start."
    n = A.shape[1]
    x = np.full(n, 1.0 - q) + eps
    s = np.full(n, q) + eps

    r = -(A.T @ beta)
    z = np.maximum(r, 0.0) + eps
    w = z - r

    rhs = A @ (c - z + w)
    y = _solve_ADAt(rhs, D=np.ones(n), chol=chol, P=P)
    # y = np.linalg.solve(A @ A.T, rhs)

    return x, s, z, w, y


# @nb.njit
def frisch_newton_solver(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
    chol: np.ndarray,
    P: np.ndarray,
    backoff: float = 0.9995,
    beta_init: Optional[np.ndarray] = None,
) -> tuple[
    np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Solve
        min_x  c^T x
        s.t.   A x = b,
               0 <= x <= u
    via the Frisch Newton interior point solver as described in
    Koenker and Ng ("A FRISCH NEWTON ALGORITHM FOR SPARSE QUANTILE
    REGRESSION").
    """
    # 1) Basic shapes
    m, n = A.shape
    c = c.ravel()
    b = b.ravel()
    u = u.ravel()

    # 2) Initialize variables
    # ---------- 2. persistent work-vectors ----------

    x, s, z, w, y = (
        cold_start(A=A, c=c, q=q, chol=chol, P=P)
        if beta_init is None
        else warm_start(A=A, c=c, beta=beta_init, q=q, chol=chol, P=P)
    )

    # reusable scratch pads (allocated once)
    r1_tilde = np.empty((n,))
    r2_tilde = np.empty((m,))
    r1_hat = np.empty_like(c)

    Qinv = np.empty_like(x)
    # rhs = np.empty_like(b)
    # work_n = np.empty_like(x)
    work_m = np.empty_like(b)

    M = np.empty((m, m))

    # predictor-/corrector direction vectors (allocated once, reused)
    dx_aff = np.empty_like(x)
    ds_aff = np.empty_like(s)
    dz_aff = np.empty_like(z)
    dw_aff = np.empty_like(w)
    dy_aff = np.empty_like(b)

    x_pred = np.empty_like(x)
    s_pred = np.empty_like(s)
    z_pred = np.empty_like(z)
    w_pred = np.empty_like(w)
    y_pred = np.empty_like(b)

    dx_cor = np.empty_like(x)
    ds_cor = np.empty_like(s)
    dz_cor = np.empty_like(z)
    dw_cor = np.empty_like(w)
    dy_cor = np.empty_like(b)

    x_pred = np.empty_like(x)
    s_pred = np.empty_like(s)
    z_pred = np.empty_like(z)
    w_pred = np.empty_like(w)
    y_pred = np.empty_like(y)

    # 6) Quick sanity checks on starting values
    if True:
        for val in [x, z, s, w]:
            if np.any(val < 0):
                raise ValueError(
                    f"Initial value {val} has negative entries, which is not allowed."
                )

    mu_curr = _duality_gap(x=x, z=z, s=s, w=w)

    has_converged = False

    for _it in range(max_iter):
        if mu_curr < tol:
            has_converged = True
            break

        # Residuals: equ. (7)
        r1_tilde[:] = c - A.T @ y
        r2_tilde[:] = b - A @ x

        # Affine-Scaling Predictor Direction (eq. (8))
        Qinv[:] = 1.0 / (z / x + w / s)  # diag
        M[:] = A @ (Qinv[:, None] * A.T)
        work_m[:] = r2_tilde + A @ (Qinv * r1_tilde)
        dy_aff[:] = np.linalg.solve(M, work_m)
        # dy_aff[:] = _solve_ADAt(rhs=work_m, D=Qinv, chol=chol, P=P)

        dx_aff[:] = Qinv * (A.T @ dy_aff - r1_tilde)
        ds_aff[:] = -dx_aff
        dz_aff[:] = -z - (z / x) * dx_aff
        dw_aff[:] = -w - (w / s) * ds_aff

        # Step lengths (eq. (9))
        alpha_p_aff = _step_length(a=(x, dx_aff), b=(s, ds_aff), backoff=backoff)
        alpha_d_aff = _step_length(a=(z, dz_aff), b=(w, dw_aff), backoff=backoff)

        # 6) Compute mu_new  and centering sigma  (eq (10))
        x_pred[:] = x + alpha_p_aff * dx_aff
        s_pred[:] = s + alpha_p_aff * ds_aff
        y_pred[:] = y + alpha_d_aff * dy_aff
        z_pred[:] = z + alpha_d_aff * dz_aff
        w_pred[:] = w + alpha_d_aff * dw_aff

        mu_aff = _duality_gap(x=x_pred, z=z_pred, s=s_pred, w=w_pred)

        ratio = mu_aff / mu_curr
        sigma = ratio**2
        mu_targ = sigma * mu_curr / n

        # corrector direction
        r1_hat = (
            mu_targ * (1 / s - 1 / x) + (dx_aff * dz_aff) / x - (ds_aff * dw_aff) / s
        )

        work_m[:] = A @ (Qinv * r1_hat)
        # dy_cor[:] = _solve_ADAt(rhs = work_m, D = Qinv, chol=chol, P = P)
        dy_cor[:] = np.linalg.solve(M, work_m)
        dx_cor[:] = Qinv * (A.T @ dy_cor - r1_hat)
        ds_cor[:] = -dx_cor
        dz_cor[:] = -(z / x) * dx_cor + (mu_targ - dx_aff * dz_aff) / x
        dw_cor[:] = -(w / s) * ds_cor + (mu_targ - ds_aff * dw_aff) / s

        # 9) Final step lengths (corrector) — eq (12)
        alpha_p_cor = _step_length(
            a=(x_pred, dx_cor), b=(s_pred, ds_cor), backoff=backoff
        )
        alpha_d_cor = _step_length(
            a=(z_pred, dz_cor), b=(w_pred, dw_cor), backoff=backoff
        )

        # 10) Update all variables / corrector step
        # Update
        # corrector (starting from the predictor point)
        x[:] = x_pred + alpha_p_cor * dx_cor
        s[:] = s_pred + alpha_p_cor * ds_cor
        y[:] = y_pred + alpha_d_cor * dy_cor
        z[:] = z_pred + alpha_d_cor * dz_cor
        w[:] = w_pred + alpha_d_cor * dw_cor

        # update
        mu_curr = _duality_gap(x=x, z=z, s=s, w=w)

    return -y, has_converged, _it, x, s, z, w, y
