import numpy as np
from numpy.linalg import solve, lstsq

def bound(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    Compute the maximum step-length factor for moving x by dx without crossing zero.
    """
    b = np.full_like(x, 1e20)
    mask = dx < 0
    b[mask] = -x[mask] / dx[mask]
    return b


def lpfnc(A1: np.ndarray,
          c1: np.ndarray,
          A2: np.ndarray,
          c2: np.ndarray,
          b: np.ndarray,
          u: np.ndarray,
          x1: np.ndarray,
          x2: np.ndarray,
          beta: float = 0.9995,
          small: float = 1e-8,
          maxit: int = 50) -> tuple[np.ndarray, int]:
    """
    Pure-Python interior-point solver for the dual quantile LP:
        min c1^T x1 + c2^T x2
        s.t. A1 x1 + A2 x2 = b,
             0 <= x1 <= u,  0 <= x2 (unbounded above)
    """
    # initialize
    s = u - x1
    n1 = A1.shape[1]
    n2 = A2.shape[1]
    # initial dual variables: OLS y solving A1 y = c1
    y = lstsq(A1, c1, rcond=None)[0]
    r1 = c1 - A1 @ y
    r2 = c2 - A2 @ y
    z1 = np.where(r1 > 0, r1, 0) + small
    w = z1 - r1 + small
    z2 = np.ones(n2)
    # initial duality gap
    gap = float(z1 @ x1 + z2 @ x2 + w @ s)
    it = 0

    while gap > small and it < maxit:
        it += 1
        # diagonal pivots
        q1 = 1.0 / (z1 / x1 + w / s)
        q2 = x2 / z2
        # residuals
        r1 = c1 - A1 @ y
        r2 = c2 - A2 @ y
        r3 = b - A1 @ x1 - A2 @ x2
        # form normal equations
        AQ1 = A1 * q1[None, :]
        AQ2 = A2 * q2[None, :]
        AQA = AQ1 @ A1.T + AQ2 @ A2.T
        rhs = r3 + AQ1 @ r1 + AQ2 @ r2
        # predictor step
        dy = solve(AQA, rhs)
        dx1 = q1 * (A1.T @ dy - r1)
        dx2 = q2 * (A2.T @ dy - r2)
        ds = -dx1
        dz1 = -z1 * (1 + dx1 / x1)
        dz2 = -z2 * (1 + dx2 / x2)
        dw = -w * (1 + ds / s)
        # step-lengths
        fx1 = bound(x1, dx1)
        fs_ = bound(s, ds)
        fx2 = bound(x2, dx2)
        fz1 = bound(z1, dz1)
        fz2 = bound(z2, dz2)
        fw = bound(w, dw)
        fp = min(np.min(np.minimum(fx1, fs_)), np.min(fx2))
        fd = min(np.min(np.minimum(fw, fz1)), np.min(fz2))
        fp = min(beta * fp, 1.0)
        fd = min(beta * fd, 1.0)
        # corrector if needed
        if min(fp, fd) < 1.0:
            # compute current gap
            mu = z1 @ x1 + z2 @ x2 + w @ s
            # predicted gap
            x1_aff = x1 + fp * dx1
            x2_aff = x2 + fp * dx2
            z1_aff = z1 + fd * dz1
            z2_aff = z2 + fd * dz2
            w_aff = w + fd * dw
            s_aff = s + fp * ds
            g = (z1_aff @ x1_aff) + (z2_aff @ x2_aff) + (w_aff @ s_aff)
            mu = mu * (g / mu)**3 / (2 * n1 + n2)
            # form corrector directions
            dxdz1 = dx1 * dz1
            dxdz2 = dx2 * dz2
            dsdw = ds * dw
            xinv1 = 1.0 / x1
            xinv2 = 1.0 / x2
            sinv = 1.0 / s
            xi1 = xinv1 * dxdz1 - sinv * dsdw - mu * (xinv1 - sinv)
            xi2 = xinv2 * dxdz2 - mu * xinv2
            rhs = rhs + A1 @ (q1 * xi1) + A2 @ (q2 * xi2)
            dy = solve(AQA, rhs)
            dx1 = q1 * (A1.T @ dy - xi1 - r1)
            dx2 = q2 * (A2.T @ dy - xi2 - r2)
            ds = -dx1
            dz1 = -z1 + xinv1 * (mu - z1 * dx1 - dxdz1)
            dz2 = -z2 + xinv2 * (mu - z2 * dx2 - dxdz2)
            dw = -w + sinv * (mu - w * ds - dsdw)
            # re-compute step-lengths
            fx1 = bound(x1, dx1)
            fs_ = bound(s, ds)
            fx2 = bound(x2, dx2)
            fz1 = bound(z1, dz1)
            fz2 = bound(z2, dz2)
            fw = bound(w, dw)
            fp = min(np.min(np.minimum(fx1, fs_)), np.min(fx2))
            fd = min(np.min(np.minimum(fw, fz1)), np.min(fz2))
            fp = min(beta * fp, 1.0)
            fd = min(beta * fd, 1.0)
        # update
        x1 += fp * dx1
        x2 += fp * dx2
        z1 += fd * dz1
        z2 += fd * dz2
        s += fp * ds
        y += fd * dy
        w += fd * dw
        gap = float(z1 @ x1 + z2 @ x2 + w @ s)

    return y, it


def rqc(X: np.ndarray,
        y: np.ndarray,
        R: np.ndarray,
        r: np.ndarray,
        tau: float = 0.5) -> dict:
    """
    Dual quantile regression via Frisch–Newton IPM.
    X: (n×p) design
    y: (n,) response
    R: (m×p) second constraint matrix
    r: (m,) second response
    tau: quantile fraction
    Returns:
      coef: dual solution (length m)
      it: number of IPM iterations
    """

    import pdb; pdb.set_trace()
    n, p = X.shape
    m, _ = R.shape
    # initial blocks
    u  = np.ones(p)
    a1 = (1 - tau) * np.ones(p)
    a2 = np.ones(m)
    b = X.T @ a1
    # transpose to match A1 shape = (p×n), A2 = (m×n)
    A1 = X.T
    A2 = R.T
    c1 = -y
    c2 = -r
    # solve
    coef, it = lpfnc(A1, c1, A2, c2, b, u, a1, a2)
    return {'coef': -coef, 'it': it}
