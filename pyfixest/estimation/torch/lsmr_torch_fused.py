"""
On-device LSMR: all scalar state as 0-d tensors to eliminate CPU-GPU sync.

Compared to lsmr_torch.py:
- Branchless Givens via torch.hypot (no if/elif, no Python math)
- All scalar state as 0-d device tensors (no .item() in hot loop)
- Convergence check via logical indexing
- Single sync point every `check_every` iterations

Designed for CUDA/MPS where CPU-GPU synchronization is expensive.

Reference:
    D. C.-L. Fong and M. A. Saunders,
    "LSMR: An iterative algorithm for sparse least-squares problems",
    SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Branchless Givens rotation
# ---------------------------------------------------------------------------


def _sym_ortho_t(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Branchless Givens rotation for 0-d tensors on device.

    Equivalent to SciPy's ``_sym_ortho`` but implemented with
    ``torch.hypot`` (overflow-safe) and ``torch.where`` (branchless).
    No CPU-GPU synchronization occurs.
    """
    r = torch.hypot(a, b)
    # Guard division: when r == 0, return (0, 0, 0)
    safe_r = torch.where(r == 0, torch.ones_like(r), r)
    c = torch.where(r == 0, torch.zeros_like(a), a / safe_r)
    s = torch.where(r == 0, torch.zeros_like(b), b / safe_r)
    return c, s, r


# ---------------------------------------------------------------------------
# Sparse matvec helpers
# ---------------------------------------------------------------------------


def _matvec(A, v: torch.Tensor) -> torch.Tensor:
    if isinstance(A, torch.Tensor):
        return A @ v
    return A.mv(v)


def _rmatvec(A, u: torch.Tensor) -> torch.Tensor:
    if isinstance(A, torch.Tensor):
        return A.t() @ u
    return A.t().mv(u)


# ---------------------------------------------------------------------------
# On-device LSMR
# ---------------------------------------------------------------------------


def lsmr_torch_fused(
    A,
    b: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
    check_every: int = 10,
) -> tuple[torch.Tensor, int, int, float, float, float, float, float]:
    """
    LSMR iterative solver with minimal CPU-GPU synchronization.

    Same algorithm as :func:`lsmr_torch`, but keeps **all** scalar state
    as 0-d tensors on the compute device.  CPU-GPU sync only happens once
    every *check_every* iterations (for the convergence check), reducing
    pipeline stalls from 3/iteration to 1/N.

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR tensor of shape (m, n).
    b : torch.Tensor
        Dense vector of shape (m,).
    damp : float
        Damping factor for regularized least-squares.
    atol, btol : float
        Stopping tolerances (see SciPy LSMR docs).
    conlim : float
        Condition number limit.
    maxiter : int or None
        Maximum iterations. Defaults to min(m, n).
    check_every : int
        How often to sync to CPU for convergence check (default: 10).
        Higher values reduce syncs but may overshoot convergence by up to
        ``check_every - 1`` iterations.

    Returns
    -------
    x, istop, itn, normr, normar, normA, condA, normx
        Same signature as :func:`lsmr_torch`.
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # Scalar constant factory
    def _s(val: float) -> torch.Tensor:
        return torch.tensor(val, device=device, dtype=dtype)

    _TINY = _s(1e-30)

    # --- Initialize Golub-Kahan bidiagonalization ---
    u = b.clone()
    normb = torch.linalg.norm(b)

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb.clone()

    # Branchless safe-normalize: u /= beta  (or zero if beta == 0)
    u = u * torch.where(beta > 0, 1.0 / torch.clamp(beta, min=1e-30), _s(0.0))

    v = _rmatvec(A, u)
    alpha = torch.linalg.norm(v)
    v = v * torch.where(alpha > 0, 1.0 / torch.clamp(alpha, min=1e-30), _s(0.0))

    # --- Scalar state (all 0-d device tensors) ---
    zetabar = alpha * beta
    alphabar = alpha.clone()
    rho = _s(1.0)
    rhobar = _s(1.0)
    cbar = _s(1.0)
    sbar = _s(0.0)

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    # ||r|| estimation
    betadd = beta.clone()
    betad = _s(0.0)
    rhodold = _s(1.0)
    tautildeold = _s(0.0)
    thetatilde = _s(0.0)
    zeta = _s(0.0)
    d = _s(0.0)

    # ||A|| and cond(A) estimation
    normA2 = alpha * alpha
    maxrbar = _s(0.0)
    minrbar = _s(1e100)
    normA = torch.sqrt(normA2)
    condA = _s(1.0)
    normx_est = _s(0.0)

    # Stopping
    ctol = _s(1.0 / conlim if conlim > 0 else 0.0)
    normr = beta.clone()
    normar = alpha * beta

    # Pre-create tolerance tensors
    atol_t = _s(atol)
    btol_t = _s(btol)
    damp_t = _s(damp)

    # Early exit (syncs once at init — unavoidable)
    if normar.item() == 0.0:
        return x, 0, 0, normr.item(), normar.item(), normA.item(), condA.item(), 0.0
    if normb.item() == 0.0:
        x.zero_()
        return x, 0, 0, normr.item(), normar.item(), normA.item(), condA.item(), 0.0

    # --- Main iteration loop (zero syncs inside, except periodic check) ---
    itn = 0
    istop = 0

    while itn < maxiter:
        itn += 1

        # Bidiagonalization step
        u = _matvec(A, v) - alpha * u
        beta = torch.linalg.norm(u)
        u = u * torch.where(beta > 0, 1.0 / torch.clamp(beta, min=1e-30), _s(0.0))

        v = _rmatvec(A, u) - beta * v
        alpha = torch.linalg.norm(v)
        v = v * torch.where(alpha > 0, 1.0 / torch.clamp(alpha, min=1e-30), _s(0.0))

        # Givens rotations (branchless, on device)
        chat, shat, alphahat = _sym_ortho_t(alphabar, damp_t)

        rhoold = rho
        c, s, rho = _sym_ortho_t(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho_t(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Vector updates (on device)
        hbar = h + hbar * (-(thetabar * rho) / (rhoold * rhobarold))
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + h * (-(thetanew / rho))

        # ||r|| estimation
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho_t(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = torch.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # ||A|| estimation
        normA2 = normA2 + beta * beta
        normA = torch.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # cond(A) estimation
        maxrbar = torch.maximum(maxrbar, rhobarold)
        if itn > 1:
            minrbar = torch.minimum(minrbar, rhobarold)
        condA = torch.maximum(maxrbar, rhotemp) / torch.minimum(minrbar, rhotemp)

        # Convergence check (all logical ops on device, no sync)
        normar = torch.abs(zetabar)
        normx_est = torch.linalg.norm(x)

        test1 = normr / normb
        test2 = normar / torch.clamp(normA * normr, min=1e-30)
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx_est / normb)
        rtol = btol_t + atol_t * normA * normx_est / normb

        converged = (
            (test1 <= rtol)
            | (test2 <= atol_t)
            | (test3 <= ctol)
            | (1.0 + t1 <= 1.0)
            | (1.0 + test2 <= 1.0)
            | (1.0 + test3 <= 1.0)
        )

        # --- Periodic sync: single .item() every check_every iterations ---
        if itn % check_every == 0 or itn >= maxiter:
            if converged.item():
                # Determine exact istop code (one-time sync at exit)
                _test1 = test1.item()
                _test2 = test2.item()
                _test3 = test3.item()
                _t1 = t1.item()
                _rtol = rtol.item()
                _ctol = ctol.item()

                if 1.0 + _test3 <= 1.0:
                    istop = 6
                elif 1.0 + _test2 <= 1.0:
                    istop = 5
                elif 1.0 + _t1 <= 1.0:
                    istop = 4
                elif _test3 <= _ctol:
                    istop = 3
                elif _test2 <= atol:
                    istop = 2
                elif _test1 <= _rtol:
                    istop = 1
                break

    if itn >= maxiter and istop == 0:
        istop = 7

    return (
        x,
        istop,
        itn,
        normr.item(),
        normar.item(),
        normA.item(),
        condA.item(),
        normx_est.item(),
    )
