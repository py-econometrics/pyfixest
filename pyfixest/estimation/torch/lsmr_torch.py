"""
Pure PyTorch implementation of the LSMR algorithm.

Ported from SciPy's `scipy.sparse.linalg.lsmr` (Fong & Saunders, 2011).
All vector operations use torch tensors (staying on-device for GPU),
while scalar Givens rotations use Python `math` to avoid autograd overhead.

Reference:
    D. C.-L. Fong and M. A. Saunders,
    "LSMR: An iterative algorithm for sparse least-squares problems",
    SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
"""

from __future__ import annotations

import math

import torch

from pyfixest.estimation.torch.lsmr_torch_compiled import lsmr_torch as _lsmr_compiled


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """
    Stable Givens rotation (SymOrtho).

    Given scalars a and b, compute c, s, r such that:
        [ c  s ] [ a ] = [ r ]
        [-s  c ] [ b ]   [ 0 ]

    This is the same algorithm as SciPy's `_sym_ortho` from LSQR,
    using pure Python math for speed on scalar values.
    """
    if b == 0.0:
        # math.copysign(1, 0) = 1.0 but np.sign(0) = 0.0; match SciPy's behavior
        c = 0.0 if a == 0.0 else math.copysign(1.0, a)
        return c, 0.0, abs(a)
    elif a == 0.0:
        return 0.0, math.copysign(1.0, b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = math.copysign(1.0, b) / math.sqrt(1.0 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = math.copysign(1.0, a) / math.sqrt(1.0 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


def _matvec(A, v: torch.Tensor) -> torch.Tensor:
    """Compute A @ v for both torch.Tensor (dense/sparse) and duck-typed wrappers."""
    if isinstance(A, torch.Tensor):
        return A @ v
    return A.mv(v)


def _rmatvec(A, u: torch.Tensor) -> torch.Tensor:
    """A^T @ u — works for both torch.Tensor (dense/sparse) and duck-typed wrappers."""
    if isinstance(A, torch.Tensor):
        return A.t() @ u
    return A.t().mv(u)


def _lsmr_scalar(
    A,
    b: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
) -> tuple[torch.Tensor, int, int, float, float, float, float, float]:
    """
    LSMR iterative solver for sparse least-squares problems, in pure PyTorch.

    Solves ``min ||b - Ax||_2`` (or the damped variant) where A is a sparse
    CSR (COO) tensor and b is a dense vector. All vector ops stay on the tensor's
    device (CPU/CUDA/MPS).

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

    Returns
    -------
    x : torch.Tensor
        Solution vector of shape (n,).
    istop : int
        Reason for stopping (0-7, same codes as SciPy LSMR).
    itn : int
        Number of iterations used.
    normr : float
        ``||b - Ax||``
    normar : float
        ``||A^T(b - Ax)||``
    normA : float
        Estimate of Frobenius norm of A.
    condA : float
        Estimate of condition number of A.
    normx : float
        ``||x||``
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # --- Initialize Golub-Kahan bidiagonalization ---
    u = b.clone()
    normb = torch.linalg.norm(b).item()

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb

    if beta > 0:
        u = u * (1.0 / beta)
        v = _rmatvec(A, u)
        alpha = torch.linalg.norm(v).item()
    else:
        v = torch.zeros(n, device=device, dtype=dtype)
        alpha = 0.0

    if alpha > 0:
        v = v * (1.0 / alpha)

    # --- Scalar state for iteration ---
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    # Estimation of ||r||
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    # Estimation of ||A|| and cond(A)
    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = math.sqrt(normA2)
    condA = 1.0
    normx = 0.0

    # Stopping
    istop = 0
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normr = beta
    normar = alpha * beta

    if normar == 0.0:
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0.0:
        x.zero_()
        return x, istop, itn, normr, normar, normA, condA, normx

    # --- Main iteration loop ---
    while itn < maxiter:
        itn += 1

        # Bidiagonalization step: get next beta, u, alpha, v
        u = _matvec(A, v) - alpha * u
        beta = torch.linalg.norm(u).item()

        if beta > 0:
            u *= 1.0 / beta
            v = _rmatvec(A, u) - beta * v
            alpha = torch.linalg.norm(v).item()
            if alpha > 0:
                v *= 1.0 / alpha

        # Construct rotation Qhat_{k,2k+1}
        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use plane rotation Q_i to turn B_i to R_i
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use plane rotation Qbar_i to turn R_i^T to R_i^bar
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, hbar, x  (vector ops — stay on device)
        hbar = h + hbar * (-(thetabar * rho) / (rhoold * rhobarold))
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + h * (-(thetanew / rho))

        # Estimate ||r||
        betaacute = chat * betadd
        betacheck = -shat * betadd

        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||
        normA2 = normA2 + beta * beta
        normA = math.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A)
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Convergence tests
        normar = abs(zetabar)
        normx = torch.linalg.norm(x).item()

        test1 = normr / normb
        test2 = normar / (normA * normr) if normA * normr != 0 else float("inf")
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        if itn >= maxiter:
            istop = 7
        if 1.0 + test3 <= 1.0:
            istop = 6
        if 1.0 + test2 <= 1.0:
            istop = 5
        if 1.0 + t1 <= 1.0:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break

    return x, istop, itn, normr, normar, normA, condA, normx


def lsmr_torch(
    A,
    b: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
    use_compile: bool | None = None,
) -> tuple[torch.Tensor, int, int, float, float, float, float, float]:
    """
    LSMR solver — unified entry point.

    Auto-selects implementation based on device:
    - CUDA: compiled (torch.compile fuses scalar step into 1 kernel)
    - CPU/MPS: scalar (Python-float math, no compilation overhead)

    Pass use_compile=True to force compilation on any device.
    """
    device = b.device
    if use_compile is None:
        use_compile = device.type == "cuda"

    if use_compile:
        return _lsmr_compiled(
            A,
            b,
            damp=damp,
            atol=atol,
            btol=btol,
            conlim=conlim,
            maxiter=maxiter,
            use_compile=True,
        )
    return _lsmr_scalar(
        A,
        b,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
    )
