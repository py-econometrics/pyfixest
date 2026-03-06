"""
Pure PyTorch LSMR iterative solver with optional torch.compile kernel fusion.

Four implementations live in this file:

0. ``_lsmr_batched`` — eager batched LSMR for K right-hand sides via SpMM.
   Uses vectorized (K,) Givens rotations with ``_sym_ortho_vec``.

1. ``_lsmr_eager`` — eager single-RHS LSMR, Python-float Givens rotations.
   Best for CPU.

2. ``_lsmr_compiled`` — packs scalar state into a 1-D tensor and runs
   the Givens / norm / convergence work through a ``torch.compile``-d
   kernel.  On CUDA this fuses ~60 per-iteration kernel launches into one.

3. ``_lsmr_compiled_batched`` — compiled batched LSMR for K RHS via SpMM.
   Packs scalar state into a (_STATE_SIZE, K) tensor — the same compiled
   ``_scalar_step`` serves both single-RHS and batched paths since all ops
   are shape-agnostic.  Fuses scalar work into one kernel per iteration.

Public entry points:
- ``lsmr_torch()`` dispatches: CUDA → compiled, CPU/MPS → eager.
- ``lsmr_torch_batched()`` dispatches: CUDA → compiled batched,
  CPU/MPS → eager batched.  Pass ``use_compile=True/False`` to override.

Reference:
    D. C.-L. Fong and M. A. Saunders,
    "LSMR: An iterative algorithm for sparse least-squares problems",
    SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
"""

from __future__ import annotations

import math
import threading

import torch

# ---------------------------------------------------------------------------
# Sparse matvec helpers
# ---------------------------------------------------------------------------


def _matvec(A, v: torch.Tensor) -> torch.Tensor:
    """Compute A @ v for both torch.Tensor (dense/sparse) and duck-typed wrappers."""
    if isinstance(A, torch.Tensor):
        return A @ v
    return A.mv(v)


def _rmatvec(At, u: torch.Tensor) -> torch.Tensor:
    """Multiply A^T @ u using a pre-computed transpose."""
    if isinstance(At, torch.Tensor):
        return At @ u
    return At.mv(u)


def _precompute_transpose(A):
    """Pre-compute A^T in a GPU-friendly layout to avoid per-iteration reconversion."""
    if isinstance(A, torch.Tensor) and A.is_sparse_csr:
        return A.t().to_sparse_csr()
    elif isinstance(A, torch.Tensor):
        return A.t().contiguous()
    return A.t()  # LinearOperator / wrapper — assume .mv() is efficient


# ---------------------------------------------------------------------------
# Scalar Givens rotation (Python math — no autograd overhead)
# ---------------------------------------------------------------------------


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """
    Stable Givens rotation (SymOrtho).

    Given scalars a and b, compute c, s, r such that:
        [ c  s ] [ a ] = [ r ]
        [-s  c ] [ b ]   [ 0 ]

    This is the same algorithm as SciPy's ``_sym_ortho`` from LSQR,
    using pure Python math for speed on scalar values.
    """
    if b == 0.0:
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


# ---------------------------------------------------------------------------
# Batched matvec helpers (SpMM: sparse @ dense matrix)
# ---------------------------------------------------------------------------


def _matvec_batched(A, V: torch.Tensor) -> torch.Tensor:
    """A @ V where V is (n, K). SpMM for sparse A, mm() for wrappers."""
    if isinstance(A, torch.Tensor):
        return A @ V
    return A.mm(V)


def _rmatvec_batched(At, U: torch.Tensor) -> torch.Tensor:
    """A^T @ U where U is (m, K). SpMM."""
    if isinstance(At, torch.Tensor):
        return At @ U
    return At.mm(U)


# ---------------------------------------------------------------------------
# Vectorized Givens rotation for (K,) tensors
# ---------------------------------------------------------------------------


def _sym_ortho_vec(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stable Givens rotation (SymOrtho) vectorized over K columns.

    Given (K,) tensors a and b, compute c, s, r such that for each k:
        [ c_k  s_k ] [ a_k ] = [ r_k ]
        [-s_k  c_k ] [ b_k ]   [ 0   ]

    Uses torch.where for branchless execution on GPU. Division guards
    use ones_like (not clamp) to preserve sign in dead lanes.
    """
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)

    # Safe divisors: replace zeros with ones to prevent NaN in dead lanes.
    # The result of the dead-lane computation is discarded by torch.where.
    safe_a = torch.where(a != 0, a, one)
    safe_b = torch.where(b != 0, b, one)

    # Case 1: b == 0
    c_b0 = torch.where(a == 0, zero, torch.sign(a))
    s_b0 = zero
    r_b0 = abs_a

    # Case 2: a == 0
    c_a0 = zero
    s_a0 = torch.sign(b)
    r_a0 = abs_b

    # Case 3: |b| > |a| (neither zero)
    tau_3 = a / safe_b
    s_3 = torch.sign(b) / torch.sqrt(one + tau_3 * tau_3)
    safe_s_3 = torch.where(s_3 != 0, s_3, one)
    c_3 = s_3 * tau_3
    r_3 = b / safe_s_3

    # Case 4: |a| >= |b| (neither zero)
    tau_4 = b / safe_a
    c_4 = torch.sign(a) / torch.sqrt(one + tau_4 * tau_4)
    safe_c_4 = torch.where(c_4 != 0, c_4, one)
    s_4 = c_4 * tau_4
    r_4 = a / safe_c_4

    # Select: b==0 → case1, a==0 → case2, |b|>|a| → case3, else → case4
    is_b0 = b == 0
    is_a0 = a == 0
    is_b_gt_a = abs_b > abs_a

    # Build from innermost to outermost condition
    c = torch.where(is_b_gt_a, c_3, c_4)
    s = torch.where(is_b_gt_a, s_3, s_4)
    r = torch.where(is_b_gt_a, r_3, r_4)

    c = torch.where(is_a0, c_a0, c)
    s = torch.where(is_a0, s_a0, s)
    r = torch.where(is_a0, r_a0, r)

    c = torch.where(is_b0, c_b0, c)
    s = torch.where(is_b0, s_b0, s)
    r = torch.where(is_b0, r_b0, r)

    return c, s, r


# ---------------------------------------------------------------------------
# Shared batched helpers
# ---------------------------------------------------------------------------


def _safe_normalize_cols(
    M: torch.Tensor, norms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divide each column of (m, K) matrix ``M`` by its (K,) ``norms``,
    zeroing columns where ``norms == 0``.

    Returns ``(M_normalized, norms)`` so the caller has the norms for later use.
    """
    nonzero = norms > 0
    safe = torch.where(nonzero, norms, torch.ones_like(norms))
    M = M / safe.unsqueeze(0)
    M[:, ~nonzero] = 0.0
    return M, norms


def _make_initial_state(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    normb: torch.Tensor,
    damp: float,
    dtype: torch.dtype,
    device: torch.device,
    *,
    K: int | None = None,
) -> torch.Tensor:
    """
    Pack the 20-element LSMR scalar state into a tensor.

    For single-RHS: ``K=None`` → shape ``(_STATE_SIZE,)``.
    For batched:     ``K=int`` → shape ``(_STATE_SIZE, K)``.
    """
    shape = (_STATE_SIZE,) if K is None else (_STATE_SIZE, K)
    state = torch.zeros(shape, device=device, dtype=dtype)
    state[_I_ALPHABAR] = alpha
    state[_I_DAMP] = damp
    state[_I_BETA] = beta
    state[_I_ALPHA] = alpha
    state[_I_SBAR] = 0.0
    state[_I_CBAR] = 1.0
    state[_I_ZETABAR] = alpha * beta
    state[_I_RHO] = 1.0
    state[_I_RHOBAR] = 1.0
    state[_I_RHODOLD] = 1.0
    state[_I_TAUTILDEOLD] = 0.0
    state[_I_THETATILDE] = 0.0
    state[_I_BETADD] = beta
    state[_I_BETAD] = 0.0
    state[_I_D] = 0.0
    state[_I_NORMA2] = alpha * alpha
    state[_I_MAXRBAR] = 0.0
    state[_I_MINRBAR] = 1e100 if dtype == torch.float64 else 1e10
    state[_I_NORMB] = normb
    state[_I_ZETA] = 0.0
    return state


def _check_convergence_batched(
    istop: torch.Tensor,
    test1: torch.Tensor,
    rtol: torch.Tensor,
    test2: torch.Tensor,
    test3: torch.Tensor,
    t1: torch.Tensor,
    atol: float,
    ctol: float,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Per-column convergence check for batched LSMR.

    Sets istop per column using an only-set-once latch: once a column's
    istop becomes non-zero, it is never overwritten.  Returns updated istop.
    """
    not_yet = istop == 0
    new_stop = torch.zeros(K, device=device, dtype=torch.long)
    new_stop = torch.where(test1 <= rtol, torch.ones_like(new_stop), new_stop)
    new_stop = torch.where(
        (test2 <= atol) & (new_stop == 0),
        2 * torch.ones_like(new_stop), new_stop,
    )
    new_stop = torch.where(
        (test3 <= ctol) & (new_stop == 0),
        3 * torch.ones_like(new_stop), new_stop,
    )
    new_stop = torch.where(
        (1.0 + t1 <= 1.0) & (new_stop == 0),
        4 * torch.ones_like(new_stop), new_stop,
    )
    new_stop = torch.where(
        (1.0 + test2 <= 1.0) & (new_stop == 0),
        5 * torch.ones_like(new_stop), new_stop,
    )
    new_stop = torch.where(
        (1.0 + test3 <= 1.0) & (new_stop == 0),
        6 * torch.ones_like(new_stop), new_stop,
    )
    return torch.where(not_yet, new_stop, istop)


def _mark_maxiter_batched(istop: torch.Tensor, itn: int, maxiter: int) -> torch.Tensor:
    """Set istop=7 for columns that did not converge before maxiter."""
    return torch.where(
        (istop == 0) & (itn >= maxiter),
        7 * torch.ones_like(istop),
        istop,
    )


# ===========================================================================
# Implementation 0: batched LSMR — K right-hand sides via SpMM
# ===========================================================================


def _lsmr_batched(
    A,
    B: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Batched LSMR: solve min ||B - A X||_F for K RHS simultaneously.

    Replaces K sequential SpMV with SpMM for GPU throughput.
    All K columns run in lock-step; converged columns do harmless work.

    Parameters
    ----------
    A : sparse tensor or LinearOperator-like
        Matrix of shape (m, n).
    B : torch.Tensor
        Dense matrix of shape (m, K).
    damp : float
        Damping factor.
    atol, btol : float
        Stopping tolerances (same for all columns).
    conlim : float
        Condition number limit.
    maxiter : int or None
        Maximum iterations. Defaults to min(m, n).

    Returns
    -------
    X : (n, K) solution matrix
    istop : (K,) int tensor — per-column stopping reason
    itn : int — iterations used
    normr : (K,) — per-column ||b - Ax||
    normar : (K,) — per-column ||A^T(b - Ax)||
    normA : (K,) — per-column estimate of ||A||_F
    condA : (K,) — per-column estimate of cond(A)
    normx : (K,) — per-column ||x||
    """
    m, n = A.shape
    K = B.shape[1]
    device = B.device
    dtype = B.dtype

    if maxiter is None:
        maxiter = min(m, n)

    At = _precompute_transpose(A)

    # --- Initialize Golub-Kahan bidiagonalization ---
    U = B.clone()                                       # (m, K)
    normb = torch.linalg.norm(U, dim=0)                 # (K,)

    X = torch.zeros(n, K, device=device, dtype=dtype)
    beta = normb.clone()                                # (K,)

    U, beta = _safe_normalize_cols(U, beta)

    V = _rmatvec_batched(At, U)                         # (n, K)  — SpMM
    alpha = torch.linalg.norm(V, dim=0)                 # (K,)

    V, alpha = _safe_normalize_cols(V, alpha)

    # --- Scalar state as (K,) tensors ---
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha.clone()
    rho = torch.ones(K, device=device, dtype=dtype)
    rhobar = torch.ones(K, device=device, dtype=dtype)
    cbar = torch.ones(K, device=device, dtype=dtype)
    sbar = torch.zeros(K, device=device, dtype=dtype)

    H = V.clone()                                       # (n, K)
    Hbar = torch.zeros(n, K, device=device, dtype=dtype)

    # ||r|| estimation state
    betadd = beta.clone()
    betad = torch.zeros(K, device=device, dtype=dtype)
    rhodold = torch.ones(K, device=device, dtype=dtype)
    tautildeold = torch.zeros(K, device=device, dtype=dtype)
    thetatilde = torch.zeros(K, device=device, dtype=dtype)
    zeta = torch.zeros(K, device=device, dtype=dtype)
    d = torch.zeros(K, device=device, dtype=dtype)

    # ||A|| and cond(A) estimation
    normA2 = alpha * alpha
    maxrbar = torch.zeros(K, device=device, dtype=dtype)
    minrbar_init = 1e100 if dtype == torch.float64 else 1e10
    minrbar = torch.full((K,), minrbar_init, device=device, dtype=dtype)

    # Convergence tracking
    istop = torch.zeros(K, device=device, dtype=torch.long)
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normr = beta.clone()
    normar = alpha * beta

    damp_vec = torch.full((K,), damp, device=device, dtype=dtype)

    # Early exit: if all normar == 0 or all normb == 0
    if (normar == 0).all():
        return (X, istop, itn, normr, normar,
                torch.sqrt(normA2), torch.ones(K, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype))

    if (normb == 0).all():
        X.zero_()
        return (X, istop, itn, normr, normar,
                torch.sqrt(normA2), torch.ones(K, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype))

    # --- Main iteration loop ---
    while itn < maxiter:
        itn += 1

        # Bidiagonalization step: SpMM replaces SpMV
        U = _matvec_batched(A, V) - alpha.unsqueeze(0) * U     # (m, K)
        beta = torch.linalg.norm(U, dim=0)                     # (K,)
        U, beta = _safe_normalize_cols(U, beta)

        V = _rmatvec_batched(At, U) - beta.unsqueeze(0) * V    # (n, K)
        alpha = torch.linalg.norm(V, dim=0)                    # (K,)
        V, alpha = _safe_normalize_cols(V, alpha)

        # Givens rotation 1: (alphabar, damp)
        chat, shat, alphahat = _sym_ortho_vec(alphabar, damp_vec)

        # Givens rotation 2: (alphahat, beta)
        rhoold = rho
        c, s, rho = _sym_ortho_vec(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Givens rotation 3: rhobar update
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho_vec(rhotemp, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Vector updates: broadcast (K,) scalars over (n, K) matrices
        # Guard divisions: when a column has zero RHS, rho/rhobar can be 0.
        # Using clamp ensures 0/0 → 0 instead of NaN (numerator is also 0).
        _eps = 1e-30
        hbar_coeff = -(thetabar * rho) / torch.clamp(rhoold * rhobarold, min=_eps)
        Hbar = H + Hbar * hbar_coeff.unsqueeze(0)
        x_coeff = zeta / torch.clamp(rho * rhobar, min=_eps)
        X = X + x_coeff.unsqueeze(0) * Hbar
        h_coeff = -(thetanew / torch.clamp(rho, min=_eps))
        H = V + H * h_coeff.unsqueeze(0)

        # ||r|| estimation
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho_vec(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        safe_rhotildeold = torch.clamp(rhotildeold, min=1e-30)
        tautildeold = (zetaold - thetatildeold * tautildeold) / safe_rhotildeold
        safe_rhodold = torch.clamp(rhodold, min=1e-30)
        taud = (zeta - thetatilde * tautildeold) / safe_rhodold
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
        condA = torch.maximum(maxrbar, rhotemp) / torch.clamp(
            torch.minimum(minrbar, rhotemp), min=1e-30
        )

        # Per-column convergence check
        normar = torch.abs(zetabar)
        normx = torch.linalg.norm(X, dim=0)        # (K,)

        safe_normb = torch.clamp(normb, min=1e-30)
        test1 = normr / safe_normb
        safe_normA_normr = torch.clamp(normA * normr, min=1e-30)
        test2 = normar / safe_normA_normr
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / safe_normb)
        rtol = btol + atol * normA * normx / safe_normb

        istop = _check_convergence_batched(
            istop, test1, rtol, test2, test3, t1, atol, ctol, K, device,
        )
        if (istop > 0).all():
            break

    istop = _mark_maxiter_batched(istop, itn, maxiter)

    return X, istop, itn, normr, normar, normA, condA, normx


# ===========================================================================
# Implementation 1: scalar-state LSMR (CPU / MPS)
# ===========================================================================


def _lsmr_eager(
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

    At = _precompute_transpose(A)

    # --- Initialize Golub-Kahan bidiagonalization ---
    u = b.clone()
    normb = torch.linalg.norm(b).item()

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb

    if beta > 0:
        u = u * (1.0 / beta)
        v = _rmatvec(At, u)
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
            v = _rmatvec(At, u) - beta * v
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


# ===========================================================================
# Implementation 2: compiled-state LSMR (CUDA)
# ===========================================================================

# ---------------------------------------------------------------------------
# Packed scalar state layout
# ---------------------------------------------------------------------------
# All scalar state is packed into a single 1-D tensor to minimize Metal buffer
# slots (hardware limit: 31 per kernel).
#
# Input state (20 elements):
_I_ALPHABAR = 0
_I_DAMP = 1
_I_BETA = 2
_I_ALPHA = 3
_I_SBAR = 4
_I_CBAR = 5
_I_ZETABAR = 6
_I_RHO = 7
_I_RHOBAR = 8
_I_RHODOLD = 9
_I_TAUTILDEOLD = 10
_I_THETATILDE = 11
_I_BETADD = 12
_I_BETAD = 13
_I_D = 14
_I_NORMA2 = 15
_I_MAXRBAR = 16
_I_MINRBAR = 17
_I_NORMB = 18
_I_ZETA = 19  # previous iteration's zeta (for normr estimation)

# Constants (3 elements): atol, btol, ctol

# Output adds extra slots for vector update coefficients:
_O_THETANEW = 20
_O_THETABAR = 21
_O_ZETA = 22
_O_RHOOLD = 23
_O_RHOBAROLD = 24
_O_CONVERGED = 25
_O_NORMR = 26
_O_NORMAR = 27
_O_NORMA = 28
_O_CONDA = 29
_O_NORMX_EST = 30  # placeholder, actual normx computed from vector

_STATE_SIZE = 20


# ---------------------------------------------------------------------------
# Overflow-safe hypot (replaces torch.hypot for Metal compatibility)
# ---------------------------------------------------------------------------


def _safe_hypot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Overflow-safe hypot: ``sqrt(a** + b**)`` without intermediate overflow.

    Uses max/min scaling: ``hypot(a,b) = max(|a|,|b|) * sqrt(1 + (min/max)**)``.
    Since ``min/max <= 1``, the argument to sqrt never exceeds 2.
    Compiles to ~6 Metal/CUDA ops that fuse into the surrounding kernel.
    """
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)
    big = torch.maximum(abs_a, abs_b)
    small = torch.minimum(abs_a, abs_b)
    safe_big = torch.where(big == 0, torch.ones_like(big), big)
    ratio = small / safe_big
    return torch.where(
        big == 0,
        torch.zeros_like(big),
        big * torch.sqrt(1.0 + ratio * ratio),
    )


# ---------------------------------------------------------------------------
# Compiled scalar step (single Metal/CUDA kernel after fusion)
# ---------------------------------------------------------------------------


def _scalar_step(state: torch.Tensor, consts: torch.Tensor) -> torch.Tensor:
    """
    All scalar work for one LSMR iteration: 4 Givens rotations, norm/cond
    estimation, and convergence check.

    Packed I/O keeps Metal buffer count to 3 (state_in, consts, state_out).
    Uses overflow-safe hypot (no torch.hypot — unsupported in Metal codegen).
    """
    # Unpack
    alphabar = state[_I_ALPHABAR]
    damp = state[_I_DAMP]
    beta = state[_I_BETA]
    alpha = state[_I_ALPHA]
    sbar = state[_I_SBAR]
    cbar = state[_I_CBAR]
    zetabar = state[_I_ZETABAR]
    rho = state[_I_RHO]
    rhobar = state[_I_RHOBAR]
    rhodold = state[_I_RHODOLD]
    tautildeold = state[_I_TAUTILDEOLD]
    thetatilde = state[_I_THETATILDE]
    betadd = state[_I_BETADD]
    betad = state[_I_BETAD]
    d = state[_I_D]
    normA2 = state[_I_NORMA2]
    maxrbar = state[_I_MAXRBAR]
    minrbar = state[_I_MINRBAR]
    normb = state[_I_NORMB]
    zetaold = state[_I_ZETA]  # zeta from previous iteration (for normr estimation)

    atol_t = consts[0]
    ctol = consts[2]

    _ZERO = state[_I_ALPHABAR] * 0.0  # device-local zero
    _ONE = _ZERO + 1.0

    # --- Givens 1: (alphabar, damp) ---
    r1 = _safe_hypot(alphabar, damp)
    safe_r1 = torch.where(r1 == _ZERO, _ONE, r1)
    chat = torch.where(r1 == _ZERO, _ZERO, alphabar / safe_r1)
    shat = torch.where(r1 == _ZERO, _ZERO, damp / safe_r1)

    # --- Givens 2: (alphahat=r1, beta) ---
    rhoold = rho
    r2 = _safe_hypot(r1, beta)
    safe_r2 = torch.where(r2 == _ZERO, _ONE, r2)
    c = torch.where(r2 == _ZERO, _ZERO, r1 / safe_r2)
    s = torch.where(r2 == _ZERO, _ZERO, beta / safe_r2)
    rho_new = r2
    thetanew = s * alpha
    alphabar_new = c * alpha

    # --- Givens 3: rhobar ---
    rhobarold = rhobar
    thetabar = sbar * rho_new
    rhotemp = cbar * rho_new
    r3 = _safe_hypot(rhotemp, thetanew)
    safe_r3 = torch.where(r3 == _ZERO, _ONE, r3)
    cbar_new = torch.where(r3 == _ZERO, _ZERO, rhotemp / safe_r3)
    sbar_new = torch.where(r3 == _ZERO, _ZERO, thetanew / safe_r3)
    rhobar_new = r3
    zeta = cbar_new * zetabar
    zetabar_new = -sbar_new * zetabar

    # --- ||r|| estimation ---
    betaacute = chat * betadd
    betacheck = -shat * betadd
    betahat = c * betaacute
    betadd_new = -s * betaacute

    # Givens 4: rhotilde
    r4 = _safe_hypot(rhodold, thetabar)
    safe_r4 = torch.where(r4 == _ZERO, _ONE, r4)
    ctildeold = torch.where(r4 == _ZERO, _ZERO, rhodold / safe_r4)
    stildeold = torch.where(r4 == _ZERO, _ZERO, thetabar / safe_r4)

    thetatilde_new = stildeold * rhobar_new
    rhodold_new = ctildeold * rhobar_new
    betad_new = -stildeold * betad + ctildeold * betahat

    tautildeold_new = (zetaold - thetatilde * tautildeold) / torch.clamp(r4, min=1e-30)
    taud = (zeta - thetatilde_new * tautildeold_new) / torch.clamp(
        rhodold_new, min=1e-30
    )
    d_new = d + betacheck * betacheck
    normr = torch.sqrt(d_new + (betad_new - taud) ** 2 + betadd_new * betadd_new)

    # --- ||A|| estimation ---
    normA2_new = normA2 + beta * beta
    normA = torch.sqrt(normA2_new)
    normA2_final = normA2_new + alpha * alpha

    # --- cond(A) estimation ---
    maxrbar_new = torch.maximum(maxrbar, rhobarold)
    # Match SciPy: only update minrbar from iteration 2 onward.
    # maxrbar == 0 on the first call (initial state), so use it as guard.
    minrbar_new = torch.where(maxrbar > 0, torch.minimum(minrbar, rhobarold), minrbar)
    condA = torch.maximum(maxrbar_new, rhotemp) / torch.clamp(
        torch.minimum(minrbar_new, rhotemp), min=1e-30
    )

    # --- Convergence check ---
    normar = torch.abs(zetabar_new)
    test2 = normar / torch.clamp(normA * normr, min=1e-30)
    test3 = _ONE / condA

    converged_flag = torch.where(
        (test2 <= atol_t)
        | (test3 <= ctol)
        | (_ONE + test2 <= _ONE)
        | (_ONE + test3 <= _ONE),
        _ONE,
        _ZERO,
    )

    # --- Pack output ---
    return torch.stack(
        [
            alphabar_new,  # 0  _I_ALPHABAR
            damp,  # 1  _I_DAMP (pass through)
            beta,  # 2  _I_BETA (pass through, updated by caller)
            alpha,  # 3  _I_ALPHA (pass through, updated by caller)
            sbar_new,  # 4  _I_SBAR
            cbar_new,  # 5  _I_CBAR
            zetabar_new,  # 6  _I_ZETABAR
            rho_new,  # 7  _I_RHO
            rhobar_new,  # 8  _I_RHOBAR
            rhodold_new,  # 9  _I_RHODOLD
            tautildeold_new,  # 10 _I_TAUTILDEOLD
            thetatilde_new,  # 11 _I_THETATILDE
            betadd_new,  # 12 _I_BETADD
            betad_new,  # 13 _I_BETAD
            d_new,  # 14 _I_D
            normA2_final,  # 15 _I_NORMA2
            maxrbar_new,  # 16 _I_MAXRBAR
            minrbar_new,  # 17 _I_MINRBAR
            normb,  # 18 _I_NORMB (pass through)
            zeta,  # 19 _I_ZETA (saved for next iteration's zetaold)
            thetanew,  # 20 _O_THETANEW (for vector update)
            thetabar,  # 21 _O_THETABAR (for vector update)
            zeta,  # 22 _O_ZETA (for vector update — same as slot 19)
            rhoold,  # 23 _O_RHOOLD (for vector update)
            rhobarold,  # 24 _O_RHOBAROLD (for vector update)
            converged_flag,  # 25 _O_CONVERGED
            normr,  # 26 _O_NORMR
            normar,  # 27 _O_NORMAR
            normA,  # 28 _O_NORMA
            condA,  # 29 _O_CONDA
            _ZERO,  # 30 _O_NORMX_EST (placeholder)
        ]
    )


# ---------------------------------------------------------------------------
# Module-level compilation cache
# ---------------------------------------------------------------------------
_compiled_step_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def _get_compiled_step(device_type: str):
    """Get or create compiled scalar step for the given device type."""
    if device_type in _compiled_step_cache:
        return _compiled_step_cache[device_type]
    with _cache_lock:
        # Double-check after acquiring lock
        if device_type not in _compiled_step_cache:
            try:
                _compiled_step_cache[device_type] = torch.compile(
                    _scalar_step, backend="inductor", fullgraph=True
                )
            except Exception:
                # Fallback: no compilation available
                _compiled_step_cache[device_type] = _scalar_step
    return _compiled_step_cache[device_type]


def _lsmr_compiled(
    A,
    b: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
    use_compile: bool = True,
) -> tuple[torch.Tensor, int, int, float, float, float, float, float]:
    """
    LSMR with packed-tensor scalar state and optional torch.compile fusion.

    On CUDA the scalar Givens rotations, norm estimation, and convergence
    check are fused into a **single GPU kernel** via ``torch.compile`` +
    Inductor, eliminating ~60 per-iteration kernel launches.

    Called by the ``lsmr_torch`` dispatcher; ``use_compile`` is already
    resolved by the caller (no auto-detection here).
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # Get compiled or uncompiled step function
    step_fn = _get_compiled_step(device.type) if use_compile else _scalar_step

    At = _precompute_transpose(A)

    # --- Initialize Golub-Kahan bidiagonalization ---
    u = b.clone()
    normb = torch.linalg.norm(b)

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb.clone()

    # Safe normalize
    u = u * torch.where(beta > 0, 1.0 / torch.clamp(beta, min=1e-30), beta * 0.0)

    v = _rmatvec(At, u)
    alpha = torch.linalg.norm(v)
    v = v * torch.where(alpha > 0, 1.0 / torch.clamp(alpha, min=1e-30), alpha * 0.0)

    state = _make_initial_state(alpha, beta, normb, damp, dtype, device)

    ctol = 1.0 / conlim if conlim > 0 else 0.0
    consts = torch.tensor([atol, btol, ctol], device=device, dtype=dtype)

    # Early exit check
    normar_init = (alpha * beta).item()
    if normar_init == 0.0:
        return x, 0, 0, beta.item(), 0.0, alpha.item(), 1.0, 0.0
    if normb.item() == 0.0:
        x.zero_()
        return x, 0, 0, beta.item(), 0.0, alpha.item(), 1.0, 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    # --- Main iteration loop ---
    itn = 0
    istop = 0

    while itn < maxiter:
        itn += 1

        # Phase 1: Sparse matvec (not compilable)
        # state[_I_ALPHA] holds the current alpha (passed through by _scalar_step)
        u = _matvec(A, v) - state[_I_ALPHA] * u
        beta_new = torch.linalg.norm(u)
        u = u * torch.where(
            beta_new > 0,
            1.0 / torch.clamp(beta_new, min=1e-30),
            beta_new * 0.0,
        )

        v = _rmatvec(At, u) - beta_new * v
        alpha_new = torch.linalg.norm(v)
        v = v * torch.where(
            alpha_new > 0,
            1.0 / torch.clamp(alpha_new, min=1e-30),
            alpha_new * 0.0,
        )

        # Update beta/alpha in state for the scalar step
        state[_I_BETA] = beta_new
        state[_I_ALPHA] = alpha_new

        # Phase 2: Compiled scalar step (single GPU kernel on CUDA)
        out = step_fn(state, consts)

        # Phase 3: Vector updates using scalar results from compiled step
        thetanew = out[_O_THETANEW]
        thetabar = out[_O_THETABAR]
        zeta = out[_O_ZETA]
        rho_new = out[_I_RHO]
        rhobar_new = out[_I_RHOBAR]
        rhoold = out[_O_RHOOLD]
        rhobarold = out[_O_RHOBAROLD]

        hbar = h + hbar * (-(thetabar * rho_new) / (rhoold * rhobarold))
        x = x + (zeta / (rho_new * rhobar_new)) * hbar
        h = v + h * (-(thetanew / rho_new))

        # Propagate state for next iteration
        state = out[:_STATE_SIZE]

        # Convergence check — single .item() sync per iteration.
        # The compiled _scalar_step checks test2 (atol) and test3 (ctol).
        # The btol-based test1 depends on normx (a vector quantity) and is
        # computed here on-device, then combined with the scalar step's flag.
        # Only one .item() call reads the combined boolean; all other tensor
        # ops queue on the GPU stream without forcing a pipeline stall.
        normx_t = torch.linalg.norm(x)
        normr_t = out[_O_NORMR]
        normA_t = out[_O_NORMA]
        normb_t = out[_I_NORMB]

        test1_t = normr_t / torch.clamp(normb_t, min=1e-30)
        t1_t = test1_t / (1.0 + normA_t * normx_t / torch.clamp(normb_t, min=1e-30))
        rtol_t = btol + atol * normA_t * normx_t / torch.clamp(normb_t, min=1e-30)

        converged_btol = (test1_t <= rtol_t) | (1.0 + t1_t <= 1.0)
        converged_any = (out[_O_CONVERGED] > 0.5) | converged_btol

        if converged_any.item():
            # Pull scalars to CPU only at exit (one-time cost)
            normr_val = normr_t.item()
            normA_val = normA_t.item()
            normx_val = normx_t.item()
            normb_val = normb_t.item()
            normar_val = out[_O_NORMAR].item()
            condA_val = out[_O_CONDA].item()

            test1 = normr_val / max(normb_val, 1e-30)
            test2 = normar_val / max(normA_val * normr_val, 1e-30)
            test3 = 1.0 / condA_val
            t1 = test1 / (1.0 + normA_val * normx_val / max(normb_val, 1e-30))
            _rtol = btol + atol * normA_val * normx_val / max(normb_val, 1e-30)

            # Priority order matches SciPy LSMR (lowest istop wins)
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
            if test1 <= _rtol:
                istop = 1
            break

    if itn >= maxiter and istop == 0:
        istop = 7

    # Handle case where loop never ran (maxiter=0 or similar)
    if itn == 0:
        return x, istop, 0, normb.item(), normar_init, alpha.item(), 1.0, 0.0

    # normx_val was already computed inside the convergence block (line 725);
    # only recompute if the loop exhausted maxiter without converging.
    if istop == 0 or istop == 7:
        normx_val = torch.linalg.norm(x).item()
    return (
        x,
        istop,
        itn,
        out[_O_NORMR].item(),
        out[_O_NORMAR].item(),
        out[_O_NORMA].item(),
        out[_O_CONDA].item(),
        normx_val,
    )


# ===========================================================================
# Implementation 3: compiled batched LSMR — K RHS via SpMM + torch.compile
# ===========================================================================


def _lsmr_compiled_batched(
    A,
    B: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
    use_compile: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Compiled batched LSMR: solve min ||B - A X||_F for K RHS simultaneously.

    Mirrors ``_lsmr_compiled`` but with (m, K) vectors and SpMM.  The scalar
    state is packed into a (_STATE_SIZE, K) tensor — each of the 20 scalar
    quantities becomes a (K,) vector.  ``_scalar_step`` is shape-agnostic:
    its indexing and element-wise ops broadcast over the K dimension without
    any code changes.

    Called by ``lsmr_torch_batched``; ``use_compile`` is already resolved.
    """
    m, n = A.shape
    K = B.shape[1]
    device = B.device
    dtype = B.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # Get compiled or uncompiled step function
    step_fn = _get_compiled_step(device.type) if use_compile else _scalar_step

    At = _precompute_transpose(A)

    # --- Initialize Golub-Kahan bidiagonalization ---
    U = B.clone()                                           # (m, K)
    normb = torch.linalg.norm(U, dim=0)                     # (K,)

    X = torch.zeros(n, K, device=device, dtype=dtype)
    beta = normb.clone()                                    # (K,)

    U, beta = _safe_normalize_cols(U, beta)

    V = _rmatvec_batched(At, U)                             # (n, K)  — SpMM
    alpha = torch.linalg.norm(V, dim=0)                     # (K,)
    V, alpha = _safe_normalize_cols(V, alpha)

    state = _make_initial_state(alpha, beta, normb, damp, dtype, device, K=K)

    ctol = 1.0 / conlim if conlim > 0 else 0.0
    # consts (3,) — _scalar_step indexes as consts[0]/consts[2] which broadcast
    # against (K,) state elements automatically.
    consts = torch.tensor([atol, btol, ctol], device=device, dtype=dtype)

    # Early exit check
    normar_init = alpha * beta  # (K,)
    if (normar_init == 0).all():
        return (X, torch.zeros(K, device=device, dtype=torch.long), 0,
                beta, torch.zeros(K, device=device, dtype=dtype),
                alpha, torch.ones(K, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype))
    if (normb == 0).all():
        X.zero_()
        return (X, torch.zeros(K, device=device, dtype=torch.long), 0,
                beta, torch.zeros(K, device=device, dtype=dtype),
                alpha, torch.ones(K, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype))

    H = V.clone()                                           # (n, K)
    Hbar = torch.zeros(n, K, device=device, dtype=dtype)

    # Convergence tracking: per-column istop, only-set-once latch
    istop = torch.zeros(K, device=device, dtype=torch.long)

    # --- Main iteration loop ---
    itn = 0
    _eps = 1e-30
    normx_t = torch.zeros(K, device=device, dtype=dtype)

    while itn < maxiter:
        itn += 1

        # Phase 1: SpMM bidiagonalization (not compilable)
        U = _matvec_batched(A, V) - state[_I_ALPHA].unsqueeze(0) * U   # (m, K)
        beta_new = torch.linalg.norm(U, dim=0)                          # (K,)
        U, beta_new = _safe_normalize_cols(U, beta_new)

        V = _rmatvec_batched(At, U) - beta_new.unsqueeze(0) * V        # (n, K)
        alpha_new = torch.linalg.norm(V, dim=0)                         # (K,)
        V, alpha_new = _safe_normalize_cols(V, alpha_new)

        # Update beta/alpha in state for the scalar step
        state[_I_BETA] = beta_new
        state[_I_ALPHA] = alpha_new

        # Phase 2: Compiled scalar step — (_STATE_SIZE, K) → (_OUTPUT_SIZE, K)
        out = step_fn(state, consts)

        # Phase 3: Vector updates using scalar results from compiled step
        thetanew = out[_O_THETANEW]          # (K,)
        thetabar = out[_O_THETABAR]          # (K,)
        zeta = out[_O_ZETA]                  # (K,)
        rho_new = out[_I_RHO]               # (K,)
        rhobar_new = out[_I_RHOBAR]         # (K,)
        rhoold = out[_O_RHOOLD]             # (K,)
        rhobarold = out[_O_RHOBAROLD]       # (K,)

        # Safe divisions: some columns may have zero RHS → zero denominators
        hbar_coeff = -(thetabar * rho_new) / torch.clamp(rhoold * rhobarold, min=_eps)
        Hbar = H + Hbar * hbar_coeff.unsqueeze(0)
        x_coeff = zeta / torch.clamp(rho_new * rhobar_new, min=_eps)
        X = X + x_coeff.unsqueeze(0) * Hbar
        h_coeff = -(thetanew / torch.clamp(rho_new, min=_eps))
        H = V + H * h_coeff.unsqueeze(0)

        # Propagate state for next iteration
        state = out[:_STATE_SIZE]

        # Phase 4: Per-column convergence — single .item() sync per iteration.
        # No not_yet.any() guard: the torch.where inside _check_convergence_batched
        # already protects converged columns, and the guard would add a second
        # host-device sync that costs more than the fused tensor ops it skips.
        normx_t = torch.linalg.norm(X, dim=0)    # (K,)
        normr_t = out[_O_NORMR]                   # (K,)
        normA_t = out[_O_NORMA]                   # (K,)
        normb_t = out[_I_NORMB]                   # (K,)

        safe_normb = torch.clamp(normb_t, min=_eps)
        test1_t = normr_t / safe_normb
        safe_normA_normr = torch.clamp(normA_t * normr_t, min=_eps)
        test2_t = out[_O_NORMAR] / safe_normA_normr
        test3_t = 1.0 / out[_O_CONDA]
        t1_t = test1_t / (1.0 + normA_t * normx_t / safe_normb)
        rtol_t = btol + atol * normA_t * normx_t / safe_normb

        istop = _check_convergence_batched(
            istop, test1_t, rtol_t, test2_t, test3_t, t1_t, atol, ctol, K, device,
        )

        # Single .item() sync: check if all columns have converged
        if (istop > 0).all().item():
            break

    istop = _mark_maxiter_batched(istop, itn, maxiter)

    # Handle case where loop never ran
    if itn == 0:
        return (X, istop, 0, normb, normar_init, alpha,
                torch.ones(K, device=device, dtype=dtype),
                torch.zeros(K, device=device, dtype=dtype))

    return (
        X,
        istop,
        itn,
        out[_O_NORMR],
        out[_O_NORMAR],
        out[_O_NORMA],
        out[_O_CONDA],
        normx_t,  # reuse from last iteration, avoids redundant norm
    )


# ===========================================================================
# Public API — dispatcher
# ===========================================================================


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
    return _lsmr_eager(
        A,
        b,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
    )


def lsmr_torch_batched(
    A,
    B: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
    use_compile: bool | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Batched LSMR solver — solve K right-hand sides simultaneously via SpMM.

    Solves ``min ||B - A X||_F`` where B has K columns. Instead of K
    sequential sparse matrix-vector products (SpMV), each iteration uses
    a single sparse matrix-matrix product (SpMM), which loads the sparse
    matrix once and streams through K dense columns. For K >= 2 on GPU,
    this is significantly faster than K sequential ``lsmr_torch`` calls.

    All K columns run in lock-step — converged columns continue doing
    harmless arithmetic until all columns converge or maxiter is reached.

    When ``use_compile=True`` (auto-detected for CUDA/MPS), scalar Givens
    rotations are fused into a single compiled kernel via ``torch.compile``,
    further reducing per-iteration kernel launches.

    Parameters
    ----------
    A : sparse tensor or LinearOperator-like
        Matrix of shape (m, n). Must support ``A @ V`` for dense V of
        shape (n, K). For ``_PreconditionedSparse``, this requires an
        ``mm()`` method.
    B : torch.Tensor
        Dense RHS matrix of shape (m, K).
    damp : float
        Damping factor for regularized least-squares.
    atol, btol : float
        Stopping tolerances (applied identically to all columns).
    conlim : float
        Condition number limit.
    maxiter : int or None
        Maximum iterations. Defaults to min(m, n).
    use_compile : bool or None
        Whether to use ``torch.compile`` for scalar step fusion.
        ``None`` (default) auto-detects: compiled for CUDA, eager for
        CPU/MPS.  Pass ``True`` to force compilation on MPS.

    Returns
    -------
    X : torch.Tensor, shape (n, K)
        Solution matrix.
    istop : torch.Tensor, shape (K,), dtype long
        Per-column stopping reason (0-7, same codes as ``lsmr_torch``).
    itn : int
        Number of iterations used (max across all columns).
    normr : torch.Tensor, shape (K,)
        Per-column ``||b_k - A x_k||``.
    normar : torch.Tensor, shape (K,)
        Per-column ``||A^T(b_k - A x_k)||``.
    normA : torch.Tensor, shape (K,)
        Per-column estimate of Frobenius norm of A.
    condA : torch.Tensor, shape (K,)
        Per-column estimate of condition number of A.
    normx : torch.Tensor, shape (K,)
        Per-column ``||x_k||``.
    """
    device = B.device
    if use_compile is None:
        use_compile = device.type == "cuda"

    if use_compile:
        return _lsmr_compiled_batched(
            A, B, damp=damp, atol=atol, btol=btol, conlim=conlim,
            maxiter=maxiter, use_compile=True,
        )
    return _lsmr_batched(
        A, B, damp=damp, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter,
    )
