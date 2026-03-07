"""
Pure PyTorch LSMR iterative solver with optional torch.compile kernel fusion.

Two implementations live in this file:

1. ``_lsmr_eager`` — eager-mode PyTorch, Python-float Givens rotations.
   Best for CPU and MPS (Metal command-buffer batching already amortizes
   kernel-launch overhead).

2. ``_lsmr_compiled`` — packs all scalar state into a 1-D tensor and runs
   the Givens / norm / convergence work through a ``torch.compile``-d
   kernel.  On CUDA this fuses ~60 per-iteration kernel launches into one.

The public entry point ``lsmr_torch()`` dispatches automatically:
CUDA → compiled, CPU/MPS → scalar.  Pass ``use_compile=True`` to override.

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

    # --- Pack initial scalar state ---
    state = torch.zeros(_STATE_SIZE, device=device, dtype=dtype)
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
    state[_I_ZETA] = 0.0  # initial zeta (no previous iteration)

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
