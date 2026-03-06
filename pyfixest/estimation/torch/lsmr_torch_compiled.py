"""
LSMR iterative solver in pure PyTorch with optional torch.compile kernel fusion.

The solver splits each iteration into three phases:
  1. Sparse matvec (A @ v, A.T @ u) — cannot be compiled (sparse CSR unsupported)
  2. Scalar Givens rotations + norm estimation + convergence — compiled on GPU
  3. Vector updates (h, hbar, x) — use scalar results from phase 2

Phase 2 involves ~60 scalar operations that, without compilation, dispatch as
~60 individual GPU kernels (~15μs each on MPS). torch.compile fuses them into
a SINGLE kernel via the Inductor backend (Metal shaders on MPS, CUDA kernels
on NVIDIA GPUs).

Workarounds for MPS/Metal limitations:
  - torch.hypot not in Metal codegen → overflow-safe manual hypot via max/min scaling
  - Metal kernel limited to 31 buffer args → pack all scalars into 1-D tensors

The safe manual hypot and packed layout work uniformly across all backends
(CPU, MPS, CUDA) with negligible overhead (~5%) after fusion.

On CPU the scalar step runs without compilation (no kernel launch overhead
to eliminate), so the packed layout is the only difference from a traditional
scalar-state LSMR.

Reference:
    D. C.-L. Fong and M. A. Saunders,
    "LSMR: An iterative algorithm for sparse least-squares problems",
    SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
"""

from __future__ import annotations

import threading

import torch

# ---------------------------------------------------------------------------
# Sparse matvec helpers (outside compiled region)
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
# Packed scalar state layout
# ---------------------------------------------------------------------------
# We pack all scalar state into a single 1-D tensor to minimize Metal buffer
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
    Overflow-safe hypot: ``sqrt(a² + b²)`` without intermediate overflow.

    Uses max/min scaling: ``hypot(a,b) = max(|a|,|b|) * sqrt(1 + (min/max)²)``.
    Since ``min/max ≤ 1``, the argument to sqrt never exceeds 2.
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
    minrbar_new = torch.minimum(minrbar, rhobarold)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    LSMR iterative solver for sparse least-squares, in pure PyTorch.

    Solves ``min ||b - Ax||_2`` (or the damped variant) where *A* is a sparse
    CSR (or dense) tensor and *b* is a dense vector.  All vector operations
    stay on the tensor's device (CPU / CUDA / MPS).

    On GPU (MPS or CUDA) the scalar Givens rotations, norm estimation, and
    convergence check are fused into a **single GPU kernel** via
    ``torch.compile`` + Inductor, eliminating ~60 per-iteration kernel
    launches.  On CPU the scalar step runs without compilation (no kernel-
    launch overhead to eliminate).

    The while loop itself cannot be compiled because sparse CSR matvec is
    not supported in ``torch.compile``.  The single remaining CPU-GPU sync
    per iteration (reading the convergence flag) is negligible after
    compilation fuses the scalar step into one kernel.

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR tensor (or dense tensor / LinearOperator with ``.mv``
        and ``.t().mv`` support) of shape ``(m, n)``.
    b : torch.Tensor
        Dense vector of shape ``(m,)``.
    damp : float
        Damping factor for regularised least-squares.
    atol, btol : float
        Stopping tolerances (see SciPy LSMR documentation).
    conlim : float
        Condition-number limit.
    maxiter : int or None
        Maximum iterations.  Defaults to ``min(m, n)``.
    use_compile : bool or None
        Whether to ``torch.compile`` the scalar step.  ``None`` (default)
        auto-selects: **True** on GPU, **False** on CPU.

    Returns
    -------
    x : torch.Tensor
        Solution vector of shape ``(n,)``.
    istop : int
        Reason for stopping (0-7, same codes as SciPy LSMR).
    itn : int
        Number of iterations used.
    normr, normar, normA, condA, normx : float
        Diagnostic norms (see SciPy LSMR documentation).
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    # Auto-detect compilation
    if use_compile is None:
        use_compile = device.type in ("cuda", "mps")

    # Get compiled or uncompiled step function
    step_fn = _get_compiled_step(device.type) if use_compile else _scalar_step

    # --- Initialize Golub-Kahan bidiagonalization ---
    u = b.clone()
    normb = torch.linalg.norm(b)

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb.clone()

    # Safe normalize
    u = u * torch.where(beta > 0, 1.0 / torch.clamp(beta, min=1e-30), beta * 0.0)

    v = _rmatvec(A, u)
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
        u = _matvec(A, v) - state[_I_ALPHA] * u
        beta_new = torch.linalg.norm(u)
        u = u * torch.where(
            beta_new > 0,
            1.0 / torch.clamp(beta_new, min=1e-30),
            beta_new * 0.0,
        )

        v = _rmatvec(A, u) - beta_new * v
        alpha_new = torch.linalg.norm(v)
        v = v * torch.where(
            alpha_new > 0,
            1.0 / torch.clamp(alpha_new, min=1e-30),
            alpha_new * 0.0,
        )

        # Update beta/alpha in state for the scalar step
        state[_I_BETA] = beta_new
        state[_I_ALPHA] = alpha_new

        # Phase 2: Compiled scalar step (single GPU kernel on MPS/CUDA)
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
        # After compilation the scalar step is one kernel, so this sync
        # (reading a boolean from GPU memory) is negligible.
        #
        # The compiled _scalar_step checks test2 (atol) and test3 (ctol).
        # The btol-based test1 depends on normx (a vector quantity) and is
        # checked here in Python since it can't be part of the compiled kernel.
        converged_scalar = out[_O_CONVERGED].item() > 0.5

        # btol / test1 convergence (requires normx from vector x)
        normr_val = out[_O_NORMR].item()
        normA_val = out[_O_NORMA].item()
        normx_val = torch.linalg.norm(x).item()
        normb_val = out[_I_NORMB].item()

        test1 = normr_val / max(normb_val, 1e-30)
        t1 = test1 / (1.0 + normA_val * normx_val / max(normb_val, 1e-30))
        rtol = btol + atol * normA_val * normx_val / max(normb_val, 1e-30)
        converged_btol = test1 <= rtol
        converged_btol_machine = 1.0 + t1 <= 1.0

        if converged_scalar or converged_btol or converged_btol_machine:
            normar_val = out[_O_NORMAR].item()
            condA_val = out[_O_CONDA].item()

            test2 = normar_val / max(normA_val * normr_val, 1e-30)
            test3 = 1.0 / condA_val

            # Priority order matches SciPy LSMR (highest istop wins)
            if 1.0 + test3 <= 1.0:
                istop = 6
            if 1.0 + test2 <= 1.0:
                istop = 5
            if converged_btol_machine:
                istop = 4
            if test3 <= ctol:
                istop = 3
            if test2 <= atol:
                istop = 2
            if converged_btol:
                istop = 1
            break

    if itn >= maxiter and istop == 0:
        istop = 7

    # Handle case where loop never ran (maxiter=0 or similar)
    if itn == 0:
        return x, istop, 0, normb.item(), normar_init, alpha.item(), 1.0, 0.0

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
