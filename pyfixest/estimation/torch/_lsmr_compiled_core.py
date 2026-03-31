"""
Compiled LSMR scalar-step kernel and packed-state protocol.

This module contains the densest part of the LSMR implementation:
the packed tensor state layout (31 index constants), the fused
``_scalar_step`` kernel that ``torch.compile`` turns into a single
Metal/CUDA kernel, and the helpers that create and cache it.

Separated from ``lsmr_torch.py`` so the protocol is co-located with
the only code that reads/writes it, and can be unit-tested in isolation.
"""

from __future__ import annotations

import threading
import warnings

import torch

# Guard value for divisions that can be zero (e.g. normb, rho, rhodold).
# Using clamp(..., min=_DIV_GUARD) or max(..., _DIV_GUARD) prevents 0/0 → NaN.
_DIV_GUARD = 1e-30

# ---------------------------------------------------------------------------
# Packed state layout
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
    Overflow-safe hypot: ``sqrt(a^2 + b^2)`` without intermediate overflow.

    Uses max/min scaling: ``hypot(a,b) = max(|a|,|b|) * sqrt(1 + (min/max)^2)``.
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
# Initial state packing
# ---------------------------------------------------------------------------


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

    tautildeold_new = (zetaold - thetatilde * tautildeold) / torch.clamp(
        r4, min=_DIV_GUARD
    )
    taud = (zeta - thetatilde_new * tautildeold_new) / torch.clamp(
        rhodold_new, min=_DIV_GUARD
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
        torch.minimum(minrbar_new, rhotemp), min=_DIV_GUARD
    )

    # --- Convergence check ---
    normar = torch.abs(zetabar_new)
    test2 = normar / torch.clamp(normA * normr, min=_DIV_GUARD)
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
                warnings.warn(
                    "torch.compile failed for LSMR scalar step; "
                    "falling back to eager mode.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                _compiled_step_cache[device_type] = _scalar_step
    return _compiled_step_cache[device_type]
