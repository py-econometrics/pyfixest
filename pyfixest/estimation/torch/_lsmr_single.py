from __future__ import annotations

import math

import torch

from pyfixest.estimation.torch._lsmr_compiled_core import (
    _DIV_GUARD,
    _I_ALPHA,
    _I_BETA,
    _I_NORMB,
    _I_RHO,
    _I_RHOBAR,
    _O_CONDA,
    _O_CONVERGED,
    _O_NORMA,
    _O_NORMAR,
    _O_NORMR,
    _O_RHOBAROLD,
    _O_RHOOLD,
    _O_THETABAR,
    _O_THETANEW,
    _O_ZETA,
    _STATE_SIZE,
    _get_compiled_step,
    _make_initial_state,
    _scalar_step,
)
from pyfixest.estimation.torch._lsmr_helpers import (
    _matvec,
    _precompute_transpose,
    _rmatvec,
    _sym_ortho,
)


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
    CSR (COO) tensor and b is a dense vector.
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    At = _precompute_transpose(A)

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

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = math.sqrt(normA2)
    condA = 1.0
    normx = 0.0

    istop = 0
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normr = beta
    normar = alpha * beta

    if normar == 0.0:
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0.0:
        x.zero_()
        return x, istop, itn, normr, normar, normA, condA, normx

    while itn < maxiter:
        itn += 1

        u = _matvec(A, v) - alpha * u
        beta = torch.linalg.norm(u).item()

        if beta > 0:
            u *= 1.0 / beta
            v = _rmatvec(At, u) - beta * v
            alpha = torch.linalg.norm(v).item()
            if alpha > 0:
                v *= 1.0 / alpha

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = h + hbar * (-(thetabar * rho) / (rhoold * rhobarold))
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + h * (-(thetanew / rho))

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

        normA2 = normA2 + beta * beta
        normA = math.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

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
    check are fused into a single GPU kernel.
    """
    m, n = A.shape
    device = b.device
    dtype = b.dtype

    if maxiter is None:
        maxiter = min(m, n)

    step_fn = _get_compiled_step(device.type) if use_compile else _scalar_step

    At = _precompute_transpose(A)

    u = b.clone()
    normb = torch.linalg.norm(b)

    x = torch.zeros(n, device=device, dtype=dtype)
    beta = normb.clone()

    u = u * torch.where(beta > 0, 1.0 / torch.clamp(beta, min=_DIV_GUARD), beta * 0.0)

    v = _rmatvec(At, u)
    alpha = torch.linalg.norm(v)
    v = v * torch.where(
        alpha > 0, 1.0 / torch.clamp(alpha, min=_DIV_GUARD), alpha * 0.0
    )

    state = _make_initial_state(alpha, beta, normb, damp, dtype, device)

    ctol = 1.0 / conlim if conlim > 0 else 0.0
    consts = torch.tensor([atol, btol, ctol], device=device, dtype=dtype)

    normar_init = (alpha * beta).item()
    if normar_init == 0.0:
        return x, 0, 0, beta.item(), 0.0, alpha.item(), 1.0, 0.0
    if normb.item() == 0.0:
        x.zero_()
        return x, 0, 0, beta.item(), 0.0, alpha.item(), 1.0, 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=dtype)

    itn = 0
    istop = 0

    while itn < maxiter:
        itn += 1

        u = _matvec(A, v) - state[_I_ALPHA] * u
        beta_new = torch.linalg.norm(u)
        u = u * torch.where(
            beta_new > 0,
            1.0 / torch.clamp(beta_new, min=_DIV_GUARD),
            beta_new * 0.0,
        )

        v = _rmatvec(At, u) - beta_new * v
        alpha_new = torch.linalg.norm(v)
        v = v * torch.where(
            alpha_new > 0,
            1.0 / torch.clamp(alpha_new, min=_DIV_GUARD),
            alpha_new * 0.0,
        )

        state[_I_BETA] = beta_new
        state[_I_ALPHA] = alpha_new

        out = step_fn(state, consts)

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

        state = out[:_STATE_SIZE]

        normx_t = torch.linalg.norm(x)
        normr_t = out[_O_NORMR]
        normA_t = out[_O_NORMA]
        normb_t = out[_I_NORMB]

        test1_t = normr_t / torch.clamp(normb_t, min=_DIV_GUARD)
        t1_t = test1_t / (
            1.0 + normA_t * normx_t / torch.clamp(normb_t, min=_DIV_GUARD)
        )
        rtol_t = btol + atol * normA_t * normx_t / torch.clamp(normb_t, min=_DIV_GUARD)

        converged_btol = (test1_t <= rtol_t) | (1.0 + t1_t <= 1.0)
        converged_any = (out[_O_CONVERGED] > 0.5) | converged_btol

        if converged_any.item():
            normr_val = normr_t.item()
            normA_val = normA_t.item()
            normx_val = normx_t.item()
            normb_val = normb_t.item()
            normar_val = out[_O_NORMAR].item()
            condA_val = out[_O_CONDA].item()

            test1 = normr_val / max(normb_val, _DIV_GUARD)
            test2 = normar_val / max(normA_val * normr_val, _DIV_GUARD)
            test3 = 1.0 / condA_val
            t1 = test1 / (1.0 + normA_val * normx_val / max(normb_val, _DIV_GUARD))
            _rtol = btol + atol * normA_val * normx_val / max(normb_val, _DIV_GUARD)

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

    if itn == 0:
        return x, istop, 0, normb.item(), normar_init, alpha.item(), 1.0, 0.0

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
