from __future__ import annotations

import torch

from pyfixest.estimation.torch._lsmr_compiled_core import (
    _DIV_GUARD,
    _I_ALPHA,
    _I_BETA,
    _I_NORMB,
    _I_RHO,
    _I_RHOBAR,
    _O_CONDA,
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
    _check_convergence_batched,
    _mark_maxiter_batched,
    _matvec_batched,
    _precompute_transpose,
    _rmatvec_batched,
    _safe_normalize_cols,
    _sym_ortho_vec,
)


def _lsmr_batched(
    A,
    B: torch.Tensor,
    damp: float = 0.0,
    atol: float = 1e-8,
    btol: float = 1e-8,
    conlim: float = 1e8,
    maxiter: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Batched LSMR: solve min ||B - A X||_F for K RHS simultaneously.

    Replaces K sequential SpMV with SpMM for GPU throughput.
    """
    m, n = A.shape
    K = B.shape[1]
    device = B.device
    dtype = B.dtype

    if maxiter is None:
        maxiter = min(m, n)

    At = _precompute_transpose(A)

    U = B.clone()
    normb = torch.linalg.norm(U, dim=0)

    X = torch.zeros(n, K, device=device, dtype=dtype)
    beta = normb.clone()

    U, beta = _safe_normalize_cols(U, beta)

    V = _rmatvec_batched(At, U)
    alpha = torch.linalg.norm(V, dim=0)

    V, alpha = _safe_normalize_cols(V, alpha)

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha.clone()
    rho = torch.ones(K, device=device, dtype=dtype)
    rhobar = torch.ones(K, device=device, dtype=dtype)
    cbar = torch.ones(K, device=device, dtype=dtype)
    sbar = torch.zeros(K, device=device, dtype=dtype)

    H = V.clone()
    Hbar = torch.zeros(n, K, device=device, dtype=dtype)

    betadd = beta.clone()
    betad = torch.zeros(K, device=device, dtype=dtype)
    rhodold = torch.ones(K, device=device, dtype=dtype)
    tautildeold = torch.zeros(K, device=device, dtype=dtype)
    thetatilde = torch.zeros(K, device=device, dtype=dtype)
    zeta = torch.zeros(K, device=device, dtype=dtype)
    d = torch.zeros(K, device=device, dtype=dtype)

    normA2 = alpha * alpha
    maxrbar = torch.zeros(K, device=device, dtype=dtype)
    minrbar_init = 1e100 if dtype == torch.float64 else 1e10
    minrbar = torch.full((K,), minrbar_init, device=device, dtype=dtype)

    istop = torch.zeros(K, device=device, dtype=torch.long)
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normr = beta.clone()
    normar = alpha * beta

    damp_vec = torch.full((K,), damp, device=device, dtype=dtype)

    if (normar == 0).all():
        return (
            X,
            istop,
            itn,
            normr,
            normar,
            torch.sqrt(normA2),
            torch.ones(K, device=device, dtype=dtype),
            torch.zeros(K, device=device, dtype=dtype),
        )

    if (normb == 0).all():
        X.zero_()
        return (
            X,
            istop,
            itn,
            normr,
            normar,
            torch.sqrt(normA2),
            torch.ones(K, device=device, dtype=dtype),
            torch.zeros(K, device=device, dtype=dtype),
        )

    while itn < maxiter:
        itn += 1

        U = _matvec_batched(A, V) - alpha.unsqueeze(0) * U
        beta = torch.linalg.norm(U, dim=0)
        U, beta = _safe_normalize_cols(U, beta)

        V = _rmatvec_batched(At, U) - beta.unsqueeze(0) * V
        alpha = torch.linalg.norm(V, dim=0)
        V, alpha = _safe_normalize_cols(V, alpha)

        chat, shat, alphahat = _sym_ortho_vec(alphabar, damp_vec)

        rhoold = rho
        c, s, rho = _sym_ortho_vec(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho_vec(rhotemp, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar_coeff = -(thetabar * rho) / torch.clamp(rhoold * rhobarold, min=_DIV_GUARD)
        Hbar = H + Hbar * hbar_coeff.unsqueeze(0)
        x_coeff = zeta / torch.clamp(rho * rhobar, min=_DIV_GUARD)
        X = X + x_coeff.unsqueeze(0) * Hbar
        h_coeff = -(thetanew / torch.clamp(rho, min=_DIV_GUARD))
        H = V + H * h_coeff.unsqueeze(0)

        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho_vec(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        safe_rhotildeold = torch.clamp(rhotildeold, min=_DIV_GUARD)
        tautildeold = (zetaold - thetatildeold * tautildeold) / safe_rhotildeold
        safe_rhodold = torch.clamp(rhodold, min=_DIV_GUARD)
        taud = (zeta - thetatilde * tautildeold) / safe_rhodold
        d = d + betacheck * betacheck
        normr = torch.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        normA2 = normA2 + beta * beta
        normA = torch.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        maxrbar = torch.maximum(maxrbar, rhobarold)
        if itn > 1:
            minrbar = torch.minimum(minrbar, rhobarold)
        condA = torch.maximum(maxrbar, rhotemp) / torch.clamp(
            torch.minimum(minrbar, rhotemp), min=_DIV_GUARD
        )

        normar = torch.abs(zetabar)
        normx = torch.linalg.norm(X, dim=0)

        safe_normb = torch.clamp(normb, min=_DIV_GUARD)
        test1 = normr / safe_normb
        safe_normA_normr = torch.clamp(normA * normr, min=_DIV_GUARD)
        test2 = normar / safe_normA_normr
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / safe_normb)
        rtol = btol + atol * normA * normx / safe_normb

        istop = _check_convergence_batched(
            istop,
            test1,
            rtol,
            test2,
            test3,
            t1,
            atol,
            ctol,
            K,
            device,
        )
        if (istop > 0).all():
            break

    istop = _mark_maxiter_batched(istop, itn, maxiter)

    return X, istop, itn, normr, normar, normA, condA, normx


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
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compiled batched LSMR: solve min ||B - A X||_F for K RHS simultaneously.

    Mirrors ``_lsmr_compiled`` but with (m, K) vectors and SpMM.
    """
    m, n = A.shape
    K = B.shape[1]
    device = B.device
    dtype = B.dtype

    if maxiter is None:
        maxiter = min(m, n)

    step_fn = _get_compiled_step(device.type) if use_compile else _scalar_step

    At = _precompute_transpose(A)

    U = B.clone()
    normb = torch.linalg.norm(U, dim=0)

    X = torch.zeros(n, K, device=device, dtype=dtype)
    beta = normb.clone()

    U, beta = _safe_normalize_cols(U, beta)

    V = _rmatvec_batched(At, U)
    alpha = torch.linalg.norm(V, dim=0)
    V, alpha = _safe_normalize_cols(V, alpha)

    state = _make_initial_state(alpha, beta, normb, damp, dtype, device, K=K)

    ctol = 1.0 / conlim if conlim > 0 else 0.0
    consts = torch.tensor([atol, btol, ctol], device=device, dtype=dtype)

    normar_init = alpha * beta
    if (normar_init == 0).all():
        return (
            X,
            torch.zeros(K, device=device, dtype=torch.long),
            0,
            beta,
            torch.zeros(K, device=device, dtype=dtype),
            alpha,
            torch.ones(K, device=device, dtype=dtype),
            torch.zeros(K, device=device, dtype=dtype),
        )
    if (normb == 0).all():
        X.zero_()
        return (
            X,
            torch.zeros(K, device=device, dtype=torch.long),
            0,
            beta,
            torch.zeros(K, device=device, dtype=dtype),
            alpha,
            torch.ones(K, device=device, dtype=dtype),
            torch.zeros(K, device=device, dtype=dtype),
        )

    H = V.clone()
    Hbar = torch.zeros(n, K, device=device, dtype=dtype)

    istop = torch.zeros(K, device=device, dtype=torch.long)

    itn = 0
    normx_t = torch.zeros(K, device=device, dtype=dtype)

    while itn < maxiter:
        itn += 1

        U = _matvec_batched(A, V) - state[_I_ALPHA].unsqueeze(0) * U
        beta_new = torch.linalg.norm(U, dim=0)
        U, beta_new = _safe_normalize_cols(U, beta_new)

        V = _rmatvec_batched(At, U) - beta_new.unsqueeze(0) * V
        alpha_new = torch.linalg.norm(V, dim=0)
        V, alpha_new = _safe_normalize_cols(V, alpha_new)

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

        hbar_coeff = -(thetabar * rho_new) / torch.clamp(
            rhoold * rhobarold, min=_DIV_GUARD
        )
        Hbar = H + Hbar * hbar_coeff.unsqueeze(0)
        x_coeff = zeta / torch.clamp(rho_new * rhobar_new, min=_DIV_GUARD)
        X = X + x_coeff.unsqueeze(0) * Hbar
        h_coeff = -(thetanew / torch.clamp(rho_new, min=_DIV_GUARD))
        H = V + H * h_coeff.unsqueeze(0)

        state = out[:_STATE_SIZE]

        normx_t = torch.linalg.norm(X, dim=0)
        normr_t = out[_O_NORMR]
        normA_t = out[_O_NORMA]
        normb_t = out[_I_NORMB]

        safe_normb = torch.clamp(normb_t, min=_DIV_GUARD)
        test1_t = normr_t / safe_normb
        safe_normA_normr = torch.clamp(normA_t * normr_t, min=_DIV_GUARD)
        test2_t = out[_O_NORMAR] / safe_normA_normr
        test3_t = 1.0 / out[_O_CONDA]
        t1_t = test1_t / (1.0 + normA_t * normx_t / safe_normb)
        rtol_t = btol + atol * normA_t * normx_t / safe_normb

        istop = _check_convergence_batched(
            istop,
            test1_t,
            rtol_t,
            test2_t,
            test3_t,
            t1_t,
            atol,
            ctol,
            K,
            device,
        )

        if (istop > 0).all().item():
            break

    istop = _mark_maxiter_batched(istop, itn, maxiter)

    if itn == 0:
        return (
            X,
            istop,
            0,
            normb,
            normar_init,
            alpha,
            torch.ones(K, device=device, dtype=dtype),
            torch.zeros(K, device=device, dtype=dtype),
        )

    return (
        X,
        istop,
        itn,
        out[_O_NORMR],
        out[_O_NORMAR],
        out[_O_NORMA],
        out[_O_CONDA],
        normx_t,
    )
