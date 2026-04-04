from __future__ import annotations

import math

import torch


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
    return A.t()


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """
    Stable Givens rotation (SymOrtho).

    Given scalars a and b, compute c, s, r such that:
        [ c  s ] [ a ] = [ r ]
        [-s  c ] [ b ]   [ 0 ]
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


def _matvec_batched(A, V: torch.Tensor) -> torch.Tensor:
    """Compute A @ V where V is (n, K). Use SpMM for sparse A, mm() for wrappers."""
    if isinstance(A, torch.Tensor):
        return A @ V
    return A.mm(V)


def _rmatvec_batched(At, U: torch.Tensor) -> torch.Tensor:
    """A^T @ U where U is (m, K). SpMM."""
    if isinstance(At, torch.Tensor):
        return At @ U
    return At.mm(U)


def _sym_ortho_vec(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stable Givens rotation (SymOrtho) vectorized over K columns.

    Given (K,) tensors a and b, compute c, s, r such that for each k:
        [ c_k  s_k ] [ a_k ] = [ r_k ]
        [-s_k  c_k ] [ b_k ]   [ 0   ]
    """
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)

    safe_a = torch.where(a != 0, a, one)
    safe_b = torch.where(b != 0, b, one)

    c_b0 = torch.where(a == 0, zero, torch.sign(a))
    s_b0 = zero
    r_b0 = abs_a

    c_a0 = zero
    s_a0 = torch.sign(b)
    r_a0 = abs_b

    tau_3 = a / safe_b
    s_3 = torch.sign(b) / torch.sqrt(one + tau_3 * tau_3)
    safe_s_3 = torch.where(s_3 != 0, s_3, one)
    c_3 = s_3 * tau_3
    r_3 = b / safe_s_3

    tau_4 = b / safe_a
    c_4 = torch.sign(a) / torch.sqrt(one + tau_4 * tau_4)
    safe_c_4 = torch.where(c_4 != 0, c_4, one)
    s_4 = c_4 * tau_4
    r_4 = a / safe_c_4

    is_b0 = b == 0
    is_a0 = a == 0
    is_b_gt_a = abs_b > abs_a

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


def _safe_normalize_cols(
    M: torch.Tensor, norms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divide each column of (m, K) matrix ``M`` by its (K,) ``norms``,
    zeroing columns where ``norms == 0``.
    """
    nonzero = norms > 0
    safe = torch.where(nonzero, norms, torch.ones_like(norms))
    mask = nonzero.unsqueeze(0).to(M.dtype)
    M = (M / safe.unsqueeze(0)) * mask
    return M, norms


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
    """Per-column convergence check for batched LSMR."""
    not_yet = istop == 0
    new_stop = torch.zeros(K, device=device, dtype=torch.long)
    new_stop = torch.where(test1 <= rtol, torch.ones_like(new_stop), new_stop)
    new_stop = torch.where(
        (test2 <= atol) & (new_stop == 0),
        2 * torch.ones_like(new_stop),
        new_stop,
    )
    new_stop = torch.where(
        (test3 <= ctol) & (new_stop == 0),
        3 * torch.ones_like(new_stop),
        new_stop,
    )
    new_stop = torch.where(
        (1.0 + t1 <= 1.0) & (new_stop == 0),
        4 * torch.ones_like(new_stop),
        new_stop,
    )
    new_stop = torch.where(
        (1.0 + test2 <= 1.0) & (new_stop == 0),
        5 * torch.ones_like(new_stop),
        new_stop,
    )
    new_stop = torch.where(
        (1.0 + test3 <= 1.0) & (new_stop == 0),
        6 * torch.ones_like(new_stop),
        new_stop,
    )
    return torch.where(not_yet, new_stop, istop)


def _mark_maxiter_batched(istop: torch.Tensor, itn: int, maxiter: int) -> torch.Tensor:
    """Set istop=7 for columns that did not converge before maxiter."""
    return torch.where(
        (istop == 0) & (itn >= maxiter),
        7 * torch.ones_like(istop),
        istop,
    )
