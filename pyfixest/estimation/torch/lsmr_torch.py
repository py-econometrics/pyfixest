"""
Public PyTorch LSMR API and dispatchers.

Implementation modules:
- ``_lsmr_single.py`` for single-RHS eager/compiled variants
- ``_lsmr_batched.py`` for multi-RHS eager/compiled variants
- ``_lsmr_helpers.py`` for shared sparse/Givens helpers
- ``_lsmr_compiled_core.py`` for packed-state compiled scalar kernels

Public entry points:
- ``lsmr_torch()`` dispatches: CUDA -> compiled, CPU/MPS -> eager
- ``lsmr_torch_batched()`` dispatches: CUDA -> compiled batched,
  CPU/MPS -> eager batched
"""

from __future__ import annotations

import torch

from pyfixest.estimation.torch._lsmr_batched import (
    _lsmr_batched,
    _lsmr_compiled_batched,
)
from pyfixest.estimation.torch._lsmr_single import _lsmr_compiled, _lsmr_eager


@torch.no_grad()
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
    LSMR solver - unified single-RHS entry point.

    Auto-selects implementation based on device:
    - CUDA: compiled
    - CPU/MPS: eager
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


@torch.no_grad()
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
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Solve K right-hand sides simultaneously via batched LSMR with SpMM."""
    device = B.device
    if use_compile is None:
        use_compile = device.type == "cuda"

    if use_compile:
        return _lsmr_compiled_batched(
            A,
            B,
            damp=damp,
            atol=atol,
            btol=btol,
            conlim=conlim,
            maxiter=maxiter,
            use_compile=True,
        )
    return _lsmr_batched(
        A,
        B,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
    )


__all__ = [
    "_lsmr_batched",
    "_lsmr_compiled",
    "_lsmr_compiled_batched",
    "_lsmr_eager",
    "lsmr_torch",
    "lsmr_torch_batched",
]
