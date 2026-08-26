"""
FWL demeaning via LSMR in pure PyTorch.

Builds a sparse dummy matrix D directly from integer-encoded fixed effects
(no formulaic/pandas dependency), then solves D @ theta = x via LSMR per column.
Diagonal preconditioning normalizes column norms when group sizes vary.

Sparse format is chosen per device: CSR on CUDA/CPU (cuSPARSE / native),
COO on MPS (Metal does not support sparse CSR).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from pyfixest.estimation.torch._preconditioned_sparse import _PreconditionedSparse
from pyfixest.estimation.torch._sparse_dummy import (
    _build_sparse_dummy,
    _scale_sparse_rows,
)
from pyfixest.estimation.torch.lsmr_torch import lsmr_torch, lsmr_torch_batched

# Minimum K (number of RHS columns) for batched SpMM to beat sequential SpMV.
# Benchmarked breakeven is device-specific.
_BATCHED_K_THRESHOLD_CUDA = 2
_BATCHED_K_THRESHOLD_MPS = 5


def _lsmr_istop_is_success(istop: int) -> bool:
    """Treat standard LSMR stopping codes 1-6 as successful termination."""
    return 1 <= int(istop) <= 6


def _should_use_batched_lsmr(device: torch.device, K: int) -> bool:
    """Use batched LSMR only when device-specific benchmarks show a benefit."""
    if device.type == "cuda":
        return K >= _BATCHED_K_THRESHOLD_CUDA
    if device.type == "mps":
        return K >= _BATCHED_K_THRESHOLD_MPS
    return False


def _get_device(dtype: torch.dtype = torch.float64) -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU.

    MPS does not support float64, so we fall back to CPU when float64 is needed.
    When MPS is available but dtype is float64, a hint is issued to use float32.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if dtype != torch.float64:
            return torch.device("mps")
        warnings.warn(
            "MPS GPU is available but requires float32. "
            "Pass `dtype=torch.float32` to `demean_torch` for GPU acceleration. "
            "Falling back to CPU.",
            UserWarning,
            stacklevel=3,
        )
        return torch.device("cpu")
    warnings.warn(
        "No GPU available — torch demeaning will run on CPU, which is slower "
        "than the scipy backend. Consider using `demean_scipy` instead.",
        UserWarning,
        stacklevel=3,
    )
    return torch.device("cpu")


@torch.no_grad()
def _demean_torch_on_device_impl(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64],
    weights: NDArray[np.float64] | None,
    tol: float,
    maxiter: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[NDArray[np.float64], bool]:
    """Inner implementation wrapped in torch.no_grad() to skip autograd overhead."""
    if flist is None:
        raise ValueError("flist cannot be None")
    if weights is None:
        weights = np.ones(x.shape[0], dtype=np.float64)

    # Track original shape to restore on output
    was_1d = x.ndim == 1
    x_2d = x[:, None] if was_1d else x
    weights_1d = weights.ravel()

    # Move to torch — use from_numpy for zero-copy when staying on CPU
    x_c = x_2d if x_2d.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x_2d)
    w_c = (
        weights_1d
        if weights_1d.flags["C_CONTIGUOUS"]
        else np.ascontiguousarray(weights_1d)
    )
    x_t = torch.from_numpy(x_c).to(dtype=dtype, device=device)
    w_t = torch.from_numpy(w_c).to(dtype=dtype, device=device)

    # Ensure flist is 2D
    flist_2d = flist if flist.ndim == 2 else flist[:, None]

    # Build sparse dummy matrix (unweighted)
    D_unweighted = _build_sparse_dummy(flist_2d, device, dtype)
    _, D_cols = D_unweighted.shape
    K = x_t.shape[1]

    # Apply sqrt-weights
    sqrt_w = torch.sqrt(w_t)
    x_w = x_t * sqrt_w[:, None]

    # Weight the dummy matrix: D_weighted = diag(sqrt_w) @ D_unweighted
    D_weighted = _scale_sparse_rows(D_unweighted, sqrt_w)

    # Diagonal preconditioning: M_inv = 1 / sqrt(D_unweighted^T @ w)
    # D_unweighted^T @ w gives the sum of weights per group
    group_weights = D_unweighted.t() @ w_t

    # Guard against zero-weight groups (would produce inf in M_inv)
    zero_weight_mask = group_weights <= 0.0
    if zero_weight_mask.any():
        bad_groups = zero_weight_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        raise ValueError(
            f"Fixed effect groups {bad_groups} have zero total weight. "
            f"Check your weights or fixed effect encoding."
        )

    M_inv = 1.0 / torch.sqrt(group_weights)

    # Build preconditioned operator once (not per column)
    A_precond = _PreconditionedSparse(D_weighted, M_inv)

    # Solve for each column — batched SpMM for K >= threshold, sequential otherwise.
    theta = torch.zeros(D_cols, K, dtype=dtype, device=device)

    if _should_use_batched_lsmr(device, K):
        # Batched: single call, K columns solved simultaneously via SpMM
        Z, istop_vec, _itn, *_ = lsmr_torch_batched(
            A_precond,
            x_w,
            damp=0.0,
            atol=tol,
            btol=tol,
            maxiter=maxiter,
        )
        theta = M_inv.unsqueeze(1) * Z
        success = all(_lsmr_istop_is_success(code) for code in istop_vec.tolist())
    else:
        # Sequential: K < threshold, per-column single-RHS path is faster
        success = True
        for k in range(K):
            z, istop, _itn, _normr, _normar, _normA, _condA, _normx = lsmr_torch(
                A_precond,
                x_w[:, k],
                damp=0.0,
                atol=tol,
                btol=tol,
                maxiter=maxiter,
            )
            theta[:, k] = M_inv * z
            success = success and _lsmr_istop_is_success(istop)

    # Compute residuals: x_demeaned = x - D_unweighted @ theta
    x_demeaned = x_t - D_unweighted @ theta

    result = x_demeaned.cpu().numpy()
    if was_1d:
        result = result[:, 0]

    return result, success


def demean_torch(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64] | None = None,
    weights: NDArray[np.float64] | None = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
    dtype: torch.dtype = torch.float64,
) -> tuple[NDArray[np.float64], bool]:
    """
    Demean x by projecting out fixed effects via FWL + LSMR.

    Auto-detects the best available device (CUDA > MPS > CPU).
    For explicit device control, use the device-specific variants:
    `demean_torch_cpu`, `demean_torch_mps`, `demean_torch_cuda`,
    `demean_torch_cuda32`.

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N, K)
        Variables to demean.
    flist : np.ndarray, shape (N, n_factors), dtype uint64
        Integer-encoded fixed effects.
    weights : np.ndarray, shape (N,)
        Observation weights (1.0 for equal weighting).
    tol : float
        Convergence tolerance for LSMR (used as both atol and btol).
    maxiter : int
        Maximum LSMR iterations per column.
    dtype : torch.dtype
        Tensor dtype. Use ``torch.float32`` to enable MPS (Apple GPU)
        acceleration. Default ``torch.float64`` for full precision.

    Returns
    -------
    x_demeaned : np.ndarray
        Residuals after projecting out fixed effects. Same shape as input x.
    success : bool
        True if LSMR converged for all columns.
    """
    if flist is None:
        raise ValueError("flist cannot be None")
    if weights is None:
        weights = np.ones(x.shape[0], dtype=np.float64)

    device = _get_device(dtype)
    return _demean_torch_on_device_impl(x, flist, weights, tol, maxiter, device, dtype)


def _make_demean_variant(
    device_str: str,
    dtype: torch.dtype,
    doc: str,
):
    """Create a device-specific demean wrapper."""

    def _demean(
        x: NDArray[np.float64],
        flist: NDArray[np.uint64] | None = None,
        weights: NDArray[np.float64] | None = None,
        tol: float = 1e-8,
        maxiter: int = 100_000,
    ) -> tuple[NDArray[np.float64], bool]:
        return _demean_torch_on_device_impl(
            x,
            flist,
            weights,
            tol,
            maxiter,
            device=torch.device(device_str),
            dtype=dtype,
        )

    _demean.__doc__ = doc
    _demean.__qualname__ = f"demean_torch_{device_str}"
    return _demean


demean_torch_cpu = _make_demean_variant(
    "cpu", torch.float64, "Torch demeaner on CPU, float64."
)
demean_torch_mps = _make_demean_variant(
    "mps", torch.float32, "Torch demeaner on MPS (Apple GPU), float32."
)
demean_torch_cuda = _make_demean_variant(
    "cuda", torch.float64, "Torch demeaner on CUDA GPU, float64."
)
demean_torch_cuda32 = _make_demean_variant(
    "cuda", torch.float32, "Torch demeaner on CUDA GPU, float32."
)
