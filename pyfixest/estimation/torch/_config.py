"""Shared configuration for the PyFixest Torch demeaner backends."""

from __future__ import annotations

import warnings

import torch

# Minimum K (number of RHS columns) for batched SpMM to beat sequential SpMV.
# Benchmarked breakeven is device-specific.
_BATCHED_K_THRESHOLD_CUDA: int = 2
_BATCHED_K_THRESHOLD_MPS: int = 5


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
