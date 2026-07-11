"""Normalize legacy and typed demeaner options."""

from __future__ import annotations

import warnings

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrDemeaner,
    MapDemeaner,
)


def _resolve_demeaner(demeaner: AnyDemeaner | None) -> AnyDemeaner:
    """Return the typed demeaner configuration, defaulting to MapDemeaner."""
    return demeaner if demeaner is not None else MapDemeaner()


def _warn_if_experimental_torch_demeaner(demeaner: object) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "torch":
        warnings.warn(
            (
                "The torch LSMR demeaner backend is experimental. "
                "Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=3,
        )


def _warn_if_deprecated_demeaner_backend(demeaner: object) -> None:
    """Warn when the resolved demeaner uses a deprecated cupy/scipy backend."""
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "cupy":
        if demeaner.device == "cpu":
            warnings.warn(
                (
                    "The `scipy` LSMR demeaner backend (LsmrDemeaner with "
                    "backend='cupy' and device='cpu') is deprecated and will "
                    "be removed in a future release. Switch to "
                    "`LsmrDemeaner()` (the default within backend)."
                ),
                DeprecationWarning,
                stacklevel=3,
            )
        elif demeaner.device == "auto":
            warnings.warn(
                (
                    "The `cupy` LSMR demeaner backend is deprecated and will "
                    "be removed in a future release. With device='auto', this "
                    "backend may run on CuPy/CUDA or fall back to SciPy CPU. "
                    "If you were running on GPU, switch to "
                    "`LsmrDemeaner(backend='torch', device='cuda')`; if you "
                    "were running on CPU, switch to "
                    "`LsmrDemeaner()` (the default within backend)."
                ),
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                (
                    "The `cupy` LSMR demeaner backend is deprecated and will "
                    "be removed in a future release. Switch to "
                    "`LsmrDemeaner(backend='torch', device='cuda')` for GPU "
                    "acceleration."
                ),
                DeprecationWarning,
                stacklevel=3,
            )
