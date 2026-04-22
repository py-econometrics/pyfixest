from __future__ import annotations

import warnings
from typing import cast

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrBackend,
    LsmrDemeaner,
    LsmrPrecision,
    MapBackend,
    MapDemeaner,
    TorchDevice,
    WithinDemeaner,
)

# Legacy string backend → (LsmrBackend, TorchDevice, LsmrPrecision)
_LSMR_PRESETS: dict[str, tuple[LsmrBackend, TorchDevice, LsmrPrecision]] = {
    "cupy": ("cupy", "auto", "float64"),
    "cupy64": ("cupy", "auto", "float64"),
    "cupy32": ("cupy", "auto", "float32"),
    "scipy": ("cupy", "cpu", "float64"),
    "torch": ("torch", "auto", "float64"),
    "torch_cpu": ("torch", "cpu", "float64"),
    "torch_mps": ("torch", "mps", "float32"),
    "torch_cuda": ("torch", "cuda", "float64"),
    "torch_cuda32": ("torch", "cuda", "float32"),
}


def _resolve_demeaner(
    *,
    demeaner: AnyDemeaner | None,
    demeaner_backend: str | None = None,
    fixef_tol: float | None = None,
    fixef_maxiter: int | None = None,
) -> AnyDemeaner:
    legacy_args_used = any(
        arg is not None for arg in (demeaner_backend, fixef_tol, fixef_maxiter)
    )

    if demeaner is not None:
        if legacy_args_used:
            raise ValueError(
                "Pass either `demeaner` or the deprecated legacy arguments "
                "`demeaner_backend`, `fixef_tol`, and `fixef_maxiter`, not both."
            )
        return demeaner

    if not legacy_args_used:
        return MapDemeaner()

    if fixef_tol is not None:
        if isinstance(fixef_tol, bool) or not isinstance(fixef_tol, float):
            raise TypeError("fixef_tol must be a float")
        if fixef_tol <= 0 or fixef_tol >= 1:
            raise ValueError("fixef_tol must be greater than zero and less than one")

    if fixef_maxiter is not None:
        if isinstance(fixef_maxiter, bool) or not isinstance(fixef_maxiter, int):
            raise TypeError("fixef_maxiter must be an int")
        if fixef_maxiter <= 0:
            raise ValueError("fixef_maxiter must be greater than zero")

    warnings.warn(
        (
            "The `demeaner_backend`, `fixef_tol`, and `fixef_maxiter` arguments "
            "are deprecated and will be removed in a future release. "
            "Pass a typed `demeaner=` configuration instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )

    backend = "numba" if demeaner_backend is None else demeaner_backend
    effective_fixef_tol = 1e-06 if fixef_tol is None else fixef_tol
    effective_fixef_maxiter = 10_000 if fixef_maxiter is None else fixef_maxiter

    if backend in {"numba", "rust", "jax"}:
        return MapDemeaner(
            backend=cast(MapBackend, backend),
            fixef_tol=effective_fixef_tol,
            fixef_maxiter=effective_fixef_maxiter,
        )

    if backend in {"rust-cg", "within"}:
        return WithinDemeaner(
            fixef_tol=effective_fixef_tol,
            fixef_maxiter=effective_fixef_maxiter,
        )

    if backend in _LSMR_PRESETS:
        lsmr_backend, device, precision = _LSMR_PRESETS[backend]
        return LsmrDemeaner(
            backend=lsmr_backend,
            device=device,
            precision=precision,
            fixef_atol=effective_fixef_tol,
            fixef_btol=effective_fixef_tol,
            fixef_maxiter=effective_fixef_maxiter,
        )

    raise ValueError(f"Invalid demeaner backend: {backend}")


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
