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


def _build_map_demeaner(
    backend: MapBackend,
    *,
    fixef_tol: float | None,
    fixef_maxiter: int | None,
) -> MapDemeaner:
    if fixef_tol is None and fixef_maxiter is None:
        return MapDemeaner(backend=backend)
    if fixef_tol is None:
        assert fixef_maxiter is not None
        return MapDemeaner(backend=backend, fixef_maxiter=fixef_maxiter)
    if fixef_maxiter is None:
        return MapDemeaner(backend=backend, fixef_tol=fixef_tol)
    return MapDemeaner(
        backend=backend,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )


def _build_within_demeaner(
    *,
    fixef_tol: float | None,
    fixef_maxiter: int | None,
) -> WithinDemeaner:
    if fixef_tol is None and fixef_maxiter is None:
        return WithinDemeaner()
    if fixef_tol is None:
        assert fixef_maxiter is not None
        return WithinDemeaner(fixef_maxiter=fixef_maxiter)
    if fixef_maxiter is None:
        return WithinDemeaner(fixef_tol=fixef_tol)
    return WithinDemeaner(
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )


def _build_lsmr_demeaner(
    *,
    backend: LsmrBackend,
    device: TorchDevice,
    precision: LsmrPrecision,
    fixef_tol: float | None,
    fixef_maxiter: int | None,
) -> LsmrDemeaner:
    if fixef_tol is None and fixef_maxiter is None:
        return LsmrDemeaner(backend=backend, device=device, precision=precision)
    if fixef_tol is None:
        assert fixef_maxiter is not None
        return LsmrDemeaner(
            backend=backend,
            device=device,
            precision=precision,
            fixef_maxiter=fixef_maxiter,
        )
    if fixef_maxiter is None:
        return LsmrDemeaner(
            backend=backend,
            device=device,
            precision=precision,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    return LsmrDemeaner(
        backend=backend,
        device=device,
        precision=precision,
        fixef_atol=fixef_tol,
        fixef_btol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )


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
                "Please either pass a typed `demeaner` or the deprecated legacy arguments "
                "`demeaner_backend`, `fixef_tol`, and `fixef_maxiter`, not both. Here we prioritize the typed `demeaner` argument and ignore the legacy arguments, but in a future release passing legacy arguments with a typed `demeaner` will raise an error."
                ""
            )
        return demeaner

    if not legacy_args_used:
        return MapDemeaner()

    if fixef_tol is not None:
        if isinstance(fixef_tol, bool) or not isinstance(fixef_tol, float):
            raise TypeError("fixef_tol must be a float")
        if fixef_tol <= 0 or fixef_tol >= 1:
            raise ValueError("fixef_tol must be greater than zero and less than one")

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

    if backend in {"numba", "rust", "jax"}:
        return _build_map_demeaner(
            cast(MapBackend, backend),
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )

    if backend in {"rust-cg", "within"}:
        return _build_within_demeaner(
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )

    if backend in {"cupy", "cupy64"}:
        return _build_lsmr_demeaner(
            backend="cupy",
            device="cuda",
            precision="float64",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "cupy32":
        return _build_lsmr_demeaner(
            backend="cupy",
            device="cuda",
            precision="float32",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "scipy":
        return _build_lsmr_demeaner(
            backend="cupy",
            device="cpu",
            precision="float64",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "torch":
        return _build_lsmr_demeaner(
            backend="torch",
            device="auto",
            precision="float64",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "torch_cpu":
        return _build_lsmr_demeaner(
            backend="torch",
            device="cpu",
            precision="float64",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "torch_mps":
        return _build_lsmr_demeaner(
            backend="torch",
            device="mps",
            precision="float32",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "torch_cuda":
        return _build_lsmr_demeaner(
            backend="torch",
            device="cuda",
            precision="float64",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if backend == "torch_cuda32":
        return _build_lsmr_demeaner(
            backend="torch",
            device="cuda",
            precision="float32",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
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
