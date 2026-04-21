from __future__ import annotations

import warnings
from typing import Literal, cast

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrDemeaner,
    MapDemeaner,
    WithinDemeaner,
)
from pyfixest.estimation.internals.literals import DemeanerBackendOptions

_DEMEANER_TYPES = (
    MapDemeaner,
    WithinDemeaner,
    LsmrDemeaner,
)


def normalize_demeaner_backend(
    demeaner_backend: DemeanerBackendOptions,
) -> DemeanerBackendOptions:
    """Map legacy backend aliases to their canonical demeaner backend."""
    if demeaner_backend == "rust-cg":
        return "within"
    if demeaner_backend == "cupy64":
        return "cupy"
    return demeaner_backend


def resolve_demeaner(
    demeaner: AnyDemeaner | None,
    demeaner_backend: DemeanerBackendOptions,
    fixef_tol: float,
    fixef_maxiter: int,
) -> AnyDemeaner:
    """Resolve user input into a fully specified demeaner object.

    If a typed ``demeaner`` is supplied, it takes precedence over the legacy
    ``demeaner_backend`` shorthand. The backend argument is only used when no
    typed demeaner configuration is provided.
    """
    backend = normalize_demeaner_backend(demeaner_backend)

    if demeaner is not None:
        if not isinstance(demeaner, _DEMEANER_TYPES):
            raise TypeError("`demeaner` must be a supported pyfixest demeaner.")
        if demeaner_backend != "numba":
            warnings.warn(
                (
                    "`demeaner_backend` is ignored when `demeaner` is provided. "
                    "The typed `demeaner` configuration takes precedence."
                ),
                UserWarning,
                stacklevel=3,
            )
        _warn_if_experimental_torch_demeaner(demeaner)
        return demeaner

    shorthand = _from_backend_shorthand(
        demeaner_backend=backend,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )
    _warn_if_experimental_torch_demeaner(shorthand)
    return shorthand


def get_demeaner_backend(demeaner: AnyDemeaner) -> DemeanerBackendOptions:
    """Return the backend key associated with a resolved demeaner."""
    if isinstance(demeaner, MapDemeaner):
        return cast(
            Literal["numba", "rust", "jax"],
            demeaner.backend,
        )
    if isinstance(demeaner, LsmrDemeaner):
        if demeaner.backend == "torch":
            if demeaner.device == "cpu":
                return "torch_cpu"
            if demeaner.device == "mps":
                return "torch_mps"
            if demeaner.device == "cuda":
                return (
                    "torch_cuda32" if demeaner.precision == "float32" else "torch_cuda"
                )
            return "torch"
        if demeaner.device == "cpu":
            return "scipy"
        return "cupy32" if demeaner.precision == "float32" else "cupy"
    return "within"


def get_resolved_fixef_controls(demeaner: AnyDemeaner) -> tuple[float, int]:
    """Return FE control values used by API-level validation and wiring."""
    if isinstance(demeaner, LsmrDemeaner):
        return max(demeaner.fixef_atol, demeaner.fixef_btol), demeaner.fixef_maxiter
    return demeaner.fixef_tol, demeaner.fixef_maxiter


def _from_backend_shorthand(
    demeaner_backend: DemeanerBackendOptions,
    fixef_tol: float,
    fixef_maxiter: int,
) -> AnyDemeaner:
    """Construct a demeaner object from a legacy backend shorthand."""
    if demeaner_backend in {"numba", "rust", "jax"}:
        return MapDemeaner(
            backend=cast(Literal["numba", "rust", "jax"], demeaner_backend),
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "within":
        return WithinDemeaner(fixef_tol=fixef_tol, fixef_maxiter=fixef_maxiter)
    if demeaner_backend == "cupy":
        return LsmrDemeaner(
            backend="cupy",
            precision="float64",
            device="cuda",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "cupy32":
        return LsmrDemeaner(
            backend="cupy",
            precision="float32",
            device="cuda",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "scipy":
        return LsmrDemeaner(
            backend="cupy",
            precision="float64",
            device="cpu",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "torch":
        return LsmrDemeaner(
            backend="torch",
            precision="float64",
            device="auto",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "torch_cpu":
        return LsmrDemeaner(
            backend="torch",
            precision="float64",
            device="cpu",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "torch_mps":
        return LsmrDemeaner(
            backend="torch",
            precision="float32",
            device="mps",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "torch_cuda":
        return LsmrDemeaner(
            backend="torch",
            precision="float64",
            device="cuda",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )
    if demeaner_backend == "torch_cuda32":
        return LsmrDemeaner(
            backend="torch",
            precision="float32",
            device="cuda",
            fixef_maxiter=fixef_maxiter,
            fixef_atol=fixef_tol,
            fixef_btol=fixef_tol,
        )

    raise ValueError(f"Unknown demeaner backend {demeaner_backend!r}.")


def _warn_if_experimental_torch_demeaner(demeaner: AnyDemeaner) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "torch":
        warnings.warn(
            (
                f"The `{get_demeaner_backend(demeaner)}` backend uses experimental "
                "torch algorithms. Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "get_demeaner_backend",
    "get_resolved_fixef_controls",
    "normalize_demeaner_backend",
    "resolve_demeaner",
]
