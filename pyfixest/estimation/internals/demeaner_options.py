from __future__ import annotations

import warnings
from typing import Literal, TypeAlias, cast

from pyfixest.demeaners import (
    AnyDemeaner,
    BaseDemeaner,
    LsmrDemeaner,
    MapDemeaner,
    WithinDemeaner,
)
from pyfixest.estimation.internals.literals import DemeanerBackendOptions

ResolvedDemeaner: TypeAlias = AnyDemeaner

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
    demeaner: BaseDemeaner | None,
    demeaner_backend: DemeanerBackendOptions,
    fixef_tol: float,
    fixef_maxiter: int,
) -> ResolvedDemeaner:
    """Resolve user input into a fully specified demeaner object.

    If a typed ``demeaner`` is supplied, it takes precedence over the legacy
    ``demeaner_backend`` shorthand. The backend argument is only used when no
    typed demeaner configuration is provided.
    """
    backend = normalize_demeaner_backend(demeaner_backend)

    if demeaner is not None:
        if not isinstance(demeaner, _DEMEANER_TYPES):
            raise TypeError("`demeaner` must be a supported pyfixest demeaner.")
        resolved = cast(ResolvedDemeaner, demeaner)
        _warn_if_experimental_torch_demeaner(resolved)
        return resolved

    shorthand = _from_backend_shorthand(
        demeaner_backend=backend,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )
    resolved = cast(ResolvedDemeaner, shorthand)
    _warn_if_experimental_torch_demeaner(resolved)
    return resolved


def get_demeaner_backend(demeaner: ResolvedDemeaner) -> DemeanerBackendOptions:
    """Return the backend key associated with a resolved demeaner."""
    if isinstance(demeaner, MapDemeaner):
        return cast(
            Literal["numba", "rust", "jax"],
            demeaner.backend,
        )
    if isinstance(demeaner, LsmrDemeaner):
        if demeaner.implementation == "torch":
            if demeaner.device == "cpu":
                return "torch_cpu"
            if demeaner.device == "mps":
                return "torch_mps"
            if demeaner.device == "cuda":
                return (
                    "torch_cuda32" if demeaner.precision == "float32" else "torch_cuda"
                )
            return "torch"
        if demeaner.use_gpu is False:
            return "scipy"
        return "cupy32" if demeaner.precision == "float32" else "cupy"
    return "within"


def _from_backend_shorthand(
    demeaner_backend: DemeanerBackendOptions,
    fixef_tol: float,
    fixef_maxiter: int,
) -> ResolvedDemeaner:
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
            implementation="cupy",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float64",
            use_gpu=True,
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "cupy32":
        return LsmrDemeaner(
            implementation="cupy",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float32",
            use_gpu=True,
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "scipy":
        return LsmrDemeaner(
            implementation="cupy",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            use_gpu=False,
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "torch":
        return LsmrDemeaner(
            implementation="torch",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float64",
            device="auto",
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "torch_cpu":
        return LsmrDemeaner(
            implementation="torch",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float64",
            device="cpu",
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "torch_mps":
        return LsmrDemeaner(
            implementation="torch",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float32",
            device="mps",
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "torch_cuda":
        return LsmrDemeaner(
            implementation="torch",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float64",
            device="cuda",
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    if demeaner_backend == "torch_cuda32":
        return LsmrDemeaner(
            implementation="torch",
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float32",
            device="cuda",
            solver_atol=fixef_tol,
            solver_btol=fixef_tol,
            solver_maxiter=fixef_maxiter,
        )
    raise ValueError(f"Unknown demeaner backend {demeaner_backend!r}.")


def _warn_if_experimental_torch_demeaner(demeaner: ResolvedDemeaner) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.implementation == "torch":
        warnings.warn(
            (
                f"The `{get_demeaner_backend(demeaner)}` backend uses experimental "
                "torch algorithms. Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "ResolvedDemeaner",
    "get_demeaner_backend",
    "normalize_demeaner_backend",
    "resolve_demeaner",
]
