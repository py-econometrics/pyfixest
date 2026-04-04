from __future__ import annotations

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
        return cast(ResolvedDemeaner, demeaner)

    shorthand = _from_backend_shorthand(
        demeaner_backend=backend,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
    )
    return cast(ResolvedDemeaner, shorthand)


def get_demeaner_backend(demeaner: ResolvedDemeaner) -> DemeanerBackendOptions:
    """Return the backend key associated with a resolved demeaner."""
    if isinstance(demeaner, MapDemeaner):
        return cast(
            Literal["numba", "rust", "jax"],
            demeaner.backend,
        )
    if isinstance(demeaner, LsmrDemeaner):
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
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float64",
            use_gpu=True,
        )
    if demeaner_backend == "cupy32":
        return LsmrDemeaner(
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            precision="float32",
            use_gpu=True,
        )
    if demeaner_backend == "scipy":
        return LsmrDemeaner(
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            use_gpu=False,
        )
    raise ValueError(f"Unknown demeaner backend {demeaner_backend!r}.")


__all__ = [
    "ResolvedDemeaner",
    "get_demeaner_backend",
    "normalize_demeaner_backend",
    "resolve_demeaner",
]
