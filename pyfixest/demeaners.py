from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Literal, get_args

MapBackend = Literal["numba", "rust", "jax"]
LsmrBackend = Literal["within", "cupy", "torch"]
LsmrPrecision = Literal["float32", "float64"]
TorchDevice = Literal["auto", "cpu", "mps", "cuda"]
LsmrPreconditioner = Literal["auto", "none", "schwarz", "diag"]


def _validate_unit_interval_float(value: float, name: str) -> None:

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    if value <= 0:
        raise ValueError(f"`{name}` must be strictly positive.")
    if value >= 1:
        raise ValueError(f"`{name}` must be less than one.")


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an int.")
    if value <= 0:
        raise ValueError(f"`{name}` must be strictly positive.")


@dataclass(frozen=True, slots=True)
class BaseDemeaner:
    """Base configuration shared by all fixed-effects demeaners."""

    fixef_maxiter: int = 10_000
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_positive_int(self.fixef_maxiter, "fixef_maxiter")


@dataclass(frozen=True, slots=True)
class MapDemeaner(BaseDemeaner):
    """Method of Alternating Projections (MAP) demeaner."""

    fixef_tol: float = 1e-06
    backend: MapBackend = "rust"
    kind: ClassVar[str] = "map"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        _validate_unit_interval_float(self.fixef_tol, "fixef_tol")
        if not isinstance(self.backend, str):
            raise TypeError("`backend` must be a string.")
        if self.backend not in get_args(MapBackend):
            raise ValueError(f"`backend` must be one of {get_args(MapBackend)}.")


@dataclass(frozen=True, slots=True)
class LsmrDemeaner(BaseDemeaner):
    """Sparse LSMR demeaner.

    Notes
    -----
    The ``within`` backend takes a single tolerance, so ``fixef_atol`` and
    ``fixef_btol`` are collapsed to ``max(fixef_atol, fixef_btol)`` for that
    backend. The ``cupy`` and ``torch`` backends use both tolerances
    independently (SciPy LSMR convention).

    The ``local_size`` field only applies to ``backend="within"``; the
    ``cupy`` and ``torch`` backends ignore it.

    ``preconditioner`` selects the preconditioner. Supported values:

    - ``"auto"`` (default) — pick the natural choice for the backend
      (``"schwarz"`` for ``within``; ``"diag"`` for ``torch`` / ``cupy``).
    - ``"none"`` — disable preconditioning. Supported by ``within`` and
      ``cupy``; not supported by ``torch``.
    - ``"schwarz"`` — additive Schwarz preconditioner. Only supported by the
      ``within`` backend.
    - ``"diag"`` — diagonal (Jacobi) preconditioner. Supported by ``torch``
      and ``cupy``; not supported by ``within``.

    If a value is incompatible with the chosen backend, a ``UserWarning`` is
    emitted at solve time and the backend's natural choice is used.
    """

    fixef_maxiter: int = 1_000
    backend: LsmrBackend = "within"
    precision: LsmrPrecision = "float64"
    device: TorchDevice = "auto"
    fixef_atol: float = 1e-8
    fixef_btol: float = 1e-8
    warn_on_cpu_fallback: bool = True
    preconditioner: LsmrPreconditioner = "auto"
    local_size: int | None = None
    kind: ClassVar[str] = "lsmr"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        if not isinstance(self.backend, str):
            raise TypeError("`backend` must be a string.")
        if self.backend not in get_args(LsmrBackend):
            raise ValueError(f"`backend` must be one of {get_args(LsmrBackend)}.")

        if not isinstance(self.precision, str):
            raise TypeError("`precision` must be a string.")
        if self.precision not in get_args(LsmrPrecision):
            raise ValueError(f"`precision` must be one of {get_args(LsmrPrecision)}.")

        if not isinstance(self.device, str):
            raise TypeError("`device` must be a string.")
        if self.device not in get_args(TorchDevice):
            raise ValueError(f"`device` must be one of {get_args(TorchDevice)}.")
        _validate_unit_interval_float(self.fixef_atol, "fixef_atol")
        _validate_unit_interval_float(self.fixef_btol, "fixef_btol")

        if not isinstance(self.warn_on_cpu_fallback, bool):
            raise TypeError("`warn_on_cpu_fallback` must be a bool.")
        if not isinstance(self.preconditioner, str):
            raise TypeError("`preconditioner` must be a string.")
        if self.preconditioner not in get_args(LsmrPreconditioner):
            raise ValueError(
                f"`preconditioner` must be one of {get_args(LsmrPreconditioner)}."
            )

        if self.local_size is not None:
            _validate_positive_int(self.local_size, "local_size")

        if self.backend == "cupy" and self.device == "mps":
            raise ValueError("The CuPy backend does not support MPS devices.")

        if (
            self.backend == "torch"
            and self.device == "mps"
            and self.precision != "float32"
        ):
            raise ValueError("The MPS torch backend requires `precision='float32'`.")


AnyDemeaner = MapDemeaner | LsmrDemeaner

__all__ = [
    "AnyDemeaner",
    "BaseDemeaner",
    "LsmrBackend",
    "LsmrDemeaner",
    "LsmrPrecision",
    "LsmrPreconditioner",
    "MapBackend",
    "MapDemeaner",
    "TorchDevice",
]
