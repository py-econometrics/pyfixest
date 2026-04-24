from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Literal, get_args

MapBackend = Literal["numba", "rust", "jax"]
LsmrBackend = Literal["cupy", "torch"]
LsmrPrecision = Literal["float32", "float64"]
TorchDevice = Literal["auto", "cpu", "mps", "cuda"]
WithinKrylov = Literal["cg", "gmres"]
WithinPreconditioner = Literal["additive", "multiplicative"]


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
    """Alternating-projections demeaner with selectable implementation backend."""

    fixef_tol: float = 1e-06
    backend: MapBackend = "numba"
    kind: ClassVar[str] = "map"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        _validate_unit_interval_float(self.fixef_tol, "fixef_tol")
        if not isinstance(self.backend, str):
            raise TypeError("`backend` must be a string.")
        if self.backend not in get_args(MapBackend):
            raise ValueError(f"`backend` must be one of {get_args(MapBackend)}.")


@dataclass(frozen=True, slots=True)
class WithinDemeaner(BaseDemeaner):
    """Demeaner configuration for the Rust `within` backend."""

    fixef_tol: float = 1e-06
    fixef_maxiter: int = 1_000
    krylov: WithinKrylov = "cg"
    preconditioner: WithinPreconditioner = "additive"
    gmres_restart: int = 30
    kind: ClassVar[str] = "within"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        _validate_unit_interval_float(self.fixef_tol, "fixef_tol")
        if not isinstance(self.krylov, str):
            raise TypeError("`krylov` must be a string.")
        if self.krylov not in get_args(WithinKrylov):
            raise ValueError(f"`krylov` must be one of {get_args(WithinKrylov)}.")

        if not isinstance(self.preconditioner, str):
            raise TypeError("`preconditioner` must be a string.")
        if self.preconditioner not in get_args(WithinPreconditioner):
            raise ValueError(
                f"`preconditioner` must be one of {get_args(WithinPreconditioner)}."
            )

        _validate_positive_int(self.gmres_restart, "gmres_restart")

        if self.krylov == "cg" and self.preconditioner == "multiplicative":
            raise ValueError(
                "`preconditioner='multiplicative'` requires `krylov='gmres'`."
            )


@dataclass(frozen=True, slots=True)
class LsmrDemeaner(BaseDemeaner):
    """Sparse LSMR demeaner for CuPy/SciPy and PyTorch backends."""

    backend: LsmrBackend = "cupy"
    precision: LsmrPrecision = "float64"
    device: TorchDevice = "auto"
    fixef_atol: float = 1e-8
    fixef_btol: float = 1e-8
    warn_on_cpu_fallback: bool = True
    use_preconditioner: bool = True
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
        if not isinstance(self.use_preconditioner, bool):
            raise TypeError("`use_preconditioner` must be a bool.")

        if self.backend == "cupy" and self.device == "mps":
            raise ValueError("The CuPy backend does not support MPS devices.")

        if self.backend == "torch":
            if self.device == "mps" and self.precision != "float32":
                raise ValueError(
                    "The MPS torch backend requires `precision='float32'`."
                )
            if not self.use_preconditioner:
                raise ValueError(
                    "The torch LSMR backend currently always uses preconditioning."
                )


AnyDemeaner = MapDemeaner | WithinDemeaner | LsmrDemeaner

__all__ = [
    "AnyDemeaner",
    "BaseDemeaner",
    "LsmrBackend",
    "LsmrDemeaner",
    "LsmrPrecision",
    "MapBackend",
    "MapDemeaner",
    "TorchDevice",
    "WithinDemeaner",
    "WithinKrylov",
    "WithinPreconditioner",
]
