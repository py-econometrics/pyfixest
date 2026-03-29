from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Literal


def _validate_positive_float(value: float, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if value <= 0:
        raise ValueError(f"`{name}` must be strictly positive.")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int.")
    if value <= 0:
        raise ValueError(f"`{name}` must be strictly positive.")
    return value


@dataclass(frozen=True, slots=True)
class BaseDemeaner:
    """Base configuration shared by all fixed-effects demeaners."""

    fixef_tol: float = 1e-06
    fixef_maxiter: int = 10_000
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixef_tol",
            _validate_positive_float(self.fixef_tol, "fixef_tol"),
        )
        object.__setattr__(
            self,
            "fixef_maxiter",
            _validate_positive_int(self.fixef_maxiter, "fixef_maxiter"),
        )


@dataclass(frozen=True, slots=True)
class MapDemeaner(BaseDemeaner):
    """Alternating-projections demeaner with selectable implementation backend."""

    backend: Literal["numba", "rust", "jax"] = "numba"
    kind: ClassVar[str] = "map"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        if not isinstance(self.backend, str):
            raise TypeError("`backend` must be a string.")
        backend = self.backend.lower()
        if backend not in {"numba", "rust", "jax"}:
            raise ValueError("`backend` must be one of 'numba', 'rust', or 'jax'.")
        object.__setattr__(self, "backend", backend)


@dataclass(frozen=True, slots=True)
class WithinDemeaner(BaseDemeaner):
    """Krylov-based demeaner configuration for the Rust `within` backend."""

    krylov_method: str = "cg"
    gmres_restart: int = 30
    preconditioner_type: str = "additive"
    kind: ClassVar[str] = "within"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        if not isinstance(self.krylov_method, str):
            raise TypeError("`krylov_method` must be a string.")
        krylov_method = self.krylov_method.lower()
        if krylov_method not in {"cg", "gmres"}:
            raise ValueError("`krylov_method` must be either 'cg' or 'gmres'.")
        object.__setattr__(self, "krylov_method", krylov_method)

        gmres_restart = _validate_positive_int(self.gmres_restart, "gmres_restart")
        object.__setattr__(self, "gmres_restart", gmres_restart)

        if not isinstance(self.preconditioner_type, str):
            raise TypeError("`preconditioner_type` must be a string.")
        preconditioner_type = self.preconditioner_type.lower()
        if preconditioner_type not in {"additive", "multiplicative"}:
            raise ValueError(
                "`preconditioner_type` must be either 'additive' or 'multiplicative'."
            )
        if preconditioner_type == "multiplicative" and krylov_method != "gmres":
            raise ValueError("Multiplicative Schwarz requires `krylov_method='gmres'`.")
        object.__setattr__(self, "preconditioner_type", preconditioner_type)


@dataclass(frozen=True, slots=True)
class LsmrDemeaner(BaseDemeaner):
    """Sparse LSMR demeaner for CPU and GPU backends."""

    precision: str = "float64"
    use_gpu: bool | None = None
    solver_atol: float = 1e-8
    solver_btol: float = 1e-8
    solver_maxiter: int = 100_000
    warn_on_cpu_fallback: bool = True
    use_preconditioner: bool = True
    kind: ClassVar[str] = "lsmr"

    def __post_init__(self) -> None:
        BaseDemeaner.__post_init__(self)
        if not isinstance(self.precision, str):
            raise TypeError("`precision` must be a string.")
        precision = self.precision.lower()
        if precision not in {"float32", "float64"}:
            raise ValueError("`precision` must be either 'float32' or 'float64'.")
        object.__setattr__(self, "precision", precision)

        if self.use_gpu is not None and not isinstance(self.use_gpu, bool):
            raise TypeError("`use_gpu` must be a bool or None.")

        object.__setattr__(
            self,
            "solver_atol",
            _validate_positive_float(self.solver_atol, "solver_atol"),
        )
        object.__setattr__(
            self,
            "solver_btol",
            _validate_positive_float(self.solver_btol, "solver_btol"),
        )
        object.__setattr__(
            self,
            "solver_maxiter",
            _validate_positive_int(self.solver_maxiter, "solver_maxiter"),
        )

        if not isinstance(self.warn_on_cpu_fallback, bool):
            raise TypeError("`warn_on_cpu_fallback` must be a bool.")
        if not isinstance(self.use_preconditioner, bool):
            raise TypeError("`use_preconditioner` must be a bool.")


AnyDemeaner = MapDemeaner | WithinDemeaner | LsmrDemeaner

__all__ = [
    "AnyDemeaner",
    "BaseDemeaner",
    "LsmrDemeaner",
    "MapDemeaner",
    "WithinDemeaner",
]
