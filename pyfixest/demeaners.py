from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from numbers import Integral, Real
from typing import ClassVar, Literal, cast, get_args

import numpy as np

from pyfixest.core.demean import (
    Preconditioner,
    WithinPreconditionerName,
    demean_within,
)

MapBackend = Literal["numba", "rust"]
LsmrBackend = Literal["within", "cupy", "torch"]
LsmrPrecision = Literal["float32", "float64"]
TorchDevice = Literal["auto", "cpu", "mps", "cuda"]
LsmrPreconditioner = Literal["auto", "off", "additive", "diagonal"]

_PRECONDITIONER_SUPPORT: dict[LsmrBackend, tuple[set[str], str]] = {
    "within": (set(get_args(WithinPreconditionerName)), "additive"),
    "torch": ({"diagonal"}, "diagonal"),
}


def _resolve_preconditioner(backend: LsmrBackend, requested: LsmrPreconditioner) -> str:
    """Resolve `preconditioner` against the supported set of Preconditioners for the
    specified backend.

    `"auto"` always resolves silently to the backend's default.
    An explicit but unsupported value emits a `UserWarning` and is replaced
    with the natural default.
    """
    supported, default = _PRECONDITIONER_SUPPORT[backend]
    if requested == "auto" or requested in supported:
        return default if requested == "auto" else requested
    warnings.warn(
        (
            f"preconditioner={requested!r} is not supported by the {backend!r} "
            f"LSMR backend; falling back to {default!r}."
        ),
        UserWarning,
        stacklevel=3,
    )
    return default


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


def _get_numba_demean() -> Callable[..., tuple[np.ndarray, bool]]:
    try:
        from pyfixest.estimation.numba.demean_nb import demean as demean_nb
    except ImportError as exc:
        raise ImportError(
            "The Numba MAP backend requires the optional `numba` extra. "
            "Install it with `pip install pyfixest[numba]`, or use the default "
            "`MapDemeaner(backend='rust')` backend."
        ) from exc

    return cast(Callable[..., tuple[np.ndarray, bool]], demean_nb)


@dataclass(frozen=True, slots=True)
class BaseDemeaner:
    """
    Base configuration shared by all fixed-effects demeaners.

    Holds the settings shared by all backends, currently `fixef_maxiter`. This
    class is not passed to an estimation function directly. Use one of the
    concrete demeaners instead:
    [MapDemeaner](/reference/demeaners.MapDemeaner.qmd), the default, or
    [LsmrDemeaner](/reference/demeaners.LsmrDemeaner.qmd). See
    [Choosing a Demeaner Backend](/how-to/demeaner-backends.qmd) for a
    comparison.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    pf.MapDemeaner(), pf.LsmrDemeaner()
    ```
    """

    fixef_maxiter: int = 10_000
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_positive_int(self.fixef_maxiter, "fixef_maxiter")


@dataclass(frozen=True, slots=True)
class MapDemeaner(BaseDemeaner):
    """
    Method of Alternating Projections (MAP) demeaner.

    The default backend. Sweeps out the fixed effects by alternately projecting
    on each of them until convergence. See
    [Choosing a Demeaner Backend](/how-to/demeaner-backends.qmd) for when to use
    which backend.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.feols("Y ~ X1 | f1 + f2", data, demeaner=pf.MapDemeaner(fixef_tol=1e-08))
    fit.tidy()
    ```
    """

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

    def with_tol(self, tol: float | None) -> MapDemeaner:
        """Override the `fixef_tol`, used for IWLS acceleration."""
        if tol is None or tol == self.fixef_tol:
            return self
        return replace(self, fixef_tol=tol)

    def demean(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None = None,
        cached_preconditioner: Preconditioner | None = None,
    ) -> tuple[np.ndarray, bool, Preconditioner | None]:
        """Demean `x` by the fixed effects in `flist` via MAP.

        `cached_preconditioner` is accepted for interface uniformity with
        :meth:`LsmrDemeaner.demean` and ignored: MAP does not use a
        preconditioner, so the third return value is always `None`.
        """
        if weights is None:
            weights = np.ones(x.shape[0], dtype=np.float64)

        if self.backend == "numba":
            demean_func = _get_numba_demean()
        elif self.backend == "rust":
            from pyfixest.core.demean import demean as demean_rs

            demean_func = demean_rs
        else:
            raise ValueError(f"Unknown MapDemeaner backend: {self.backend!r}")

        result, success = demean_func(
            x=x,
            flist=flist.astype(np.uintp, copy=False),
            weights=weights,
            tol=self.fixef_tol,
            maxiter=self.fixef_maxiter,
        )
        return result, success, None


@dataclass(frozen=True, slots=True)
class LsmrDemeaner(BaseDemeaner):
    """Sparse LSMR demeaner.

    Solves the demeaning problem as a single sparse least squares system with
    LSMR instead of alternating projections. See
    [Choosing a Demeaner Backend](/how-to/demeaner-backends.qmd) for when to use
    which backend.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.feols("Y ~ X1 | f1 + f2", data, demeaner=pf.LsmrDemeaner())
    fit.tidy()
    ```

    Notes
    -----
    The `within`` backend takes a single tolerance, so `fixef_atol` and
    `fixef_btol` are collapsed to `max(fixef_atol, fixef_btol)` for that
    backend. The `torch` backend uses both tolerances independently
    (SciPy LSMR convention).

    The `local_size` field only applies to `backend="within"`; the
    `torch` backend ignores it.

    The `precision``, `device``, and `warn_on_cpu_fallback`` fields are
    only relevant for the `torch` backend. The `within` backend always
    runs on CPU in float64 and ignores these fields.

    `preconditioner` selects the preconditioner. Supported values:

    - `"auto"` (default): selects different preconditioners for different
      backend implementations: `"additive"` for `"within"`; `"diagonal"`
      for `"torch"`.
    - `"off"`: disables preconditioning. Supported by `"within"`; not
      supported by `"torch"`.
    - `"additive"`: additive Schwarz preconditioner. Only supported by the
      `"within"` backend.
    - `"diagonal"`: diagonal (Jacobi) preconditioner. Supported by
      `"within"` and `"torch"`.
    - A :class:`pyfixest.Preconditioner` instance: a previously built
      preconditioner (typically obtained via `fit.preconditioner` or
      pickled across sessions). Only supported by `backend='within'`;
      preconditioners are only computed and applied for two or more
      fixed-effect factors because single-factor problems run MAP as the within algo
      provides no benefits. Passing a preconditioner to any other backend raises `ValueError`
      at construction time.

    If a *string* value is incompatible with the chosen backend, a
    `UserWarning` is emitted at solve time and the backend's default is
    used. A `Preconditioner` paired with a non-`within` backend is
    rejected eagerly with `ValueError` because there is no sensible
    fallback for a prebuilt object.
    """

    fixef_maxiter: int = 1_000
    backend: LsmrBackend = "within"
    precision: LsmrPrecision = "float64"
    device: TorchDevice = "auto"
    fixef_atol: float = 1e-8
    fixef_btol: float = 1e-8
    warn_on_cpu_fallback: bool = True
    preconditioner: LsmrPreconditioner | Preconditioner = "auto"
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
        if isinstance(self.preconditioner, Preconditioner):
            if self.backend != "within":
                raise ValueError(
                    "A Preconditioner can only be reused with `backend='within'`."
                )
        elif not isinstance(self.preconditioner, str):
            raise TypeError("`preconditioner` must be a string or a Preconditioner.")
        elif self.preconditioner not in get_args(LsmrPreconditioner):
            raise ValueError(
                f"`preconditioner` must be one of {get_args(LsmrPreconditioner)} "
                "or a Preconditioner."
            )

        if self.local_size is not None:
            _validate_positive_int(self.local_size, "local_size")

        if (
            self.backend == "torch"
            and self.device == "mps"
            and self.precision != "float32"
        ):
            raise ValueError("The MPS torch backend requires `precision='float32'`.")

    def with_tol(self, tol: float | None) -> LsmrDemeaner:
        """Overwrite LSMR tolerances (used for IWLS acceleration)."""
        if tol is None or (tol == self.fixef_atol and tol == self.fixef_btol):
            return self
        return replace(self, fixef_atol=tol, fixef_btol=tol)

    def demean(
        self,
        x: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray | None = None,
        cached_preconditioner: Preconditioner | None = None,
    ) -> tuple[np.ndarray, bool, Preconditioner | None]:
        """Demean ``x`` by the fixed effects in ``flist`` via LSMR.

        Parameters
        ----------
        cached_preconditioner : Preconditioner or None, optional
            A preconditioner saved by the caller from an earlier within solve
            on the same fixed-effect design. This is separate from
            ``self.preconditioner``: the latter is the user's requested
            configuration, while ``cached_preconditioner`` is the model's
            internal "reuse this if it still matches" handle. The cache is
            used only when the current request is a string preconditioner
            with the same variant (``"additive"`` or ``"diagonal"``). If the
            user explicitly supplied a ``Preconditioner`` on the demeaner,
            that object is passed through and the model cache is ignored.

        Returns
        -------
        tuple[np.ndarray, bool, Preconditioner | None]
            The demeaned array, a convergence flag, and the within
            preconditioner actually used during the solve. The third element
            is ``None`` for non-within backends, when
            ``preconditioner='off'`` was requested, or when the single-FE MAP
            fallback path was taken inside ``demean_within`` — in those cases
            no preconditioner participated in the solve. Callers (e.g. the
            ``DemeanCache``) can cache the returned instance to amortise
            setup across subsequent solves on the same design.
        """
        if self.backend == "within":
            preconditioner: WithinPreconditionerName | Preconditioner
            if isinstance(self.preconditioner, Preconditioner):
                preconditioner = self.preconditioner
            else:
                preconditioner = cast(
                    WithinPreconditionerName,
                    _resolve_preconditioner("within", self.preconditioner),
                )

            if (
                cached_preconditioner is not None
                and isinstance(preconditioner, str)
                and cached_preconditioner.variant.lower() == preconditioner
            ):
                preconditioner = cached_preconditioner

            return demean_within(
                x=x,
                flist=flist.astype(np.uint32, copy=False),
                weights=weights,
                tol=max(self.fixef_atol, self.fixef_btol),
                maxiter=self.fixef_maxiter,
                local_size=self.local_size,
                preconditioner=preconditioner,
            )

        if weights is None:
            weights = np.ones(x.shape[0], dtype=np.float64)

        # Non-within branches never produce a Preconditioner, so their
        # third return value is always None.
        if self.backend == "torch":
            # Torch LSMR always uses its built-in diagonal preconditioner.
            # Call resolver for its UserWarning side effect on incompatible
            # requests; the returned value is intentionally unused.
            _ = _resolve_preconditioner(
                "torch", cast(LsmrPreconditioner, self.preconditioner)
            )
            try:
                torch = import_module("torch")
                torch_demean_module = import_module(
                    "pyfixest.estimation.torch.demean_torch_"
                )
            except ImportError:
                from pyfixest.core.demean import demean as demean_rs

                result, success = demean_rs(
                    x=x,
                    flist=flist.astype(np.uintp, copy=False),
                    weights=weights,
                    tol=max(self.fixef_atol, self.fixef_btol),
                    maxiter=self.fixef_maxiter,
                )
                return result, success, None

            dtype = torch.float32 if self.precision == "float32" else torch.float64
            tol = max(self.fixef_atol, self.fixef_btol)
            flist_uint64 = flist.astype(np.uint64, copy=False)

            if self.device == "auto":
                demean_torch = cast(
                    Callable[..., tuple[np.ndarray, bool]],
                    torch_demean_module.demean_torch,
                )
                result, success = demean_torch(
                    x=x,
                    flist=flist_uint64,
                    weights=weights,
                    tol=tol,
                    maxiter=self.fixef_maxiter,
                    dtype=dtype,
                )
                return result, success, None

            demean_torch_on_device = cast(
                Callable[..., tuple[np.ndarray, bool]],
                torch_demean_module._demean_torch_on_device_impl,
            )
            result, success = demean_torch_on_device(
                x=x,
                flist=flist_uint64,
                weights=weights,
                tol=tol,
                maxiter=self.fixef_maxiter,
                device=torch.device(self.device),
                dtype=dtype,
            )
            return result, success, None

        raise ValueError(f"Unknown LsmrDemeaner backend: {self.backend!r}")


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
