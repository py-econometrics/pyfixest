from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Literal, get_args

MapBackend = Literal["numba", "rust", "jax"]
LsmrBackend = Literal["within", "cupy", "torch"]
LsmrPrecision = Literal["float32", "float64"]
TorchDevice = Literal["auto", "cpu", "mps", "cuda"]
LsmrPreconditioner = Literal["auto", "additive", "diagonal", "off"]


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
    """
    Base configuration shared by all fixed-effects demeaners.

    This class is not used directly. Instantiate one of its subclasses
    (`MapDemeaner` or `LsmrDemeaner`) and pass it as the
    ``demeaner`` argument to ``feols``, ``fepois``, or related estimators.

    Parameters
    ----------
    fixef_maxiter : int
        Maximum number of iterations before the demeaning loop is declared
        diverged. The meaning of one *iteration* differs by algorithm:

        - **MapDemeaner**: one full sweep over all fixed effects (subtract
          the group mean for every FE once).
        - **LsmrDemeaner**: one LSMR iteration
          (matrix-vector product pair against the fixed-effects design).

        When the limit is reached without convergence a ``ValueError`` is
        raised.  Increase this value if you encounter convergence failures on
        large or weakly-identified FE structures.
    """

    fixef_maxiter: int = 10_000
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_positive_int(self.fixef_maxiter, "fixef_maxiter")


@dataclass(frozen=True, slots=True)
class MapDemeaner(BaseDemeaner):
    """
    Method of Alternating Projections (MAP) demeaner.

    Removes fixed effects by iteratively subtracting each group's
    (weighted) mean in turn - one pass per fixed effect per iteration -
    until the solution stops changing.  This is the classic Gauss-Seidel /
    "ZigZag" approach and is the default demeaner for ``feols``.

    All three backend variants run the same mathematical algorithm; only
    the execution engine differs.

    **When to use**

    ``MapDemeaner`` works well when the fixed-effect graph is *dense*
    (every group appears many times) and well-connected (e.g. the standard
    two-way FE with many firms *and* many workers per firm).  For sparse,
    weakly-connected graphs (e.g. matched employer-employee data with
    many singleton movers) prefer :class:`LsmrDemeaner` with
    ``backend="within"``.

    Parameters
    ----------
    backend : {"rust", "numba", "jax"}
        Execution engine for the alternating-projections loop.

        ``"rust"`` *(default)*
            Same algorithm compiled in Rust.  No extra installation required.
            Performance is roughly equivalent to ``"numba"``.

        ``"numba"``
            JIT-compiled multi-threaded Python.  Requires
            ``pip install pyfixest[numba]``.  Columns of the data matrix are
            processed in parallel across CPU threads.

        ``"jax"``
            Runs on the GPU (or CPU) via `JAX
            <https://jax.readthedocs.io>`_.  Requires
            ``pip install pyfixest[jax]``. Usually slower than
            ``"numba"`` and ``"rust"`` on the CPU, but can be faster
            on GPU for large datasets. If a GPU is available, we
            nevertheless recommend :class:`LsmrDemeaner` with the
            ``"torch"`` or ``"cupy"`` backend, as performance will
            generally be better.

    fixef_tol : float
        Convergence tolerance.  After each full sweep over all fixed
        effects, the algorithm checks whether the maximum absolute
        change in any element of the demeaned vector is below this
        threshold::

            max_i |x_curr[i] - x_prev[i]| < fixef_tol

        Smaller values yield more accurate removal of FE means at the
        cost of more iterations.  The default of ``1e-6`` matches the
        precision of ``fixest`` (R).

    fixef_maxiter : int
        Maximum number of full sweeps before declaring failure.
        Inherited from :class:`BaseDemeaner`; default is ``10_000``.

    Examples
    --------
    >>> import pyfixest as pf
    >>> from pyfixest.demeaners import MapDemeaner
    >>> df = pf.get_data()
    >>> # Default - rust backend
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df)
    >>> # Rust backend, tighter tolerance
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2", data=df, demeaner=MapDemeaner(backend="rust", fixef_tol=1e-10)
    ... )
    >>> # JAX backend (requires pip install pyfixest[jax])
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=MapDemeaner(backend="jax"))
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


@dataclass(frozen=True, slots=True)
class LsmrDemeaner(BaseDemeaner):
    """
    Sparse LSMR demeaner.

    Solves the demeaning problem as a sparse least-squares system using
    the LSMR algorithm (Fong & Saunders, 2011). The result is
    mathematically identical to alternating projections but the convergence
    profile can differ substantially.

    Two execution backends are available:

    * ``"within"`` - bundled Rust solver via the ``within`` crate. This is
      the default and preferred CPU backend for sparse or weakly connected
      fixed-effect structures.
    * ``"torch"`` - experimental PyTorch-based LSMR (CPU or GPU).

    **When to use**

    Can be faster than MAP on problems where MAP struggles, especially
    with ``backend="within"`` on sparse multi-way fixed effects.

    **Convergence criteria**

    LSMR uses two separate tolerance parameters that together govern when
    the solver declares success:

    * ``fixef_atol``: controls convergence of the *least-squares residual*.
      The solver stops when::

          ||A^T r_k|| / (||A|| * ||r_k||) <= fixef_atol

      where *r_k* = *b* - *A x_k* is the current residual and *A* is the
      weighted design matrix.  This checks that the gradient of the
      least-squares objective is small relative to the problem scale.

    * ``fixef_btol``: controls convergence of the *solution residual*.
      The solver stops when::

          ||r_k|| <= fixef_atol * ||A|| * ||x_k|| + fixef_btol * ||b||

      This checks that the absolute residual is small relative to the
      right-hand side.

    For ``backend="within"``, the Rust solver currently accepts a single
    relative tolerance; PyFixest passes ``max(fixef_atol, fixef_btol)``.

    Parameters
    ----------
    backend : {"within", "cupy", "torch"}
        Execution engine.

        ``"within"`` *(default)*
            Rust LSMR-Schwarz backend from the
            `within <https://github.com/py-econometrics/within>`_ crate.
            This backend works on CPU and does not require optional Python
            GPU dependencies.

        ``"torch"``
            Experimental PyTorch-based LSMR.  Supports CUDA and Apple
            MPS devices in addition to the CPU.  Always uses diagonal
            preconditioning.  Requires ``pip install pyfixest[torch]``.

    precision : {"float64", "float32"}
        Floating-point precision used on the GPU.

        ``"float64"`` *(default)*
            Double precision.  Recommended for most use cases; matches
            the precision of CPU-based estimators.

        ``"float32"``
            Single precision.  Roughly 2x faster on modern NVIDIA GPUs
            (which have wider float32 throughput), but may introduce
            numerical noise for ill-conditioned problems.  **Required**
            when ``backend="torch"`` and ``device="mps"`` (Apple
            Silicon does not support float64 in MPS kernels).

    device : {"auto", "cpu", "cuda", "mps"}
        Target device for the ``"torch"`` backend.

        ``"auto"`` *(default)*
            Auto-detect: use the GPU if CUDA is available,
            otherwise fall back to CPU.

        ``"cpu"``
            Force execution on the CPU (SciPy LSMR for the ``"cupy"``
            backend; PyTorch CPU for ``"torch"``).

        ``"cuda"``
            Force NVIDIA CUDA GPU.  Falls back to CPU with a warning
            (if ``warn_on_cpu_fallback=True``) when CuPy or a CUDA
            device is unavailable.

        ``"mps"``
            Apple Metal Performance Shaders (Apple Silicon GPU).
            Only supported by the ``"torch"`` backend, and requires
            ``precision="float32"``.

    fixef_atol : float
        Absolute tolerance for the LSMR gradient stopping criterion.
        See the *Convergence criteria* section above.  Default ``1e-8``.

    fixef_btol : float
        Relative tolerance for the LSMR residual stopping criterion.
        See the *Convergence criteria* section above.  Default ``1e-8``.

    fixef_maxiter : int
        Maximum LSMR iterations.  Inherited from :class:`BaseDemeaner`;
        default is ``10_000``.

    warn_on_cpu_fallback : bool
        If ``True`` (default), emit a ``UserWarning`` when
        ``device="cuda"`` is requested but a CUDA-capable GPU
        is not available and the solver silently falls back to CPU.
        Set to ``False`` to suppress the warning (e.g. in test suites
        that run on CPU-only machines).

    use_preconditioner : bool
        If ``True`` (default), use backend-specific preconditioning. The
        effective variant is controlled by ``preconditioner``. For
        ``backend="within"``, the default is additive Schwarz. For the
        ``"torch"`` backend, the default is diagonal preconditioning, which
        scales the design matrix *D* by the inverse square root of each group's
        total weight before passing it to LSMR::

            M_inv = 1 / sqrt(diag(D^T W D))   # W = diag(weights)

        Preconditioning significantly improves convergence when group
        sizes are highly imbalanced (some fixed-effect levels have many
        more observations than others).

        Always ``True`` for the ``"torch"`` backend (cannot be disabled).

    preconditioner : {"auto", "additive", "diagonal", "off"}
        Preconditioner variant. ``"auto"`` (default) resolves to
        ``"additive"`` for ``backend="within"`` and ``"diagonal"`` for
        ``backend="torch"``. ``"additive"`` uses
        additive Schwarz over local fixed-effect blocks and is only
        supported by ``backend="within"``. ``"diagonal"`` is supported by
        Torch LSMR backends. ``"off"`` disables
        preconditioning where the backend supports doing so; Torch always
        uses diagonal preconditioning.

    local_size : int | None
        Optional reorthogonalization window for ``backend="within"``.
        ``None`` disables reorthogonalization.

    Examples
    --------
    >>> import pyfixest as pf
    >>> from pyfixest.demeaners import LsmrDemeaner
    >>> df = pf.get_data()
    >>> # Rust LSMR-Schwarz backend
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner())
    >>> # Apple Silicon (MPS), requires float32
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2",
    ...     data=df,
    ...     demeaner=LsmrDemeaner(
    ...         backend="torch",
    ...         device="mps",
    ...         precision="float32",
    ...     ),
    ... )
    """

    backend: LsmrBackend = "within"
    precision: LsmrPrecision = "float64"
    device: TorchDevice = "auto"
    fixef_atol: float = 1e-8
    fixef_btol: float = 1e-8
    warn_on_cpu_fallback: bool = True
    use_preconditioner: bool = True
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
        if not isinstance(self.use_preconditioner, bool):
            raise TypeError("`use_preconditioner` must be a bool.")

        if not isinstance(self.preconditioner, str):
            raise TypeError("`preconditioner` must be a string.")
        if self.preconditioner not in get_args(LsmrPreconditioner):
            raise ValueError(
                f"`preconditioner` must be one of {get_args(LsmrPreconditioner)}."
            )
        resolved_preconditioner: LsmrPreconditioner
        if self.preconditioner == "auto":
            if self.backend == "within":
                resolved_preconditioner = (
                    "additive" if self.use_preconditioner else "off"
                )
            elif self.backend == "cupy":
                resolved_preconditioner = (
                    "diagonal" if self.use_preconditioner else "off"
                )
            else:
                resolved_preconditioner = "diagonal"
        else:
            resolved_preconditioner = self.preconditioner

        if self.backend == "within":
            if resolved_preconditioner == "diagonal":
                raise ValueError(
                    "The within LSMR backend supports "
                    "`preconditioner='additive'` or `preconditioner='off'`; "
                    "diagonal preconditioning is only supported by the "
                    "CuPy/SciPy and Torch backends."
                )
            if not self.use_preconditioner:
                resolved_preconditioner = "off"
        elif self.backend == "cupy":
            if resolved_preconditioner == "additive":
                raise ValueError(
                    "Additive Schwarz preconditioning is only supported by "
                    "`backend='within'`. The CuPy/SciPy LSMR backend uses "
                    "diagonal preconditioning."
                )
            if not self.use_preconditioner:
                resolved_preconditioner = "off"
        else:
            if not self.use_preconditioner:
                raise ValueError(
                    "The torch LSMR backend always uses diagonal preconditioning."
                )
            if resolved_preconditioner != "diagonal":
                raise ValueError(
                    "The torch LSMR backend always uses diagonal "
                    "preconditioning. Additive Schwarz preconditioning is "
                    "only supported by `backend='within'`."
                )

        object.__setattr__(self, "preconditioner", resolved_preconditioner)
        object.__setattr__(self, "use_preconditioner", resolved_preconditioner != "off")

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
