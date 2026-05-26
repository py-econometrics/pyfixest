from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Literal, get_args

MapBackend = Literal["numba", "rust", "jax"]
LsmrBackend = Literal["cupy", "torch"]
LsmrPrecision = Literal["float32", "float64"]
TorchDevice = Literal["auto", "cpu", "mps", "cuda"]
WithinKrylov = Literal["cg", "gmres"]
WithinPreconditioner = Literal["additive", "multiplicative", "off"]


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
    (`MapDemeaner`, `WithinDemeaner`, or `LsmrDemeaner`) and pass it as the
    ``demeaner`` argument to ``feols``, ``fepois``, or related estimators.

    Parameters
    ----------
    fixef_maxiter : int
        Maximum number of iterations before the demeaning loop is declared
        diverged. The meaning of one *iteration* differs by algorithm:

        - **MapDemeaner / WithinDemeaner**: one full sweep over all fixed
          effects (subtract the group mean for every FE once).
        - **LsmrDemeaner**: one LSMR iteration (matrix-vector product
          pair against the stacked design matrix).

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
    many singleton movers) prefer :class:`WithinDemeaner`.

    Parameters
    ----------
    backend : {"rust", "numba", "jax"}
        Execution engine for the alternating-projections loop.

        ``"rust"`` *(default)*
            Algorithm compiled in Rust.  No optional dependencies required.
            Ships with the base ``pyfixest`` install.

        ``"numba"``
            JIT-compiled multi-threaded Python.  Requires the optional
            ``numba`` extra: ``pip install pyfixest[numba]``.  Columns of
            the data matrix are processed in parallel across CPU threads.
            Performance is roughly equivalent to ``"rust"``.

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
    >>> # Numba backend (requires pip install pyfixest[numba])
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=MapDemeaner(backend="numba"))
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
class WithinDemeaner(BaseDemeaner):
    """
    Krylov-subspace demeaner implemented in Rust via the ``within`` library.

    Instead of alternating projections, this backend solves the demeaning
    problem as a linear system whose coefficient matrix is the block-diagonal
    normal-equations matrix of the fixed-effects design.  A Krylov solver
    (CG or GMRES) is applied with an optional Schwarz preconditioner that
    inverts the diagonal blocks locally.

    **When to use**

    ``WithinDemeaner`` is the preferred choice for *sparse* or
    *weakly-connected* fixed-effect structures - for example matched
    employer-employee panels where each worker appears in only a handful of
    firms.  In these cases alternating projections (``MapDemeaner``) can
    converge very slowly, while a Krylov solver with a good preconditioner
    can converge in far fewer iterations.

    For dense, well-connected structures ``MapDemeaner`` is usually faster.

    Parameters
    ----------
    krylov : {"cg", "gmres"}
        Krylov solver for the normal equations.

        ``"cg"`` *(default)*
            Conjugate Gradient.  Requires a *symmetric positive-definite*
            preconditioner, so only ``preconditioner="additive"`` or
            ``preconditioner="off"`` are valid.

        ``"gmres"``
            Generalized Minimum Residual.  Supports both ``"additive"``
            and ``"multiplicative"`` preconditioners.  Use when CG stalls
            or when you want to try multiplicative Schwarz.

    preconditioner : {"additive", "multiplicative", "off"}
        Schwarz preconditioner applied at each Krylov step.  A Schwarz
        preconditioner inverts the diagonal block for each fixed-effect
        group independently, then combines the local solutions.

        ``"additive"`` *(default)*
            Additive Schwarz: solves each FE group in
            parallel and sums the corrections.  Symmetric, so compatible
            with both CG and GMRES.  Generally robust and fast.

        ``"multiplicative"``
            Multiplicative Schwarz (block-Gauss-Seidel): applies
            corrections sequentially, each one seeing the residual
            updated by previous blocks.  Typically converges in fewer
            iterations than additive Schwarz but is *not symmetric*,
            so it **requires** ``krylov="gmres"``.

        ``"off"``
            No preconditioning.  Rarely beneficial; mainly useful for
            debugging or comparing with preconditioned runs.

    gmres_restart : int
        Restart dimension for GMRES (the "m" in GMRES(m)).  Only used
        when ``krylov="gmres"``. Default is ``30``.

    fixef_tol : float
        Convergence tolerance for the Krylov solver.  The solver stops
        when the relative residual norm satisfies::

            ||r_k|| / ||r_0|| < fixef_tol

        Defaults to ``1e-6``.

    fixef_maxiter : int
        Maximum Krylov iterations.  Defaults to ``1_000`` (lower than
        ``MapDemeaner`` because each iteration is more expensive, and
        generally fewer iterations are needed).

    Examples
    --------
    >>> import pyfixest as pf
    >>> from pyfixest.demeaners import WithinDemeaner
    >>> df = pf.get_data()
    >>> # CG + additive Schwarz (default)
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=WithinDemeaner())
    >>> # GMRES + multiplicative Schwarz - better for weakly connected graphs
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2",
    ...     data=df,
    ...     demeaner=WithinDemeaner(
    ...         krylov="gmres",
    ...         preconditioner="multiplicative",
    ...         gmres_restart=50,
    ...     ),
    ... )
    >>> # Unpreconditioned CG
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2",
    ...     data=df,
    ...     demeaner=WithinDemeaner(krylov="cg", preconditioner="off"),
    ... )
    """

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
    """
    Sparse LSMR demeaner.

    Rather than working with integer-encoded group IDs, this backend
    constructs the sparse dummy-variable design matrix *D* for all
    fixed effects and solves the demeaning problem as a sparse least-squares
    system using the LSMR algorithm (Fong & Saunders, 2011).  The result
    is mathematically identical to alternating projections but the
    convergence profile can differ substantially.

    Two execution backends are available:

    * ``"cupy"`` - uses CuPy's CUDA-accelerated LSMR on a GPU, or falls
      back to SciPy's CPU LSMR when no GPU / CuPy is available.
    * ``"torch"`` - experimental PyTorch-based LSMR (CPU or GPU).

    **When to use**

    Can be faster than MAP on problems where MAP struggles, but generally
    at mildly lower performance than :class:`WithinDemeaner`.

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

    Parameters
    ----------
    backend : {"cupy", "torch"}
        Execution engine.

        ``"cupy"`` *(default)*
            Builds the sparse FE design matrix once (via formulaic) and
            solves with LSMR from `CuPy <https://cupy.dev>`_ on a CUDA
            GPU, or with SciPy on the CPU if ``device="cpu"`` or no GPU
            is found.  Requires ``pip install pyfixest[cupy]`` for GPU
            use; SciPy is always available.

        ``"torch"``
            Experimental PyTorch-based LSMR.  Supports CUDA and Apple
            MPS devices in addition to the CPU.  Always uses a
            preconditioner.  Requires ``pip install pyfixest[torch]``.

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
        Target device.

        ``"auto"`` *(default)*
            Auto-detect: use the GPU if CuPy/CUDA is available,
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
        ``device="cuda"`` is requested but CuPy or a CUDA-capable GPU
        is not available and the solver silently falls back to CPU.
        Set to ``False`` to suppress the warning (e.g. in test suites
        that run on CPU-only machines).

    use_preconditioner : bool
        If ``True`` (default), scale the design matrix *D* by the
        inverse square root of each group's total weight before passing
        it to LSMR::

            M_inv = 1 / sqrt(diag(D^T W D))   # W = diag(weights)

        Preconditioning significantly improves convergence when group
        sizes are highly imbalanced (some fixed-effect levels have many
        more observations than others).

        Always ``True`` for the ``"torch"`` backend (cannot be
        disabled).

    Examples
    --------
    >>> import pyfixest as pf
    >>> from pyfixest.demeaners import LsmrDemeaner
    >>> df = pf.get_data()
    >>> # CuPy GPU (falls back to SciPy CPU if no GPU)
    >>> pf.feols("Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner(backend="cupy"))
    >>> # Force CPU via SciPy
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2", data=df, demeaner=LsmrDemeaner(backend="cupy", device="cpu")
    ... )
    >>> # CUDA GPU, float32, tight tolerance
    >>> pf.feols(
    ...     "Y ~ X1 | f1 + f2",
    ...     data=df,
    ...     demeaner=LsmrDemeaner(
    ...         backend="cupy",
    ...         device="cuda",
    ...         precision="float32",
    ...         fixef_atol=1e-10,
    ...         fixef_btol=1e-10,
    ...     ),
    ... )
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
