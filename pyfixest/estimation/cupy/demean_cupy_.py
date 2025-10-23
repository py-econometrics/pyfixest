"""
CuPy-accelerated FWL demeaner using sparse solvers.

This module implements the Frisch-Waugh-Lovell theorem for demeaning using:
1. Normal equations (D'D @ theta = D'x) with spsolve (vectorized, fast)
2. Fallback to LSMR on rectangular D if normal equations fail

Advantages over alternating projections:
- GPU acceleration via CuPy (NVIDIA CUDA)
- Direct solve (no iterative convergence for normal equations)
- Handles high-dimensional fixed effects efficiently
- Graceful CPU fallback when GPU unavailable
"""

from typing import Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Conditional imports for CuPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_sparse = None
    cp_lsmr = None

# Always import scipy for CPU fallback
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import lsmr as sp_lsmr, spsolve as sp_spsolve

# Import formulaic for sparse matrix creation
try:
    from formulaic import Formula

    FORMULAIC_AVAILABLE = True
except ImportError:
    FORMULAIC_AVAILABLE = False
    Formula = None


class CupyFWLDemeaner:
    """
    Frisch-Waugh-Lovell theorem demeaner using sparse solvers.

    Implements two strategies:
    1. Normal equations: (D'D) @ theta = D'x, solve with spsolve (vectorized)
    2. LSMR: Solve min ||Dx - b||² directly (sequential, more stable)

    Strategy selection is automatic based on problem size and sparsity.
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        solver_atol: float = 1e-8,
        solver_btol: float = 1e-8,
        solver_maxiter: Optional[int] = None,
        cache_operators: bool = True,
        warn_on_cpu_fallback: bool = True,
        normal_eq_threshold: int = 10000,
        sparsity_threshold: float = 0.5,
    ):
        """
        Initialize CuPy FWL demeaner.

        Parameters
        ----------
        use_gpu : bool, optional
            Force GPU usage (True), CPU usage (False), or auto-detect (None).
            Auto-detect checks if CuPy is available and GPU is accessible.
        solver_atol : float, default=1e-8
            Absolute tolerance for LSMR stopping criterion.
        solver_btol : float, default=1e-8
            Relative tolerance for LSMR stopping criterion.
        solver_maxiter : int, optional
            Maximum LSMR iterations. If None, uses LSMR's default.
        cache_operators : bool, default=True
            Cache sparse FE matrices to avoid reconstruction.
        warn_on_cpu_fallback : bool, default=True
            Warn when falling back to CPU despite use_gpu=True.
        normal_eq_threshold : int, default=10000
            Max number of FE parameters to use normal equations.
            Above this, always use LSMR for stability.
        sparsity_threshold : float, default=0.5
            If D'D density > this threshold, fall back to LSMR.
        """
        # Determine GPU availability
        if use_gpu is None:
            # Auto-detect
            self.use_gpu = CUPY_AVAILABLE and self._gpu_available()
        elif use_gpu and not CUPY_AVAILABLE:
            if warn_on_cpu_fallback:
                warnings.warn(
                    "CuPy not available. Falling back to CPU (scipy). "
                    "Install CuPy for GPU acceleration: pip install cupy-cuda12x",
                    UserWarning,
                )
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu

        # Solver configuration
        self.solver_atol = solver_atol
        self.solver_btol = solver_btol
        self.solver_maxiter = solver_maxiter

        # Strategy configuration
        self.normal_eq_threshold = normal_eq_threshold
        self.sparsity_threshold = sparsity_threshold

        # Caching
        self.cache_operators = cache_operators
        self._operator_cache = {}

        # Module references (for easier testing/mocking)
        if self.use_gpu:
            self.xp = cp
            self.sparse = cp_sparse
            self.lsmr = cp_lsmr
            self.spsolve = cp_sparse.linalg.spsolve
        else:
            self.xp = np
            self.sparse = sp_sparse
            self.lsmr = sp_lsmr
            self.spsolve = sp_spsolve

    @staticmethod
    def _gpu_available() -> bool:
        """Check if GPU is actually accessible (not just CuPy installed)."""
        if not CUPY_AVAILABLE:
            return False
        try:
            cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    def _create_fe_sparse_matrix(
        self, flist: NDArray[np.uint64]
    ) -> "sp_sparse.csr_matrix | cp_sparse.csr_matrix":
        """
        Create sparse CSR matrix for fixed effects from integer codes.

        IMPORTANT: Creates full dummy matrix with ALL levels (no level dropping).
        For FE with n groups, creates n columns (not n-1).

        Parameters
        ----------
        flist : np.ndarray, shape (n_obs, n_factors) or (n_obs,)
            Integer-encoded fixed effects (from factorize()).

        Returns
        -------
        D : csr_matrix
            Sparse dummy matrix [D_fe1 | D_fe2 | ...] in CSR format.
            Shape: (n_obs, total_groups) where total_groups = sum of n_groups per FE.
        """
        # Handle 1D flist
        if flist.ndim == 1:
            flist = flist.reshape(-1, 1)

        # Generate cache key
        cache_key = hash(flist.tobytes()) if self.cache_operators else None
        if cache_key in self._operator_cache:
            return self._operator_cache[cache_key]

        n_obs = flist.shape[0]
        n_factors = flist.shape[1]

        # Build COO arrays for all FE dimensions at once
        row_indices = []
        col_indices = []
        data = []
        col_offset = 0

        for j in range(n_factors):
            fe_codes = flist[:, j].astype(np.int32)
            n_groups = int(fe_codes.max()) + 1  # ALL groups (no level dropping)

            # COO format for this FE dimension
            row_indices.append(np.arange(n_obs, dtype=np.int32))
            col_indices.append(fe_codes + col_offset)
            data.append(np.ones(n_obs, dtype=np.float64))

            col_offset += n_groups

        # Concatenate all FE dimensions
        row_indices = np.concatenate(row_indices)
        col_indices = np.concatenate(col_indices)
        data = np.concatenate(data)

        # Create sparse matrix in COO format, then convert to CSR
        D_cpu = sp_sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_obs, col_offset),
            dtype=np.float64,
        ).tocsr()

        # Transfer to GPU if needed
        if self.use_gpu:
            D = self._scipy_to_cupy_sparse(D_cpu)
        else:
            D = D_cpu

        # Cache result
        if cache_key is not None:
            self._operator_cache[cache_key] = D

        return D

    def _scipy_to_cupy_sparse(
        self, scipy_matrix: sp_sparse.csr_matrix
    ) -> "cp_sparse.csr_matrix":
        """
        Convert scipy.sparse.csr_matrix to cupyx.scipy.sparse.csr_matrix.

        Parameters
        ----------
        scipy_matrix : scipy.sparse.csr_matrix
            Sparse matrix on CPU.

        Returns
        -------
        cupy_matrix : cupyx.scipy.sparse.csr_matrix
            Sparse matrix on GPU.
        """
        # Extract CSR components
        data = cp.asarray(scipy_matrix.data)
        indices = cp.asarray(scipy_matrix.indices)
        indptr = cp.asarray(scipy_matrix.indptr)

        # Reconstruct on GPU
        return cp_sparse.csr_matrix(
            (data, indices, indptr), shape=scipy_matrix.shape, dtype=np.float64
        )

    def _solve_normal_equations(
        self,
        D_weighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_weighted: "np.ndarray | cp.ndarray",
        D_unweighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_unweighted: "np.ndarray | cp.ndarray",
    ) -> Tuple["np.ndarray | cp.ndarray", bool]:
        """
        Solve using normal equations: (D'D) @ theta = D'x.

        This is VECTORIZED and works for all columns at once.

        Parameters
        ----------
        D_weighted : sparse matrix
            Weighted fixed effects dummy matrix (for solving).
        x_weighted : array
            Weighted variable(s) (for solving).
        D_unweighted : sparse matrix
            Unweighted fixed effects dummy matrix (for residuals).
        x_unweighted : array
            Unweighted variable(s) (for residuals).

        Returns
        -------
        x_demeaned : array
            Demeaned variable (residuals).
        success : bool
            True if solve succeeded.
        """
        try:
            # Form normal equations using WEIGHTED matrices
            DtD = D_weighted.T @ D_weighted
            Dtx = D_weighted.T @ x_weighted  # Shape: (n_fe_params,) or (n_fe_params, n_cols)

            # Solve using sparse solver (VECTORIZED for multiple columns!)
            theta = self.spsolve(DtD, Dtx)

            # Check if solve produced valid result (no NaN or Inf)
            if self.xp.any(self.xp.isnan(theta)) or self.xp.any(self.xp.isinf(theta)):
                warnings.warn(
                    "Normal equations produced invalid values (NaN/Inf), falling back to LSMR",
                    UserWarning,
                )
                return None, False

            # Compute residuals using UNWEIGHTED matrices
            D_theta = D_unweighted @ theta
            # Ensure D_theta has same shape as x_unweighted for broadcasting
            if x_unweighted.ndim == 2 and D_theta.ndim == 1:
                D_theta = D_theta.reshape(-1, 1)
            x_demeaned = x_unweighted - D_theta

            return x_demeaned, True

        except Exception as e:
            # spsolve failed (singular matrix, memory issues, etc.)
            warnings.warn(
                f"Normal equations failed ({e}), falling back to LSMR", UserWarning
            )
            return None, False

    def _solve_lsmr_loop(
        self,
        D_weighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_weighted: "np.ndarray | cp.ndarray",
        D_unweighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_unweighted: "np.ndarray | cp.ndarray",
    ) -> Tuple["np.ndarray | cp.ndarray", bool]:
        """
        Solve using LSMR on rectangular D (sequential loop over columns).

        More numerically stable than normal equations but not vectorized.

        Parameters
        ----------
        D_weighted : sparse matrix
            Weighted fixed effects dummy matrix (for solving).
        x_weighted : array
            Weighted variable(s) (for solving).
        D_unweighted : sparse matrix
            Unweighted fixed effects dummy matrix (for residuals).
        x_unweighted : array
            Unweighted variable(s) (for residuals).

        Returns
        -------
        x_demeaned : array
            Demeaned variable (residuals).
        success : bool
            True if LSMR converged for all columns.
        """
        if x_weighted.ndim == 1:
            # Single column
            result = self.lsmr(
                D_weighted,
                x_weighted,
                damp=0.0,
                atol=self.solver_atol,
                btol=self.solver_btol,
                maxiter=self.solver_maxiter,
            )
            theta = result[0]
            istop = result[1]
            # Compute residuals using UNWEIGHTED matrices
            D_theta = D_unweighted @ theta
            # Ensure D_theta has same shape as x_unweighted for broadcasting
            if x_unweighted.ndim == 2 and D_theta.ndim == 1:
                D_theta = D_theta.reshape(-1, 1)
            x_demeaned = x_unweighted - D_theta
            # LSMR istop: 1,2,3 = converged
            success = istop in [1, 2, 3]
        else:
            # Multiple columns - sequential loop
            x_demeaned = self.xp.zeros_like(x_unweighted)
            success = True

            for k in range(x_weighted.shape[1]):
                result = self.lsmr(
                    D_weighted,
                    x_weighted[:, k],
                    damp=0.0,
                    atol=self.solver_atol,
                    btol=self.solver_btol,
                    maxiter=self.solver_maxiter,
                )
                theta = result[0]
                istop = result[1]
                # Compute residuals using UNWEIGHTED matrices
                # Note: D @ theta is 1D, need to flatten x_unweighted[:, k] too
                x_demeaned[:, k] = x_unweighted[:, k].flatten() - (D_unweighted @ theta)
                success = success and (istop in [1, 2, 3])

        return x_demeaned, success

    def demean(
        self,
        x: NDArray[np.float64],
        flist: NDArray[np.uint64],
        weights: NDArray[np.float64],
        tol: float = 1e-8,
        maxiter: int = 100_000,
        fe_sparse_matrix: Optional["sp_sparse.csr_matrix"] = None,
    ) -> Tuple[NDArray[np.float64], bool]:
        """
        Demean variable x by projecting out fixed effects using FWL theorem.

        Standard signature matching other pyfixest demean backends.

        Strategy:
        1. Try normal equations with spsolve (fast, vectorized)
        2. Fall back to LSMR if normal equations fail or problem is too large

        Parameters
        ----------
        x : np.ndarray, shape (n_obs,) or (n_obs, n_vars)
            Variable(s) to demean.
        flist : np.ndarray, shape (n_obs, n_factors) or (n_obs,)
            Integer-encoded fixed effects. Ignored if fe_sparse_matrix provided.
        weights : np.ndarray, shape (n_obs,)
            Observation weights (1.0 for equal weighting).
        tol : float, default=1e-8
            Convergence tolerance (unused for normal eq; used for LSMR fallback).
        maxiter : int, default=100_000
            Maximum iterations (used for LSMR fallback).
        fe_sparse_matrix : scipy.sparse.csr_matrix, optional
            Pre-computed sparse FE dummy matrix (from formulaic).
            If provided, uses this directly (faster and more accurate).
            Should have reference levels dropped for full-rank system.

        Returns
        -------
        x_demeaned : np.ndarray
            Demeaned variable (residuals after projecting out FEs).
        success : bool
            True if solver converged/succeeded.
        """
        # Override maxiter if not set in __init__
        if self.solver_maxiter is None:
            original_maxiter = self.solver_maxiter
            self.solver_maxiter = maxiter
        else:
            original_maxiter = None

        try:
            # Create sparse FE matrix
            if fe_sparse_matrix is not None:
                # Use pre-computed sparse matrix (from formulaic)
                D = fe_sparse_matrix
            else:
                # Fall back to creating from integer codes
                D = self._create_fe_sparse_matrix(flist)

            # Transfer data to GPU if needed
            if self.use_gpu:
                x_device = cp.asarray(x)
                weights_device = cp.asarray(weights)
            else:
                x_device = x
                weights_device = weights

            # Handle weights via WLS transform: √w * x and √w * D
            has_weights = weights is not None and not np.allclose(weights, 1.0)
            if has_weights:
                sqrt_w = self.xp.sqrt(weights_device)
                if x_device.ndim == 2:
                    x_weighted = x_device * sqrt_w[:, None]
                else:
                    x_weighted = x_device * sqrt_w

                # Weight the FE matrix (multiply rows by sqrt_w)
                D_weighted = D.multiply(sqrt_w[:, None])
            else:
                x_weighted = x_device
                D_weighted = D

            # Decide strategy based on problem size
            n_fe_params = D.shape[1]
            use_normal_eq = n_fe_params < self.normal_eq_threshold

            if use_normal_eq:
                # Try normal equations first (fast, vectorized)
                # Check D'D sparsity
                DtD = D_weighted.T @ D_weighted
                density = DtD.nnz / (n_fe_params**2)

                if density < self.sparsity_threshold:
                    # D'D is sparse enough, use spsolve
                    x_demeaned, success = self._solve_normal_equations(
                        D_weighted, x_weighted, D, x_device
                    )

                    if success:
                        # Success! Transfer back to CPU if needed
                        if self.use_gpu:
                            x_demeaned = cp.asnumpy(x_demeaned)
                        return x_demeaned, success
                    # else: fall through to LSMR

            # Fall back to LSMR (large problem or normal eq failed)
            x_demeaned, success = self._solve_lsmr_loop(
                D_weighted, x_weighted, D, x_device
            )

            # Transfer back to CPU
            if self.use_gpu:
                x_demeaned = cp.asnumpy(x_demeaned)

            return x_demeaned, success

        finally:
            # Restore original maxiter setting
            if original_maxiter is not None:
                self.solver_maxiter = original_maxiter


def create_fe_sparse_matrix(
    fe: pd.DataFrame, drop_reference: bool = True
) -> sp_sparse.csr_matrix:
    """
    Create sparse fixed effects matrix using formulaic.

    This function encodes fixed effects as a sparse one-hot matrix with proper
    reference level handling. By default, it drops one reference level per FE
    to ensure the system is full-rank, which matches the alternating projections
    approach used by other backends.

    Parameters
    ----------
    fe : pd.DataFrame
        DataFrame with fixed effects columns. Each column represents one
        categorical fixed effect.
    drop_reference : bool, default=True
        If True, drops one reference level per FE for full-rank encoding.
        If False, keeps all levels (use "-1+" formula syntax).

    Returns
    -------
    D : scipy.sparse.csr_matrix, shape (n_obs, n_fe_params)
        Sparse one-hot encoded FE matrix.

    Raises
    ------
    ImportError
        If formulaic is not installed.

    Examples
    --------
    >>> import pandas as pd
    >>> fe = pd.DataFrame({'fe1': [0, 0, 1, 1], 'fe2': [0, 1, 0, 1]})
    >>> D = create_fe_sparse_matrix(fe, drop_reference=True)
    >>> D.shape  # (4, 3) - dropped 2 reference levels
    """
    if not FORMULAIC_AVAILABLE:
        raise ImportError(
            "formulaic is required for sparse FE matrix creation. "
            "Install with: pip install formulaic"
        )

    # Build formula: C(fe1) + C(fe2) + ... for categorical encoding
    fe_terms = " + ".join([f"C({col})" for col in fe.columns])

    # Always remove intercept for FE-only model
    formula_str = f"-1 + {fe_terms}"

    # Create model matrix using formulaic
    from formulaic import model_matrix

    # ensure_full_rank controls reference level dropping
    # - True (default): drops one level per FE for full-rank system
    # - False: keeps all levels (rank-deficient, needs LSMR)
    # We want ensure_full_rank=True when drop_reference=True
    mm = model_matrix(formula_str, fe, ensure_full_rank=drop_reference)

    # Convert to scipy sparse matrix
    if hasattr(mm, "sparse"):
        # Formulaic returns sparse-aware matrix
        D = mm.sparse.to_csr()
    else:
        # Fall back to dense conversion (should not happen with C())
        D = sp_sparse.csr_matrix(mm.to_numpy())

    return D


# Module-level singleton for functional interface
_default_demeaner = None


def demean_cupy(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
    fe: Optional[pd.DataFrame] = None,
    drop_reference: bool = True,
) -> Tuple[NDArray[np.float64], bool]:
    """
    Functional interface for CuPy FWL demeaner.

    Matches standard pyfixest demean signature for backend compatibility.
    Uses module-level singleton with default configuration.

    For custom configuration (GPU control, solver tolerances), instantiate
    CupyFWLDemeaner directly.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs,) or (n_obs, n_vars)
        Variable(s) to demean.
    flist : np.ndarray, shape (n_obs, n_factors) or (n_obs,), optional
        Integer-encoded fixed effects. Deprecated in favor of `fe`.
        If provided, will fall back to internal FE matrix construction.
    weights : np.ndarray, shape (n_obs,), optional
        Observation weights. If None, uses uniform weights.
    tol : float, default=1e-8
        Convergence tolerance for LSMR fallback.
    maxiter : int, default=100_000
        Maximum iterations for LSMR fallback.
    fe : pd.DataFrame, optional
        Fixed effects as DataFrame. Preferred over `flist` for efficiency.
        Each column represents one categorical fixed effect.
    drop_reference : bool, default=True
        If True, drops one reference level per FE for full-rank encoding.
        Only used when `fe` is provided.

    Returns
    -------
    x_demeaned : np.ndarray
        Demeaned variable.
    success : bool
        True if solver succeeded.

    Notes
    -----
    Prefer passing `fe` DataFrame over `flist` for:
    - Faster sparse matrix construction via formulaic
    - Proper reference level handling (matches alternating projections)
    - Better numerical stability for high-dimensional FE

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> x = np.random.randn(100, 2)
    >>> fe = pd.DataFrame({'fe1': np.random.randint(0, 10, 100)})
    >>> x_demeaned, success = demean_cupy(x, fe=fe)
    """
    global _default_demeaner
    if _default_demeaner is None:
        _default_demeaner = CupyFWLDemeaner()

    # Set up default weights
    if weights is None:
        weights = np.ones(x.shape[0] if x.ndim > 1 else len(x))

    # Create sparse FE matrix if fe DataFrame provided
    fe_sparse_matrix = None
    if fe is not None:
        fe_sparse_matrix = create_fe_sparse_matrix(fe, drop_reference=drop_reference)
    elif flist is None:
        raise ValueError("Either `fe` or `flist` must be provided")

    # For backward compatibility with flist, we need to handle it
    if flist is None:
        # Create dummy flist - won't be used since fe_sparse_matrix is provided
        flist = np.zeros((x.shape[0], 1), dtype=np.uint64)

    return _default_demeaner.demean(
        x, flist, weights, tol, maxiter, fe_sparse_matrix=fe_sparse_matrix
    )
