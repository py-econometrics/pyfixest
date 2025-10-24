from typing import Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from numpy.typing import np.ndarray
from formulaic import Formula

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

import scipy.sparse as sp_sparse
from scipy.sparse.linalg import lsmr as sp_lsmr, spsolve as sp_spsolve



class CupyFWLDemeaner:
    """
    Frisch-Waugh-Lovell theorem demeaner using sparse solvers.

    Solves via the LSMR solver. 

    Strategy selection is automatic based on problem size and sparsity.
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        solver_atol: float = 1e-8,
        solver_btol: float = 1e-8,
        solver_maxiter: Optional[int] = None,
        warn_on_cpu_fallback: bool = True,
        dtype: type = np.float64,
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
        warn_on_cpu_fallback : bool, default=True
            Warn when falling back to CPU despite use_gpu=True.
        dtype : type, default=np.float64
            Data type for GPU computations (np.float32 or np.float64).
        """
        if use_gpu is None:
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

        self.solver_atol = solver_atol
        self.solver_btol = solver_btol
        self.solver_maxiter = solver_maxiter
        self.warn_on_cpu_fallback = warn_on_cpu_fallback
        self.dtype = dtype

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

    def _solve_lsmr_loop(
        self,
        D_weighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_weighted: "np.np.ndarray | cp.np.ndarray",
        D_unweighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_unweighted: "np.ndarray | cp.ndarray",
    ) -> Tuple["np.ndarray | cp.ndarray", bool]:
        "Solve OLS Equations via LSMR solver."

        X_k = x_unweighted.shape[1]
        D_k = D_weighted.shape[1]
        x_demeaned = self.xp.zeros_like(x_unweighted)
        theta = self.xp.zeros((D_k, X_k), dtype=x_unweighted.dtype)
        success = True

        for k in range(X_k):
            result = self.lsmr(
                D_weighted,
                x_weighted[:, k],
                damp=0.0,
                atol=self.solver_atol,
                btol=self.solver_btol,
                maxiter=self.solver_maxiter,
            )
            theta[:, k] = result[0]
            istop = result[1]
            success = success and (istop in [1, 2, 3])

        x_demeaned = x_unweighted - (D_unweighted @ theta)

        return x_demeaned, success

    def demean(
        self,
        x: np.ndarray[np.float64],
        flist: np.ndarray[np.uint64],
        weights: np.ndarray[np.float64],
        tol: float = 1e-8,
        maxiter: int = 100_000,
        fe_sparse_matrix: Optional["sp_sparse.csr_matrix"] = None,
    ) -> Tuple[np.ndarray[np.float64], bool]:
        """
        Demean variable x by projecting out fixed effects using FWL theorem.

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

        D = fe_sparse_matrix
        if self.use_gpu:
            x_device = cp.asarray(x, dtype=self.dtype)
            weights_device = cp.asarray(weights, dtype=self.dtype)
            D_device = cp_sparse.csr_matrix(D)
            D_device = D_device.astype(self.dtype)
        else:
            x_device = x
            weights_device = weights
            D_device = D

        if weights is not None:
            sqrt_w = self.xp.sqrt(weights_device)
            if x_device.ndim == 2:
                x_weighted = x_device * sqrt_w[:, None]
            else:
                x_weighted = x_device * sqrt_w
            D_weighted = D_device.multiply(sqrt_w[:, None])
        else:
            x_weighted = x_device
            D_weighted = D_device

        x_demeaned, success = self._solve_lsmr_loop(
            D_weighted, x_weighted, D_device, x_device
        )

        if self.use_gpu:
            # Convert back to CPU and ensure float64 for downstream precision
            x_demeaned = cp.asnumpy(x_demeaned).astype(np.float64)

        return x_demeaned, success


def create_fe_sparse_matrix(
    fe: pd.DataFrame
) -> sp_sparse.csr_matrix:
    "Create sparse fixed effects matrix using formulaic."

    fe_fml = " + ".join([f"C({col})" for col in fe.columns])
    FML = Formula(fe_fml)
    D = FML.get_model_matrix(data = fe, output = "sparse")
    return D.tocsr()


def demean_cupy(
    x: np.ndarray[np.float64],
    flist: Optional[np.ndarray[np.uint64]] = None,
    weights: Optional[np.ndarray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
    dtype: type = np.float64,
) -> Tuple[np.ndarray[np.float64], bool]:
    """
    Function interface for CuPy demeaner.

    Parameters
    ----------
    dtype : type, default=np.float64
        Data type for GPU computations (np.float32 or np.float64).
    """
    if weights is None:
        weights = np.ones(x.shape[0] if x.ndim > 1 else len(x))

    n_fe = flist.shape[1] if flist.ndim > 1 else 1
    fe_df = pd.DataFrame(flist, columns=[f"f{i+1}" for i in range(n_fe)], copy=False)
    fe_sparse_matrix = create_fe_sparse_matrix(fe_df)

    return CupyFWLDemeaner(dtype=dtype).demean(
        x, flist, weights, tol, maxiter, fe_sparse_matrix=fe_sparse_matrix
    )


def demean_cupy32(
    x: np.ndarray[np.float64],
    flist: Optional[np.ndarray[np.uint64]] = None,
    weights: Optional[np.ndarray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> Tuple[np.ndarray[np.float64], bool]:
    """
    CuPy demeaner using float32 precision (faster on GPU, ~2x speedup).

    May have numerical stability issues for ill-conditioned problems.
    Results are converted back to float64 for downstream precision.
    """
    return demean_cupy(x, flist, weights, tol, maxiter, dtype=np.float32)


def demean_cupy64(
    x: np.ndarray[np.float64],
    flist: Optional[np.ndarray[np.uint64]] = None,
    weights: Optional[np.ndarray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> Tuple[np.ndarray[np.float64], bool]:
    """
    CuPy demeaner using float64 precision (more accurate, safer default).

    Slower than float32 but provides better numerical stability.
    """
    return demean_cupy(x, flist, weights, tol, maxiter, dtype=np.float64)
