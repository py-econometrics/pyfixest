import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from formulaic import Formula
from numpy.typing import NDArray

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.sparse.linalg import LinearOperator as cp_LinearOperator
    from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_sparse = None
    cp_LinearOperator = None
    cp_lsmr = None

import scipy.sparse as sp_sparse
from scipy.sparse.linalg import LinearOperator as sp_LinearOperator
from scipy.sparse.linalg import lsmr as sp_lsmr
from scipy.sparse.linalg import spsolve as sp_spsolve


class CupyFWLDemeaner:
    """
    Frisch-Waugh-Lovell theorem demeaner using sparse solvers.

    Solves via the LSMR solver.
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        solver_atol: float = 1e-8,
        solver_btol: float = 1e-8,
        solver_maxiter: Optional[int] = None,
        warn_on_cpu_fallback: bool = True,
        dtype: type = np.float64,
        use_preconditioner: bool = True,
    ):
        """
        Initialize CuPy FWL demeaner.

        Parameters
        ----------
        use_gpu : bool, optional
            Force GPU usage (True), CPU usage (False), or auto-detect (None).
            Auto-detect checks if CuPy is available and GPU is accessible. If
            both are True, runs on the GPU via CuPy.
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
        use_preconditioner : bool, default=True
            Whether to use diagonal preconditioning for LSMR. Preconditioning
            improves convergence when fixed effect group sizes vary.
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
        self.use_preconditioner = use_preconditioner

        if self.use_gpu:
            self.xp = cp
            self.sparse = cp_sparse
            self.lsmr = cp_lsmr
            self.spsolve = cp_sparse.linalg.spsolve
            self.LinearOperator = cp_LinearOperator
        else:
            self.xp = np
            self.sparse = sp_sparse
            self.lsmr = sp_lsmr
            self.spsolve = sp_spsolve
            self.LinearOperator = sp_LinearOperator

    @staticmethod
    def _gpu_available() -> bool:
        """Check if GPU is actually accessible (not just CuPy installed)."""
        if not CUPY_AVAILABLE:
            return False
        try:
            _ = cp.cuda.Device(0).compute_capability
        except Exception:
            return False
        else:
            return True

    def _compute_column_scale(
        self,
        D: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        weights: "NDArray[Any] | Any",
    ) -> "NDArray[Any] | Any":
        """
        Compute M_inv = 1/sqrt(sum of weights per group) for preconditioning.

        Computes from the sparse design matrix D using D.T @ weights,
        which gives the sum of weights for each column (group) efficiently.

        Parameters
        ----------
        D : sparse matrix of shape (n_obs, n_groups)
            Fixed effects design matrix (binary indicator matrix).
        weights : array of shape (n_obs,)
            Observation weights.

        Returns
        -------
        M_inv : array of shape (n_groups,)
            Inverse scaling factors for each column of D.
        """
        # D.T @ weights gives sum of weights per group (since D is binary)
        # This is O(nnz) and matches D's column structure exactly
        group_weights = D.T @ weights

        # Convert sparse matrix result to dense array if needed
        group_weights = self.xp.asarray(group_weights).ravel()

        # Check for empty groups - raise error
        if self.xp.any(group_weights == 0):
            raise ValueError(
                "Empty fixed effect groups detected. "
                "This may indicate singleton observations that should be dropped."
            )

        M_inv = 1.0 / self.xp.sqrt(group_weights)
        return M_inv

    def _create_preconditioned_operator(
        self,
        D: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        M_inv: "NDArray[Any] | Any",
    ) -> "sp_LinearOperator | Any":
        """
        Create LinearOperator for D @ diag(M_inv).

        matvec:  (D @ M) @ z = D @ (M_inv * z)
        rmatvec: (D @ M).T @ u = M_inv * (D.T @ u)
        """
        n_rows, n_cols = D.shape

        def matvec(z):
            return D @ (M_inv * z)

        def rmatvec(u):
            return M_inv * (D.T @ u)

        return self.LinearOperator(
            shape=(n_rows, n_cols),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=D.dtype,
        )

    def _solve_lsmr_loop(
        self,
        D_weighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_weighted: "NDArray[Any] | Any",
        D_unweighted: "sp_sparse.csr_matrix | cp_sparse.csr_matrix",
        x_unweighted: "NDArray[Any] | Any",
        weights: "NDArray[Any] | Any",
    ) -> tuple["NDArray[Any] | Any", bool]:
        "Solve OLS Equations via LSMR solver with optional diagonal preconditioning."
        X_k = x_unweighted.shape[1]
        D_k = D_weighted.shape[1]
        x_demeaned = self.xp.zeros_like(x_unweighted)
        theta = self.xp.zeros((D_k, X_k), dtype=x_unweighted.dtype)
        success = True

        # Setup operator and scaling based on preconditioning setting
        if self.use_preconditioner:
            M_inv = self._compute_column_scale(D_unweighted, weights)
            A_op = self._create_preconditioned_operator(D_weighted, M_inv)
        else:
            M_inv = None
            A_op = D_weighted

        for k in range(X_k):
            result = self.lsmr(
                A_op,
                x_weighted[:, k],
                damp=0.0,
                atol=self.solver_atol,
                btol=self.solver_btol,
                maxiter=self.solver_maxiter,
            )
            z = result[0]

            # Recover theta from preconditioned solution: theta = M_inv * z
            if M_inv is not None:
                theta[:, k] = M_inv * z
            else:
                theta[:, k] = z

            istop = result[1]
            success = success and (istop in [1, 2, 3])

        x_demeaned = x_unweighted - (D_unweighted @ theta)

        return x_demeaned, success

    def demean(
        self,
        x: NDArray[Any],
        flist: NDArray[Any],
        weights: NDArray[Any],
        tol: float = 1e-8,
        maxiter: int = 100_000,
        fe_sparse_matrix: Optional["sp_sparse.csr_matrix"] = None,
    ) -> tuple[NDArray[Any], bool]:
        """
        Demean variable x by projecting out fixed effects using FWL theorem.

        Parameters
        ----------
        x : np.ndarray.
            Variable(s) to demean.
        flist : np.ndarray.
            Integer-encoded fixed effects. Ignored if fe_sparse_matrix provided.
            Usually not used within pyfixest internals.
        weights : np.ndarray, shape (n_obs,)
            Weights (1.0 for equal weighting).
        tol : float, default=1e-8
            Convergence tolerance. Used for both atol and btol of lsmr algo.
        maxiter : int, default=100_000
            Maximum iterations for lsmr iterations.
        fe_sparse_matrix : scipy.sparse.csr_matrix, optional
            Pre-computed sparse FE dummy matrix.

        Returns
        -------
        x_demeaned : np.ndarray
            Demeaned variable (residuals after projecting out FEs).
        success : bool
            True if solver converged/succeeded.
        """
        # Override maxiter if not set in __init__
        if self.solver_maxiter is None:
            self.solver_maxiter = maxiter

        D = fe_sparse_matrix
        if self.use_gpu:
            if x.dtype != self.dtype:
                x_converted: NDArray[Any] = x.astype(self.dtype, copy=False)
                x_device = cp.asarray(x_converted)
            else:
                x_device = cp.asarray(x)

            if weights.dtype != self.dtype:
                weights_converted: NDArray[Any] = weights.astype(self.dtype, copy=False)
                weights_device = cp.asarray(weights_converted)
            else:
                weights_device = cp.asarray(weights)

            if D is not None and D.dtype != self.dtype:
                D_converted = D.astype(self.dtype)
                D_device = cp_sparse.csr_matrix(D_converted)
            elif D is not None:
                D_device = cp_sparse.csr_matrix(D)
            else:
                raise ValueError("fe_sparse_matrix cannot be None")
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
            D_weighted, x_weighted, D_device, x_device, weights_device
        )

        if self.use_gpu:
            if self.dtype == np.float64:
                x_demeaned = cp.asnumpy(x_demeaned)
            else:
                x_demeaned_f64 = x_demeaned.astype(np.float64)
                x_demeaned = cp.asnumpy(x_demeaned_f64)

        return x_demeaned, success


def create_fe_sparse_matrix(fe: pd.DataFrame) -> sp_sparse.csr_matrix:
    "Create sparse fixed effects matrix using formulaic."
    fe_fml = " + ".join([f"C({col})" for col in fe.columns])
    FML = Formula(fe_fml)
    D = FML.get_model_matrix(data=fe, output="sparse")
    return D.tocsr()


def demean_cupy(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
    dtype: type = np.float64,
) -> tuple[NDArray[np.float64], bool]:
    """
    Functional interface for CuPy demeaner.

    Parameters
    ----------
    dtype : type, default=np.float64
        Data type for GPU computations (np.float32 or np.float64).
    """
    if weights is None:
        weights = np.ones(x.shape[0] if x.ndim > 1 else len(x))

    if flist is None:
        raise ValueError("flist cannot be None")

    n_fe = flist.shape[1] if flist.ndim > 1 else 1
    fe_df = pd.DataFrame(flist, columns=[f"f{i + 1}" for i in range(n_fe)], copy=False)
    fe_sparse_matrix = create_fe_sparse_matrix(fe_df)

    return CupyFWLDemeaner(dtype=dtype).demean(
        x, flist, weights, tol, maxiter, fe_sparse_matrix=fe_sparse_matrix
    )


def demean_cupy32(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> tuple[NDArray[np.float64], bool]:
    """
    CuPy demeaner using float32 precision (faster on GPU, ~2x speedup).

    May have numerical stability issues for ill-conditioned problems.
    Results are converted back to float64 for downstream precision.
    """
    return demean_cupy(x, flist, weights, tol, maxiter, dtype=np.float32)


def demean_cupy64(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> tuple[NDArray[np.float64], bool]:
    """
    CuPy demeaner using float64 precision (more accurate, safer default).

    Slower than float32 but provides better numerical stability.
    """
    return demean_cupy(x, flist, weights, tol, maxiter, dtype=np.float64)


def demean_scipy(
    x: NDArray[np.float64],
    flist: Optional[NDArray[np.uint64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> tuple[NDArray[np.float64], bool]:
    "Scipy demeaner using float64 precision (CPU-only, no GPU)."
    if weights is None:
        weights = np.ones(x.shape[0] if x.ndim > 1 else len(x))

    if flist is None:
        raise ValueError("flist cannot be None")

    n_fe = flist.shape[1] if flist.ndim > 1 else 1
    fe_df = pd.DataFrame(flist, columns=[f"f{i + 1}" for i in range(n_fe)], copy=False)
    fe_sparse_matrix = create_fe_sparse_matrix(fe_df)

    # Force CPU usage (use_gpu=False) and disable warnings
    return CupyFWLDemeaner(
        use_gpu=False, warn_on_cpu_fallback=False, dtype=np.float64
    ).demean(x, flist, weights, tol, maxiter, fe_sparse_matrix=fe_sparse_matrix)
