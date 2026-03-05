"""
FWL demeaning via LSMR in pure PyTorch.

Builds a sparse dummy matrix D directly from integer-encoded fixed effects
(no formulaic/pandas dependency), then solves D @ theta = x via LSMR per column.
Diagonal preconditioning normalizes column norms when group sizes vary.

Sparse format is chosen per device: CSR on CUDA/CPU (cuSPARSE / native),
COO on MPS (Metal does not support sparse CSR).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from pyfixest.estimation.torch.lsmr_torch import lsmr_torch


def _get_device(dtype: torch.dtype = torch.float64) -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU.

    MPS does not support float64, so we fall back to CPU when float64 is needed.
    When MPS is available but dtype is float64, a hint is issued to use float32.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if dtype != torch.float64:
            return torch.device("mps")
        warnings.warn(
            "MPS GPU is available but requires float32. "
            "Pass `dtype=torch.float32` to `demean_torch` for GPU acceleration. "
            "Falling back to CPU.",
            UserWarning,
            stacklevel=3,
        )
        return torch.device("cpu")
    warnings.warn(
        "No GPU available — torch demeaning will run on CPU, which is slower "
        "than the scipy backend. Consider using `demean_scipy` instead.",
        UserWarning,
        stacklevel=3,
    )
    return torch.device("cpu")


def _build_sparse_dummy(
    flist: NDArray[np.uint64],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build sparse dummy matrix D from integer-encoded FE array.

    For n_factors factors, D has shape (N, total_groups) where
    total_groups = sum of unique groups per factor. Columns are stacked:
    [factor0_groups | factor1_groups | ...].

    No reference-level dropping — LSMR finds the min-norm solution,
    which gives correct residuals for rank-deficient systems.

    Returns COO on MPS (Metal has no CSR kernels), CSR otherwise.

    Parameters
    ----------
    flist : np.ndarray, shape (N, n_factors), dtype uint64
        Integer-encoded fixed effects. Must be contiguous 0-based integers
        per factor (i.e., values in [0, n_groups_j) with no gaps).
    device : torch.device
        Target device.
    dtype : torch.dtype
        Value dtype (e.g. torch.float64).

    Returns
    -------
    D : torch.Tensor
        Sparse tensor of shape (N, total_groups). COO on MPS, CSR otherwise.

    Raises
    ------
    ValueError
        If any factor has non-contiguous group IDs (gaps in the integer encoding).
    """
    N, n_factors = flist.shape

    row_indices = []
    col_indices = []
    col_offset = 0

    for j in range(n_factors):
        col_j = flist[:, j]
        unique_vals = np.unique(col_j)
        n_unique_j = len(unique_vals)
        n_groups_j = int(unique_vals[-1]) + 1

        if n_groups_j != n_unique_j:
            raise ValueError(
                f"Factor {j} has non-contiguous group IDs: "
                f"max ID is {n_groups_j - 1} but only {n_unique_j} unique values found. "
                f"Re-encode to contiguous 0-based integers."
            )

        rows = np.arange(N, dtype=np.int64)
        cols = col_j.astype(np.int64) + col_offset

        row_indices.append(rows)
        col_indices.append(cols)
        col_offset += n_groups_j

    all_rows = np.concatenate(row_indices)
    all_cols = np.concatenate(col_indices)

    indices = torch.tensor(
        np.stack([all_rows, all_cols]), dtype=torch.long, device=device
    )
    values = torch.ones(len(all_rows), dtype=dtype, device=device)

    D_coo = torch.sparse_coo_tensor(indices, values, size=(N, col_offset))

    if device.type == "mps":
        return D_coo.coalesce()
    return D_coo.to_sparse_csr()


def _scale_sparse_rows(D: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale rows of a sparse matrix by a dense vector (row-wise multiply).

    Dispatches to the appropriate implementation based on sparse layout:
    CSR on CUDA/CPU, COO on MPS.
    """
    if D.layout == torch.sparse_csr:
        return _scale_csr_rows(D, scale)
    return _scale_coo_rows(D, scale)


def _scale_csr_rows(D: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale rows of a sparse CSR matrix.

    Operates directly on CSR internal arrays, avoiding COO format roundtrip.
    Uses repeat_interleave to expand per-row scales to per-nonzero scales.
    """
    crow = D.crow_indices()
    col = D.col_indices()
    row_counts = crow[1:] - crow[:-1]
    val = D.values() * torch.repeat_interleave(scale, row_counts)

    return torch.sparse_csr_tensor(
        crow, col, val, D.shape, dtype=D.dtype, device=D.device
    )


def _scale_coo_rows(D: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale rows of a sparse COO matrix.

    Indexes into the scale vector using COO row indices to expand
    per-row scales to per-nonzero scales.
    """
    d_indices = D.indices()
    new_values = D.values() * scale[d_indices[0]]
    return torch.sparse_coo_tensor(
        d_indices, new_values, D.shape, device=D.device
    ).coalesce()


def demean_torch(
    x: NDArray[np.float64],
    flist: NDArray[np.uint64] | None = None,
    weights: NDArray[np.float64] | None = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
    dtype: torch.dtype = torch.float64,
) -> tuple[NDArray[np.float64], bool]:
    """
    Demean x by projecting out fixed effects via FWL + LSMR.

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N, K)
        Variables to demean.
    flist : np.ndarray, shape (N, n_factors), dtype uint64
        Integer-encoded fixed effects.
    weights : np.ndarray, shape (N,)
        Observation weights (1.0 for equal weighting).
    tol : float
        Convergence tolerance for LSMR (used as both atol and btol).
    maxiter : int
        Maximum LSMR iterations per column.
    dtype : torch.dtype
        Tensor dtype. Use ``torch.float32`` to enable MPS (Apple GPU)
        acceleration. Default ``torch.float64`` for full precision.

    Returns
    -------
    x_demeaned : np.ndarray
        Residuals after projecting out fixed effects. Same shape as input x.
    success : bool
        True if LSMR converged for all columns.
    """
    if flist is None:
        raise ValueError("flist cannot be None")
    if weights is None:
        weights = np.ones(x.shape[0], dtype=np.float64)

    # Track original shape to restore on output
    was_1d = x.ndim == 1
    x_2d = x[:, None] if was_1d else x
    weights_1d = weights.ravel()

    device = _get_device(dtype)

    # Move to torch — use from_numpy for zero-copy when staying on CPU
    x_c = x_2d if x_2d.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x_2d)
    w_c = weights_1d if weights_1d.flags["C_CONTIGUOUS"] else np.ascontiguousarray(weights_1d)
    x_t = torch.from_numpy(x_c).to(dtype=dtype, device=device)
    w_t = torch.from_numpy(w_c).to(dtype=dtype, device=device)

    # Ensure flist is 2D
    flist_2d = flist if flist.ndim == 2 else flist[:, None]

    # Build sparse dummy matrix (unweighted)
    D_unweighted = _build_sparse_dummy(flist_2d, device, dtype)
    _, D_cols = D_unweighted.shape
    K = x_t.shape[1]

    # Apply sqrt-weights
    sqrt_w = torch.sqrt(w_t)
    x_w = x_t * sqrt_w[:, None]

    # Weight the dummy matrix: D_weighted = diag(sqrt_w) @ D_unweighted
    D_weighted = _scale_sparse_rows(D_unweighted, sqrt_w)

    # Diagonal preconditioning: M_inv = 1 / sqrt(D_unweighted^T @ w)
    # D_unweighted^T @ w gives the sum of weights per group
    group_weights = D_unweighted.t() @ w_t

    # Guard against zero-weight groups (would produce inf in M_inv)
    zero_weight_mask = group_weights <= 0.0
    if zero_weight_mask.any():
        bad_groups = zero_weight_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        raise ValueError(
            f"Fixed effect groups {bad_groups} have zero total weight. "
            f"Check your weights or fixed effect encoding."
        )

    M_inv = 1.0 / torch.sqrt(group_weights)

    # Build preconditioned operator once (not per column)
    A_precond = _PreconditionedSparse(D_weighted, M_inv)

    # Solve for each column
    theta = torch.zeros(D_cols, K, dtype=dtype, device=device)
    success = True

    for k in range(K):
        z, istop, itn, normr, normar, normA, condA, normx = lsmr_torch(
            A_precond,
            x_w[:, k],
            damp=0.0,
            atol=tol,
            btol=tol,
            maxiter=maxiter,
        )

        # Recover theta from preconditioned solution: theta = M_inv * z
        theta[:, k] = M_inv * z
        success = success and (istop in (1, 2, 3))

    # Compute residuals: x_demeaned = x - D_unweighted @ theta
    x_demeaned = x_t - D_unweighted @ theta

    result = x_demeaned.cpu().numpy()
    if was_1d:
        result = result[:, 0]

    return result, success


class _PreconditionedSparse:
    """
    Wraps a sparse matrix D and diagonal preconditioner M_inv
    to present A_precond = D @ diag(M_inv) for LSMR.

    This avoids forming the preconditioned matrix explicitly —
    just element-wise multiply before/after matvec.

    The transpose view is cached and returned by `.t()`, so LSMR's
    repeated `A.t().mv(u)` calls don't allocate a new object each time.
    """

    def __init__(
        self, D: torch.Tensor, M_inv: torch.Tensor, *, _transposed: bool = False
    ):
        m, n = D.shape
        self.shape = (n, m) if _transposed else (m, n)
        self._D = D
        self._M_inv = M_inv
        self._transposed = _transposed
        self._T: _PreconditionedSparse | None = None

    def mv(self, v: torch.Tensor) -> torch.Tensor:
        if self._transposed:
            # Compute M_inv * (D^T @ u)
            return self._M_inv * (self._D.t() @ v)
        # Compute D @ (M_inv * v)
        return self._D @ (self._M_inv * v)

    def t(self) -> _PreconditionedSparse:
        """Return cached transpose view."""
        if self._T is None:
            self._T = _PreconditionedSparse(
                self._D, self._M_inv, _transposed=not self._transposed
            )
            self._T._T = self  # cross-link so .t().t() returns self
        return self._T
