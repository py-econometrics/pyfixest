from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


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

    No reference-level dropping - LSMR finds the min-norm solution,
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
    """Scale rows of a sparse matrix by a dense vector (row-wise multiply)."""
    if D.layout == torch.sparse_csr:
        return _scale_csr_rows(D, scale)
    return _scale_coo_rows(D, scale)


def _scale_csr_rows(D: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale rows of a sparse CSR matrix."""
    crow = D.crow_indices()
    col = D.col_indices()
    row_counts = crow[1:] - crow[:-1]
    val = D.values() * torch.repeat_interleave(scale, row_counts)

    return torch.sparse_csr_tensor(
        crow, col, val, D.shape, dtype=D.dtype, device=D.device
    )


def _scale_coo_rows(D: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale rows of a sparse COO matrix."""
    d_indices = D.indices()
    new_values = D.values() * scale[d_indices[0]]
    return torch.sparse_coo_tensor(
        d_indices, new_values, D.shape, device=D.device
    ).coalesce()
