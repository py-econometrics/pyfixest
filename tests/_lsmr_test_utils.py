from __future__ import annotations

import numpy as np
import pytest

from tests._torch_test_utils import HAS_CUDA as HAS_CUDA
from tests._torch_test_utils import HAS_MPS as HAS_MPS
from tests._torch_test_utils import torch

if torch is None:  # pragma: no cover - environment dependent
    pytest.skip("torch not available", allow_module_level=True)


def make_sparse_problem(m: int, n: int, density: float = 0.1, seed: int = 42):
    """Create a sparse CSR system for LSMR tests."""
    rng = np.random.default_rng(seed)
    nnz = int(m * n * density)
    rows = rng.integers(0, m, nnz)
    cols = rng.integers(0, n, nnz)
    vals = rng.standard_normal(nnz)

    A_coo = torch.sparse_coo_tensor(
        torch.tensor(np.stack([rows, cols])),
        torch.tensor(vals, dtype=torch.float64),
        size=(m, n),
    )
    return A_coo.to_sparse_csr()


def make_rhs(m: int, k: int, seed: int = 42) -> torch.Tensor:
    """Create a dense RHS matrix of shape ``(m, k)``."""
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((m, k)), dtype=torch.float64)
