"""
Correctness and timing tests for lsmr_torch_fused vs lsmr_torch.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE pixi run -e dev python -m pytest tests/test_lsmr_fused.py -v -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from pyfixest.estimation.torch.lsmr_torch import lsmr_torch
from pyfixest.estimation.torch.lsmr_torch_fused import lsmr_torch_fused

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_problem(m: int, n: int, density: float = 0.01, seed: int = 42):
    """Create a sparse CSR system A and dense rhs b on the given device."""
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
    A_csr = A_coo.to_sparse_csr()
    b = torch.tensor(rng.standard_normal(m), dtype=torch.float64)
    return A_csr, b


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,n", [(200, 100), (500, 300), (1000, 500)])
def test_fused_matches_original(m, n):
    """The fused solver should produce the same solution as the original."""
    A, b = _make_sparse_problem(m, n)

    x_orig, istop_orig, itn_orig, *_ = lsmr_torch(A, b)
    x_fused, istop_fused, itn_fused, *_ = lsmr_torch_fused(A, b, check_every=1)

    # Solutions should match closely
    assert torch.allclose(x_orig, x_fused, atol=1e-6, rtol=1e-6), (
        f"Solutions differ: max_diff={torch.max(torch.abs(x_orig - x_fused)).item():.2e}"
    )
    # Iteration count may differ slightly due to convergence check frequency
    assert abs(itn_orig - itn_fused) <= 1, f"itn differs: {itn_orig} vs {itn_fused}"


@pytest.mark.parametrize("check_every", [1, 5, 10, 50])
def test_check_every_correctness(check_every):
    """Different check_every values should all converge to the same solution."""
    A, b = _make_sparse_problem(500, 300)
    x_ref, *_ = lsmr_torch(A, b)
    x_fused, _, itn, *_ = lsmr_torch_fused(A, b, check_every=check_every)

    assert torch.allclose(x_ref, x_fused, atol=1e-6, rtol=1e-6), (
        f"check_every={check_every}: max_diff="
        f"{torch.max(torch.abs(x_ref - x_fused)).item():.2e}"
    )


def test_zero_rhs():
    """b = 0 should return x = 0."""
    A, _ = _make_sparse_problem(100, 50)
    b = torch.zeros(100, dtype=torch.float64)
    x, istop, itn, *_ = lsmr_torch_fused(A, b)
    assert torch.all(x == 0)
    assert itn == 0


def test_damping():
    """Damped solve should differ from undamped."""
    A, b = _make_sparse_problem(200, 100)
    x_undamped, *_ = lsmr_torch_fused(A, b, damp=0.0)
    x_damped, *_ = lsmr_torch_fused(A, b, damp=1.0)
    assert not torch.allclose(x_undamped, x_damped, atol=1e-3)


# ---------------------------------------------------------------------------
# Branchless _sym_ortho tests
# ---------------------------------------------------------------------------


def test_sym_ortho_matches_scipy():
    """Branchless _sym_ortho_t should match SciPy's convention."""
    import math

    from pyfixest.estimation.torch.lsmr_torch import _sym_ortho
    from pyfixest.estimation.torch.lsmr_torch_fused import _sym_ortho_t

    cases = [
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (0.0, 5.0),
        (5.0, 0.0),
        (0.0, 0.0),
        (1e-300, 1e-300),
        (1e300, 1e300),
        (1.0, 1e-15),
    ]
    for a_val, b_val in cases:
        c_ref, s_ref, r_ref = _sym_ortho(a_val, b_val)
        a_t = torch.tensor(a_val, dtype=torch.float64)
        b_t = torch.tensor(b_val, dtype=torch.float64)
        c_t, s_t, r_t = _sym_ortho_t(a_t, b_t)
        assert abs(c_t.item() - c_ref) < 1e-10, f"c mismatch for ({a_val}, {b_val})"
        assert abs(s_t.item() - s_ref) < 1e-10, f"s mismatch for ({a_val}, {b_val})"
        assert abs(r_t.item() - r_ref) < 1e-10, f"r mismatch for ({a_val}, {b_val})"


# ---------------------------------------------------------------------------
# Timing benchmark (not a test — run with -s to see output)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,n", [(5000, 2000), (10000, 5000)])
def test_timing_comparison(m, n):
    """Compare wall time of original vs fused LSMR."""
    A, b = _make_sparse_problem(m, n, density=0.005)

    # Warmup
    lsmr_torch(A, b, maxiter=5)
    lsmr_torch_fused(A, b, maxiter=5)

    # Original
    t0 = time.perf_counter()
    x_orig, _, itn_orig, *_ = lsmr_torch(A, b)
    t_orig = time.perf_counter() - t0

    # Fused (check every 10)
    t0 = time.perf_counter()
    x_fused, _, itn_fused, *_ = lsmr_torch_fused(A, b, check_every=10)
    t_fused = time.perf_counter() - t0

    speedup = t_orig / t_fused if t_fused > 0 else float("inf")
    print(
        f"\n  [{m}x{n}] original: {t_orig:.3f}s ({itn_orig} iters) | "
        f"fused: {t_fused:.3f}s ({itn_fused} iters) | "
        f"speedup: {speedup:.2f}x"
    )

    # Correctness sanity check
    assert torch.allclose(x_orig, x_fused, atol=1e-5, rtol=1e-5)
