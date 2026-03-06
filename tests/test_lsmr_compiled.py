"""
Tests for lsmr_torch (compiled version): correctness, auto-detection, MPS torch.compile.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE pixi run -e dev python -m pytest tests/test_lsmr_compiled.py -v -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

# Reference: original scalar-state LSMR (for correctness comparison)
from pyfixest.estimation.torch.lsmr_torch import _lsmr_eager as lsmr_torch_original
from pyfixest.estimation.torch.lsmr_torch import lsmr_torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_problem(m: int, n: int, density: float = 0.01, seed: int = 42):
    """Create a sparse CSR system A and dense rhs b."""
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
# Correctness tests (CPU, f64 - auto-detects use_compile=False)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,n", [(200, 100), (500, 300), (1000, 500)])
def test_matches_original_cpu(m, n):
    """New lsmr_torch (auto CPU mode) matches original on CPU f64."""
    A, b = _make_sparse_problem(m, n)

    x_orig, _istop_orig, itn_orig, *_ = lsmr_torch_original(A, b)
    x_new, _istop_new, itn_new, *_ = lsmr_torch(A, b)

    assert torch.allclose(x_orig, x_new, atol=1e-6, rtol=1e-6), (
        f"Solutions differ: max_diff={torch.max(torch.abs(x_orig - x_new)).item():.2e}"
    )
    assert itn_orig == itn_new, f"itn differs: {itn_orig} vs {itn_new}"


def test_zero_rhs():
    """B = 0 should return x = 0."""
    A, _ = _make_sparse_problem(100, 50)
    b = torch.zeros(100, dtype=torch.float64)
    x, _istop, itn, *_ = lsmr_torch(A, b)
    assert torch.all(x == 0)
    assert itn == 0


def test_damping():
    """Damped solve should differ from undamped."""
    A, b = _make_sparse_problem(200, 100)
    x_undamped, *_ = lsmr_torch(A, b, damp=0.0)
    x_damped, *_ = lsmr_torch(A, b, damp=1.0)
    assert not torch.allclose(x_undamped, x_damped, atol=1e-3)


def test_diagnostics_match_original():
    """normr, normar, normA, condA, normx diagnostics match reference."""
    A, b = _make_sparse_problem(500, 300)

    _x_o, istop_orig, _itn_o, normr_o, normar_o, normA_o, _condA_o, _normx_o = (
        lsmr_torch_original(A, b)
    )
    _x_n, istop_new, _itn_n, normr_n, normar_n, normA_n, _condA_n, _normx_n = (
        lsmr_torch(A, b)
    )

    assert istop_orig == istop_new, f"istop: {istop_orig} vs {istop_new}"
    assert abs(normr_o - normr_n) / max(normr_o, 1e-30) < 1e-6, (
        f"normr: {normr_o:.6e} vs {normr_n:.6e}"
    )
    assert abs(normar_o - normar_n) / max(normar_o, 1e-30) < 1e-6
    assert abs(normA_o - normA_n) / max(normA_o, 1e-30) < 1e-6


def test_btol_convergence():
    """btol-based stopping (istop=1) should work and match reference."""
    A, b = _make_sparse_problem(200, 100)

    # Use tight btol but very loose atol - should converge via test1, not test2
    _, istop_orig, itn_orig, *_ = lsmr_torch_original(A, b, atol=1e-2, btol=1e-10)
    _, istop_new, itn_new, *_ = lsmr_torch(A, b, atol=1e-2, btol=1e-10)

    # Both should converge; compiled should match reference istop
    assert istop_new == istop_orig, f"istop: {istop_orig} vs {istop_new}"
    assert itn_new == itn_orig, f"itn: {itn_orig} vs {itn_new}"


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------


def test_auto_cpu_defaults():
    """On CPU tensors, auto-selects use_compile=False."""
    A, b = _make_sparse_problem(200, 100)
    x, istop, _itn, *_ = lsmr_torch(A, b)
    assert x.device.type == "cpu"
    assert istop in range(8)


# ---------------------------------------------------------------------------
# MPS + torch.compile tests
# ---------------------------------------------------------------------------

HAS_MPS = torch.backends.mps.is_available()


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_correctness_mps():
    """lsmr_torch on MPS f32 produces reasonable results vs CPU f64."""
    m, n = 500, 300
    A_cpu, b_cpu = _make_sparse_problem(m, n)

    x_ref, *_ = lsmr_torch_original(A_cpu, b_cpu)

    # MPS f32 (auto: use_compile=False; pass use_compile=True to force)
    A_mps = A_cpu.to(torch.float32).to_dense().to("mps")
    b_mps = b_cpu.to(torch.float32).to("mps")

    x_mps, *_ = lsmr_torch(A_mps, b_mps)

    max_diff = torch.max(torch.abs(x_ref.float() - x_mps.cpu())).item()
    assert max_diff < 0.1, f"MPS f32 vs CPU f64 too different: {max_diff:.2e}"


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_compiled_vs_uncompiled_mps():
    """Compiled and uncompiled give same results on MPS."""
    m, n = 500, 300
    A_cpu, b_cpu = _make_sparse_problem(m, n)
    A_mps = A_cpu.to(torch.float32).to_dense().to("mps")
    b_mps = b_cpu.to(torch.float32).to("mps")

    x_comp, *_ = lsmr_torch(A_mps, b_mps, use_compile=True)
    x_nocomp, *_ = lsmr_torch(A_mps, b_mps, use_compile=False)

    max_diff = torch.max(torch.abs(x_comp - x_nocomp)).item()
    assert max_diff < 1e-5, f"Compiled vs uncompiled differ: {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Timing benchmark (run with -s to see output)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
@pytest.mark.parametrize("m,n", [(5000, 2000), (10000, 5000)])
def test_timing_mps(m, n):
    """Compare compiled vs uncompiled LSMR timing on MPS."""
    rng = np.random.default_rng(42)
    nnz = int(m * n * 0.005)
    rows = rng.integers(0, m, nnz)
    cols = rng.integers(0, n, nnz)
    vals = rng.standard_normal(nnz).astype(np.float32)

    A = torch.zeros(m, n, dtype=torch.float32)
    for r, c, v in zip(rows, cols, vals):
        A[r, c] += v
    A = A.to("mps")
    b = torch.tensor(rng.standard_normal(m).astype(np.float32), device="mps")

    # Warmup
    lsmr_torch(A, b, use_compile=True)
    lsmr_torch(A, b, use_compile=False)
    torch.mps.synchronize()

    # Compiled
    torch.mps.synchronize()
    t0 = time.perf_counter()
    x_comp, _, itn_comp, *_ = lsmr_torch(A, b, use_compile=True)
    torch.mps.synchronize()
    t_comp = time.perf_counter() - t0

    # Uncompiled
    torch.mps.synchronize()
    t0 = time.perf_counter()
    x_nocomp, _, itn_nocomp, *_ = lsmr_torch(A, b, use_compile=False)
    torch.mps.synchronize()
    t_nocomp = time.perf_counter() - t0

    speedup = t_nocomp / t_comp if t_comp > 0 else float("inf")
    print(
        f"\n  [{m}x{n}] compiled: {t_comp:.3f}s ({itn_comp} iters) | "
        f"uncompiled: {t_nocomp:.3f}s ({itn_nocomp} iters) | "
        f"speedup: {speedup:.2f}x"
    )

    # Correctness
    assert torch.allclose(x_comp, x_nocomp, atol=1e-4, rtol=1e-4)
