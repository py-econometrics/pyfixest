"""Tests for compiled-vs-eager LSMR behavior."""

from __future__ import annotations

import pytest

from pyfixest.estimation.torch.lsmr_torch import (
    _lsmr_batched,
    _lsmr_compiled,
    _lsmr_compiled_batched,
    _lsmr_eager,
    lsmr_torch,
    lsmr_torch_batched,
)
from tests._lsmr_test_utils import (
    HAS_CUDA,
    HAS_MPS,
    make_rhs,
    make_sparse_problem,
    torch,
)


@pytest.mark.parametrize("m,n", [(200, 100), (500, 300), (1000, 500)])
def test_single_rhs_compiled_matches_eager_on_cpu(m, n):
    """The compiled single-RHS core should match the eager core on CPU."""
    A = make_sparse_problem(m, n, density=0.01)
    b = make_rhs(m, 1).squeeze(1)

    x_eager, istop_eager, itn_eager, *_ = _lsmr_eager(A, b)
    x_comp, istop_comp, itn_comp, *_ = _lsmr_compiled(A, b, use_compile=False)

    assert torch.allclose(x_eager, x_comp, atol=1e-6, rtol=1e-6)
    assert istop_eager == istop_comp
    assert itn_eager == itn_comp


@pytest.mark.parametrize("k", [1, 5, 20])
def test_batched_compiled_matches_eager_on_cpu(k):
    """The compiled batched core should match the eager batched core on CPU."""
    m, n = 300, 150
    A = make_sparse_problem(m, n, seed=77)
    B = make_rhs(m, k, seed=88)

    X_eager, istop_eager, itn_eager, *_ = _lsmr_batched(A, B)
    X_comp, istop_comp, itn_comp, *_ = _lsmr_compiled_batched(A, B, use_compile=False)

    assert torch.allclose(X_eager, X_comp, atol=1e-6, rtol=1e-6)
    assert torch.equal(istop_eager, istop_comp)
    assert itn_eager == itn_comp


@pytest.mark.parametrize(
    ("solver", "rhs_factory", "kwargs"),
    [
        (_lsmr_eager, lambda m: make_rhs(m, 1, seed=321).squeeze(1), {}),
        (
            _lsmr_compiled,
            lambda m: make_rhs(m, 1, seed=321).squeeze(1),
            {"use_compile": False},
        ),
        (_lsmr_batched, lambda m: make_rhs(m, 3, seed=321), {}),
        (
            _lsmr_compiled_batched,
            lambda m: make_rhs(m, 3, seed=321),
            {"use_compile": False},
        ),
    ],
)
def test_compiled_and_eager_paths_reject_invalid_maxiter(solver, rhs_factory, kwargs):
    """All eager/compiled internal solvers should reject non-positive maxiter."""
    A = make_sparse_problem(50, 20)
    rhs = rhs_factory(50)

    with pytest.raises(ValueError, match="maxiter must be a positive integer"):
        solver(A, rhs, maxiter=0, **kwargs)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param(
            "mps",
            torch.float32,
            marks=pytest.mark.skipif(not HAS_MPS, reason="MPS not available"),
        ),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available"),
        ),
    ],
)
def test_single_rhs_compiled_vs_uncompiled_on_device(device, dtype):
    """On supported accelerators, compiled and uncompiled single-RHS solves should agree."""
    A_cpu = make_sparse_problem(500, 300, density=0.01, seed=42)
    b_cpu = make_rhs(500, 1, seed=123).squeeze(1)

    A_device = A_cpu.to(dtype).to_dense().to(device)
    b_device = b_cpu.to(dtype).to(device)

    x_comp, *_ = lsmr_torch(A_device, b_device, use_compile=True)
    x_nocomp, *_ = lsmr_torch(A_device, b_device, use_compile=False)

    assert torch.allclose(x_comp, x_nocomp, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param(
            "mps",
            torch.float32,
            marks=pytest.mark.skipif(not HAS_MPS, reason="MPS not available"),
        ),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available"),
        ),
    ],
)
def test_batched_compiled_vs_uncompiled_on_device(device, dtype):
    """On supported accelerators, compiled and uncompiled batched solves should agree."""
    m, n, k = 300, 150, 5
    A_cpu = make_sparse_problem(m, n, density=0.1, seed=42)
    B_cpu = make_rhs(m, k, seed=123)

    A_device = A_cpu.to(dtype).to_dense().to(device)
    B_device = B_cpu.to(dtype).to(device)

    X_comp, *_ = lsmr_torch_batched(A_device, B_device, use_compile=True)
    X_nocomp, *_ = lsmr_torch_batched(A_device, B_device, use_compile=False)

    assert torch.allclose(X_comp, X_nocomp, atol=1e-4, rtol=1e-4)
