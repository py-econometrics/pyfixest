"""Tests for batched-vs-single LSMR behavior."""

from __future__ import annotations

import pytest

from pyfixest.estimation.torch.lsmr_torch import lsmr_torch, lsmr_torch_batched
from tests._lsmr_test_utils import make_rhs, make_sparse_problem, torch


@pytest.mark.parametrize("k", [1, 5, 20])
def test_batched_matches_single_rhs_solves(k):
    """Batched LSMR should match K sequential single-RHS solves."""
    m, n = 300, 150
    A = make_sparse_problem(m, n, seed=77)
    B = make_rhs(m, k, seed=88)

    X_batch, *_ = lsmr_torch_batched(A, B)

    for col in range(k):
        x_single, *_ = lsmr_torch(A, B[:, col])
        assert torch.allclose(X_batch[:, col], x_single, atol=1e-6, rtol=1e-6), (
            f"K={k}, col={col}, max diff="
            f"{torch.max(torch.abs(X_batch[:, col] - x_single)).item():.2e}"
        )


def test_batched_zero_rhs_column_matches_single():
    """A zero RHS column should stay zero while other columns still match single solves."""
    m, n = 100, 50
    A = make_sparse_problem(m, n)
    B = make_rhs(m, 3, seed=42)
    B[:, 1] = 0.0

    X_batch, *_ = lsmr_torch_batched(A, B)

    assert torch.allclose(
        X_batch[:, 1], torch.zeros(n, dtype=torch.float64), atol=1e-12
    )

    for col in (0, 2):
        x_single, *_ = lsmr_torch(A, B[:, col])
        assert torch.allclose(X_batch[:, col], x_single, atol=1e-6, rtol=1e-6)


def test_batched_all_zero_rhs_returns_zero_solution():
    """All-zero RHS should return immediately with an all-zero solution."""
    m, n, k = 100, 50, 3
    A = make_sparse_problem(m, n)
    B = torch.zeros(m, k, dtype=torch.float64)

    X_batch, _istop, itn, *_ = lsmr_torch_batched(A, B)

    assert torch.allclose(X_batch, torch.zeros(n, k, dtype=torch.float64), atol=1e-12)
    assert itn == 0


def test_batched_damped_matches_single_rhs_solves():
    """Damped batched LSMR should match damped single-RHS solves."""
    m, n, k = 200, 100, 3
    A = make_sparse_problem(m, n)
    B = make_rhs(m, k, seed=55)
    damp = 5.0

    X_batch, *_ = lsmr_torch_batched(A, B, damp=damp)

    for col in range(k):
        x_single, *_ = lsmr_torch(A, B[:, col], damp=damp)
        assert torch.allclose(X_batch[:, col], x_single, atol=1e-6, rtol=1e-6), (
            f"damped col={col}, max diff="
            f"{torch.max(torch.abs(X_batch[:, col] - x_single)).item():.2e}"
        )


def test_batched_invalid_maxiter_raises():
    """The public batched solver should reject non-positive maxiter."""
    m, n, k = 50, 20, 3
    A = make_sparse_problem(m, n)
    B = make_rhs(m, k, seed=321)

    with pytest.raises(ValueError, match="maxiter must be a positive integer"):
        lsmr_torch_batched(A, B, maxiter=0)
