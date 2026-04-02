"""
Tests for standalone PyTorch LSMR solver and torch-based FWL demeaning.

Three test levels:
1. Bare LSMR: verify the solver on known linear systems
2. Demeaning vs pyhdfe: compare demean_torch against pyhdfe reference
3. Demeaning vs demean_scipy: compare against SciPy LSMR-based backend
"""

import numpy as np
import pyhdfe
import pytest

torch = pytest.importorskip("torch")

from pyfixest.estimation.cupy.demean_cupy_ import demean_scipy  # noqa: E402
from pyfixest.estimation.torch.demean_torch_ import demean_torch  # noqa: E402
from pyfixest.estimation.torch.lsmr_torch import lsmr_torch  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demean_data():
    """Shared data fixture for all demeaning tests (matches test_demean.py pattern)."""
    rng = np.random.default_rng(929291)
    N = 1000
    M = 10
    x = rng.normal(0, 1, M * N).reshape((N, M))
    f1 = rng.choice(list(range(M)), N).reshape((N, 1))
    f2 = rng.choice(list(range(M)), N).reshape((N, 1))
    flist = np.concatenate((f1, f2), axis=1).astype(np.uint64)
    # Weights drawn from the *same* advanced RNG (not a fresh seed)
    weights = rng.uniform(0, 1, N)
    return x, flist, weights


# ---------------------------------------------------------------------------
# Level 1: Bare LSMR tests
# ---------------------------------------------------------------------------


class TestLSMR:
    """Unit tests for the pure-torch LSMR solver."""

    def test_overdetermined_known_solution(self):
        """Overdetermined system with exact solution: LSMR should recover it."""
        A_dense = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        )
        x_true = torch.tensor([1.0, -1.0], dtype=torch.float64)
        b = A_dense @ x_true

        A_sparse = A_dense.to_sparse_csr()
        x_sol, istop, _itn, normr, _normar, *_ = lsmr_torch(A_sparse, b)

        assert istop in (1, 2), f"LSMR did not converge, istop={istop}"
        assert torch.allclose(x_sol, x_true, atol=1e-10), (
            f"Solution mismatch: {x_sol} vs {x_true}"
        )
        assert normr < 1e-10, f"Residual too large: {normr}"

    def test_underdetermined_min_norm(self):
        """Underdetermined system (m < n): LSMR should find min-norm solution."""
        torch.manual_seed(42)
        A_dense = torch.randn(2, 4, dtype=torch.float64)
        b = torch.randn(2, dtype=torch.float64)

        A_sparse = A_dense.to_sparse_csr()
        x_sol, _istop, _itn, _normr, _normar, *_ = lsmr_torch(A_sparse, b)

        residual = torch.norm(A_dense @ x_sol - b).item()
        assert residual < 1e-8, f"Residual too large: {residual}"

        x_lstsq = torch.linalg.lstsq(A_dense, b).solution
        assert torch.allclose(x_sol, x_lstsq, atol=1e-6), (
            f"Not min-norm: ||x_lsmr||={torch.norm(x_sol):.6f}, "
            f"||x_lstsq||={torch.norm(x_lstsq):.6f}"
        )

    def test_larger_sparse_system(self):
        """Larger sparse system to exercise the iteration loop."""
        rng = np.random.default_rng(123)
        m, n = 200, 50
        density = 0.1

        nnz = int(m * n * density)
        rows = rng.integers(0, m, nnz)
        cols = rng.integers(0, n, nnz)
        vals = rng.standard_normal(nnz)

        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float64)
        A_sparse = torch.sparse_coo_tensor(indices, values, (m, n)).to_sparse_csr()

        x_true = torch.tensor(rng.standard_normal(n), dtype=torch.float64)
        A_dense = A_sparse.to_dense()
        b = A_dense @ x_true

        x_sol, istop, _itn, _normr, _normar, *_ = lsmr_torch(A_sparse, b)

        assert istop in (1, 2, 3), f"LSMR did not converge, istop={istop}"
        residual = torch.norm(A_dense @ x_sol - b).item()
        assert residual < 1e-5, f"Residual too large: {residual}"

    def test_zero_rhs(self):
        """B = 0 should give x = 0."""
        A_sparse = torch.eye(3, dtype=torch.float64).to_sparse_csr()
        b = torch.zeros(3, dtype=torch.float64)

        x_sol, istop, _itn, _normr, _normar, *_ = lsmr_torch(A_sparse, b)

        assert torch.allclose(x_sol, torch.zeros(3, dtype=torch.float64), atol=1e-12)
        assert istop == 0

    def test_damped_regularization(self):
        """Damping should shrink the solution toward zero."""
        A_sparse = torch.eye(3, dtype=torch.float64).to_sparse_csr()
        b = torch.ones(3, dtype=torch.float64)

        x_undamped, *_ = lsmr_torch(A_sparse, b, damp=0.0)
        x_damped, *_ = lsmr_torch(A_sparse, b, damp=10.0)

        assert torch.norm(x_damped) < torch.norm(x_undamped), (
            "Damped solution should have smaller norm"
        )

    def test_maxiter_exhaustion_returns_istop_7(self):
        """Forcing maxiter=2 on an ill-conditioned system must return istop=7."""
        # Use a system that genuinely needs many iterations
        # (identity converges in 1 step, so we use a harder matrix)
        torch.manual_seed(99)
        A_dense = torch.randn(20, 10, dtype=torch.float64)
        b = torch.randn(20, dtype=torch.float64)
        A_sparse = A_dense.to_sparse_csr()

        _, istop, itn, *_ = lsmr_torch(A_sparse, b, maxiter=2, atol=1e-15, btol=1e-15)

        assert istop == 7, f"Expected istop=7 (maxiter hit), got {istop}"
        assert itn == 2

    def test_full_return_tuple(self):
        """Verify all 8 return values are present and sensible."""
        A_sparse = torch.eye(3, dtype=torch.float64).to_sparse_csr()
        b = torch.ones(3, dtype=torch.float64)

        _x, _istop, _itn, _normr, _normar, normA, condA, normx = lsmr_torch(A_sparse, b)

        assert isinstance(normA, float) and normA > 0
        assert isinstance(condA, float) and condA >= 1.0
        assert isinstance(normx, float) and normx > 0


# ---------------------------------------------------------------------------
# Level 2: Demeaning vs pyhdfe
# ---------------------------------------------------------------------------


class TestDemeanVsPyhdfe:
    """Compare demean_torch against pyhdfe reference."""

    def test_unweighted(self, demean_data):
        """Verify unweighted demeaning matches pyhdfe."""
        x, flist, _ = demean_data
        N = x.shape[0]
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "demean_torch did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-6,
            atol=1e-8,
            err_msg="demean_torch vs pyhdfe mismatch (unweighted)",
        )

    def test_weighted(self, demean_data):
        """Verify weighted demeaning matches pyhdfe."""
        x, flist, weights = demean_data
        N = x.shape[0]

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x, weights.reshape(N, 1))

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "demean_torch did not converge (weighted)"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-6,
            atol=1e-8,
            err_msg="demean_torch vs pyhdfe mismatch (weighted)",
        )


# ---------------------------------------------------------------------------
# Level 3: Demeaning vs demean_scipy
# ---------------------------------------------------------------------------


class TestDemeanVsScipy:
    """Compare demean_torch against demean_scipy (both use LSMR)."""

    def test_unweighted_vs_scipy(self, demean_data):
        """Verify unweighted demeaning matches scipy LSMR."""
        x, flist, _ = demean_data
        N = x.shape[0]
        weights = np.ones(N)

        res_scipy, success_scipy = demean_scipy(x, flist, weights, tol=1e-10)
        res_torch, success_torch = demean_torch(x, flist, weights, tol=1e-10)

        assert success_scipy, "demean_scipy did not converge"
        assert success_torch, "demean_torch did not converge"

        # Different D construction (formulaic drops reference level, we don't)
        # so tolerances are slightly looser than torch-vs-pyhdfe.
        # Both produce valid demeaned residuals, but the underlying theta
        # coefficients differ because the systems have different null spaces.
        np.testing.assert_allclose(
            res_torch,
            res_scipy,
            rtol=1e-5,
            atol=1e-7,
            err_msg="demean_torch vs demean_scipy mismatch (unweighted)",
        )

    def test_weighted_vs_scipy(self, demean_data):
        """Verify weighted demeaning matches scipy LSMR."""
        x, flist, weights = demean_data

        res_scipy, success_scipy = demean_scipy(x, flist, weights, tol=1e-10)
        res_torch, success_torch = demean_torch(x, flist, weights, tol=1e-10)

        assert success_scipy, "demean_scipy did not converge"
        assert success_torch, "demean_torch did not converge"

        np.testing.assert_allclose(
            res_torch,
            res_scipy,
            rtol=1e-5,
            atol=1e-7,
            err_msg="demean_torch vs demean_scipy mismatch (weighted)",
        )


# ---------------------------------------------------------------------------
# Level 4: Edge cases for demean_torch
# ---------------------------------------------------------------------------


class TestDemeanEdgeCases:
    """Edge case tests for demean_torch."""

    def test_1d_x_input(self):
        """1D x input should return 1D output with same shape."""
        rng = np.random.default_rng(42)
        N = 100
        x_1d = rng.normal(0, 1, N)
        flist = rng.choice(5, N).astype(np.uint64).reshape(N, 1)
        weights = np.ones(N)

        res, success = demean_torch(x_1d, flist, weights, tol=1e-10)
        assert success
        assert res.ndim == 1, f"Expected 1D output, got shape {res.shape}"
        assert res.shape == x_1d.shape

    def test_single_fe_factor(self):
        """Single fixed effect factor should work and match pyhdfe."""
        rng = np.random.default_rng(42)
        N = 200
        x = rng.normal(0, 1, (N, 3))
        flist = rng.choice(10, N).astype(np.uint64).reshape(N, 1)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-6, atol=1e-8)

    def test_three_fe_factors(self):
        """Three FE factors should work and match pyhdfe."""
        rng = np.random.default_rng(42)
        N = 500
        x = rng.normal(0, 1, (N, 2))
        f1 = rng.choice(8, N).reshape(N, 1)
        f2 = rng.choice(6, N).reshape(N, 1)
        f3 = rng.choice(4, N).reshape(N, 1)
        flist = np.concatenate([f1, f2, f3], axis=1).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-6, atol=1e-8)

    def test_non_contiguous_group_ids_raises(self):
        """Non-contiguous group IDs (gaps) should raise ValueError."""
        N = 50
        x = np.ones((N, 1))
        # Groups [0, 5, 10] — non-contiguous, max+1=11 but only 3 unique
        flist = np.array(
            [0, 5, 10] * (N // 3) + [0] * (N % 3), dtype=np.uint64
        ).reshape(N, 1)
        weights = np.ones(N)

        with pytest.raises(ValueError, match="non-contiguous group IDs"):
            demean_torch(x, flist, weights)

    def test_zero_weight_group_raises(self):
        """If any group has zero total weight, should raise ValueError."""
        N = 50
        x = np.ones((N, 1))
        # All observations in group 0 get zero weight
        flist = np.zeros(N, dtype=np.uint64).reshape(N, 1)
        flist[N // 2 :, 0] = 1
        weights = np.ones(N)
        weights[: N // 2] = 0.0  # group 0 has zero total weight

        with pytest.raises(ValueError, match="zero total weight"):
            demean_torch(x, flist, weights)

    def test_flist_none_raises(self):
        """flist=None should raise ValueError."""
        with pytest.raises(ValueError, match="flist cannot be None"):
            demean_torch(np.ones((10, 2)), flist=None, weights=np.ones(10))


# ---------------------------------------------------------------------------
# Level 5: High-dimensional stress tests
# ---------------------------------------------------------------------------


class TestHighDimensional:
    """Stress tests with larger N, many groups, and unbalanced designs."""

    def test_many_groups(self):
        """N=10K, 200 groups per factor — exercises LSMR on a larger system."""
        rng = np.random.default_rng(7777)
        N = 10_000
        G = 200
        x = rng.normal(0, 1, (N, 3))
        f1 = rng.choice(G, N).reshape(N, 1)
        f2 = rng.choice(G, N).reshape(N, 1)
        flist = np.concatenate([f1, f2], axis=1).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "demean_torch did not converge on high-D problem"
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-5, atol=1e-7)

    def test_unbalanced_groups(self):
        """Highly unbalanced groups: one large group, many tiny groups.

        This stresses the diagonal preconditioner because group_weights
        vary by orders of magnitude.
        """
        rng = np.random.default_rng(8888)
        N = 5_000
        # Group 0 gets 80% of observations, groups 1-49 share the rest
        groups = np.zeros(N, dtype=np.uint64)
        small_start = int(N * 0.8)
        groups[small_start:] = rng.choice(49, N - small_start).astype(np.uint64) + 1
        flist = groups.reshape(N, 1)

        x = rng.normal(0, 1, (N, 2))
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "demean_torch did not converge on unbalanced groups"
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-5, atol=1e-7)

    def test_many_columns(self):
        """Many dependent variables (K=50) — checks per-column loop scales."""
        rng = np.random.default_rng(9999)
        N = 2_000
        K = 50
        x = rng.normal(0, 1, (N, K))
        flist = rng.choice(20, (N, 2)).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-5, atol=1e-7)

    def test_weighted_high_d(self):
        """Weighted demeaning with many groups — weighted preconditioner stress test."""
        rng = np.random.default_rng(1234)
        N = 10_000
        G = 100
        x = rng.normal(0, 1, (N, 5))
        flist = rng.choice(G, (N, 2)).astype(np.uint64)
        # Highly skewed weights (log-normal)
        weights = rng.lognormal(0, 2, N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x, weights.reshape(N, 1))

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "demean_torch did not converge with skewed weights"
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Level 6: float32 / MPS tests
# ---------------------------------------------------------------------------

HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class TestFloat32:
    """Tests for float32 dtype path (COO on MPS, COO-converted-to-CSR on CPU)."""

    def test_cpu_float32_vs_pyhdfe(self, demean_data):
        """float32 on CPU should match pyhdfe within single-precision tolerance."""
        x, flist, _ = demean_data
        N = x.shape[0]
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(
            x, flist, weights, tol=1e-5, dtype=torch.float32
        )
        assert success, "demean_torch (f32) did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-3,
            atol=1e-3,
            err_msg="demean_torch (f32 CPU) vs pyhdfe mismatch",
        )

    def test_cpu_float32_weighted(self, demean_data):
        """Weighted float32 on CPU should match pyhdfe."""
        x, flist, weights = demean_data
        N = x.shape[0]

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x, weights.reshape(N, 1))

        res_torch, success = demean_torch(
            x, flist, weights, tol=1e-5, dtype=torch.float32
        )
        assert success, "demean_torch (f32 weighted) did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-3,
            atol=1e-3,
            err_msg="demean_torch (f32 CPU weighted) vs pyhdfe mismatch",
        )

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_mps_float32_vs_pyhdfe(self, demean_data):
        """float32 on MPS should match pyhdfe within single-precision tolerance."""
        x, flist, _ = demean_data
        N = x.shape[0]
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(
            x, flist, weights, tol=1e-5, dtype=torch.float32
        )
        assert success, "demean_torch (MPS f32) did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-3,
            atol=1e-3,
            err_msg="demean_torch (MPS f32) vs pyhdfe mismatch",
        )

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_mps_float32_weighted(self, demean_data):
        """Weighted float32 on MPS should match pyhdfe."""
        x, flist, weights = demean_data
        N = x.shape[0]

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x, weights.reshape(N, 1))

        res_torch, success = demean_torch(
            x, flist, weights, tol=1e-5, dtype=torch.float32
        )
        assert success, "demean_torch (MPS f32 weighted) did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-3,
            atol=1e-3,
            err_msg="demean_torch (MPS f32 weighted) vs pyhdfe mismatch",
        )

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_mps_float32_high_d(self):
        """MPS float32 with many groups — exercises COO path at scale."""
        rng = np.random.default_rng(5555)
        N = 10_000
        G = 100
        x = rng.normal(0, 1, (N, 3))
        flist = rng.choice(G, (N, 2)).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(
            x, flist, weights, tol=1e-5, dtype=torch.float32
        )
        assert success, "demean_torch (MPS f32 high-D) did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-3,
            atol=1e-3,
            err_msg="demean_torch (MPS f32 high-D) vs pyhdfe mismatch",
        )
