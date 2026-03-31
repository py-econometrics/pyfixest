"""
Tests for batched LSMR solver (K right-hand sides via SpMM).

Verifies that lsmr_torch_batched produces identical results to K sequential
lsmr_torch calls, and that the batched demeaning path matches the sequential one.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE pixi run -e dev pytest tests/test_batched_lsmr.py -v -s
"""

from __future__ import annotations

import numpy as np
import pyhdfe
import pytest

torch = pytest.importorskip("torch")

from pyfixest.estimation.torch.demean_torch_ import (  # noqa: E402
    _build_sparse_dummy,
    _PreconditionedSparse,
    demean_torch,
)
from pyfixest.estimation.torch.lsmr_torch import (  # noqa: E402
    lsmr_torch,
    lsmr_torch_batched,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_problem(m: int, n: int, density: float = 0.1, seed: int = 42):
    """Create a sparse CSR system A and dense rhs b.

    Default density=0.1 ensures the system is well-conditioned enough
    for LSMR to converge within min(m, n) iterations.
    """
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
    return A_csr


def _make_rhs(m: int, K: int, seed: int = 42) -> torch.Tensor:
    """Create a dense RHS matrix B of shape (m, K)."""
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((m, K)), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Core batched LSMR tests
# ---------------------------------------------------------------------------


class TestBatchedLSMR:
    """Unit tests for lsmr_torch_batched."""

    def test_matches_sequential(self):
        """K=5: batched result should match K sequential lsmr_torch calls.

        Tolerance is 1e-6 because the batched path uses vectorized tensor
        Givens rotations while sequential uses Python-float math. Accumulated
        rounding differences over many iterations produce ~1e-7 divergence.
        """
        m, n, K = 200, 100, 5
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K, seed=123)

        # Batched solve
        X_batch, istop_batch, itn_batch, *_ = lsmr_torch_batched(A, B)

        # Sequential solve
        for k in range(K):
            x_seq, istop_seq, itn_seq, *_ = lsmr_torch(A, B[:, k])
            assert torch.allclose(X_batch[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"Column {k}: max diff = "
                f"{torch.max(torch.abs(X_batch[:, k] - x_seq)).item():.2e}"
            )

    @pytest.mark.parametrize("K", [2, 10, 20])
    def test_matches_sequential_various_K(self, K):
        """Batched matches sequential for various K values."""
        m, n = 300, 150
        A = _make_sparse_problem(m, n, seed=77)
        B = _make_rhs(m, K, seed=88)

        X_batch, *_ = lsmr_torch_batched(A, B)

        for k in range(K):
            x_seq, *_ = lsmr_torch(A, B[:, k])
            assert torch.allclose(X_batch[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"K={K}, col {k}: max diff = "
                f"{torch.max(torch.abs(X_batch[:, k] - x_seq)).item():.2e}"
            )

    def test_single_column(self):
        """K=1 batched should match single-RHS lsmr_torch."""
        m, n = 200, 100
        A = _make_sparse_problem(m, n)
        b = _make_rhs(m, 1, seed=42)

        X_batch, istop_batch, itn_batch, *_ = lsmr_torch_batched(A, b)
        x_single, istop_single, itn_single, *_ = lsmr_torch(A, b[:, 0])

        assert torch.allclose(X_batch[:, 0], x_single, atol=1e-6)

    def test_convergence_per_column(self):
        """Columns with different difficulty should converge independently."""
        m, n = 200, 100
        A = _make_sparse_problem(m, n)

        rng = np.random.default_rng(999)
        # Column 0: easy (small values), Column 1: harder (large values)
        B = torch.zeros(m, 2, dtype=torch.float64)
        B[:, 0] = torch.tensor(rng.standard_normal(m) * 0.01, dtype=torch.float64)
        B[:, 1] = torch.tensor(rng.standard_normal(m) * 100.0, dtype=torch.float64)

        X, istop, itn, *_ = lsmr_torch_batched(A, B)

        # Both should converge (istop in 1-3)
        assert (istop >= 1).all() and (istop <= 3).all(), (
            f"Not all columns converged: istop = {istop}"
        )
        # Solutions should match sequential (convergence correctness)
        for k in range(2):
            x_seq, *_ = lsmr_torch(A, B[:, k])
            assert torch.allclose(X[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"Column {k}: batched vs sequential max diff = "
                f"{torch.max(torch.abs(X[:, k] - x_seq)).item():.2e}"
            )

    def test_zero_rhs_column(self):
        """B with a zero column should produce zero solution for that column."""
        m, n = 100, 50
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, 3, seed=42)
        B[:, 1] = 0.0  # Zero out middle column

        X, istop, *_ = lsmr_torch_batched(A, B)

        assert torch.allclose(
            X[:, 1], torch.zeros(n, dtype=torch.float64), atol=1e-12
        ), (
            f"Zero-RHS column has non-zero solution: ||x|| = {torch.norm(X[:, 1]).item()}"
        )

    def test_all_zero_rhs(self):
        """All-zero B should return all-zero X with istop=0."""
        m, n, K = 100, 50, 3
        A = _make_sparse_problem(m, n)
        B = torch.zeros(m, K, dtype=torch.float64)

        X, istop, itn, *_ = lsmr_torch_batched(A, B)

        assert torch.allclose(X, torch.zeros(n, K, dtype=torch.float64), atol=1e-12)
        assert itn == 0

    def test_damp(self):
        """Damped batched should match damped sequential."""
        m, n, K = 200, 100, 3
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K, seed=55)
        damp = 5.0

        X_batch, *_ = lsmr_torch_batched(A, B, damp=damp)

        for k in range(K):
            x_seq, *_ = lsmr_torch(A, B[:, k], damp=damp)
            assert torch.allclose(X_batch[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"Damped col {k}: max diff = "
                f"{torch.max(torch.abs(X_batch[:, k] - x_seq)).item():.2e}"
            )

    def test_overdetermined_known_solution(self):
        """Overdetermined system with exact solution, K=3 columns."""
        A_dense = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        )
        A_sparse = A_dense.to_sparse_csr()

        X_true = torch.tensor([[1.0, 2.0, -1.0], [-1.0, 0.5, 3.0]], dtype=torch.float64)
        B = A_dense @ X_true  # (3, 3)

        X_sol, istop, *_ = lsmr_torch_batched(A_sparse, B)

        assert torch.allclose(X_sol, X_true, atol=1e-10), (
            f"Solution mismatch: max diff = "
            f"{torch.max(torch.abs(X_sol - X_true)).item():.2e}"
        )
        assert ((istop >= 1) & (istop <= 3)).all()

    def test_maxiter_exhaustion(self):
        """Forcing maxiter=2 should return istop=7 for all columns."""
        m, n, K = 100, 50, 3
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K, seed=42)

        _, istop, itn, *_ = lsmr_torch_batched(A, B, maxiter=2, atol=1e-15, btol=1e-15)

        assert (istop == 7).all(), f"Expected istop=7, got {istop}"
        assert itn == 2

    def test_return_shapes(self):
        """Verify all return values have correct shapes."""
        m, n, K = 200, 100, 5
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K)

        X, istop, itn, normr, normar, normA, condA, normx = lsmr_torch_batched(A, B)

        assert X.shape == (n, K)
        assert istop.shape == (K,)
        assert isinstance(itn, int)
        assert normr.shape == (K,)
        assert normar.shape == (K,)
        assert normA.shape == (K,)
        assert condA.shape == (K,)
        assert normx.shape == (K,)


# ---------------------------------------------------------------------------
# PreconditionedSparse.mm() tests
# ---------------------------------------------------------------------------


class TestPreconditionedSparseMM:
    """Tests for _PreconditionedSparse.mm() batched matvec."""

    def test_mm_matches_mv_loop(self):
        """mm(V) should equal column-wise mv(v_k) for each k."""
        rng = np.random.default_rng(42)
        N, G, K = 200, 20, 5

        flist = rng.choice(G, (N, 1)).astype(np.uint64)
        D = _build_sparse_dummy(flist, torch.device("cpu"), torch.float64)
        M_inv = 1.0 / torch.sqrt(
            torch.tensor(np.bincount(flist[:, 0], minlength=G), dtype=torch.float64)
        )

        A = _PreconditionedSparse(D, M_inv)
        V = torch.randn(G, K, dtype=torch.float64)

        # Batched
        result_mm = A.mm(V)

        # Sequential
        for k in range(K):
            result_mv = A.mv(V[:, k])
            assert torch.allclose(result_mm[:, k], result_mv, atol=1e-12), (
                f"mm vs mv mismatch at column {k}"
            )

    def test_mm_transpose_matches_mv_transpose(self):
        """A^T.mm(U) should match column-wise A^T.mv(u_k)."""
        rng = np.random.default_rng(42)
        N, G, K = 200, 20, 5

        flist = rng.choice(G, (N, 1)).astype(np.uint64)
        D = _build_sparse_dummy(flist, torch.device("cpu"), torch.float64)
        M_inv = 1.0 / torch.sqrt(
            torch.tensor(np.bincount(flist[:, 0], minlength=G), dtype=torch.float64)
        )

        At = _PreconditionedSparse(D, M_inv).t()
        U = torch.randn(N, K, dtype=torch.float64)

        result_mm = At.mm(U)

        for k in range(K):
            result_mv = At.mv(U[:, k])
            assert torch.allclose(result_mm[:, k], result_mv, atol=1e-12), (
                f"transpose mm vs mv mismatch at column {k}"
            )


# ---------------------------------------------------------------------------
# Integration: batched demeaning
# ---------------------------------------------------------------------------


class TestDemeanBatched:
    """Verify batched demeaning path produces correct results."""

    def test_batched_demean_matches_pyhdfe(self):
        """Batched demean (K>1) should match pyhdfe reference."""
        rng = np.random.default_rng(929291)
        N, K = 1000, 10
        x = rng.normal(0, 1, (N, K))
        flist = np.column_stack(
            [
                rng.choice(10, N),
                rng.choice(10, N),
            ]
        ).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "Batched demean did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Batched demean vs pyhdfe mismatch",
        )

    def test_batched_demean_weighted_matches_pyhdfe(self):
        """Weighted batched demean should match pyhdfe."""
        rng = np.random.default_rng(929291)
        N, K = 1000, 5
        x = rng.normal(0, 1, (N, K))
        flist = np.column_stack(
            [
                rng.choice(10, N),
                rng.choice(10, N),
            ]
        ).astype(np.uint64)
        weights = rng.uniform(0.1, 2.0, N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x, weights.reshape(N, 1))

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success, "Weighted batched demean did not converge"
        np.testing.assert_allclose(
            res_torch,
            res_pyhdfe,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Weighted batched demean vs pyhdfe mismatch",
        )

    def test_single_column_still_works(self):
        """K=1 demean should still work (uses sequential path)."""
        rng = np.random.default_rng(42)
        N = 500
        x = rng.normal(0, 1, (N, 1))
        flist = rng.choice(10, N).astype(np.uint64).reshape(N, 1)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-6, atol=1e-8)

    def test_many_columns(self):
        """K=50 batched demean should match pyhdfe."""
        rng = np.random.default_rng(9999)
        N, K = 2000, 50
        x = rng.normal(0, 1, (N, K))
        flist = rng.choice(20, (N, 2)).astype(np.uint64)
        weights = np.ones(N)

        algorithm = pyhdfe.create(flist)
        res_pyhdfe = algorithm.residualize(x)

        res_torch, success = demean_torch(x, flist, weights, tol=1e-10)
        assert success
        np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Compiled batched LSMR tests
# ---------------------------------------------------------------------------

HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class TestCompiledBatchedLSMR:
    """Tests for the compiled batched LSMR path (_lsmr_compiled_batched)."""

    def test_compiled_batched_matches_eager_batched(self):
        """Packed-state batched matches vectorized eager batched on CPU f64, K=5.

        Uses use_compile=False to avoid Inductor C++ backend issues on macOS.
        This still validates the _lsmr_compiled_batched logic (packed state,
        _scalar_step on (STATE_SIZE, K) tensors, vector updates with clamp guards).
        Actual torch.compile fusion is tested in the MPS tests below.
        """
        from pyfixest.estimation.torch.lsmr_torch import (
            _lsmr_batched,
            _lsmr_compiled_batched,
        )

        m, n, K = 200, 100, 5
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K, seed=123)

        X_eager, istop_eager, itn_eager, *_ = _lsmr_batched(A, B)
        X_comp, istop_comp, itn_comp, *_ = _lsmr_compiled_batched(
            A, B, use_compile=False
        )

        assert torch.allclose(X_eager, X_comp, atol=1e-6, rtol=1e-6), (
            f"max diff = {torch.max(torch.abs(X_eager - X_comp)).item():.2e}"
        )
        assert itn_eager == itn_comp, f"itn: {itn_eager} vs {itn_comp}"

    @pytest.mark.parametrize("K", [2, 10, 20])
    def test_compiled_matches_sequential_various_K(self, K):
        """Packed-state batched matches K sequential lsmr_torch calls."""
        from pyfixest.estimation.torch.lsmr_torch import _lsmr_compiled_batched

        m, n = 300, 150
        A = _make_sparse_problem(m, n, seed=77)
        B = _make_rhs(m, K, seed=88)

        X_comp, *_ = _lsmr_compiled_batched(A, B, use_compile=False)

        for k in range(K):
            x_seq, *_ = lsmr_torch(A, B[:, k])
            assert torch.allclose(X_comp[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"K={K}, col {k}: max diff = "
                f"{torch.max(torch.abs(X_comp[:, k] - x_seq)).item():.2e}"
            )

    def test_compiled_single_column(self):
        """Packed-state K=1 matches single-RHS lsmr_torch."""
        from pyfixest.estimation.torch.lsmr_torch import _lsmr_compiled_batched

        m, n = 200, 100
        A = _make_sparse_problem(m, n)
        b = _make_rhs(m, 1, seed=42)

        X_comp, *_ = _lsmr_compiled_batched(A, b, use_compile=False)
        x_single, *_ = lsmr_torch(A, b[:, 0])

        assert torch.allclose(X_comp[:, 0], x_single, atol=1e-6), (
            f"max diff = {torch.max(torch.abs(X_comp[:, 0] - x_single)).item():.2e}"
        )

    def test_compiled_zero_rhs_column(self):
        """Zero RHS column handled correctly in packed-state path."""
        from pyfixest.estimation.torch.lsmr_torch import _lsmr_compiled_batched

        m, n = 100, 50
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, 3, seed=42)
        B[:, 1] = 0.0  # Zero out middle column

        X, istop, *_ = _lsmr_compiled_batched(A, B, use_compile=False)

        assert torch.allclose(
            X[:, 1], torch.zeros(n, dtype=torch.float64), atol=1e-12
        ), (
            f"Zero-RHS column has non-zero solution: ||x|| = {torch.norm(X[:, 1]).item()}"
        )

        # Non-zero columns should still solve correctly
        for k in [0, 2]:
            x_seq, *_ = lsmr_torch(A, B[:, k])
            assert torch.allclose(X[:, k], x_seq, atol=1e-6, rtol=1e-6)

    def test_compiled_damp(self):
        """Damped packed-state batched matches damped sequential."""
        from pyfixest.estimation.torch.lsmr_torch import _lsmr_compiled_batched

        m, n, K = 200, 100, 3
        A = _make_sparse_problem(m, n)
        B = _make_rhs(m, K, seed=55)
        damp = 5.0

        X_comp, *_ = _lsmr_compiled_batched(A, B, damp=damp, use_compile=False)

        for k in range(K):
            x_seq, *_ = lsmr_torch(A, B[:, k], damp=damp)
            assert torch.allclose(X_comp[:, k], x_seq, atol=1e-6, rtol=1e-6), (
                f"Damped col {k}: max diff = "
                f"{torch.max(torch.abs(X_comp[:, k] - x_seq)).item():.2e}"
            )

    def test_compiled_all_zero_rhs(self):
        """All-zero B returns all-zero X in packed-state path."""
        from pyfixest.estimation.torch.lsmr_torch import _lsmr_compiled_batched

        m, n, K = 100, 50, 3
        A = _make_sparse_problem(m, n)
        B = torch.zeros(m, K, dtype=torch.float64)

        X, istop, itn, *_ = _lsmr_compiled_batched(A, B, use_compile=False)

        assert torch.allclose(X, torch.zeros(n, K, dtype=torch.float64), atol=1e-12)
        assert itn == 0

    # --- MPS-specific tests ---

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_compiled_mps_correctness(self):
        """Compiled batched on MPS f32 vs CPU f64 reference."""
        m, n, K = 300, 150, 5
        A_cpu = _make_sparse_problem(m, n, density=0.1, seed=42)
        B_cpu = _make_rhs(m, K, seed=123)

        # CPU f64 reference (eager)
        X_ref, *_ = lsmr_torch_batched(A_cpu, B_cpu, use_compile=False)

        # MPS f32 compiled
        A_mps = A_cpu.to(torch.float32).to_dense().to("mps")
        B_mps = B_cpu.to(torch.float32).to("mps")

        X_mps, *_ = lsmr_torch_batched(A_mps, B_mps, use_compile=True)

        max_diff = torch.max(torch.abs(X_ref.float() - X_mps.cpu())).item()
        assert max_diff < 0.1, (
            f"MPS f32 compiled vs CPU f64 too different: {max_diff:.2e}"
        )

    @pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
    def test_compiled_vs_uncompiled_mps(self):
        """Compiled and uncompiled give same results on MPS."""
        m, n, K = 300, 150, 5
        A_cpu = _make_sparse_problem(m, n, density=0.1, seed=42)
        B_cpu = _make_rhs(m, K, seed=123)

        A_mps = A_cpu.to(torch.float32).to_dense().to("mps")
        B_mps = B_cpu.to(torch.float32).to("mps")

        X_comp, *_ = lsmr_torch_batched(A_mps, B_mps, use_compile=True)
        X_nocomp, *_ = lsmr_torch_batched(A_mps, B_mps, use_compile=False)

        max_diff = torch.max(torch.abs(X_comp - X_nocomp)).item()
        assert max_diff < 1e-4, f"Compiled vs uncompiled differ on MPS: {max_diff:.2e}"
