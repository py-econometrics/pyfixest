import numpy as np
import pytest

from pyfixest.core.crv1 import crv1_vcov_loop as crv1_vcov_loop_rs
from pyfixest.estimation.internals.vcov_utils import (
    _crv1_vcov_loop as crv1_vcov_loop_nb,
)


def _run_rust(X, clustid, cluster_col, q, u_hat, delta):
    A, B = crv1_vcov_loop_rs(X, clustid, cluster_col, q, u_hat, delta)
    B_inv = np.linalg.inv(B)
    return B_inv @ A @ B_inv


def _run_numba(X, clustid, cluster_col, q, u_hat, delta):
    A, B = crv1_vcov_loop_nb(X, clustid, cluster_col, q, u_hat, delta)
    B_inv = np.linalg.inv(B)
    return B_inv @ A @ B_inv


@pytest.fixture
def small_data():
    """Small dataset with 6 observations, 2 clusters, 2 regressors."""
    X = np.array(
        [
            [1.0, 0.5],
            [2.0, 1.0],
            [1.5, 0.8],
            [3.0, 1.5],
            [0.5, 2.0],
            [1.0, 1.0],
        ]
    )
    cluster_col = np.array([0, 0, 0, 1, 1, 1])
    clustid = np.array([0, 1])
    u_hat = np.array([0.3, -0.5, 0.0001, 0.8, -0.2, 0.1])
    q = 0.5
    delta = 0.5
    return X, clustid, cluster_col, q, u_hat, delta


@pytest.fixture
def random_data():
    """Larger random dataset for stress testing."""
    rng = np.random.default_rng(42)
    n_obs = 500
    n_clusters = 20
    k = 4
    X = rng.standard_normal((n_obs, k))
    cluster_col = np.sort(rng.integers(0, n_clusters, size=n_obs))
    clustid = np.unique(cluster_col)
    u_hat = rng.standard_normal(n_obs)
    q = 0.25
    delta = 0.4
    return X, clustid, cluster_col, q, u_hat, delta


def test_crv1_vcov_loop_small(small_data):
    """Rust and numba produce the same result on a small dataset."""
    X, clustid, cluster_col, q, u_hat, delta = small_data

    result_nb = _run_numba(X, clustid, cluster_col, q, u_hat, delta)
    result_rs = _run_rust(
        X,
        clustid.astype(np.uintp),
        cluster_col.astype(np.uintp),
        q,
        u_hat,
        delta,
    )

    assert np.allclose(result_rs, result_nb, atol=1e-10), (
        f"Results differ.\nRust:\n{result_rs}\nNumba:\n{result_nb}"
    )


def test_crv1_vcov_loop_random(random_data):
    """Rust and numba produce the same result on a larger random dataset."""
    X, clustid, cluster_col, q, u_hat, delta = random_data

    result_nb = _run_numba(X, clustid, cluster_col, q, u_hat, delta)
    result_rs = _run_rust(
        X,
        clustid.astype(np.uintp),
        cluster_col.astype(np.uintp),
        q,
        u_hat,
        delta,
    )

    assert np.allclose(result_rs, result_nb, atol=1e-10), (
        f"Results differ.\nRust:\n{result_rs}\nNumba:\n{result_nb}"
    )


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_crv1_vcov_loop_quantiles(q):
    """Rust and numba agree across different quantile values."""
    rng = np.random.default_rng(123)
    n_obs = 200
    n_clusters = 10
    k = 3
    X = rng.standard_normal((n_obs, k))
    cluster_col = np.sort(rng.integers(0, n_clusters, size=n_obs))
    clustid = np.unique(cluster_col)
    u_hat = rng.standard_normal(n_obs)
    delta = 0.5

    result_nb = _run_numba(X, clustid, cluster_col, q, u_hat, delta)
    result_rs = _run_rust(
        X,
        clustid.astype(np.uintp),
        cluster_col.astype(np.uintp),
        q,
        u_hat,
        delta,
    )

    assert np.allclose(result_rs, result_nb, atol=1e-10), (
        f"Results differ for q={q}.\nRust:\n{result_rs}\nNumba:\n{result_nb}"
    )


@pytest.mark.parametrize("func", [_run_numba, _run_rust], ids=["numba", "rust"])
def test_crv1_vcov_loop_benchmark(benchmark, func):
    rng = np.random.default_rng(42)
    n_obs = 5000
    n_clusters = 50
    k = 5
    X = rng.standard_normal((n_obs, k))
    cluster_col = np.sort(rng.integers(0, n_clusters, size=n_obs)).astype(np.uintp)
    clustid = np.unique(cluster_col).astype(np.uintp)
    u_hat = rng.standard_normal(n_obs)
    q = 0.5
    delta = 0.5

    result = benchmark(func, X, clustid, cluster_col, q, u_hat, delta)

    assert result.shape == (k, k)


@pytest.mark.parametrize("func", [_run_numba, _run_rust], ids=["numba", "rust"])
def test_crv1_vcov_loop_benchmark_large(benchmark, func):
    rng = np.random.default_rng(42)
    n_obs = 100_000
    n_clusters = 200
    k = 10
    X = rng.standard_normal((n_obs, k))
    cluster_col = np.sort(rng.integers(0, n_clusters, size=n_obs)).astype(np.uintp)
    clustid = np.unique(cluster_col).astype(np.uintp)
    u_hat = rng.standard_normal(n_obs)
    q = 0.5
    delta = 0.5

    result = benchmark(func, X, clustid, cluster_col, q, u_hat, delta)

    assert result.shape == (k, k)
