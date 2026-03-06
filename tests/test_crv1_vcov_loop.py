import numpy as np
import pytest

from pyfixest.core.crv1 import crv1_vcov_loop as crv1_vcov_loop_rs
from pyfixest.estimation.quantreg.quantreg_ import (
    _crv1_vcov_loop as crv1_vcov_loop_numba,
)


@pytest.mark.parametrize("n", [20, 100])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("n_clusters", [4, 10])
@pytest.mark.parametrize("q", [0.25, 0.5, 0.9])
def test_crv1_vcov_rust_vs_python(n, k, n_clusters, q):
    """Compare Rust and Python (Numba) implementations of _crv1_vcov_loop."""
    np.random.seed(42)

    # Ensure n is divisible by n_clusters
    n = (n // n_clusters) * n_clusters

    X = np.random.randn(n, k)
    cluster_col = np.repeat(np.arange(n_clusters), n // n_clusters)
    clustid = np.arange(n_clusters)
    u_hat = np.random.randn(n) * 0.5
    delta = 0.1

    # Python implementation returns inv(B) @ A @ inv(B)
    py_result = crv1_vcov_loop_numba(
        X, clustid.astype(np.int64), cluster_col.astype(np.int64), q, u_hat, delta
    )

    # Rust implementation returns (A, B) with B already divided by 2*delta
    A_rs, B_rs = crv1_vcov_loop_rs(
        X, clustid.astype(np.uintp), cluster_col.astype(np.uintp), q, u_hat, delta
    )
    rs_result = np.linalg.inv(B_rs) @ A_rs @ np.linalg.inv(B_rs)

    np.testing.assert_allclose(py_result, rs_result, rtol=1e-10, atol=1e-10)
