import numpy as np
import pytest

from pyfixest.core.crv1 import crv1_vcov_loop as crv1_vcov_loop_rs
from pyfixest.estimation.internals.vcov_utils import (
    _crv1_vcov_loop as crv1_vcov_loop_nb,
)


@pytest.mark.parametrize("func", [crv1_vcov_loop_nb, crv1_vcov_loop_rs])
def test_crv1_vcov_loop(benchmark, func):
    # Input data: 6 observations, 2 clusters, 2 regressors
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
    clustid = np.array([0, 1])
    cluster_col = np.array([0, 0, 0, 1, 1, 1])
    u_hat = np.array([0.3, -0.5, 0.0001, 0.8, -0.2, 0.1])
    q = 0.5
    delta = 0.5

    # Precomputed expected A and B matrices (from numba reference)
    expected_A = np.array([[3.125, 0.475], [0.475, 0.085]])
    expected_B = np.array([[4.5, 3.7], [3.7, 5.89]])

    A, B = benchmark(
        func,
        X,
        clustid.astype(np.uintp),
        cluster_col.astype(np.uintp),
        q,
        u_hat,
        delta,
    )

    np.testing.assert_allclose(A, expected_A, atol=1e-10)
    np.testing.assert_allclose(B, expected_B, atol=1e-10)
