import numpy as np
import pytest

from pyfixest.core import crv1_meat_loop as crv1_meat_loop_rs
from pyfixest.estimation.internals.vcov_utils import _crv1_meat_loop


@pytest.mark.parametrize("func", [_crv1_meat_loop, crv1_meat_loop_rs])
def test_crv1_meat_loop(benchmark, func):
    # Input data
    scores = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    clustid = np.array([0, 1])
    cluster_col = np.array([0, 0, 1, 1])

    # Expected:
    # For group 0: indices [0, 1], sum = [4.0, 6.0]
    # outer = [[16, 24], [24, 36]]
    # For group 1: indices [2, 3], sum = [12.0, 14.0]
    # outer = [[144, 168], [168, 196]]
    # Total = sum of the two
    expected = np.array([[160, 192], [192, 232]])

    result = benchmark(
        func, scores, clustid.astype(np.uint), cluster_col.astype(np.uint)
    )

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
