import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyfixest.core import find_collinear_variables
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb,
)


@pytest.mark.parametrize("fn", [find_collinear_variables, _find_collinear_variables_nb])
def test_find_collinear_variables(benchmark, fn):
    """Test the find_collinear_variables function with various test cases."""
    # =========================================================================
    # Test Case 1: Simple collinearity
    # =========================================================================
    # Create a matrix with a simple collinearity: last column is sum of first two
    N = 100
    dim = 1000
    X1 = np.random.RandomState(495).randn(dim, N)
    X1 = np.concat([X1, X1[:, [1]] + X1[:, [2]]], axis=1)
    X1 = X1.T @ X1
    # Test with default tolerance
    collinear_flags, n_collinear, all_collinear = benchmark(fn, X1)

    # Third column should be flagged as collinear
    expected_flags = np.array(N * [False] + [True])
    assert_array_equal(collinear_flags, expected_flags)
    assert n_collinear == 1
    assert not all_collinear
