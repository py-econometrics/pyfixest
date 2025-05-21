import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb,
)
from pyfixest_core import find_collinear_variables_rs


@pytest.mark.parametrize(
    "fn", [find_collinear_variables_rs, _find_collinear_variables_nb]
)
def test_find_collinear_variables(benchmark, fn):
    """Test the find_collinear_variables function with various test cases."""
    # =========================================================================
    # Test Case 1: Simple collinearity
    # =========================================================================
    # Create a matrix with a simple collinearity: third column is sum of first two
    X1 = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 11.0, 8.0], [9.0, 10.0, 19.0, 12.0]]
    ).T
    X1 = X1.T @ X1
    # Test with default tolerance
    collinear_flags, n_collinear, all_collinear = benchmark(fn, X1, tol=1e-10)

    # Third column should be flagged as collinear
    expected_flags = np.array([False, False, True])
    assert_array_equal(collinear_flags, expected_flags)
    assert n_collinear == 1
    assert not all_collinear
    """
    # =========================================================================
    # Test Case 2: Multiple collinear columns
    # =========================================================================
    # Create a matrix with multiple collinearities
    X2 = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 15.0, 10.0, 11.0, 18.0],
        [13.0, 14.0, 27.0, 16.0, 29.0, 30.0]
    ])
    # Column 2 = col0 + col1
    # Column 4 = col0 + col3

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X2, tol=1e-10)

    # Columns 2 and 4 should be flagged as collinear
    expected_flags = np.array([[False], [False], [True], [False], [True], [False]])
    assert_array_equal(collinear_flags, expected_flags)
    assert n_collinear == 2
    assert not all_collinear

    # =========================================================================
    # Test Case 3: All columns collinear
    # =========================================================================
    # Create a matrix where all columns are collinear
    X3 = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X3, tol=1e-10)

    # At least k-1 columns should be flagged as collinear
    assert n_collinear >= 2  # The algorithm might keep one column as reference
    assert all_collinear or n_collinear == 2  # Either all_collinear=True or n_collinear=2

    # =========================================================================
    # Test Case 4: No collinearity
    # =========================================================================
    # Create a matrix with no collinearity
    X4 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X4, tol=1e-10)

    # No columns should be flagged as collinear
    expected_flags = np.array([[False], [False], [False]])
    assert_array_equal(collinear_flags, expected_flags)
    assert n_collinear == 0
    assert not all_collinear

    # =========================================================================
    # Test Case 5: Near collinearity with tolerance
    # =========================================================================
    # Create a matrix with near collinearity
    X5 = np.array([
        [1.0, 2.0, 3.0 + 1e-5],
        [4.0, 5.0, 9.0 - 1e-5],
        [7.0, 8.0, 15.0 + 1e-5]
    ])

    # Test with default tolerance (should detect collinearity)
    collinear_flags, n_collinear, all_collinear = benchmark(fn, X5, tol=1e-10)
    assert collinear_flags[2, 0]  # Third column should be collinear

    # Test with stricter tolerance (should not detect collinearity)
    collinear_flags, n_collinear, all_collinear = benchmark(fn, X5, tol=1e-12)
    assert not collinear_flags[2, 0]  # Third column should not be collinear with strict tolerance

    # =========================================================================
    # Test Case 6: Random data matrix
    # =========================================================================
    # Create a random data matrix with known collinearity
    np.random.seed(42)  # For reproducibility
    X6 = np.random.randn(20, 5)
    # Make column 3 a linear combination of columns 0 and 1
    X6[:, 3] = 0.7 * X6[:, 0] + 0.3 * X6[:, 1]

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X6, tol=1e-10)

    # Column 3 should be flagged as collinear
    assert collinear_flags[3, 0]
    assert n_collinear == 1
    assert not all_collinear

    # =========================================================================
    # Test Case 7: Correlation matrix input
    # =========================================================================
    # Create a correlation matrix with known collinearity
    X7 = np.array([
        [1.0, 0.5, 0.8, 0.2],
        [0.5, 1.0, 0.9, 0.3],
        [0.8, 0.9, 1.0, 0.4],
        [0.2, 0.3, 0.4, 1.0]
    ])

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X7, tol=1e-10)

    # At least one column should be flagged as collinear
    assert n_collinear > 0

    # =========================================================================
    # Test Case 8: Edge case - very large matrix
    # =========================================================================
    # Test with a moderately large matrix to verify performance
    np.random.seed(42)
    X8 = np.random.randn(100, 50)
    # Make every 10th column a linear combination of previous columns
    for i in range(9, 50, 10):
        X8[:, i] = 0.5 * X8[:, i-2] + 0.5 * X8[:, i-1]

    collinear_flags, n_collinear, all_collinear = benchmark(fn, X8, tol=1e-10)

    # Expected number of collinear columns: 4 (columns 9, 19, 29, 39, 49)
    assert n_collinear == 5
    assert not all_collinear

    # Check if the expected columns are flagged
    expected_collinear = [9, 19, 29, 39, 49]
    actual_collinear = np.where(collinear_flags.flatten())[0]
    assert set(expected_collinear) == set(actual_collinear)

    # =========================================================================
    # Test Case 9: Test utility functions
    # =========================================================================
    # Test get_independent_columns
    np.random.seed(42)
    X9 = np.random.randn(20, 5)
    # Make column 3 a linear combination of columns 0 and 1
    X9[:, 3] = 0.7 * X9[:, 0] + 0.3 * X9[:, 1]

    # Test with invalid inputs
    with pytest.raises(ValueError):
        # 1D array should raise error
        benchmark(fn, np.array([1, 2, 3]), tol=1e-10)

    with pytest.raises(ValueError):
        # More columns than rows in data matrix should raise error
        benchmark(fn, np.random.randn(3, 10), tol=1e-10)
    """
