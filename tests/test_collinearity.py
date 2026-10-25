import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyfixest.core import find_collinear_variables
from pyfixest.estimation.models.feiv_ import _projected_design_has_full_rank
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb,
)


@pytest.mark.parametrize(
    "case, expected",
    [
        ("full", True),
        ("rescaled-control", True),
        ("dependent-fits", False),
        ("zero-fit", False),
        ("tiny-fit", False),
        ("too-many-columns", False),
    ],
)
def test_projected_iv_design_rank(case, expected):
    """Known orthogonal columns cover the internal joint-rank contract.

    Multiple endogenous regressors are not publicly supported, so their joint
    projected-rank failure must be exercised at this array-only seam.
    """
    controls = np.ones(8)
    fitted = np.tile([-1.0, 1.0], 4)
    if case == "rescaled-control":
        controls *= 1e14
    elif case == "zero-fit":
        fitted *= 0.0
    elif case == "tiny-fit":
        fitted *= 1e-9
    design = np.column_stack((controls, fitted))
    if case == "dependent-fits":
        design = np.column_stack((design, fitted))
    elif case == "too-many-columns":
        design = design[:1]
    assert (
        _projected_design_has_full_rank(projected_design=design, collin_tol=1e-9)
        is expected
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
