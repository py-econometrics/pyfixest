import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyfixest.core import find_collinear_variables
from pyfixest.core.collinear import collinear_cholesky
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _collinear_cholesky_nb,
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


@pytest.mark.parametrize("fn", [collinear_cholesky, _collinear_cholesky_nb])
@pytest.mark.parametrize(
    "case",
    [
        "full",
        "zero_first",
        "duplicate_middle",
        "duplicate_last",
        "near",
        "all",
        "empty",
        "single",
    ],
)
def test_collinearity_factor_retained_order(fn, case):
    """The compact factor follows kept columns even across skipped pivots."""
    rng = np.random.default_rng(4202)
    X = rng.normal(size=(32, 4))
    expected_mask = np.zeros(4, dtype=bool)
    if case == "zero_first":
        X[:, 0] = 0
        expected_mask[0] = True
    elif case == "duplicate_middle":
        X[:, 1] = X[:, 0]
        expected_mask[1] = True
    elif case in ("duplicate_last", "near"):
        X[:, 3] = X[:, 0] + (1e-8 * X[:, 2] if case == "near" else 0)
        expected_mask[3] = True
    elif case == "all":
        X[:] = 0
        expected_mask[:] = True
    elif case in ("empty", "single"):
        k = 0 if case == "empty" else 1
        X = X[:, :k]
        expected_mask = expected_mask[:k]
    gram = X.T @ X
    original = gram.copy()
    mask, count, all_removed, upper = fn(gram, 1e-9)
    assert_array_equal(mask, expected_mask, err_msg="excluded columns")
    assert count == int(mask.sum())
    assert bool(all_removed) == (case == "all")
    kept_gram = gram[np.ix_(~mask, ~mask)]
    assert upper.shape == kept_gram.shape
    np.testing.assert_allclose(
        upper.T @ upper,
        kept_gram,
        rtol=1e-12,
        atol=1e-14,
        err_msg="retained Gram reconstruction",
    )
    np.testing.assert_allclose(
        upper,
        np.linalg.cholesky(kept_gram).T,
        rtol=1e-12,
        atol=1e-14,
        err_msg="retained factor vs NumPy Cholesky",
    )
    assert_array_equal(gram, original, err_msg="rank check mutated its input")


@pytest.mark.parametrize("weighted", [False, True])
def test_weighted_factorization_matches_selected_design(weighted):
    rng = np.random.default_rng(4203)
    X = rng.normal(size=(30, 4))
    X[:, 1] = X[:, 0]
    weights = rng.uniform(0.1, 2.0, size=30) if weighted else None
    original = X.copy()
    if weights is not None:
        weights.setflags(write=False)
    X.setflags(write=False)
    with pytest.warns(UserWarning, match="dropped due to multicollinearity"):
        selected, check, factor = drop_multicollinear_variables(
            X=X,
            names=["x0", "duplicate", "x2", "x3"],
            collin_tol=1e-9,
            weights=weights,
        )
    assert check.coefnames == ("x0", "x2", "x3")
    expected = selected.T @ (
        selected if weights is None else weights[:, None] * selected
    )
    np.testing.assert_allclose(
        factor.gram, expected, rtol=1e-12, atol=1e-14, err_msg="weighted Gram"
    )
    np.testing.assert_allclose(
        factor.upper.T @ factor.upper,
        expected,
        rtol=1e-12,
        atol=1e-14,
        err_msg="weighted factor",
    )
    assert not factor.gram.flags.writeable
    assert not factor.upper.flags.writeable
    assert_array_equal(X, original, err_msg="weighted rank check mutated X")
