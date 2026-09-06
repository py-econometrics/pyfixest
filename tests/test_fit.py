from __future__ import annotations

import numpy as np
import pytest

from pyfixest.estimation.internals.fit_ import fit_iv, fit_ols

# The numerical contract of `fit_ols` / `fit_iv` is pinned externally, by the
# weighted and IV cases of `test_single_fit_feols` against `fixest`. What those
# comparisons cannot see is whether the primitives write through to the arrays
# the model object handed them, so that is what is tested here.


@pytest.fixture
def ols_arrays() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    Y = np.array([[0.5], [1.0], [1.25], [2.5], [4.0]])
    return X, Y


@pytest.fixture
def iv_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 1.5],
        ]
    )
    Z = np.array(
        [
            [1.0, -0.5, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.5, 0.5],
            [1.0, 2.5, 2.0],
            [1.0, 1.0, -0.5],
        ]
    )
    Y = np.array([[0.5], [1.0], [2.5], [4.0], [2.0]])
    return X, Z, Y


@pytest.mark.parametrize(
    "weights",
    [None, np.array([0.5, 1.0, 1.5, 2.0, 3.0])],
    ids=("unweighted", "weighted"),
)
def test_fit_ols_does_not_mutate_inputs(
    ols_arrays: tuple[np.ndarray, np.ndarray],
    weights: np.ndarray | None,
) -> None:
    X, Y = ols_arrays
    X_before = X.copy()
    Y_before = Y.copy()
    weights_before = None if weights is None else weights.copy()
    X.setflags(write=False)
    Y.setflags(write=False)
    if weights is not None:
        weights.setflags(write=False)

    fit = fit_ols(X=X, Y=Y, weights=weights)

    assert np.all(np.isfinite(fit.beta))
    np.testing.assert_array_equal(X, X_before, err_msg="fit_ols mutated X")
    np.testing.assert_array_equal(Y, Y_before, err_msg="fit_ols mutated Y")
    if weights is not None:
        np.testing.assert_array_equal(
            weights,
            weights_before,
            err_msg="fit_ols mutated weights",
        )


@pytest.mark.parametrize(
    "weights",
    [None, np.array([[0.5], [1.0], [1.5], [2.0], [3.0]])],
    ids=("unweighted", "weighted"),
)
def test_fit_iv_does_not_mutate_inputs(
    iv_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: np.ndarray | None,
) -> None:
    X, Z, Y = iv_arrays
    X_before = X.copy()
    Z_before = Z.copy()
    Y_before = Y.copy()
    weights_before = None if weights is None else weights.copy()
    X.setflags(write=False)
    Z.setflags(write=False)
    Y.setflags(write=False)
    if weights is not None:
        weights.setflags(write=False)

    fit = fit_iv(X=X, Z=Z, Y=Y, weights=weights)

    assert np.all(np.isfinite(fit.beta))
    np.testing.assert_array_equal(X, X_before, err_msg="fit_iv mutated X")
    np.testing.assert_array_equal(Z, Z_before, err_msg="fit_iv mutated Z")
    np.testing.assert_array_equal(Y, Y_before, err_msg="fit_iv mutated Y")
    if weights is not None:
        np.testing.assert_array_equal(
            weights,
            weights_before,
            err_msg="fit_iv mutated weights",
        )
