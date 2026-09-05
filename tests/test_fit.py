from __future__ import annotations

import numpy as np
import pytest

import pyfixest.estimation.internals.fit_ as fit_module
from pyfixest.estimation.internals.fit_ import fit_iv, fit_ols


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
def test_fit_ols_matches_direct_equations_without_mutating_inputs(
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

    weight_values = np.ones(X.shape[0]) if weights is None else weights.reshape(-1)
    expected_tXX = X.T @ (weight_values[:, None] * X)
    expected_tXy = X.T @ (weight_values[:, None] * Y)
    expected_beta = np.linalg.solve(expected_tXX, expected_tXy).flatten()
    expected_residuals = Y.flatten() - X @ expected_beta

    np.testing.assert_allclose(fit.beta, expected_beta, err_msg="OLS coefficients")
    np.testing.assert_allclose(
        fit.residuals,
        expected_residuals,
        err_msg="OLS response-scale residuals",
    )
    np.testing.assert_allclose(
        fit.scores,
        X * (weight_values * expected_residuals)[:, None],
        err_msg="OLS weighted scores",
    )
    np.testing.assert_allclose(
        fit.hessian,
        expected_tXX,
        err_msg="OLS weighted Hessian",
    )
    np.testing.assert_allclose(
        fit.tZX,
        expected_tXX,
        err_msg="OLS weighted X'X",
    )
    np.testing.assert_allclose(
        fit.tZy,
        expected_tXy,
        err_msg="OLS weighted X'Y",
    )
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
def test_fit_iv_matches_direct_equations_without_mutating_inputs(
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

    weight_values = np.ones(X.shape[0]) if weights is None else weights.reshape(-1)
    expected_tZX = Z.T @ (weight_values[:, None] * X)
    expected_tXZ = X.T @ (weight_values[:, None] * Z)
    expected_tZy = Z.T @ (weight_values[:, None] * Y)
    expected_tZZ = Z.T @ (weight_values[:, None] * Z)
    expected_tZZinv = np.linalg.inv(expected_tZZ)
    projection = expected_tXZ @ expected_tZZinv
    expected_beta = np.linalg.solve(
        projection @ expected_tZX,
        projection @ expected_tZy,
    ).flatten()
    expected_residuals = Y.flatten() - X @ expected_beta

    np.testing.assert_allclose(fit.beta, expected_beta, err_msg="IV coefficients")
    np.testing.assert_allclose(
        fit.residuals,
        expected_residuals,
        err_msg="IV response-scale residuals",
    )
    np.testing.assert_allclose(
        fit.scores,
        Z * (weight_values * expected_residuals)[:, None],
        err_msg="IV weighted scores",
    )
    np.testing.assert_allclose(
        fit.hessian,
        expected_tZZ,
        err_msg="IV weighted Z'Z",
    )
    np.testing.assert_allclose(
        fit.tZX,
        expected_tZX,
        err_msg="IV weighted Z'X",
    )
    np.testing.assert_allclose(
        fit.tXZ,
        expected_tXZ,
        err_msg="IV weighted X'Z",
    )
    np.testing.assert_allclose(
        fit.tZy,
        expected_tZy,
        err_msg="IV weighted Z'Y",
    )
    np.testing.assert_allclose(
        fit.tZZinv,
        expected_tZZinv,
        err_msg="inverse IV weighted Z'Z",
    )
    np.testing.assert_array_equal(X, X_before, err_msg="fit_iv mutated X")
    np.testing.assert_array_equal(Z, Z_before, err_msg="fit_iv mutated Z")
    np.testing.assert_array_equal(Y, Y_before, err_msg="fit_iv mutated Y")
    if weights is not None:
        np.testing.assert_array_equal(
            weights,
            weights_before,
            err_msg="fit_iv mutated weights",
        )


@pytest.mark.parametrize("fit_name", ["ols", "iv"])
def test_unweighted_fit_does_not_compute_sqrt_weights(
    monkeypatch: pytest.MonkeyPatch,
    ols_arrays: tuple[np.ndarray, np.ndarray],
    iv_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    fit_name: str,
) -> None:
    def unexpected_sqrt(_: np.ndarray) -> np.ndarray:
        raise AssertionError("the unweighted path must not transform unit weights")

    monkeypatch.setattr(fit_module.np, "sqrt", unexpected_sqrt)
    if fit_name == "ols":
        X, Y = ols_arrays
        fit_ols(X=X, Y=Y, weights=None)
    else:
        X, Z, Y = iv_arrays
        fit_iv(X=X, Z=Z, Y=Y, weights=None)
