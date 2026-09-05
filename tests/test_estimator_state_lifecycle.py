"""Protect formula, within-array, and observation-weight state boundaries."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.internals.model_state import (
    ObservationWeights,
    WithinLinearData,
)


@pytest.fixture
def lifecycle_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260831)
    n_obs = 24
    fixed_effect = np.repeat(["a", "b", "c", "d"], n_obs // 4)
    group_effect = pd.Series(fixed_effect).map(
        {"a": -1.0, "b": 0.5, "c": 1.25, "d": -0.25}
    )
    instrument = rng.normal(size=n_obs)
    covariate = rng.normal(size=n_obs)
    second_covariate = rng.normal(size=n_obs)
    endogenous = 0.8 * instrument + 0.4 * covariate + rng.normal(size=n_obs)
    response = (
        1.2 * covariate
        - 0.6 * second_covariate
        + 1.5 * endogenous
        + group_effect.to_numpy()
        + rng.normal(scale=0.3, size=n_obs)
    )
    return pd.DataFrame(
        {
            "y": response,
            "x": covariate,
            "x2": second_covariate,
            "endog": endogenous,
            "z": instrument,
            "fe": fixed_effect,
            "weight": np.tile([1, 2, 4, 3, 1, 2], 4),
        }
    )


@pytest.mark.parametrize(
    ("weights_type", "expected_n"), [("aweights", 24), ("fweights", 52)]
)
def test_feols_keeps_formula_within_and_weight_domains_distinct(
    lifecycle_data: pd.DataFrame, weights_type: str, expected_n: int
) -> None:
    fit = pf.feols(
        "y ~ x | fe",
        data=lifecycle_data,
        weights="weight",
        weights_type=weights_type,
        vcov="iid",
    )
    assert isinstance(fit._model_matrix, ModelMatrix)
    assert isinstance(fit._observation_weights, ObservationWeights)
    assert isinstance(fit._within_data, WithinLinearData)
    assert fit._Y is fit._within_data.response
    assert fit._X is fit._within_data.design
    assert fit._Z is fit._within_data.design
    assert not hasattr(fit, "_Yd")
    assert not hasattr(fit, "_Xd")
    weights = lifecycle_data["weight"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(fit._observation_weights.values, weights)
    np.testing.assert_array_equal(fit._weights.flatten(), weights)
    assert fit._observation_weights.kind == weights_type
    assert expected_n == fit._N
    weighted_group_mean = (lifecycle_data["y"] * lifecycle_data["weight"]).groupby(
        lifecycle_data["fe"]
    ).transform("sum") / lifecycle_data["weight"].groupby(
        lifecycle_data["fe"]
    ).transform("sum")
    expected_y_within = lifecycle_data["y"] - weighted_group_mean
    np.testing.assert_allclose(fit._within_data.response.flatten(), expected_y_within)
    residuals = fit._within_data.response.flatten() - fit._X @ fit._beta_hat
    np.testing.assert_allclose(fit._u_hat, residuals)
    np.testing.assert_allclose(fit.resid(), residuals)
    np.testing.assert_allclose(fit._scores, fit._X * (weights * residuals)[:, None])
    np.testing.assert_allclose(fit._hessian, fit._X.T @ (weights[:, None] * fit._X))
    with pytest.raises(FrozenInstanceError):
        fit._within_data.response = fit._within_data.design  # type: ignore[misc]


def test_weighted_iv_keeps_each_econometric_role_on_within_scale(
    lifecycle_data: pd.DataFrame,
) -> None:
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        weights="weight",
        weights_type="aweights",
        vcov="iid",
    )
    within = fit._within_data
    assert isinstance(within, WithinLinearData)
    assert within.instruments is not None
    assert within.endogenous is not None
    assert fit._Y is within.response
    assert fit._X is within.design
    assert fit._Z is within.instruments
    assert fit._endogvar is within.endogenous
    weights = lifecycle_data["weight"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        fit._tZX, within.instruments.T @ (weights[:, None] * within.design)
    )
    np.testing.assert_allclose(
        fit._scores, within.instruments * (weights * fit._u_hat)[:, None]
    )


def test_formula_data_remains_canonical_after_linear_fit(
    lifecycle_data: pd.DataFrame,
) -> None:
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe", data=lifecycle_data, weights="weight", vcov="iid"
    )
    model_matrix = fit._model_matrix
    assert isinstance(model_matrix, ModelMatrix)
    assert isinstance(model_matrix.dependent, pd.DataFrame)
    assert isinstance(model_matrix.independent, pd.DataFrame)
    assert isinstance(model_matrix.instruments, pd.DataFrame)
    assert isinstance(fit._Y, np.ndarray)
    assert isinstance(fit._X, np.ndarray)
    assert isinstance(fit._Z, np.ndarray)


def test_unweighted_effective_n_remains_integer_for_prediction_errors(
    lifecycle_data: pd.DataFrame,
) -> None:
    fit = pf.feols("y ~ x", data=lifecycle_data, vcov="iid")
    assert isinstance(fit._N, int)
    assert isinstance(fit._observation_weights.n_effective, int)
    assert fit.predict(se_fit=True).shape == (len(lifecycle_data),)


@pytest.mark.parametrize(
    ("fml", "weights", "weights_type"),
    [
        ("y ~ x", None, "aweights"),
        ("y ~ x | fe", "weight", "aweights"),
        ("y ~ x | fe", "weight", "fweights"),
    ],
)
def test_gaussian_glm_performance_uses_explicit_response_domains(
    lifecycle_data: pd.DataFrame,
    fml: str,
    weights: str | None,
    weights_type: str,
) -> None:
    fit = pf.feglm(
        fml,
        data=lifecycle_data,
        family="gaussian",
        weights=weights,
        weights_type=weights_type,
        vcov="iid",
        iwls_tol=1e-10,
    )
    fit.get_performance()
    response = lifecycle_data["y"].to_numpy()
    observation_weights = fit._observation_weights.values
    residuals = fit._u_hat_response
    if observation_weights is None:
        ssu = np.sum(residuals**2)
        ssy = np.sum((response - np.mean(response)) ** 2)
    else:
        ssu = np.sum(observation_weights * residuals**2)
        center = np.average(response, weights=observation_weights)
        ssy = np.sum(observation_weights * (response - center) ** 2)
    np.testing.assert_allclose(fit._rmse, np.sqrt(ssu / fit._N))
    np.testing.assert_allclose(fit._r2, 1 - ssu / ssy)
    if fit._has_fixef:
        assert observation_weights is not None
        weighted_y = lifecycle_data["weight"] * lifecycle_data["y"]
        group_mean = weighted_y.groupby(lifecycle_data["fe"]).transform("sum")
        group_mean /= (
            lifecycle_data["weight"].groupby(lifecycle_data["fe"]).transform("sum")
        )
        response_within = response - group_mean.to_numpy()
        ssy_within = np.sum(observation_weights * response_within**2)
        np.testing.assert_allclose(fit._r2_within, 1 - ssu / ssy_within)
