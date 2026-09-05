"""Protect the estimator-state lifecycle boundaries of formula data.

These tests deliberately inspect private state: public numerical behavior is
covered by the release snapshots and live-R suites, while the representation
and row-sample seams locked here are not observable from those suites.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.model_matrix import ModelMatrix, create_model_matrix
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.model_state import (
    ObservationWeights,
    WithinLinearData,
)


@pytest.fixture
def lifecycle_data() -> pd.DataFrame:
    """Return a small full-rank weighted FE/IV data set."""
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
    ("weights_type", "expected_n"),
    [("aweights", 24), ("fweights", 52)],
)
def test_feols_keeps_formula_within_and_weight_domains_distinct(
    lifecycle_data: pd.DataFrame,
    weights_type: str,
    expected_n: int,
) -> None:
    """A weighted FE fit retains within arrays and response-scale residuals."""
    fit = pf.feols(
        "y ~ x | fe",
        data=lifecycle_data,
        weights="weight",
        weights_type=weights_type,
        vcov="iid",
    )

    assert isinstance(fit._model_matrix, ModelMatrix)
    assert isinstance(fit._model_matrix.dependent, pd.DataFrame)
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
    assert not np.allclose(
        fit._within_data.response.flatten(),
        expected_y_within * np.sqrt(weights),
    )

    residuals = fit._within_data.response.flatten() - fit._X @ fit._beta_hat
    np.testing.assert_allclose(fit._u_hat, residuals)
    np.testing.assert_allclose(fit.resid(), residuals)
    np.testing.assert_allclose(
        fit._scores,
        fit._X * (weights * residuals)[:, None],
    )
    np.testing.assert_allclose(fit._hessian, fit._X.T @ (weights[:, None] * fit._X))

    with pytest.raises(FrozenInstanceError):
        fit._within_data.response = fit._within_data.design  # type: ignore[misc]


def test_weighted_iv_keeps_each_econometric_role_on_within_scale(
    lifecycle_data: pd.DataFrame,
) -> None:
    """IV state names response, design, endogenous, and instrument roles."""
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
    assert not hasattr(fit, "_Yd")
    assert not hasattr(fit, "_Xd")
    assert not hasattr(fit, "_Zd")
    assert not hasattr(fit, "_endogvard")

    weights = lifecycle_data["weight"].to_numpy(dtype=np.float64)
    weighted_design = weights[:, None] * within.design
    weighted_response = weights[:, None] * within.response
    np.testing.assert_allclose(fit._tZX, within.instruments.T @ weighted_design)
    np.testing.assert_allclose(fit._tZy, within.instruments.T @ weighted_response)
    np.testing.assert_allclose(
        fit._scores,
        within.instruments * (weights * fit._u_hat)[:, None],
    )
    np.testing.assert_allclose(fit.resid(), fit._u_hat)


def test_formula_data_remains_canonical_after_linear_fit(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Formula roles remain tabular while compatibility aliases are transformed."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        weights="weight",
        vcov="iid",
    )

    model_matrix = fit._model_matrix
    assert isinstance(model_matrix, ModelMatrix)

    assert isinstance(model_matrix.dependent, pd.DataFrame)
    assert isinstance(model_matrix.independent, pd.DataFrame)
    assert isinstance(model_matrix.fixed_effects, pd.DataFrame)
    assert isinstance(model_matrix.instruments, pd.DataFrame)
    assert isinstance(model_matrix.weights, pd.DataFrame)
    assert isinstance(fit._Y, np.ndarray)
    assert isinstance(fit._X, np.ndarray)
    assert isinstance(fit._Z, np.ndarray)
    pd.testing.assert_frame_equal(
        model_matrix.dependent,
        lifecycle_data.loc[:, ["y"]],
    )
    pd.testing.assert_frame_equal(
        model_matrix.weights,
        lifecycle_data.loc[:, ["weight"]],
    )
    assert fit._model_spec is model_matrix.model_spec


def test_unweighted_effective_n_remains_integer_for_prediction_errors(
    lifecycle_data: pd.DataFrame,
) -> None:
    """An integer physical row count remains usable by prediction allocation."""
    fit = pf.feols("y ~ x", data=lifecycle_data, vcov="iid")

    assert isinstance(fit._N, int)
    assert isinstance(fit._observation_weights.n_effective, int)
    assert fit.predict(se_fit=True).shape == (len(lifecycle_data),)


def test_glm_separation_replaces_formula_data_with_filtered_state() -> None:
    """Canonical GLM formula data describes the post-separation sample."""
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 1, 2, 3],
            "fe": ["a", "a", "b", "b", "b", "c"],
            "x": [-1.0, 0.5, 0.25, 1.0, -0.5, 1.5],
        }
    )

    with pytest.warns(
        UserWarning, match="2 observations removed because of separation"
    ):
        fit = pf.fepois(
            "y ~ x | fe",
            data=data,
            vcov="hetero",
            separation_check=["fe"],
        )

    model_matrix = fit._model_matrix
    assert model_matrix.dependent.index.equals(fit._data.index)
    assert model_matrix.independent.index.equals(fit._data.index)
    assert model_matrix.fixed_effects is not None
    assert model_matrix.fixed_effects.index.equals(fit._data.index)
    # Row 5 is a formula-stage singleton; rows 0 and 1 are separated.
    assert model_matrix.na_index == frozenset({0, 1, 5})
    assert len(model_matrix.dependent) == fit._N_rows
    assert fit.n_separation_na == 2


def test_model_matrix_without_rows_returns_filtered_copy(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Estimator-level row filters yield a new ModelMatrix and keep the source."""
    model_matrix = create_model_matrix(
        formula=Formula.parse("y ~ x | fe")[0],
        data=lifecycle_data.copy(),
        weights="weight",
    )
    kept_index = model_matrix.dependent.index.drop([0, 5])

    filtered = model_matrix.without_rows([0, 5])

    assert model_matrix.without_rows([]) is model_matrix
    assert filtered is not model_matrix
    assert filtered.na_index == model_matrix.na_index | {0, 5}
    assert filtered.model_spec is model_matrix.model_spec
    for role in ("dependent", "independent", "fixed_effects", "weights"):
        assert getattr(filtered, role).index.equals(kept_index)
    assert filtered.endogenous is None
    assert filtered.instruments is None
    assert filtered.offset is None
    assert len(model_matrix.dependent) == len(lifecycle_data)


def test_multiple_estimation_shares_array_native_demean_cache(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Multiple fits share one ordered array cache without DataFrame round trips."""
    fit = pf.feols(
        "y ~ sw(x, x2) | fe",
        data=lifecycle_data,
        weights="weight",
        vcov="iid",
    )

    assert isinstance(fit, FixestMulti)
    models = list(fit.all_fitted_models.values())
    assert len(models) == 2

    demeaned_caches = [model._demean_cache.lookup_demeaned_data for model in models]
    preconditioner_caches = [
        model._demean_cache.lookup_preconditioner for model in models
    ]
    assert demeaned_caches[0] is demeaned_caches[1]
    assert preconditioner_caches[0] is preconditioner_caches[1]
    assert demeaned_caches[0]
    assert all(isinstance(value, DemeanedData) for value in demeaned_caches[0].values())
    assert all(isinstance(model._within_data, WithinLinearData) for model in models)
    assert all(model._Y is model._within_data.response for model in models)
    assert all(model._X is model._within_data.design for model in models)


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
