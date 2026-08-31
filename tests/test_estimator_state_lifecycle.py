"""Characterize and protect private estimator-state lifecycle boundaries.

These tests intentionally inspect implementation details. Public numerical
behavior belongs to the release snapshots and live-R suites; this module locks
the representation, scale, cache, and cleanup seams that those suites cannot
observe. Refactors should replace these assertions with equivalent assertions
about the new explicit state objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.model_matrix import FormulaData
from pyfixest.estimation.models.feols_ import Feols


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


def _clone_state(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return value.copy()
    return value


def _capture_after(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_name: str,
    state_name: str,
    attributes: tuple[str, ...],
    states: dict[str, dict[str, Any]],
) -> None:
    """Capture selected Feols attributes after one lifecycle method."""
    original: Callable[..., Any] = getattr(Feols, method_name)

    def wrapped(self: Feols, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        states[state_name] = {
            attribute: _clone_state(getattr(self, attribute))
            for attribute in attributes
        }
        return result

    monkeypatch.setattr(Feols, method_name, wrapped)


@pytest.mark.parametrize(
    ("weights_type", "expected_n"),
    [("aweights", 24), ("fweights", 52)],
)
def test_feols_fields_change_type_and_scale_during_fit(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_data: pd.DataFrame,
    weights_type: str,
    expected_n: int,
) -> None:
    """Record the DataFrame -> within array -> solver array mutation."""
    states: dict[str, dict[str, Any]] = {}
    _capture_after(
        monkeypatch,
        method_name="prepare_model_matrix",
        state_name="formula",
        attributes=("_Y", "_X", "_weights"),
        states=states,
    )
    _capture_after(
        monkeypatch,
        method_name="demean",
        state_name="within_frame",
        attributes=("_Y", "_X", "_Yd", "_Xd"),
        states=states,
    )
    _capture_after(
        monkeypatch,
        method_name="to_array",
        state_name="within_array",
        attributes=("_Y", "_X"),
        states=states,
    )
    _capture_after(
        monkeypatch,
        method_name="wls_transform",
        state_name="solver",
        attributes=("_Y", "_X"),
        states=states,
    )

    fit = pf.feols(
        "y ~ x | fe",
        data=lifecycle_data,
        weights="weight",
        weights_type=weights_type,
        vcov="iid",
    )

    assert isinstance(states["formula"]["_Y"], pd.DataFrame)
    assert isinstance(states["formula"]["_X"], pd.DataFrame)
    assert isinstance(states["within_frame"]["_Yd"], pd.DataFrame)
    assert isinstance(states["within_frame"]["_Xd"], pd.DataFrame)
    assert isinstance(states["within_array"]["_Y"], np.ndarray)
    assert isinstance(states["within_array"]["_X"], np.ndarray)
    assert isinstance(states["solver"]["_Y"], np.ndarray)
    assert isinstance(states["solver"]["_X"], np.ndarray)

    np.testing.assert_allclose(
        states["within_array"]["_Y"], states["within_frame"]["_Yd"].to_numpy()
    )
    np.testing.assert_allclose(
        states["within_array"]["_X"], states["within_frame"]["_Xd"].to_numpy()
    )

    sqrt_weight = np.sqrt(states["formula"]["_weights"])
    np.testing.assert_allclose(
        states["solver"]["_Y"], states["within_array"]["_Y"] * sqrt_weight
    )
    np.testing.assert_allclose(
        states["solver"]["_X"], states["within_array"]["_X"] * sqrt_weight
    )

    weighted_group_mean = (lifecycle_data["y"] * lifecycle_data["weight"]).groupby(
        lifecycle_data["fe"]
    ).transform("sum") / lifecycle_data["weight"].groupby(
        lifecycle_data["fe"]
    ).transform("sum")
    expected_y_within = lifecycle_data["y"] - weighted_group_mean
    np.testing.assert_allclose(
        states["within_frame"]["_Yd"].to_numpy().flatten(), expected_y_within
    )

    np.testing.assert_allclose(
        fit._u_hat.flatten(), fit.resid() * np.sqrt(fit._weights.flatten())
    )
    assert expected_n == fit._N


def test_weighted_iv_fields_end_on_solver_scale(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Characterize the retained Y/X/Z domains of a weighted FE-IV fit."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        weights="weight",
        weights_type="aweights",
        vcov="iid",
    )

    assert isinstance(fit._Yd, pd.DataFrame)
    assert isinstance(fit._Xd, pd.DataFrame)
    assert isinstance(fit._Zd, pd.DataFrame)
    assert isinstance(fit._Y, np.ndarray)
    assert isinstance(fit._X, np.ndarray)
    assert isinstance(fit._Z, np.ndarray)

    sqrt_weight = np.sqrt(fit._weights)
    np.testing.assert_allclose(fit._Y, fit._Yd.to_numpy() * sqrt_weight)
    np.testing.assert_allclose(fit._X, fit._Xd.to_numpy() * sqrt_weight)
    np.testing.assert_allclose(fit._Z, fit._Zd.to_numpy() * sqrt_weight)
    np.testing.assert_allclose(
        fit._u_hat.flatten(), fit.resid() * sqrt_weight.flatten()
    )


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

    formula_data = fit._formula_data
    assert isinstance(formula_data, FormulaData)
    assert not hasattr(formula_data, "__dict__")
    with pytest.raises(FrozenInstanceError):
        formula_data.na_index = frozenset()

    assert isinstance(formula_data.dependent, pd.DataFrame)
    assert isinstance(formula_data.independent, pd.DataFrame)
    assert isinstance(formula_data.fixed_effects, pd.DataFrame)
    assert isinstance(formula_data.instruments, pd.DataFrame)
    assert isinstance(formula_data.weights, pd.DataFrame)
    assert isinstance(fit._Y, np.ndarray)
    assert isinstance(fit._X, np.ndarray)
    assert isinstance(fit._Z, np.ndarray)
    pd.testing.assert_frame_equal(
        formula_data.dependent,
        lifecycle_data.loc[:, ["y"]],
    )
    pd.testing.assert_frame_equal(
        formula_data.weights,
        lifecycle_data.loc[:, ["weight"]],
    )
    assert fit._model_spec is formula_data.model_spec


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

    formula_data = fit._formula_data
    assert formula_data.dependent.index.equals(fit._data.index)
    assert formula_data.independent.index.equals(fit._data.index)
    assert formula_data.fixed_effects is not None
    assert formula_data.fixed_effects.index.equals(fit._data.index)
    # Row 5 is a formula-stage singleton; rows 0 and 1 are separated.
    assert formula_data.na_index == frozenset({0, 1, 5})
    assert len(formula_data.dependent) == fit._N_rows
    assert fit.n_separation_na == 2


def test_multiple_estimation_shares_dataframe_demean_cache(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Record the shared mutable DataFrame cache used by FixestMulti."""
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
    assert all(isinstance(value, pd.DataFrame) for value in demeaned_caches[0].values())
    assert all(isinstance(model._Yd, pd.DataFrame) for model in models)
    assert all(isinstance(model._Xd, pd.DataFrame) for model in models)
    assert all(isinstance(model._Y, np.ndarray) for model in models)
    assert all(isinstance(model._X, np.ndarray) for model in models)


@pytest.mark.parametrize(
    ("fit_kwargs", "missing_fields"),
    [
        ({}, frozenset()),
        ({"store_data": False}, frozenset({"_data", "_formula_data"})),
        (
            {"lean": True},
            frozenset(
                {
                    "_data",
                    "_X",
                    "_Y",
                    "_Z",
                    "_Xd",
                    "_Yd",
                    "_weights",
                    "_scores",
                    "_u_hat",
                    "_formula_data",
                }
            ),
        ),
    ],
)
def test_storage_options_delete_expected_state(
    lifecycle_data: pd.DataFrame,
    fit_kwargs: dict[str, bool],
    missing_fields: frozenset[str],
) -> None:
    """Distinguish data-only cleanup from lean fit-state cleanup."""
    state_fields = frozenset(
        {
            "_data",
            "_X",
            "_Y",
            "_Z",
            "_Xd",
            "_Yd",
            "_weights",
            "_scores",
            "_u_hat",
            "_formula_data",
        }
    )
    fit = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid", **fit_kwargs)

    observed_fields = frozenset(field for field in state_fields if hasattr(fit, field))
    assert observed_fields == state_fields - missing_fields
    assert np.isfinite(fit.coef()).all()


def test_feglm_overwrites_observation_weights_with_working_weights() -> None:
    """Record the current observation/IRLS weight field collision."""
    rng = np.random.default_rng(8675309)
    n_obs = 160
    covariate = rng.normal(size=n_obs)
    probability = 1 / (1 + np.exp(-(-0.2 + 0.8 * covariate)))
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, probability),
            "x": covariate,
            "observation_weight": np.linspace(0.5, 2.0, n_obs),
        }
    )

    fit = pf.feglm(
        "y ~ x",
        data=data,
        family="logit",
        weights="observation_weight",
        vcov="iid",
        iwls_tol=1e-10,
    )

    observation_weights = fit._weights_df.to_numpy()
    assert isinstance(fit._Y_untransformed, pd.DataFrame)
    assert isinstance(fit._X, np.ndarray)
    assert isinstance(fit._Y, np.ndarray)
    np.testing.assert_allclose(fit._weights.flatten(), fit._irls_weights.flatten())
    assert not np.allclose(fit._weights.flatten(), observation_weights.flatten())
    np.testing.assert_allclose(fit._scores, fit._u_hat[:, None] * fit._X)
