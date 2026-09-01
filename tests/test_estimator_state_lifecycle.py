"""Protect estimator representation, scale, cache, and cleanup boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.model_matrix import FormulaData
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.model_state import (
    GlmWorkingState,
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

    assert isinstance(fit._formula_data, FormulaData)
    assert isinstance(fit._formula_data.dependent, pd.DataFrame)
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
    """Formula roles remain tabular after numerical arrays are prepared."""
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
        formula_data.na_index = frozenset()  # type: ignore[misc]

    assert isinstance(formula_data.dependent, pd.DataFrame)
    assert isinstance(formula_data.independent, pd.DataFrame)
    assert isinstance(formula_data.fixed_effects, pd.DataFrame)
    assert isinstance(formula_data.instruments, pd.DataFrame)
    assert isinstance(formula_data.weights, pd.DataFrame)
    pd.testing.assert_frame_equal(
        formula_data.dependent,
        lifecycle_data.loc[:, ["y"]],
    )
    pd.testing.assert_frame_equal(
        formula_data.weights,
        lifecycle_data.loc[:, ["weight"]],
    )
    assert fit._model_spec is formula_data.model_spec


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

    formula_data = fit._formula_data
    assert formula_data.dependent.index.equals(fit._data.index)
    assert formula_data.independent.index.equals(fit._data.index)
    assert formula_data.fixed_effects is not None
    assert formula_data.fixed_effects.index.equals(fit._data.index)
    # Row 5 is a formula-stage singleton; rows 0 and 1 are separated.
    assert formula_data.na_index == frozenset({0, 1, 5})
    assert len(formula_data.dependent) == fit._N_rows
    assert fit.n_separation_na == 2


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


@pytest.mark.parametrize("fit_kwargs", [{"store_data": False}, {"lean": True}])
def test_multiple_estimation_respects_storage_options(
    lifecycle_data: pd.DataFrame,
    fit_kwargs: dict[str, bool],
) -> None:
    """The result container must not retain data cleared from all child fits."""
    fit = pf.feols(
        "y ~ sw(x, x2) | fe",
        data=lifecycle_data,
        vcov="iid",
        **fit_kwargs,
    )

    assert isinstance(fit, FixestMulti)
    assert not hasattr(fit, "_data")
    assert not hasattr(fit, "_config")
    assert not hasattr(fit, "_context")
    assert all(not hasattr(model, "_data") for model in fit.to_list())


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
                    "_formula_data",
                    "_within_data",
                    "_observation_weights",
                    "_demean_cache",
                    "_X",
                    "_Y",
                    "_Z",
                    "_weights",
                    "_scores",
                    "_u_hat",
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
            "_formula_data",
            "_within_data",
            "_observation_weights",
            "_demean_cache",
            "_X",
            "_Y",
            "_Z",
            "_weights",
            "_scores",
            "_u_hat",
        }
    )
    fit = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid", **fit_kwargs)

    observed_fields = frozenset(field for field in state_fields if hasattr(fit, field))
    assert observed_fields == state_fields - missing_fields
    if fit_kwargs.get("lean"):
        for field in ("_model_spec", "_context", "_fe", "_weights_df", "_offset_df"):
            assert not hasattr(fit, field)
    assert np.isfinite(fit.coef()).all()


def test_store_data_false_allows_array_only_vcov_updates(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Array-only covariance updates do not require retained formula data."""
    stripped = pf.feols("y ~ x", data=lifecycle_data, vcov="iid", store_data=False)
    expected = pf.feols("y ~ x", data=lifecycle_data, vcov="HC1")

    stripped.vcov("HC1")

    np.testing.assert_allclose(stripped._vcov, expected._vcov)
    assert not hasattr(stripped, "_data")


def test_store_data_false_vcov_uses_explicit_estimation_sample(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Data-dependent covariance updates use the documented data argument."""
    stripped = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid", store_data=False)
    expected = pf.feols("y ~ x | fe", data=lifecycle_data, vcov={"CRV1": "fe"})

    with pytest.raises(RuntimeError, match=r"store_data=False.*Pass.*data="):
        stripped.vcov({"CRV1": "fe"})

    stripped.vcov({"CRV1": "fe"}, data=lifecycle_data)

    np.testing.assert_allclose(stripped._vcov, expected._vcov)
    assert not hasattr(stripped, "_data")


def test_vcov_rejects_unfiltered_explicit_data(
    lifecycle_data: pd.DataFrame,
) -> None:
    """An explicit covariance sample must align with the fitted row arrays."""
    data = lifecycle_data.copy()
    data.loc[data.index[0], "y"] = np.nan
    estimation_sample = data.dropna(subset=["y", "x", "fe"])
    stripped = pf.feols(
        "y ~ x | fe",
        data=data,
        vcov="iid",
        store_data=False,
    )
    expected = pf.feols(
        "y ~ x | fe",
        data=estimation_sample,
        vcov={"CRV1": "fe"},
    )

    with pytest.raises(
        ValueError,
        match=(
            r"already-filtered estimation sample.*original estimation order; "
            r"expected 23 rows, received 24"
        ),
    ):
        stripped.vcov({"CRV1": "fe"}, data=data)

    stripped.vcov({"CRV1": "fe"}, data=estimation_sample)
    np.testing.assert_allclose(stripped._vcov, expected._vcov)


def test_lean_vcov_fails_with_storage_guidance(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Lean results explain that covariance inputs were intentionally discarded."""
    fit = pf.feols("y ~ x", data=lifecycle_data, vcov="iid", lean=True)

    with pytest.raises(RuntimeError, match=r"vcov\(\).*lean=True.*discarded"):
        fit.vcov("HC1")


def test_iv_first_stage_respects_store_data_false(
    lifecycle_data: pd.DataFrame,
) -> None:
    """A retained IV first stage must not hide a copy of stripped input data."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="hetero",
        store_data=False,
    )

    first_stage = fit._model_1st_stage
    assert not hasattr(first_stage, "_data")
    assert not hasattr(first_stage, "_formula_data")
    assert hasattr(first_stage, "_within_data")
    assert hasattr(first_stage, "_X")
    fit.IV_Diag()
    assert np.isfinite(fit._eff_F)
    with pytest.raises(RuntimeError, match=r"first_stage\(\).*store_data=False"):
        fit.first_stage()


def test_iv_effective_f_explains_stripped_iid_data_requirement(
    lifecycle_data: pd.DataFrame,
) -> None:
    """IID first-stage refits fail informatively when their data were stripped."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="iid",
        store_data=False,
    )

    with pytest.raises(RuntimeError, match=r"effective F.*store_data=False"):
        fit.IV_Diag()


def test_lean_iv_discards_retained_first_stage_arrays(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Lean IV results retain diagnostics but no nested fitted-data graph."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="hetero",
        lean=True,
    )

    assert not hasattr(fit, "_model_1st_stage")
    assert not hasattr(fit, "_X_hat")
    assert not hasattr(fit, "_v_hat")
    assert not hasattr(fit, "_endogvar")
    assert np.isfinite(fit._f_stat_1st_stage)
    with pytest.raises(RuntimeError, match=r"IV_Diag\(\).*lean=True"):
        fit.IV_Diag()


def test_feglm_keeps_observation_and_working_weights_distinct() -> None:
    """The stable observation-weight alias is never replaced by IRLS weights."""
    rng = np.random.default_rng(8675309)
    n_obs = 160
    covariate = rng.normal(size=n_obs)
    probability = 1 / (1 + np.exp(-(-0.2 + 0.8 * covariate)))
    observation_weights = np.linspace(0.5, 2.0, n_obs)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, probability),
            "x": covariate,
            "observation_weight": observation_weights,
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

    working = fit._working_state
    assert isinstance(working, GlmWorkingState)
    np.testing.assert_allclose(fit._weights.flatten(), observation_weights)
    np.testing.assert_allclose(fit._observation_weights.values, observation_weights)
    np.testing.assert_allclose(fit._irls_weights, working.working_weights)
    assert not np.allclose(working.working_weights, observation_weights)
    assert fit._X is working.design_within
    assert fit._Y is working.working_response_within
    np.testing.assert_allclose(fit.resid("response"), working.response_residuals)
    np.testing.assert_allclose(fit.resid("working"), working.working_residuals)
    np.testing.assert_allclose(
        fit._scores,
        fit._X * (working.working_weights * working.working_residuals)[:, None],
    )


def test_glm_lean_discards_formula_and_working_state() -> None:
    """Lean GLM results discard both canonical input and working fit arrays."""
    data = pd.DataFrame({"y": [0, 1, 0, 1, 1, 0], "x": np.arange(6.0)})
    fit = pf.feglm("y ~ x", data=data, family="logit", lean=True, vcov="iid")

    discarded = (
        "_formula_data",
        "_observation_weights",
        "_demean_cache",
        "_working_state",
        "_irls_weights",
        "_u_hat_response",
        "_u_hat_working",
        "_scores_response",
        "_scores_working",
        "_Xbeta",
        "_offset",
    )
    assert all(not hasattr(fit, attr) for attr in discarded)
    assert np.isfinite(fit.coef()).all()
    for residual_type in ("response", "working"):
        with pytest.raises(
            RuntimeError, match=r"resid\(\).*lean=True.*residual arrays were discarded"
        ):
            fit.resid(residual_type)


def test_poisson_lean_discards_offset_and_null_fit_arrays() -> None:
    """Lean Poisson results do not retain offset or null-fit row arrays."""
    data = pd.DataFrame(
        {
            "y": [0, 1, 2, 1, 3, 2],
            "x": np.arange(6.0),
            "offset": np.linspace(0.1, 0.6, 6),
        }
    )
    fit = pf.fepois("y ~ x", data=data, offset="offset", lean=True, vcov="iid")

    assert not hasattr(fit, "_offset")
    assert not hasattr(fit, "_y_hat_null")
    assert np.isfinite(fit.coef()).all()


def test_quantreg_lean_discards_solver_arrays() -> None:
    """Lean quantile results do not retain row-sized solver outputs."""
    data = pd.DataFrame({"y": [0.2, 1.1, 1.8, 3.2, 3.9, 5.1], "x": np.arange(6.0)})
    with pytest.warns(FutureWarning, match="experimental"):
        fit = pf.quantreg("y ~ x", data=data, lean=True, vcov="iid", maxiter=100)

    solver_arrays = ("_x_final", "_s_final", "_z_final", "_w_final", "_y_final")
    assert all(not hasattr(fit, attr) for attr in solver_arrays)
    assert np.isfinite(fit.coef()).all()
