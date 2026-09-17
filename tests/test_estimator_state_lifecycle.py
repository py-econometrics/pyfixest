"""Protect the estimator-state lifecycle boundaries of formula data.

These tests deliberately inspect private state: public numerical behavior is
covered by the release snapshots and live-R suites, while the representation
and row-sample seams locked here are not observable from those suites.
"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import EmptyVcovError
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.model_matrix import ModelMatrix, create_model_matrix
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.literals import DropStageOptions
from pyfixest.estimation.internals.model_state import (
    DroppedRowCounts,
    EstimationSample,
    ObservationWeights,
    SandwichComponents,
    VarianceCovariance,
    WithinIvData,
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

    assert isinstance(fit.model_matrix, ModelMatrix)
    assert isinstance(fit.model_matrix.dependent, pd.DataFrame)
    assert isinstance(fit.observation_weights, ObservationWeights)
    assert isinstance(fit.within_data, WithinLinearData)
    assert not hasattr(fit, "_Z")
    assert not hasattr(fit, "_Yd")
    assert not hasattr(fit, "_Xd")

    weights = lifecycle_data["weight"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(fit.observation_weights.values, weights)
    assert fit.observation_weights.weights_type == weights_type
    assert expected_n == fit.sample_info.n_obs

    weighted_group_mean = (lifecycle_data["y"] * lifecycle_data["weight"]).groupby(
        lifecycle_data["fe"]
    ).transform("sum") / lifecycle_data["weight"].groupby(
        lifecycle_data["fe"]
    ).transform("sum")
    expected_y_within = lifecycle_data["y"] - weighted_group_mean
    np.testing.assert_allclose(fit.within_data.response.flatten(), expected_y_within)
    assert not np.allclose(
        fit.within_data.response.flatten(),
        expected_y_within * np.sqrt(weights),
    )

    residuals = (
        fit.within_data.response.flatten() - fit.within_data.design @ fit._beta_hat
    )
    np.testing.assert_allclose(fit._u_hat, residuals)
    np.testing.assert_allclose(fit.resid(), residuals)
    sandwich = fit.sandwich
    assert type(sandwich) is SandwichComponents
    np.testing.assert_allclose(
        sandwich.scores,
        fit.within_data.design * (weights * residuals)[:, None],
    )
    hessian = fit.within_data.design.T @ (weights[:, None] * fit.within_data.design)
    np.testing.assert_allclose(sandwich.hessian, hessian)
    # atol: off-diagonal entries of bread @ hessian are rounding noise.
    np.testing.assert_allclose(
        sandwich.bread @ hessian, np.eye(hessian.shape[0]), atol=1e-12
    )
    for name in ("_scores", "_hessian", "_bread", "_tZX", "_tXZ", "_tZy", "_tZZinv"):
        assert not hasattr(fit, name), name

    with pytest.raises(FrozenInstanceError):
        fit.within_data.response = fit.within_data.design  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sandwich.bread = hessian  # type: ignore[misc]


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

    within = fit.within_data
    assert isinstance(within, WithinIvData)
    assert not hasattr(fit, "_Yd")
    assert not hasattr(fit, "_Xd")
    assert not hasattr(fit, "_Zd")
    assert not hasattr(fit, "_endogvard")

    weights = lifecycle_data["weight"].to_numpy(dtype=np.float64)
    weighted_design = weights[:, None] * within.design
    weighted_instruments = weights[:, None] * within.instruments
    tZX = within.instruments.T @ weighted_design
    tZZ = within.instruments.T @ weighted_instruments
    tZZinv = np.linalg.inv(tZZ)
    sandwich = fit.sandwich
    assert isinstance(sandwich, SandwichComponents)
    # The IV Hessian is the 2SLS Hessian, whose inverse is the bread.
    hessian = tZX.T @ tZZinv @ tZX
    np.testing.assert_allclose(sandwich.hessian, hessian)
    # atol: off-diagonal entries of bread @ hessian are rounding noise.
    np.testing.assert_allclose(
        sandwich.bread @ hessian, np.eye(hessian.shape[0]), atol=1e-12
    )
    # 2SLS scores are the OLS scores of the first-stage projection X_hat.
    X_hat = within.instruments @ tZZinv @ tZX
    np.testing.assert_allclose(
        sandwich.scores,
        X_hat * (weights * fit._u_hat)[:, None],
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

    model_matrix = fit.model_matrix
    assert isinstance(model_matrix, ModelMatrix)

    assert isinstance(model_matrix.dependent, pd.DataFrame)
    assert isinstance(model_matrix.independent, pd.DataFrame)
    assert isinstance(model_matrix.fixed_effects, pd.DataFrame)
    assert isinstance(model_matrix.instruments, pd.DataFrame)
    assert isinstance(model_matrix.weights, pd.DataFrame)
    assert isinstance(fit.within_data.response, np.ndarray)
    assert isinstance(fit.within_data.design, np.ndarray)
    assert isinstance(fit.within_data.instruments, np.ndarray)
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

    assert isinstance(fit.sample_info.n_obs, int)
    assert fit.sample_info.n_rows == len(lifecycle_data)
    assert fit.predict(se_fit=True).shape == (len(lifecycle_data),)


def test_glm_separation_replaces_formula_data_with_filtered_state() -> None:
    """Canonical GLM formula data describes the post-separation sample_info."""
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

    model_matrix = fit.model_matrix
    assert model_matrix.dependent.index.equals(fit._data.index)
    assert model_matrix.independent.index.equals(fit._data.index)
    assert model_matrix.fixed_effects is not None
    assert model_matrix.fixed_effects.index.equals(fit._data.index)
    # Row 5 is a formula-stage singleton; rows 0 and 1 are separated.
    assert model_matrix.dropped_row_index == frozenset({0, 1, 5})
    assert len(model_matrix.dependent) == fit.sample_info.n_rows
    assert fit.sample_info.dropped_by_stage == DroppedRowCounts(
        missing=0, singleton=1, separation=2
    )
    assert fit.sample_info.n_rows == model_matrix.n_rows
    assert fit.sample_info.dropped_row_index == model_matrix.dropped_row_index
    assert fit.sample_info.dropped_by_stage == model_matrix.dropped_by_stage


@pytest.mark.parametrize("stage", ["missing", "separation"])
def test_model_matrix_without_rows_returns_filtered_copy(
    lifecycle_data: pd.DataFrame,
    stage: DropStageOptions,
) -> None:
    """Estimator-level row filters yield a new ModelMatrix and keep the source."""
    model_matrix = create_model_matrix(
        formula=Formula.parse("y ~ x | fe")[0],
        data=lifecycle_data.copy(),
        weights="weight",
    )
    kept_index = model_matrix.dependent.index.drop([0, 5])

    filtered = model_matrix.without_rows([0, 5], stage=stage)

    assert model_matrix.without_rows([], stage=stage) is model_matrix
    assert filtered is not model_matrix
    assert filtered.dropped_row_index == model_matrix.dropped_row_index | {0, 5}
    assert filtered.model_spec is model_matrix.model_spec
    for role in ("dependent", "independent", "fixed_effects", "weights"):
        assert getattr(filtered, role).index.equals(kept_index)
    assert filtered.endogenous is None
    assert filtered.instruments is None
    assert filtered.offset is None
    assert len(model_matrix.dependent) == len(lifecycle_data)

    # The source keeps its bookkeeping; the copy counts rows under the given stage.
    assert model_matrix.dropped_by_stage == DroppedRowCounts()
    assert model_matrix.n_rows == len(lifecycle_data)
    assert filtered.dropped_by_stage == DroppedRowCounts(**{stage: 2})
    assert filtered.dropped_row_index == frozenset({0, 5})
    assert filtered.n_rows == len(lifecycle_data) - 2


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
    assert all(isinstance(model.within_data, WithinLinearData) for model in models)


@pytest.mark.parametrize(
    ("fml", "weights", "weights_type"),
    [
        ("y ~ x", None, "aweights"),
        ("y ~ x | fe", "weight", "aweights"),
        ("y ~ x | fe", "weight", "fweights"),
    ],
)
@pytest.mark.parametrize("storage", [{}, {"store_data": False}, {"lean": True}])
def test_gaussian_glm_performance_uses_explicit_response_domains(
    lifecycle_data: pd.DataFrame,
    fml: str,
    weights: str | None,
    weights_type: str,
    storage: dict,
) -> None:
    fit = pf.feglm(
        fml,
        data=lifecycle_data,
        family="gaussian",
        weights=weights,
        weights_type=weights_type,
        vcov="iid",
        iwls_tol=1e-10,
        **storage,
    )
    # Gaussian fitting does not yet populate performance statistics.
    for attribute in ("_rmse", "_r2", "_adj_r2", "_r2_within", "_adj_r2_within"):
        assert np.isnan(getattr(fit, attribute)), attribute
    if storage:
        return
    fit.get_performance()
    response = lifecycle_data["y"].to_numpy()
    observation_weights = fit.observation_weights.values
    residuals = fit.working_state.response_residuals
    if observation_weights is None:
        ssu = np.sum(residuals**2)
        ssy = np.sum((response - np.mean(response)) ** 2)
    else:
        ssu = np.sum(observation_weights * residuals**2)
        center = np.average(response, weights=observation_weights)
        ssy = np.sum(observation_weights * (response - center) ** 2)
    np.testing.assert_allclose(fit._rmse, np.sqrt(ssu / fit.sample_info.n_obs))
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


@pytest.mark.parametrize("copy_data", [False, True])
@pytest.mark.parametrize(
    "estimator,formula,kwargs",
    [
        (pf.feols, "y ~ x + x2", {}),
        (pf.feols, "y ~ x + I(2 * x)", {}),
        (pf.feols, "y ~ x + [endog ~ z]", {"weights": "weight"}),
        (pf.feols, "y ~ x + [endog ~ z] | fe", {"weights": "weight"}),
        (pf.feglm, "y ~ x", {"family": "gaussian", "weights": "weight"}),
        (pf.feglm, "y ~ x | fe", {"family": "gaussian", "weights": "weight"}),
        (pf.feols, "y ~ x | fe", {"weights": "weight", "weights_type": "fweights"}),
        (pf.fepois, "weight ~ x | fe", {"offset": "x2", "weights": "weight"}),
        (pf.quantreg, "y ~ x", {}),
    ],
)
def test_published_components_preserve_inputs(
    lifecycle_data, copy_data, estimator, formula, kwargs
):
    """Publication exposes canonical components without modifying caller data."""
    from pyfixest.estimation.state import GlmWorkingState, ModelMatrix

    original = lifecycle_data.copy(deep=True)
    input_array = lifecycle_data["x"].to_numpy()
    writeable_before = input_array.flags.writeable
    fit = estimator(formula, lifecycle_data, copy_data=copy_data, **kwargs)
    assert isinstance(fit.model_matrix, ModelMatrix)
    component = fit.working_state if hasattr(fit, "working_state") else fit.within_data
    if isinstance(component, GlmWorkingState):
        assert not hasattr(fit, "within_data")
    removed = (
        "_model_matrix",
        "_observation_weights",
        "_within_data",
        "_working_state",
        "_X",
        "_Y",
        "_Z",
        "_endogvar",
        "_weights",
        "_weights_df",
        "_offset_df",
        "_offset",
        "_fe",
        "_Y_untransformed",
        "_irls_weights",
        "_Xbeta",
        "_u_hat_response",
        "_u_hat_working",
        "_scores",
        "_hessian",
        "_bread",
        "_tZX",
        "_tXZ",
        "_tZy",
        "_tZZinv",
        "_tZXinv",
    )
    assert not any(hasattr(fit, name) for name in removed)
    if estimator is pf.quantreg:
        # Quantile inference follows R quantreg and never reads a sandwich.
        assert not hasattr(fit, "sandwich")
    else:
        assert isinstance(fit.sandwich, SandwichComponents)
    pd.testing.assert_frame_equal(lifecycle_data, original)
    assert input_array.flags.writeable == writeable_before


@pytest.mark.parametrize("multi_method", ["cfm1", "cfm2"])
@pytest.mark.parametrize("store_data", [False, True])
@pytest.mark.parametrize("lean", [False, True])
def test_multi_quantile_children_follow_ols_retention(
    lifecycle_data, multi_method, store_data, lean
):
    """Both process solvers apply the OLS storage policy to every child."""
    fit = pf.quantreg(
        "y ~ x",
        lifecycle_data,
        quantile=[0.25, 0.5, 0.75],
        multi_method=multi_method,
        seed=42,
        store_data=store_data,
        lean=lean,
    )
    ols = pf.feols("y ~ x", lifecycle_data, store_data=store_data, lean=lean)
    assert hasattr(ols, "sandwich") == (not lean)
    for child in fit.to_list():
        assert not hasattr(child, "sandwich")
        for name in ("_data", "model_matrix", "within_data", "observation_weights"):
            assert hasattr(child, name) == hasattr(ols, name), name
        if lean:
            assert not hasattr(child, "within_data")
            continue
        assert isinstance(child.within_data, WithinLinearData)
        np.testing.assert_allclose(
            child.predict()[:3],
            child.predict(lifecycle_data.iloc[:3]),
            rtol=1e-12,
            atol=1e-12,
            err_msg="multi-quantile retained and newdata predictions disagree",
        )
    if multi_method == "cfm1" and not store_data and not lean:
        assert np.isfinite(fit.to_list()[0].objective_value), (
            "store_data=False made the retained quantile objective unavailable"
        )


@pytest.mark.parametrize("store_data", [False, True])
@pytest.mark.parametrize("lean", [False, True])
def test_iv_first_stage_follows_parent_retention(
    lifecycle_data: pd.DataFrame, store_data: bool, lean: bool
) -> None:
    """IV cleanup keeps completed diagnostics while stripping parent and child."""
    fit = pf.feols(
        "y ~ x + [endog ~ z]",
        lifecycle_data,
        vcov="hetero",
        store_data=store_data,
        lean=lean,
    )
    first_stage = fit._model_1st_stage

    for model in (fit, first_stage):
        assert hasattr(model, "_data") is (store_data and not lean)
        assert hasattr(model, "model_matrix") is (store_data and not lean)
        assert hasattr(model, "within_data") is (not lean)
        assert hasattr(model, "observation_weights") is (not lean)
        assert np.isfinite(model.coef()).all()
        assert np.isfinite(model.se()).all()
        # The sample survives every storage option unchanged.
        assert model.sample_info.n_rows == len(lifecycle_data)
        assert model.sample_info.dropped_by_stage == DroppedRowCounts()

    retained_f = fit._f_stat_1st_stage
    fit.IV_weakness_test(["f_stat"])
    np.testing.assert_allclose(
        fit._f_stat_1st_stage,
        retained_f,
        rtol=1e-12,
        atol=1e-12,
        err_msg="IV first-stage F statistic changed after retained-state cleanup",
    )


def test_store_data_false_retains_robust_effective_f(
    lifecycle_data: pd.DataFrame,
) -> None:
    reference = pf.feols(
        "y ~ x + [endog ~ z]",
        lifecycle_data,
        vcov="hetero",
    )
    fit = pf.feols(
        "y ~ x + [endog ~ z]",
        lifecycle_data,
        vcov="hetero",
        store_data=False,
    )

    reference.eff_F()
    fit.eff_F()

    np.testing.assert_allclose(
        fit._eff_F,
        reference._eff_F,
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed robust effective-F",
    )


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        (pf.feols, {}),
        (pf.fepois, {}),
        (pf.feglm, {"family": "gaussian"}),
        (pf.quantreg, {}),
    ],
)
def test_lean_prediction_on_new_data_without_fixed_effects(
    lifecycle_data: pd.DataFrame, estimator, kwargs
) -> None:
    data = lifecycle_data.assign(y_count=np.tile([1, 2, 3, 4], 6))
    outcome = "y_count" if estimator is pf.fepois else "y"
    reference = estimator(f"{outcome} ~ x", data, **kwargs)
    fit = estimator(f"{outcome} ~ x", data, lean=True, **kwargs)

    expected = reference.predict(newdata=data.iloc[:3])
    prediction = fit.predict(newdata=data.iloc[:3])

    np.testing.assert_allclose(
        prediction,
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="lean cleanup changed no-FE new-data predictions",
    )


def test_store_data_false_preserves_no_fe_post_estimation(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Methods needing only retained arrays stay available without raw data."""
    reference = pf.feols("y ~ x + x2", lifecycle_data)
    fit = pf.feols("y ~ x + x2", lifecycle_data, store_data=False)

    reference_boot = reference.wildboottest(param="x", reps=99, seed=42)
    stripped_boot = fit.wildboottest(param="x", reps=99, seed=42)
    np.testing.assert_allclose(
        stripped_boot[["t value", "Pr(>|t|)"]].to_numpy(dtype=float),
        reference_boot[["t value", "Pr(>|t|)"]].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed no-FE heteroskedastic bootstrap results",
    )

    reference_decomposition = reference.decompose(decomp_var="x", only_coef=True)
    stripped_decomposition = fit.decompose(decomp_var="x", only_coef=True)
    for name, expected in reference_decomposition.results.absolute.items():
        np.testing.assert_allclose(
            stripped_decomposition.results.absolute[name],
            expected,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"store_data=False changed decomposition quantity {name}",
        )


@pytest.mark.parametrize(
    "estimator,formula,kwargs,expected_separation",
    [
        (pf.feols, "y ~ x | fe", {}, 0),
        (pf.feols, "y ~ x | fe", {"weights": "weight", "weights_type": "fweights"}, 0),
        (pf.feols, "y ~ x + [endog ~ z] | fe", {}, 0),
        (pf.feols, "y ~ csw(x, x2) | fe", {}, 0),
        (pf.fepois, "count ~ x | fe", {"separation_check": ["fe"]}, 6),
        (pf.feglm, "binary ~ x | fe", {"family": "logit"}, 0),
        (pf.quantreg, "y ~ x", {"quantile": 0.5}, 0),
    ],
)
def test_estimation_sample_counts_dropped_rows_by_stage(
    lifecycle_data, estimator, formula, kwargs, expected_separation
):
    """Every estimator reports its final row sample and the stage of each dropped row."""
    data = lifecycle_data.assign(
        count=np.tile([1, 3, 2, 4], 6), binary=np.tile([0, 1], 12)
    )
    data.loc[1, "x"] = np.nan  # formula missing-value handling
    data.loc[1, "fe"] = "solo"  # would be a singleton, but is already missing
    data.loc[2, "x"] = np.inf  # infinite filter
    data.loc[[2, 4], "fe"] = "pair"  # row 4 becomes a singleton once 2 is dropped
    data.loc[3, "fe"] = "solo"  # singleton fixed-effect level
    data.loc[data["fe"] == "b", "count"] = 0  # level b is separated for fepois
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator(formula, data, **kwargs)
    models = result.to_list() if isinstance(result, FixestMulti) else [result]
    uses_fe = "| fe" in formula
    expected_dropped = DroppedRowCounts(
        missing=1,
        infinite=1,
        singleton=2 * int(uses_fe),
        separation=expected_separation,
    )
    expected_index = data.index.drop([1, 2] + ([3, 4] if uses_fe else []))
    if expected_separation:
        expected_index = expected_index.drop(
            data.index[(data["fe"] == "b") & (data["count"] == 0)]
        )
    for model in models:
        sample_info = model.sample_info
        assert isinstance(sample_info, EstimationSample)
        assert sample_info.n_rows == model.model_matrix.n_rows
        assert sample_info.dropped_row_index == model.model_matrix.dropped_row_index
        assert sample_info.dropped_by_stage == model.model_matrix.dropped_by_stage
        assert sample_info.dropped_by_stage == expected_dropped
        assert sample_info.dropped_by_stage.total == len(sample_info.dropped_row_index)
        assert sample_info.n_rows == len(data) - expected_dropped.total
        assert sample_info.n_rows == len(model.resid())
        assert set(sample_info.dropped_row_index) == set(
            data.index.difference(expected_index)
        )
        if kwargs.get("weights_type") == "fweights":
            # Singletons are physical rows; the effective count sums weights.
            assert sample_info.n_obs == data.loc[expected_index, "weight"].sum()
            assert isinstance(sample_info.n_obs, float)
        else:
            assert sample_info.n_obs == sample_info.n_rows
            assert isinstance(sample_info.n_obs, int)
        if model._is_iv:
            # The first stage is refit on the retained rows: it owns a sample
            # with no dropped rows of its own.
            first_stage = model._model_1st_stage.sample_info
            assert first_stage is not sample_info
            assert first_stage.dropped_by_stage == DroppedRowCounts()
            assert first_stage.n_rows == sample_info.n_rows
    assert len({id(model.sample_info) for model in models}) == len(models)


def test_split_samples_count_only_formula_drops(lifecycle_data: pd.DataFrame):
    """A split selects each child's rows; only formula filters count as dropped."""
    data = lifecycle_data.copy()
    data.loc[7, "x"] = np.nan
    fit = pf.feols("y ~ x", data, split="fe")
    for model in fit.to_list():
        level = model._sample_split_value
        population = data.index[data["fe"] == level]
        sample_info = model.sample_info
        assert sample_info.n_rows == len(population) - int(level == "b")
        assert sample_info.dropped_by_stage == DroppedRowCounts(
            missing=int(level == "b")
        )
        # Dropped positions count from zero in the child's input frame.
        assert sample_info.dropped_row_index == frozenset(
            np.flatnonzero(population == 7).tolist()
        )


@pytest.mark.parametrize(
    ("estimator", "formula", "vcov", "vcov_kwargs"),
    [
        (pf.feols, "y ~ x + x2 | fe", "HC1", None),
        (pf.feols, "y ~ x + x2 | fe", {"CRV1": "fe+group"}, None),
        (
            pf.feols,
            "y ~ x | fe",
            "NW",
            {"time_id": "period", "panel_id": "unit", "lag": 2},
        ),
        (pf.feols, "y ~ x | fe | endog ~ z", {"CRV1": "fe"}, None),
        (pf.fepois, "count ~ x | fe", {"CRV1": "fe"}, None),
    ],
)
def test_meat_reproduces_the_adjusted_vcov(
    lifecycle_data, estimator, formula, vcov, vcov_kwargs
):
    """The published meat sandwiches back to the published covariance.

    Nothing outside the fitted model reads the meat, so the live-R suites
    cannot catch a wrong one; this identity is its only check.
    """
    data = lifecycle_data.assign(
        group=np.tile(["g1", "g2", "g3"], 8),
        period=np.tile(np.arange(6), 4),
        unit=np.repeat(np.arange(4), 6),
        count=np.random.default_rng(3).poisson(2.0, size=len(lifecycle_data)),
    )
    fit = estimator(formula, data, vcov=vcov, vcov_kwargs=vcov_kwargs)
    covariance = fit.variance_covariance

    assert isinstance(covariance, VarianceCovariance)
    bread = fit.sandwich.bread
    np.testing.assert_allclose(
        covariance.vcov, bread @ covariance.meat @ bread, rtol=1e-12, atol=1e-14
    )
    assert covariance.ssc.shape == (len(covariance.G) or 1,)


def test_get_inference_before_vcov_raises_empty_vcov(lifecycle_data):
    """A fixed-effects-only fit skips vcov() and carries no covariance."""
    fit = pf.feols("y ~ 1 | fe", lifecycle_data)
    assert not hasattr(fit, "variance_covariance")
    with pytest.raises(EmptyVcovError):
        fit.get_inference()


def test_quantreg_rejects_multiway_clustering(lifecycle_data):
    """Quantile regression declares no multiway support before any state is read."""
    data = lifecycle_data.assign(group=np.tile(["g1", "g2", "g3"], 8))
    with pytest.raises(NotImplementedError, match="Multiway clustering"):
        pf.quantreg("y ~ x", data, vcov={"CRV1": "fe+group"})
