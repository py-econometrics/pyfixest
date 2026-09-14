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
    assert expected_n == fit._N

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
    np.testing.assert_allclose(
        fit._scores,
        fit.within_data.design * (weights * residuals)[:, None],
    )
    np.testing.assert_allclose(
        fit._hessian,
        fit.within_data.design.T @ (weights[:, None] * fit.within_data.design),
    )

    with pytest.raises(FrozenInstanceError):
        fit.within_data.response = fit.within_data.design  # type: ignore[misc]


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

    assert isinstance(fit._N, int)
    assert isinstance(fit.observation_weights.n_effective, int)
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

    model_matrix = fit.model_matrix
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
    )
    assert not any(hasattr(fit, name) for name in removed)
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
    for child in fit.to_list():
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


@pytest.mark.parametrize("store_data", [False, True])
@pytest.mark.parametrize("lean", [False, True])
def test_retention_options_remove_exact_legacy_attributes(
    lifecycle_data: pd.DataFrame, store_data: bool, lean: bool
) -> None:
    """Each storage combination follows the pre-existing deletion table."""
    fit = pf.feols(
        "y ~ x | fe",
        lifecycle_data,
        vcov={"CRV1": "fe"},
        store_data=store_data,
        lean=lean,
    )

    expected_removed = set()
    if not store_data:
        expected_removed.update({"_data", "model_matrix"})
    if lean:
        expected_removed.update(
            {
                "_data",
                "model_matrix",
                "_cluster_df",
                "_tXZ",
                "_tZy",
                "_tZX",
                "_scores",
                "_tZZinv",
                "_u_hat",
                "_Y_hat_link",
                "_Y_hat_response",
                "within_data",
                "observation_weights",
            }
        )

    checked = {
        "_data",
        "model_matrix",
        "_cluster_df",
        "_tXZ",
        "_tZy",
        "_tZX",
        "_scores",
        "_tZZinv",
        "_u_hat",
        "_Y_hat_link",
        "_Y_hat_response",
        "within_data",
        "observation_weights",
    }
    for attribute in checked:
        assert hasattr(fit, attribute) is (attribute not in expected_removed), attribute

    glm = pf.feglm(
        "y ~ x",
        lifecycle_data,
        family="gaussian",
        store_data=store_data,
        lean=lean,
    )
    assert hasattr(glm, "working_state") is (not lean)


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
        fit._model_1st_stage.coef(),
        reference._model_1st_stage.coef(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained first-stage coefficients",
    )
    np.testing.assert_allclose(
        fit._model_1st_stage.se(),
        reference._model_1st_stage.se(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained first-stage inference",
    )
    np.testing.assert_allclose(
        fit._f_stat_1st_stage,
        reference._f_stat_1st_stage,
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained first-stage F statistic",
    )
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


def test_quantile_objective_remains_lazy_when_residuals_are_retained(
    lifecycle_data: pd.DataFrame,
) -> None:
    reference = pf.quantreg("y ~ x", lifecycle_data, quantile=0.35)
    fit = pf.quantreg("y ~ x", lifecycle_data, quantile=0.35, store_data=False)

    assert not hasattr(fit, "_objective_value")
    np.testing.assert_allclose(
        fit.objective_value,
        reference.objective_value,
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed the lazy quantile objective",
    )


def test_store_data_false_preserves_no_fe_post_estimation(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Methods needing only retained arrays stay available without raw data."""
    reference = pf.feols("y ~ x + x2", lifecycle_data)
    fit = pf.feols("y ~ x + x2", lifecycle_data, store_data=False)

    np.testing.assert_allclose(
        fit.coef(),
        reference.coef(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained coefficients",
    )
    np.testing.assert_allclose(
        fit.se(),
        reference.se(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained inference",
    )
    np.testing.assert_allclose(
        fit.resid(),
        reference.resid(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="store_data=False changed retained residuals",
    )

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
