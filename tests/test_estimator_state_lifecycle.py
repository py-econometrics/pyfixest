"""Protect the estimator-state lifecycle boundaries of formula data.

These tests deliberately inspect private state: public numerical behavior is
covered by the release snapshots and live-R suites, while the representation
and row-sample seams locked here are not observable from those suites.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
import pyfixest.estimation.models.base_regression_ as base_regression_module
from pyfixest.errors import NanInClusterVarError, VcovTypeNotSupportedError
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.model_matrix import ModelMatrix, create_model_matrix
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.model_state import (
    GlmWorkingState,
    ObservationWeights,
    WithinLinearData,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.quantreg.QuantregMulti import QuantregMulti

INFERENCE_FIELDS = (
    "_vcov_type",
    "_vcov_type_detail",
    "_is_clustered",
    "_clustervar",
    "_G",
    "_bread",
    "_ssc",
    "_vcov",
    "_df_k",
    "_df_t",
    "_se",
    "_tstat",
    "_pvalue",
    "_conf_int",
    "_cluster_df",
    "_lag",
    "_time_id",
    "_panel_id",
)
_MISSING = object()


def _snapshot_inference(model) -> dict[str, object]:
    """Copy only covariance metadata and derived inference fields."""
    snapshot: dict[str, object] = {}
    for field in INFERENCE_FIELDS:
        value = getattr(model, field, _MISSING)
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, pd.DataFrame):
            value = value.copy(deep=True)
        elif isinstance(value, list):
            value = list(value)
        snapshot[field] = value
    return snapshot


def _assert_inference_unchanged(model, before: dict[str, object]) -> None:
    """Assert exact preservation of covariance metadata and inference arrays."""
    after = _snapshot_inference(model)
    assert after.keys() == before.keys()
    for field, expected in before.items():
        actual = after[field]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected, err_msg=field)
        elif isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected, obj=field)
        else:
            assert actual == expected, field


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


@pytest.mark.parametrize("alias", ["_Y", "_X", "_Z", "_weights"])
def test_array_aliases_are_read_only_views(
    lifecycle_data: pd.DataFrame,
    alias: str,
) -> None:
    """The typed state objects stay the single writable representation."""
    fit = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid")

    with pytest.raises(AttributeError):
        setattr(fit, alias, np.zeros((fit._N_rows, 1)))
    with pytest.raises(AttributeError):
        delattr(fit, alias)


def test_unweighted_weight_alias_is_materialized_on_access(
    lifecycle_data: pd.DataFrame,
) -> None:
    """An unweighted fit stores no weight vector but still exposes one."""
    fit = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid")

    assert fit._observation_weights.values is None
    np.testing.assert_array_equal(fit._weights, np.ones((fit._N_rows, 1)))
    assert fit._weights is not fit._weights


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


LEAN_REJECTION_CASES = [
    ("resid", "y ~ x | fe", lambda fit: fit.resid()),
    ("get_performance", "y ~ x | fe", lambda fit: fit.get_performance()),
    ("predict", "y ~ x | fe", lambda fit: fit.predict()),
    ("fixef", "y ~ x | fe", lambda fit: fit.fixef()),
    ("preconditioner", "y ~ x | fe", lambda fit: fit.preconditioner),
    ("ccv", "y ~ x", lambda fit: fit.ccv(treatment="x")),
    (
        "decompose",
        "y ~ x + x2",
        lambda fit: fit.decompose(decomp_var="x", only_coef=True),
    ),
    ("wildboottest", "y ~ x + x2", lambda fit: fit.wildboottest(param="x", reps=11)),
    ("ritest", "y ~ x + x2", lambda fit: fit.ritest(resampvar="x", reps=11)),
    (
        "update",
        "y ~ x + x2",
        lambda fit: fit.update(np.ones((1, 3)), np.zeros(1)),
    ),
]


@pytest.mark.parametrize(
    ("method", "fml", "call"),
    LEAN_REJECTION_CASES,
    ids=[case[0] for case in LEAN_REJECTION_CASES],
)
def test_lean_results_reject_state_dependent_methods(
    lifecycle_data: pd.DataFrame,
    method: str,
    fml: str,
    call,
) -> None:
    """Discarded fit arrays produce a remedy, not an AttributeError."""
    fit = pf.feols(fml, data=lifecycle_data, vcov="iid", lean=True)

    with pytest.raises(RuntimeError, match=rf"{method}\(\).*lean=True.*discarded"):
        call(fit)


STORE_DATA_REJECTION_CASES = [
    ("fixef", "y ~ x | fe", lambda fit, data: fit.fixef()),
    ("predict", "y ~ x | fe", lambda fit, data: fit.predict(newdata=data)),
    ("ccv", "y ~ x", lambda fit, data: fit.ccv(treatment="x")),
    (
        "decompose",
        "y ~ x + x2 | fe",
        lambda fit, data: fit.decompose(decomp_var="x", only_coef=True),
    ),
    (
        "wildboottest",
        "y ~ x + x2",
        lambda fit, data: fit.wildboottest(param="x", reps=11),
    ),
    ("ritest", "y ~ x + x2", lambda fit, data: fit.ritest(resampvar="x", reps=11)),
    (
        "first_stage",
        "y ~ x + [endog ~ z] | fe",
        lambda fit, data: fit.first_stage(),
    ),
]


@pytest.mark.parametrize(
    ("method", "fml", "call"),
    STORE_DATA_REJECTION_CASES,
    ids=[case[0] for case in STORE_DATA_REJECTION_CASES],
)
def test_store_data_false_results_reject_data_dependent_methods(
    lifecycle_data: pd.DataFrame,
    method: str,
    fml: str,
    call,
) -> None:
    """Methods that read the estimation sample say how to supply it again."""
    fit = pf.feols(fml, data=lifecycle_data, vcov="iid", store_data=False)

    with pytest.raises(RuntimeError, match=rf"{method}\(\).*store_data=False"):
        call(fit, lifecycle_data)


def test_lean_glm_rejects_residualize_and_residuals() -> None:
    """A lean GLM keeps neither its IRLS residuals nor its demeaning caches."""
    data = pd.DataFrame({"y": [0, 1, 0, 1, 1, 0], "x": np.arange(6.0)})
    fit = pf.feglm("y ~ x", data=data, family="logit", vcov="iid", lean=True)

    with pytest.raises(
        RuntimeError, match=r"resid\(\).*lean=True.*residual arrays were discarded"
    ):
        fit.resid("response")
    with pytest.raises(RuntimeError, match=r"residualize\(\).*lean=True.*discarded"):
        fit.residualize(
            v=np.zeros((6, 1)),
            X=np.ones((6, 1)),
            flist=None,
            weights=np.ones((6, 1)),
            tol=1e-06,
        )


def test_lean_quantreg_rejects_objective_value() -> None:
    """The quantile loss needs residuals that lean storage discarded."""
    data = pd.DataFrame({"y": [0.2, 1.1, 1.8, 3.2, 3.9, 5.1], "x": np.arange(6.0)})
    with pytest.warns(FutureWarning, match="experimental"):
        fit = pf.quantreg("y ~ x", data=data, lean=True, vcov="iid", maxiter=100)

    with pytest.raises(RuntimeError, match=r"objective_value\(\).*lean=True"):
        _ = fit.objective_value


@pytest.mark.parametrize("method", ["IV_Diag", "IV_weakness_test", "eff_F"])
def test_lean_iv_rejects_first_stage_diagnostics(
    lifecycle_data: pd.DataFrame,
    method: str,
) -> None:
    """First-stage diagnostics need the retained first-stage fit."""
    fit = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="hetero",
        lean=True,
    )

    assert not hasattr(fit, "_model_1st_stage")
    assert np.isfinite(fit._f_stat_1st_stage)
    with pytest.raises(RuntimeError, match=rf"{method}\(\).*lean=True"):
        getattr(fit, method)()


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


def test_lean_results_predict_new_data_without_fixed_effects(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Lean results keep the formula spec and context needed for prediction."""
    lean_fit = pf.feols("y ~ x + x2", data=lifecycle_data, vcov="iid", lean=True)
    retained_fit = pf.feols("y ~ x + x2", data=lifecycle_data, vcov="iid")

    np.testing.assert_allclose(
        lean_fit.predict(newdata=lifecycle_data),
        retained_fit.predict(newdata=lifecycle_data),
    )


def test_store_data_false_allows_array_only_vcov_updates(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Array-only covariance updates do not require retained formula data."""
    stripped = pf.feols("y ~ x", data=lifecycle_data, vcov="iid", store_data=False)
    expected = pf.feols("y ~ x", data=lifecycle_data, vcov="HC1")

    stripped.vcov("HC1")

    np.testing.assert_allclose(stripped._vcov, expected._vcov)
    assert not hasattr(stripped, "_data")


@pytest.mark.parametrize(
    "expected_fit_kwargs",
    [{}, {"store_data": False}, {"lean": True}],
    ids=["retained", "store_data_false", "lean"],
)
def test_vcov_uses_filtered_model_data_for_non_formula_cluster(
    lifecycle_data: pd.DataFrame,
    expected_fit_kwargs: dict[str, bool],
) -> None:
    """Clustering reads filtered model data before and after storage cleanup."""
    data = lifecycle_data.copy()
    data.loc[data.index[0], "x"] = np.nan
    data.loc[data.index[-1], "fe"] = "singleton"
    data["cluster"] = np.tile(["g1", "g2", "g3"], 8)

    with pytest.warns(UserWarning, match="singleton fixed effect"):
        fit = pf.feols("y ~ x | fe", data=data, vcov="iid")
    with pytest.warns(UserWarning, match="singleton fixed effect"):
        expected = pf.feols(
            "y ~ x | fe",
            data=data,
            vcov={"CRV1": "cluster"},
            **expected_fit_kwargs,
        )

    assert "cluster" not in fit._model_matrix.independent.columns
    assert fit._data.index.equals(fit._model_matrix.dependent.index)
    fit.vcov({"CRV1": "cluster"})

    np.testing.assert_allclose(
        fit._vcov,
        expected._vcov,
        err_msg="CRV1 covariance differs between post-fit update and initial fit",
    )


def test_glm_vcov_uses_post_separation_model_data() -> None:
    """A GLM covariance update uses rows left after singleton and separation filters."""
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 1, 2, 3],
            "fe": ["a", "a", "b", "b", "b", "c"],
            "x": [-1.0, 0.5, 0.25, 1.0, -0.5, 1.5],
            "cluster": ["u", "u", "v", "v", "w", "w"],
        }
    )

    with pytest.warns(UserWarning, match="observations removed because of separation"):
        fit = pf.fepois("y ~ x | fe", data=data, vcov="iid", separation_check=["fe"])
    with pytest.warns(UserWarning, match="observations removed because of separation"):
        expected = pf.fepois(
            "y ~ x | fe",
            data=data,
            vcov={"CRV1": "cluster"},
            separation_check=["fe"],
        )

    assert fit._data.index.equals(fit._model_matrix.dependent.index)
    fit.vcov({"CRV1": "cluster"})

    np.testing.assert_allclose(
        fit._vcov,
        expected._vcov,
        err_msg="Poisson CRV1 covariance differs after separation filtering",
    )


def test_fixest_multi_vcov_uses_each_split_child_sample(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Split and stepwise children cluster on their own retained row samples."""
    data = lifecycle_data.assign(
        sample=np.tile(["left", "right"], 12),
        cluster=np.tile(["g1", "g2", "g3"], 8),
    )
    data.loc[[0, 13], "x2"] = np.nan
    fit = pf.feols("y ~ sw(x, x2) | fe", data=data, split="sample", vcov="iid")
    expected = pf.feols(
        "y ~ sw(x, x2) | fe",
        data=data,
        split="sample",
        vcov={"CRV1": "cluster"},
    )

    fit.vcov({"CRV1": "cluster"})

    assert {child._N_rows for child in fit.to_list()} == {11, 12}
    for child, expected_child in zip(fit.to_list(), expected.to_list(), strict=True):
        assert child._data.index.equals(child._model_matrix.dependent.index)
        np.testing.assert_allclose(
            child._vcov,
            expected_child._vcov,
            err_msg="Split-child CRV1 covariance differs from initial fit",
        )


def test_store_data_false_rejects_data_dependent_vcov(
    lifecycle_data: pd.DataFrame,
) -> None:
    """A stripped fit cannot reconstruct cluster columns from retained arrays."""
    fit = pf.feols("y ~ x", data=lifecycle_data, vcov="iid", store_data=False)

    with pytest.raises(
        RuntimeError,
        match=r"vcov\(\).*store_data=False.*Refit with store_data=True",
    ):
        fit.vcov({"CRV1": "fe"})


def test_rejected_glm_crv3_preserves_inference_state(
    lifecycle_data: pd.DataFrame,
) -> None:
    """An unsupported covariance request leaves every published field unchanged."""
    fit = pf.feglm("y ~ x", data=lifecycle_data, family="gaussian", vcov="iid")
    before = _snapshot_inference(fit)

    with pytest.raises(VcovTypeNotSupportedError, match="CRV3 inference"):
        fit.vcov({"CRV3": "fe"})

    _assert_inference_unchanged(fit, before)


@pytest.mark.parametrize("failure", ["validation", "computation"])
def test_failed_hac_update_preserves_inference_state(
    lifecycle_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """HAC validation and kernel failures cannot publish partial metadata."""
    data = lifecycle_data.assign(time=np.arange(len(lifecycle_data)))
    if failure == "validation":
        data["time"] = pd.Series(["invalid"] * len(data), dtype=object)
    else:

        def fail_hac_computation(**kwargs):
            raise RuntimeError("forced HAC computation failure")

        monkeypatch.setattr(base_regression_module, "vcov_hac", fail_hac_computation)

    fit = pf.feols("y ~ x", data=data, vcov={"CRV1": "fe"})
    before = _snapshot_inference(fit)
    error = ValueError if failure == "validation" else RuntimeError
    message = (
        "time variable must be numeric"
        if failure == "validation"
        else "forced HAC computation failure"
    )

    with pytest.raises(error, match=message):
        fit.vcov("NW", vcov_kwargs={"time_id": "time", "lag": 1})

    _assert_inference_unchanged(fit, before)


def test_fixest_multi_vcov_prepares_every_child_before_publish(
    lifecycle_data: pd.DataFrame,
) -> None:
    """A later child's failure leaves every model in a FixestMulti unchanged."""
    data = lifecycle_data.assign(
        cluster=np.tile(["g1", "g2", "g3"], 8),
        time=np.arange(len(lifecycle_data)),
    )
    data.loc[0, ["x", "cluster"]] = np.nan
    fit = pf.feols(
        "y ~ sw(x, x2)",
        data=data,
        vcov="NW",
        vcov_kwargs={"time_id": "time", "lag": 1},
    )
    children = fit.to_list()
    child_ids = [id(child) for child in children]
    before = [_snapshot_inference(child) for child in children]

    with pytest.raises(NanInClusterVarError, match="missing values"):
        fit.vcov({"CRV1": "cluster"})

    assert [id(child) for child in fit.to_list()] == child_ids
    for child, state in zip(children, before, strict=True):
        _assert_inference_unchanged(child, state)


def test_quantreg_multi_vcov_isolates_later_child_rng_failure(
    lifecycle_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantile staging binds a private solver and advances no RNG on failure."""
    with pytest.warns(FutureWarning, match="experimental"):
        fit = pf.quantreg(
            "y ~ x",
            data=lifecycle_data,
            quantile=[0.25, 0.75],
            method="pfn",
            vcov="iid",
            seed=42,
        )
    children = fit.to_list()
    wrapper = QuantregMulti.__new__(QuantregMulti)
    wrapper.all_quantregs = {child._quantile: child for child in children}
    child_ids = [id(child) for child in children]
    inference_before = [_snapshot_inference(child) for child in children]
    rng_before = [
        deepcopy(child._fit.keywords["rng"].bit_generator.state) for child in children
    ]
    rng_objects = [child._fit.keywords["rng"] for child in children]
    original_vcov_nid = Quantreg._vcov_nid

    def fail_second_child(self: Quantreg) -> np.ndarray:
        if self._quantile == 0.75:
            assert self._fit.func.__self__ is self
            self._fit.keywords["rng"].random()
            raise RuntimeError("forced quantile covariance failure")
        return original_vcov_nid(self)

    monkeypatch.setattr(Quantreg, "_vcov_nid", fail_second_child)

    with pytest.raises(RuntimeError, match="forced quantile covariance failure"):
        wrapper.vcov("nid")

    assert [id(child) for child in wrapper.all_quantregs.values()] == child_ids
    for child, inference, rng_state in zip(
        children, inference_before, rng_before, strict=True
    ):
        _assert_inference_unchanged(child, inference)
        assert child._fit.keywords["rng"].bit_generator.state == rng_state

    def successful_nid(self: Quantreg) -> np.ndarray:
        assert self._fit.func.__self__ is self
        self._fit.keywords["rng"].random()
        return self._vcov.copy()

    monkeypatch.setattr(Quantreg, "_vcov_nid", successful_nid)
    assert wrapper.vcov("nid") is wrapper.all_quantregs
    for child, rng, rng_state in zip(children, rng_objects, rng_before, strict=True):
        assert child._fit.keywords["rng"] is rng
        assert rng.bit_generator.state != rng_state


@pytest.mark.parametrize(
    ("initial_vcov", "vcov_kwargs"),
    [({"CRV1": "fe"}, None), ("NW", {"time_id": "time", "lag": 1})],
    ids=["cluster", "hac"],
)
def test_successful_vcov_update_resets_type_specific_metadata(
    lifecycle_data: pd.DataFrame,
    initial_vcov: str | dict[str, str],
    vcov_kwargs: dict[str, str | int] | None,
) -> None:
    """Publishing IID inference clears metadata owned by prior CRV or HAC state."""
    data = lifecycle_data.assign(time=np.arange(len(lifecycle_data)))
    fit = pf.feols("y ~ x", data=data, vcov=initial_vcov, vcov_kwargs=vcov_kwargs)

    assert fit.vcov("iid") is fit
    assert fit._vcov_type == "iid"
    assert fit._is_clustered is False
    assert fit._clustervar == []
    assert fit._G == []
    assert fit._cluster_df is None
    assert fit._lag is None
    assert fit._time_id is None
    assert fit._panel_id is None


def test_lean_vcov_fails_with_storage_guidance(
    lifecycle_data: pd.DataFrame,
) -> None:
    """Lean results explain that covariance inputs were intentionally discarded."""
    fit = pf.feols("y ~ x", data=lifecycle_data, vcov="iid", lean=True)

    with pytest.raises(RuntimeError, match=r"vcov\(\).*lean=True.*discarded"):
        fit.vcov("HC1")


STORAGE_STATE_FIELDS = frozenset(
    {
        "_data",
        "_model_matrix",
        "_within_data",
        "_observation_weights",
        "_demean_cache",
        "_fe",
        "_response",
        "_X",
        "_Y",
        "_Z",
        "_weights",
        "_scores",
        "_u_hat",
    }
)


@pytest.mark.parametrize(
    ("fit_kwargs", "missing_fields"),
    [
        ({}, frozenset()),
        ({"store_data": False}, frozenset({"_data", "_model_matrix"})),
        ({"lean": True}, STORAGE_STATE_FIELDS),
    ],
    ids=["retained", "store_data_false", "lean"],
)
def test_storage_options_delete_expected_state(
    lifecycle_data: pd.DataFrame,
    fit_kwargs: dict[str, bool],
    missing_fields: frozenset[str],
) -> None:
    """Distinguish data-only cleanup from lean fit-state cleanup."""
    fit = pf.feols("y ~ x | fe", data=lifecycle_data, vcov="iid", **fit_kwargs)

    observed_fields = frozenset(
        field for field in STORAGE_STATE_FIELDS if hasattr(fit, field)
    )
    assert observed_fields == STORAGE_STATE_FIELDS - missing_fields
    if fit_kwargs.get("lean"):
        # Retained so that predict(newdata=...) keeps working without fixed
        # effects, and so that row alignment stays checkable.
        for field in ("_model_spec", "_context", "_na_index"):
            assert hasattr(fit, field)
    assert np.isfinite(fit.coef()).all()


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
    assert not hasattr(first_stage, "_model_matrix")
    assert hasattr(first_stage, "_within_data")
    assert hasattr(first_stage, "_X")
    fit.IV_Diag()
    assert np.isfinite(fit._eff_F)
    with pytest.raises(RuntimeError, match=r"first_stage\(\).*store_data=False"):
        fit.first_stage()


def test_iv_effective_f_uses_retained_arrays_without_stored_data(
    lifecycle_data: pd.DataFrame,
) -> None:
    """IID first-stage diagnostics need retained arrays, not formula data."""
    stripped = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="iid",
        store_data=False,
    )
    retained = pf.feols(
        "y ~ x + [endog ~ z] | fe",
        data=lifecycle_data,
        vcov="iid",
    )

    stripped.IV_Diag()
    retained.IV_Diag()

    np.testing.assert_allclose(stripped._eff_F, retained._eff_F)
    assert not hasattr(stripped._model_1st_stage, "_data")


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
    # `_endogvar` is a read-only view on the discarded within data.
    assert not hasattr(fit, "_endogvar")
    assert np.isfinite(fit._f_stat_1st_stage)
    with pytest.raises(RuntimeError, match=r"IV_Diag\(\).*lean=True"):
        fit.IV_Diag()


def test_glm_lean_discards_formula_and_working_state() -> None:
    """Lean GLM results discard both canonical input and working fit arrays."""
    data = pd.DataFrame({"y": [0, 1, 0, 1, 1, 0], "x": np.arange(6.0)})
    fit = pf.feglm("y ~ x", data=data, family="logit", lean=True, vcov="iid")

    discarded = (
        "_model_matrix",
        "_observation_weights",
        "_demean_cache",
        "_working_state",
        "_u_hat_response",
        "_u_hat_working",
        "_offset",
    )
    assert all(not hasattr(fit, attr) for attr in discarded)
    # The IRLS aliases are read-only views on the discarded working state.
    assert not hasattr(fit, "_irls_weights")
    assert not hasattr(fit, "_Xbeta")
    assert np.isfinite(fit.coef()).all()


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


def test_quantreg_multi_retains_default_post_estimation_state() -> None:
    """Default multi-quantile results support prediction and vcov updates."""
    rng = np.random.default_rng(20260901)
    x = rng.normal(size=200)
    data = pd.DataFrame({"y": 1 + 2 * x + rng.normal(size=200), "x": x})
    with pytest.warns(FutureWarning, match="experimental"):
        multi = pf.quantreg("y ~ x", data=data, quantile=[0.25, 0.75], vcov="iid")
        single = pf.quantreg("y ~ x", data=data, quantile=0.25, vcov="hetero")

    multi.vcov("hetero")

    for model in multi.to_list():
        assert np.isfinite(model.predict()).all()
        assert np.isfinite(model.se()).all()

    first_quantile = multi.fetch_model(0, print_fml=False)
    np.testing.assert_allclose(first_quantile.predict(), single.predict())
    np.testing.assert_allclose(first_quantile.se(), single.se())
