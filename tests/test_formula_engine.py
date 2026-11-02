"""Public OLS/IV contracts for the Rust formula engine."""

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.engine import LinearModelMatrix


@pytest.fixture
def data():
    rng = np.random.default_rng(173)
    n = 120
    x, z, u = rng.normal(size=(3, n))
    d = 0.8 * z + 0.3 * x + u
    group = np.tile(np.arange(12), 10)
    y = 1.2 + 2 * x - 0.7 * d + group / 4 + u + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, x=x, z=z, d=d, group=group, w=rng.uniform(0.5, 2, n)))


@pytest.mark.parametrize("fixed_effects", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_ols_matches_explicit_design(data, fixed_effects, weighted):
    design = np.column_stack([np.ones(len(data)), data.x, data.d])
    if fixed_effects:
        design = np.column_stack([design, pd.get_dummies(data.group, drop_first=True)])
    weights = data.w.to_numpy() if weighted else np.ones(len(data))
    expected = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), data.y * np.sqrt(weights), rcond=None
    )[0]
    fit = pf.feols(
        "y ~ x + d" + (" | group" if fixed_effects else ""),
        data,
        weights="w" if weighted else None,
    )
    assert isinstance(fit.model_matrix, LinearModelMatrix)
    assert fit._model_name_plot == fit.model.model_name
    np.testing.assert_allclose(
        fit.coef().loc[["x", "d"]],
        expected[1:3],
        atol=1e-10,
        err_msg="OLS coefficients",
    )
    # Tighten FE recovery's LSQR stopping tolerance for the dense-solve comparison.
    np.testing.assert_allclose(
        fit.predict(data.iloc[:8], atol=1e-12, btol=1e-12),
        (design @ expected)[:8],
        atol=1e-8,
        err_msg="OLS predictions",
    )


@pytest.mark.parametrize("syntax", ["y ~ x + [d ~ z]", "y ~ x | d ~ z"])
@pytest.mark.parametrize("fixed_effects", [False, True])
def test_iv_matches_explicit_2sls(data, syntax, fixed_effects):
    if fixed_effects:
        syntax = (
            syntax.replace(" | d", " | group | d")
            if " | d" in syntax
            else syntax + " | group"
        )
    X = np.column_stack([np.ones(len(data)), data.x, data.d])
    Z = np.column_stack([np.ones(len(data)), data.x, data.z])
    if fixed_effects:
        dummies = pd.get_dummies(data.group, drop_first=True).to_numpy()
        X, Z = np.column_stack([X, dummies]), np.column_stack([Z, dummies])
    projected = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    expected = np.linalg.solve(projected.T @ X, projected.T @ data.y)
    fit = pf.feols(syntax, data)
    np.testing.assert_allclose(
        fit.coef().loc[["x", "d"]],
        expected[1:3],
        atol=1e-10,
        err_msg="2SLS coefficients",
    )
    expected_first = np.linalg.lstsq(Z, data.d, rcond=None)[0]
    np.testing.assert_allclose(
        fit.first_stage.model.coef().loc[["x", "z"]],
        expected_first[1:3],
        atol=1e-10,
        err_msg="published first-stage coefficients",
    )


def test_joint_missing_sample_and_index(data):
    data.index = np.arange(len(data)) * 3 + 11
    data.iloc[0, data.columns.get_loc("z")] = np.nan
    data.iloc[1, data.columns.get_loc("w")] = np.nan
    data.iloc[2, data.columns.get_loc("y")] = np.nan
    fit = pf.feols("y ~ x + [d ~ z] | group", data, weights="w")
    reference = pf.feols("y ~ x + [d ~ z] | group", data.iloc[3:], weights="w")
    assert fit.model_matrix.dropped_row_index == frozenset([0, 1, 2])
    np.testing.assert_allclose(
        fit.coef(), reference.coef(), atol=1e-11, err_msg="joint IV estimation sample"
    )
    assert data.index[0] == 11


def test_redundant_endogenous_term_does_not_enter_instruments(data):
    fit = pf.feols("y ~ x + d + [d ~ z]", data)
    expected = pf.feols("y ~ x + [d ~ z]", data)
    assert "d" not in fit.model_matrix.instruments.columns
    np.testing.assert_allclose(
        fit.coef(), expected.coef(), atol=1e-11, err_msg="redundant endogenous term"
    )


@pytest.mark.parametrize(
    "source", ["y ~ x + [d ~ z + z2 - 1]", "y ~ x - 1 + [d ~ 1 + z + z2]"]
)
def test_iv_stage_intercepts_survive_normalization(data, source):
    data["z2"] = np.random.default_rng(432).normal(size=len(data))
    fit = pf.feols(source, data)
    replay = pf.feols(fit.model.formula, data)
    np.testing.assert_allclose(
        fit.coef(), replay.coef(), atol=1e-10, err_msg="IV formula replay"
    )
    pd.testing.assert_frame_equal(
        fit.first_stage.model.model_matrix.independent,
        fit.model_matrix.instruments,
    )


def test_prediction_reuses_polynomial_state(data):
    fit = pf.feols("y ~ poly(x, degree=2)", data)
    np.testing.assert_allclose(
        fit.predict(data.iloc[[2, 9, 18]]),
        fit.predict()[[2, 9, 18]],
        atol=1e-11,
        err_msg="fitted polynomial prediction state",
    )


def test_unseen_categories_and_fixed_effects(data):
    fit = pf.feols("y ~ x + C(group, reference=0)", data)
    new = data.iloc[:3].copy()
    new.loc[new.index[1], "group"] = 999
    assert np.isnan(fit.predict(new)[1])
    fe = pf.feols("y ~ x | group", data)
    assert np.isnan(fe.predict(new)[1])


def test_stepwise(data):
    fits = pf.feols("y ~ sw(x, d)", data)
    assert len(fits.to_list()) == 2


@pytest.mark.parametrize("source", ["y + y2 ~ x | d ~ z", "sw(y, y2) ~ x + [d ~ z]"])
def test_multiple_responses_with_iv(data, source):
    data["y2"] = 2 * data.y
    fits = pf.feols(source, data).to_list()
    assert len(fits) == 2
    np.testing.assert_allclose(
        fits[1].coef(), 2 * fits[0].coef(), atol=1e-10, err_msg="multiple IV outcomes"
    )


def test_reference_only_category_preserves_validity(data):
    data["category"] = 1.0
    data.loc[0, "category"] = np.nan
    fit = pf.feols("y ~ x + C(category, reference=1.0)", data)
    assert fit.model_matrix.dropped_row_index == frozenset([0])
    new = data.iloc[1:4].copy()
    new["category"] = [1.0, 2.0, np.nan]
    prediction = fit.predict(new)
    assert np.isfinite(prediction[0])
    assert np.isnan(prediction[1:]).all()


@pytest.mark.parametrize(
    "source", ["C(group, reference=0) ~ x", "y ~ x + [C(group, reference=0) ~ z]"]
)
def test_categorical_response_is_rejected(data, source):
    with pytest.raises(TypeError, match="must be numeric"):
        pf.feols(source, data)


def test_iv_before_fixed_effects_is_rejected(data):
    from pyfixest.errors import FormulaSyntaxError

    with pytest.raises(FormulaSyntaxError, match="final pipe part"):
        pf.feols("y ~ x | d ~ z | group", data)


def test_boolean_and_zero_intercept(data):
    data["treated"] = data.x > 0
    fit = pf.feols("y ~ 0 + treated + x", data)
    design = data[["treated", "x"]].to_numpy(dtype=float)
    expected = np.linalg.lstsq(design, data.y, rcond=None)[0]
    np.testing.assert_allclose(
        fit.coef(), expected, atol=1e-11, err_msg="Boolean regressor without intercept"
    )
    np.testing.assert_allclose(
        fit.predict(data.iloc[:5]),
        (design @ expected)[:5],
        atol=1e-11,
        err_msg="Boolean prediction",
    )


def test_quoted_columns_and_missing_numeric_categories(data):
    data["category"] = pd.Categorical(data.group)
    data.loc[0, "category"] = np.nan
    data = data.rename(columns={"x": "x + other", "group": "g | group"})
    fit = pf.feols("y ~ `x + other` + C(category, reference=0) | `g | group`", data)
    assert fit.model_matrix.dropped_row_index == frozenset([0])
    assert len(fit.predict(data.iloc[1:4])) == 3


@pytest.mark.parametrize(
    "source", ["y ~ i(group)", "y ~ x | group[x]", "y ~ x | group[[x]]"]
)
def test_unsupported_syntax_is_explicit(data, source):
    with pytest.raises(NotImplementedError, match="not supported"):
        pf.feols(source, data)


@pytest.mark.parametrize("context", [0, {}, {"custom": np.log}])
def test_context_is_rejected(data, context):
    with pytest.raises(NotImplementedError, match="does not support context"):
        pf.feols("y ~ x", data, context=context)
