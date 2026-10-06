"""Smoke tests for formulaic internals relied on by pyfixest."""

from types import SimpleNamespace

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import EndogVarsAsCovarsError, InstrumentsAsCovarsError
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.formulaic_compat import (
    FormulaicCompatibilityError,
    filter_multistage_endogenous_terms,
    formula_required_variables,
    get_first_multistage_lhs,
    i_term_columns,
    iter_i_categorical_levels,
    rows_with_unseen_contrast_levels,
    terms_without_intercept,
)
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FixedEffectEncoding,
    fixed_effect_context,
)

FORMULAIC_271 = "https://github.com/matthewwardrop/formulaic/issues/271"


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Y": rng.normal(size=100),
            "X1": rng.normal(size=100),
            "X2": rng.normal(size=100),
            "Z1": rng.normal(size=100),
            "f1": rng.integers(0, 5, size=100),
            "f2": rng.integers(0, 3, size=100),
        }
    )


def test_multistage_iv_parse_structure(data: pd.DataFrame) -> None:
    """IV formulas parse to StructuredFormula with .deps[0].lhs/.rhs."""
    fit = pf.feols("Y ~ X1 + [X2 ~ Z1]", data=data)
    rhs = fit.model.fixest_formula._right_hand_side

    import formulaic.formula

    assert fit.model.is_iv
    assert isinstance(rhs, formulaic.formula.StructuredFormula)
    assert len(rhs.deps) == 1
    assert [str(v) for v in rhs.deps[0].lhs.required_variables] == ["X2"]
    assert "Z1" in {str(v) for v in rhs.deps[0].rhs.required_variables}


@pytest.mark.parametrize(
    "formula, expected",
    [("1", []), ("X1", ["X1"]), ("0 + X1", ["X1"])],
)
def test_terms_without_intercept(formula: str, expected: list[str]) -> None:
    """Formulaic represents the intercept as the term `1`."""
    terms = terms_without_intercept(formulaic.Formula(formula))

    assert [str(term) for term in terms] == expected


@pytest.mark.parametrize(
    "rhs, expected",
    [
        ("`my var`", {"my var"}),
        ("`firm.id`", {"firm.id"}),
        ("I(`my var`)", {"my var"}),
        ("Q('my var')", {"my var"}),
        ("{X1 * X2}", {"X1", "X2"}),
    ],
)
def test_required_variables_preserve_quoted_names(rhs, expected):
    """IV dependency checks preserve literal names and transformed dependencies."""
    parsed = Formula.parse(f"Y ~ {rhs}")[0]
    assert formula_required_variables(parsed.exogenous) == expected


@pytest.mark.parametrize(
    "fml, error",
    [
        (
            "Y ~ I(`my var`) + [Q('my var') ~ Z1]",
            EndogVarsAsCovarsError,
        ),
        (
            "Y ~ Q('my var') + [I(`my var`) ~ Z1]",
            EndogVarsAsCovarsError,
        ),
        (
            "Y ~ I(`my var`) + [X2 ~ Q('my var')]",
            InstrumentsAsCovarsError,
        ),
        (
            "Y ~ Q('my var') + [X2 ~ I(`my var`)]",
            InstrumentsAsCovarsError,
        ),
    ],
)
def test_iv_rejects_quoted_variable_overlap(fml, error):
    """Quoted transformations must not hide a variable used in both IV roles."""
    with pytest.raises(error, match="my var"):
        Formula.parse(fml)


def test_hat_suffix_filtering(data: pd.DataFrame) -> None:
    """The _hat suffix from formulaic MULTISTAGE is filtered from exogenous."""
    fit = pf.feols("Y ~ X1 + [X2 ~ Z1]", data=data)

    exog_vars = {str(v) for v in fit.model.fixest_formula.exogenous.required_variables}

    assert "X1" in exog_vars
    assert "X2" not in exog_vars
    assert "X2_hat" not in exog_vars


def test_hat_suffix_filtering_with_transformed_endogenous(data: pd.DataFrame) -> None:
    """Formulaic names generated terms after the endogenous term, not its variables."""
    fit = pf.feols("Y ~ X1 + [np.exp(X2) ~ Z1]", data=data)

    exog_terms = {str(term) for term in fit.model.fixest_formula.exogenous}

    # `np.exp(X2)` generates `np.exp(X2)_hat`, never `X2_hat`.
    assert exog_terms == {"1", "X1"}
    assert fit.model.fixest_formula.second_stage == formulaic.Formula(
        "Y ~ 1 + X1 + np.exp(X2)"
    )
    assert "np.exp(X2)" in fit.coef().index


def test_transformed_endogenous_matches_precomputed_column(data: pd.DataFrame) -> None:
    """Transforming the endogenous variable inline equals transforming it in the data."""
    precomputed = data.assign(exp_X2=np.exp(data["X2"]))

    inline = pf.feols("Y ~ X1 + [np.exp(X2) ~ Z1]", data=data)
    column = pf.feols("Y ~ X1 + [exp_X2 ~ Z1]", data=precomputed)

    np.testing.assert_allclose(inline.coef().to_numpy(), column.coef().to_numpy())
    np.testing.assert_allclose(inline.se().to_numpy(), column.se().to_numpy())


def test_multisource_endogenous_term_matches_precomputed_column(
    data: pd.DataFrame,
) -> None:
    """One endogenous term may depend on multiple source columns."""
    precomputed = data.assign(X1_plus_X2=data["X1"] + data["X2"])

    inline = pf.feols("Y ~ 1 + [I(X1 + X2) ~ Z1]", data=data)
    column = pf.feols("Y ~ 1 + [X1_plus_X2 ~ Z1]", data=precomputed)

    np.testing.assert_allclose(inline.coef().to_numpy(), column.coef().to_numpy())
    np.testing.assert_allclose(inline.se().to_numpy(), column.se().to_numpy())


def test_multistage_access_guard_raises_loudly() -> None:
    """Malformed formulaic MULTISTAGE shape must fail before silent IV leakage."""
    malformed_rhs = formulaic.Formula("X1")

    with pytest.raises(FormulaicCompatibilityError, match="MULTISTAGE structure"):
        get_first_multistage_lhs(malformed_rhs)


def test_hat_suffix_guard_raises_loudly() -> None:
    """Missing formulaic _hat terms must fail before endogenous leakage."""
    exogenous = SimpleNamespace(root=["1", "X1"])

    with pytest.raises(FormulaicCompatibilityError, match="endogenous suffix"):
        filter_multistage_endogenous_terms(exogenous, ["X2"])


def test_encoder_state_tuple_shape(data: pd.DataFrame) -> None:
    """encoder_state values are (Factor.Kind, state_dict) 2-tuples."""
    fit = pf.feols("Y ~ X1 + C(f1)", data=data)

    from formulaic.parser.types import Factor

    rhs_spec = fit.model.model_spec["second_stage"].rhs
    for value in rhs_spec.encoder_state.values():
        assert isinstance(value, tuple)
        assert len(value) == 2
        kind, state = value
        assert isinstance(kind, Factor.Kind)
        assert isinstance(state, dict)


def test_encoder_state_guard_raises_loudly() -> None:
    """Unexpected encoder_state values must fail before unseen levels are skipped."""
    malformed_spec = SimpleNamespace(
        factor_contrasts={},
        factor_variables={},
        encoder_state={"i(f1)": object()},
    )

    with pytest.raises(FormulaicCompatibilityError, match="encoder_state structure"):
        list(iter_i_categorical_levels(malformed_spec, pd.DataFrame({"f1": [1]})))


@pytest.mark.parametrize(
    ("fml", "expected"),
    [
        ("Y ~ X1 + C(f1)", []),
        ("Y ~ i(f1, ref=1) + X1", ["f1::0", "f1::2", "f1::3", "f1::4"]),
        ("Y ~ X1 + i(f1, X2, ref=1)", ["f1::0:X2", "f1::2:X2", "f1::3:X2", "f1::4:X2"]),
        ("Y ~ i(f1, ref=1):X1 + X1", ["f1::0:X1", "f1::2:X1", "f1::3:X1", "f1::4:X1"]),
    ],
    ids=["no_i", "i", "i_continuous", "i_in_interaction"],
)
def test_i_term_columns(data: pd.DataFrame, fml: str, expected: list[str]) -> None:
    """i_term_columns returns exactly the columns produced by i() terms."""
    fit = pf.feols(fml, data=data)

    columns = i_term_columns(fit.model.model_spec["second_stage"].rhs)

    assert columns == expected
    assert set(columns).issubset(fit._coefnames)


def test_i_term_columns_ignores_double_colon_names(data: pd.DataFrame) -> None:
    """A user column named with '::' is not an i() term."""
    renamed = data.rename(columns={"X1": "a::b"})
    fit = pf.feols("Y ~ Q('a::b') + X2", data=renamed)

    assert "Q('a::b')" in fit._coefnames
    assert i_term_columns(fit.model.model_spec["second_stage"].rhs) == []


def test_i_term_columns_survives_lean(data: pd.DataFrame) -> None:
    """The i() registry lives on the retained model spec, not the model matrix."""
    fit = pf.feols("Y ~ i(f1, ref=1)", data=data, lean=True, store_data=False)

    assert i_term_columns(fit.model.model_spec["second_stage"].rhs) == [
        "f1::0",
        "f1::2",
        "f1::3",
        "f1::4",
    ]


def test_contrasts_state_key_format(data: pd.DataFrame) -> None:
    """i() stores contrast state under __contrasts_<var>__."""
    fit = pf.feols("Y ~ X1 + i(f1, X2)", data=data)

    rhs_spec = fit.model.model_spec["second_stage"].rhs
    i_state = None
    for factor_expr, value in rhs_spec.encoder_state.items():
        if factor_expr.startswith("i("):
            _kind, state = value
            i_state = state
            break

    assert i_state is not None
    assert any(k.startswith("__contrasts_") and k.endswith("__") for k in i_state)


def test_fe_transform_state_has_encoding(data: pd.DataFrame) -> None:
    """FE transform_state stores the fitted level indexes and combinations."""
    fit = pf.feols("Y ~ X1 | f1", data=data)

    fe_spec = fit.model.model_spec["fe"]
    (factor,) = fe_spec.formula[0].factors
    assert factor.metadata["term"].factors[0].expr == "f1"
    assert factor.expr.removeprefix("__fixed_effect__(").removesuffix(")").isdigit()
    assert fe_spec.required_variables == {"f1"}
    fe_state = fe_spec.transform_state[factor.expr]
    encoding = fe_state["__fixed_effect_encoding__"]
    assert isinstance(encoding, FixedEffectEncoding)
    assert encoding.levels[0].tolist() == sorted(data.f1.unique())


@pytest.mark.parametrize(
    "name", ["my fe", "firm.id", "a:b", 'fe"quote', "fe\\backslash"]
)
@pytest.mark.parametrize("expression", ["`{name}`", "C(`{name}`)"])
def test_fe_metadata_preserves_lookup_and_dependencies(data, name, expression):
    """FE lookup names survive materialization, recovery, and prediction."""
    renamed = data.rename(columns={"f1": name})
    renamed.loc[0, name] = np.nan
    # Stateful Python expressions containing quotes/backslashes remain limited
    # by Formulaic's own state-key escaping; plain LOOKUP factors bypass it.
    fit = pf.feols(f"Y ~ X1 | {expression.format(name=name)}:f2", data=renamed)
    spec = fit.model.model_spec["fe"]
    assert {name, "f2"}.issubset(spec.variables)
    # Formulaic normalizes dotted names when deriving required_variables;
    # the original dependency remains in variables, as in the existing hook.
    if "." not in name:
        assert spec.required_variables == {name, "f2"}
    baseline = pf.feols("Y ~ X1 | f1:f2", data=renamed.rename(columns={name: "f1"}))
    np.testing.assert_allclose(
        fit.coef(), baseline.coef(), rtol=0, atol=1e-12, err_msg="FE coefficients"
    )
    prediction = fit.predict(newdata=renamed.iloc[:10])
    assert np.isnan(prediction[0])
    np.testing.assert_allclose(
        prediction[1:],
        baseline.predict(newdata=renamed.rename(columns={name: "f1"}).iloc[:10])[1:],
        rtol=0,
        atol=1e-10,
        err_msg="FE predictions",
    )


@pytest.mark.parametrize(
    "expression", ["Q('my fe')", "C(Q('my fe'))", "I(center(Q('my fe')))"]
)
def test_fe_nested_transforms_retain_state_and_dependencies(data, expression):
    """Original Python factors retain context, dependencies, and learned state."""
    renamed = data.rename(columns={"f1": "my fe"})
    fit = pf.feols(f"Y ~ X1 | {expression}", data=renamed)
    spec = fit.model.model_spec["fe"]
    assert spec.required_variables == {"my fe"}
    newdata = renamed.iloc[:10]
    matrix = spec.get_model_matrix(
        newdata,
        context=fixed_effect_context(
            terms=spec.formula, data=newdata, context=FORMULAIC_TRANSFORMS
        ),
    )
    np.testing.assert_array_equal(
        matrix.to_numpy(),
        fit.model_matrix.fixed_effects.iloc[:10].to_numpy(),
        err_msg="persistent FE transform state",
    )


@pytest.mark.parametrize("arity", [1, 2, 3])
@pytest.mark.parametrize("kind", ["numeric", "string", "categorical", "mixed"])
def test_fe_index_codes_preserve_groupby_order(data, arity, kind):
    """Level indexes preserve fitted codes, including categorical reference order."""
    frame = data.copy()
    frame["f3"] = frame.f1 % 2
    if kind == "string":
        frame["f1"] = frame.f1.astype(str)
    elif kind == "categorical":
        frame["f1"] = pd.Categorical(
            frame.f1, categories=[4, 2, 0, 3, 1, 99], ordered=True
        )
    elif kind == "mixed":
        frame["f1"] = frame.f1.astype(object).where(frame.f1 < 3, "a")
    frame.loc[0, "f1"] = np.nan
    names = ["f1", "f2", "f3"][:arity]
    expected = frame.groupby(names).ngroup().dropna()
    fit = pf.feols(f"Y ~ X1 | {':'.join(names)}", data=frame, fixef_rm="none")
    np.testing.assert_array_equal(
        fit.model_matrix.fixed_effects.iloc[:, 0],
        expected,
        err_msg="FE group codes and reference order",
    )


@pytest.mark.parametrize(
    "kind", ["categorical", "bool_to_numeric", "numeric_to_bool", "unmatched_string"]
)
def test_fe_prediction_matches_values_across_dtypes(data, kind):
    """Numeric categoricals match integers; bool/numeric coercion is disallowed."""
    frame = data.copy()
    frame["f1"] = frame.f1 % 2
    if kind == "categorical":
        frame["f1"] = frame.f1.astype("category")
    elif kind == "bool_to_numeric":
        frame["f1"] = frame.f1.astype(bool)
    fit = pf.feols("Y ~ X1 | f1", data=frame)
    newdata = frame.iloc[:10].copy()
    if kind == "unmatched_string":
        newdata["f1"] = newdata.f1.astype(str)
        with pytest.warns(UserWarning, match="unseen level"):
            assert np.isnan(fit.predict(newdata=newdata)).all()
        return
    newdata["f1"] = newdata.f1.astype(bool if kind == "numeric_to_bool" else int)
    if kind in {"bool_to_numeric", "numeric_to_bool"}:
        with pytest.warns(UserWarning, match="unseen level"):
            assert np.isnan(fit.predict(newdata=newdata)).all()
        return
    np.testing.assert_allclose(
        fit.predict(newdata=newdata),
        fit.predict()[:10],
        rtol=0,
        atol=1e-10,
        err_msg="FE predictions across compatible dtypes",
    )


@pytest.mark.parametrize("expression", ["I(f1 * 10)", "f1:{f1 // 2}", "I(center(f1))"])
def test_fe_expression_prediction_and_labels(data, expression):
    """Expression FEs use parsed labels and evaluated values in every consumer."""
    fit = pf.feols(f"Y ~ X1 | {expression}", data=data)
    np.testing.assert_allclose(
        fit.predict(newdata=data.iloc[:10]),
        fit.predict()[:10],
        rtol=0,
        atol=1e-10,
        err_msg="expression FE predictions",
    )
    expected_label = ":".join(
        str(factor) for factor in fit.model.fixest_formula.fixed_effects[0].factors
    )
    assert set(fit.fixef().variable) == {expected_label}
    newdata = data.iloc[:10].copy()
    newdata.loc[0, "f1"] = 999
    newdata.loc[1, "f1"] = np.nan
    with pytest.warns(UserWarning, match="1 unseen level"):
        prediction = fit.predict(newdata=newdata)
    assert np.isnan(prediction[:2]).all()
    assert np.isfinite(prediction[2:]).all()


def test_fe_unseen_combination_and_factor_row_alignment(data):
    """Match factors by row index and reject new pairs of individually seen levels."""
    frame = data.copy()
    frame["f2"] = frame.f1 % 2

    def reverse_rows(values):
        return values.iloc[::-1]

    fit = pf.feols(
        "Y ~ X1 | f1:reverse_rows(f2)",
        data=frame,
        context={"reverse_rows": reverse_rows},
    )
    baseline = pf.feols("Y ~ X1 | f1:f2", data=frame)
    np.testing.assert_allclose(
        fit.coef(),
        baseline.coef(),
        rtol=0,
        atol=1e-12,
        err_msg="row-aligned FE coefficients",
    )
    newdata = frame.iloc[:10].copy()
    newdata.loc[0, "f2"] = 1 - newdata.loc[0, "f2"]
    with pytest.warns(UserWarning, match="1 unseen level"):
        prediction = fit.predict(newdata=newdata)
    assert np.isnan(prediction[0])
    np.testing.assert_allclose(
        prediction[1:],
        baseline.predict(newdata=frame.iloc[1:10]),
        rtol=0,
        atol=1e-10,
        err_msg="seen FE combinations",
    )


def test_materializer_cache_contains_evaluated_factor_values(
    data: pd.DataFrame,
) -> None:
    """The materializer cache stores evaluated rather than source values."""
    fit = pf.feols("Y ~ C(np.floor(X2))", data=data)
    rhs_spec = fit.model.model_spec["second_stage"].rhs
    context = FORMULAIC_TRANSFORMS | {**fit.options.context}

    materializer = rhs_spec.get_materializer(data, context=context)
    materializer.get_model_matrix(rhs_spec)
    factor, contrast_state = next(iter(rhs_spec.factor_contrasts.items()))
    evaluated = materializer.factor_cache[factor.expr]

    assert factor.expr == "C(np.floor(X2))"
    np.testing.assert_array_equal(np.asarray(evaluated.values), np.floor(data["X2"]))
    assert set(contrast_state.levels) == set(np.floor(data["X2"]))


def test_evaluated_factor_cache_guard_raises_loudly(data: pd.DataFrame) -> None:
    """A missing evaluated factor must fail before unseen levels are skipped."""
    fit = pf.feols("Y ~ C(np.floor(X2))", data=data)
    rhs_spec = fit.model.model_spec["second_stage"].rhs

    with pytest.raises(FormulaicCompatibilityError, match="evaluated factor"):
        rows_with_unseen_contrast_levels(rhs_spec, data, {})


@pytest.mark.xfail(
    strict=True,
    reason=f"Formulaic issue #271 remains unresolved: {FORMULAIC_271}",
)
def test_formulaic_271_unseen_levels_respect_na_action() -> None:
    """Unseen levels must not become all-zero rows after `na_action` runs."""
    train = pd.DataFrame({"y": [1, 2, 3], "x": ["a", "b", "a"]})
    newdata = pd.DataFrame({"x": ["a", "z", "b"]})
    rhs_spec = formulaic.Formula("y ~ C(x)").get_model_matrix(train).model_spec.rhs

    with pytest.raises(ValueError):
        rhs_spec.get_model_matrix(newdata, na_action="raise")


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1 + i(f1)",
        "Y ~ X1 + C(f1)",
        "Y ~ X1 | f1",
        "Y ~ X1 | f1:f2",
        # Categorical factors whose levels are *evaluated* rather than read off
        # a column: ModelSpec.factor_variables reports X2, not floor(X2).
        "Y ~ X1 + C(np.floor(X2))",
        "Y ~ X1 + C(np.floor(center(X2)))",
        "Y ~ C(np.floor(X2)):X1",
        "Y ~ X1 + C(f1 + f2)",
    ],
)
def test_model_spec_get_model_matrix_prediction_roundtrip(
    data: pd.DataFrame, fml: str
) -> None:
    """Stored ModelSpec round-trips: seen rows predict, and match in-sample fits."""
    fit = pf.feols(fml, data=data)
    pred = fit.predict(newdata=data.iloc[:20])

    assert pred.shape[0] == 20
    assert np.all(np.isfinite(pred))
    # atol covers the lsqr tolerance used to recover fixed effects.
    np.testing.assert_allclose(pred, fit.predict()[:20], atol=1e-4)


def test_unseen_level_of_transformed_categorical_is_nan(data: pd.DataFrame) -> None:
    """Only rows whose *evaluated* level is unseen are dropped to NaN."""
    fit = pf.feols("Y ~ C(np.floor(X2))", data=data)

    newdata = data.iloc[:20].copy()
    newdata.loc[newdata.index[0], "X2"] = 1e6
    pred = fit.predict(newdata=newdata)

    assert np.isnan(pred[0])
    assert np.all(np.isfinite(pred[1:]))
