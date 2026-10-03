"""Smoke tests for formulaic internals relied on by pyfixest."""

from __future__ import annotations

from types import SimpleNamespace

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.formulaic_compat import (
    FormulaicCompatibilityError,
    filter_multistage_endogenous_terms,
    get_first_multistage_lhs,
    iter_i_categorical_levels,
    rows_with_unseen_contrast_levels,
    terms_without_intercept,
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
    assert fit.model.fixest_formula.second_stage == "Y ~ 1 + X1 + np.exp(X2)"
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
    """FE transform_state stores __fixed_effect_encoding__ DataFrame."""
    fit = pf.feols("Y ~ X1 | f1", data=data)

    fe_spec = fit.model.model_spec["fe"]
    fe_state = fe_spec.transform_state["__fixed_effect__(f1)"]
    enc_df = fe_state["__fixed_effect_encoding__"]

    assert isinstance(enc_df, pd.DataFrame)
    assert "__fixed_effect_encoding__" in enc_df.columns


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


@pytest.mark.parametrize("dtype", [None, "category", "Int64", "string"])
def test_fe_encoding_cache_samples_and_prediction(data, monkeypatch, dtype):
    """Reuse full-input codes, but isolate samples and prediction mappings."""
    from functools import wraps

    from pyfixest.estimation.formula.fe_encoding_cache import FixedEffectEncodingCache
    from pyfixest.estimation.formula.model_matrix import create_model_matrix
    from pyfixest.estimation.formula.parse import Formula
    from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
        FIXED_EFFECT_ENCODING,
    )

    data.loc[0, "f1"] = 999  # singleton, even when another model loses rows
    data.loc[1, "f2"] = np.nan
    data.loc[2:5, "X2"] = np.nan
    if dtype:
        data["f1"] = data.f1.astype(dtype)
        data["f2"] = data.f2.astype(dtype)
    original = FORMULAIC_TRANSFORMS["__fixed_effect__"]
    calls = []

    @wraps(original)
    def counted(*args, **kwargs):
        calls.append(tuple(arg.name for arg in args))
        return original(*args, **kwargs)

    monkeypatch.setitem(FORMULAIC_TRANSFORMS, "__fixed_effect__", counted)
    cache = FixedEffectEncodingCache()
    matrices = []
    for formula in Formula.parse("Y ~ csw(X1, X2) | f1 + f1:f2"):
        matrices.append(
            create_model_matrix(
                formula=formula,
                data=data.copy(),
                drop_singletons=True,
                fixed_effect_encoding_cache=cache,
            )
        )
    assert len(calls) == 2  # Two FE terms, each encoded only once.
    assert matrices[0].n_rows > matrices[1].n_rows
    for matrix, formula in zip(
        matrices, Formula.parse("Y ~ csw(X1, X2) | f1 + f1:f2"), strict=True
    ):
        uncached = create_model_matrix(
            formula=formula, data=data.copy(), drop_singletons=True
        )
        pd.testing.assert_frame_equal(matrix.fixed_effects, uncached.fixed_effects)
        pd.testing.assert_frame_equal(matrix.independent, uncached.independent)
        assert matrix.dropped_row_index == uncached.dropped_row_index

    # Materialize retained FE specs on reordered rows and an unseen level.
    newdata = data.iloc[[8, 7, 6]].copy()
    if dtype == "category":
        newdata["f1"] = newdata.f1.cat.add_categories([1001])
    newdata.iloc[0, newdata.columns.get_loc("f1")] = (
        "1001" if dtype == "string" else 1001
    )
    first_spec = matrices[0].model_spec["fe"]
    second_spec = matrices[1].model_spec["fe"]
    first_prediction = first_spec.get_model_matrix(
        newdata, context=FORMULAIC_TRANSFORMS
    )
    second_prediction = second_spec.get_model_matrix(
        newdata, context=FORMULAIC_TRANSFORMS
    )
    pd.testing.assert_frame_equal(first_prediction, second_prediction)
    saved_cache = {
        key: entry[1][FIXED_EFFECT_ENCODING].copy(deep=True)
        for key, entry in cache._entries.items()
    }
    for key in first_spec.transform_state:
        state_a = first_spec.transform_state[key][FIXED_EFFECT_ENCODING]
        state_b = second_spec.transform_state[key][FIXED_EFFECT_ENCODING]
        assert state_a is not state_b
        saved = state_b.copy(deep=True)
        state_a.iloc[0, -1] = -100
        pd.testing.assert_frame_equal(state_b, saved)
    for key, saved in saved_cache.items():
        pd.testing.assert_frame_equal(
            cache._entries[key][1][FIXED_EFFECT_ENCODING], saved
        )


def test_fe_encoding_cache_expression_fallback_and_lifetime(data):
    """Evaluated arrays must not collide with a direct column's cache key."""
    import gc
    import weakref

    from pyfixest.estimation.formula.fe_encoding_cache import FixedEffectEncodingCache

    original = FORMULAIC_TRANSFORMS["__fixed_effect__"]
    cache = FixedEffectEncodingCache()
    reference = weakref.ref(cache)
    with cache.transform(data=data, original=original) as transform:
        transform(data.f1, _state={})
        assert len(cache._entries) == 1
        for value in (
            data.f1 + 1,
            data.f1.iloc[::-1].reset_index(drop=True),
            data.f1.astype("Int64"),
        ):
            actual = transform(value, _state={})
            expected = original(value, _state={})
            pd.testing.assert_series_equal(actual, expected)
        assert len(cache._entries) == 1
    del cache
    gc.collect()
    assert reference() is None
    pd.testing.assert_series_equal(
        transform(data.f1 + 2, _state={}), original(data.f1 + 2, _state={})
    )


def test_fe_encoding_cache_respects_context_override(data):
    """A user transform is evaluated independently, even for direct FE columns."""
    from functools import wraps

    from pyfixest.estimation.formula.fe_encoding_cache import FixedEffectEncodingCache
    from pyfixest.estimation.formula.model_matrix import create_model_matrix
    from pyfixest.estimation.formula.parse import Formula

    original = FORMULAIC_TRANSFORMS["__fixed_effect__"]
    calls = []

    @wraps(original)
    def custom(*args, **kwargs):
        calls.append(args[0].name)
        return original(*args, **kwargs)

    cache = FixedEffectEncodingCache()
    for formula in Formula.parse("Y ~ csw(X1, X2) | f1"):
        create_model_matrix(
            formula=formula,
            data=data.copy(),
            context={"__fixed_effect__": custom},
            fixed_effect_encoding_cache=cache,
        )
    assert calls == ["f1", "f1"]
    assert cache._entries == {}
