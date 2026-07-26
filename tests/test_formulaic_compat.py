"""Smoke tests for formulaic internals relied on by pyfixest."""

from types import SimpleNamespace

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.formulaic_compat import (
    FormulaicCompatibilityError,
    filter_multistage_endogenous_terms,
    get_first_multistage_lhs,
    iter_model_spec_categorical_levels,
)

# Known-broken behaviour, fixed by the follow-up PRs in this series. `strict`
# means the marker has to be removed together with the fix.
XFAIL_TRANSFORMED_ENDOGENOUS = (
    "the generated `_hat` term is named after the endogenous variable rather "
    "than the endogenous term, so `np.exp(X2)` looks for `X2_hat`"
)
XFAIL_TRANSFORMED_CATEGORICAL = (
    "unseen categorical levels are looked up in the factor's source columns, "
    "so raw `X2` is compared against the levels of `floor(X2)`"
)


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
    rhs = fit.FixestFormula._right_hand_side

    import formulaic.formula

    assert fit._is_iv
    assert isinstance(rhs, formulaic.formula.StructuredFormula)
    assert len(rhs.deps) == 1
    assert [str(v) for v in rhs.deps[0].lhs.required_variables] == ["X2"]
    assert "Z1" in {str(v) for v in rhs.deps[0].rhs.required_variables}


def test_hat_suffix_filtering(data: pd.DataFrame) -> None:
    """The _hat suffix from formulaic MULTISTAGE is filtered from exogenous."""
    fit = pf.feols("Y ~ X1 + [X2 ~ Z1]", data=data)

    exog_vars = {str(v) for v in fit.FixestFormula.exogenous.required_variables}

    assert "X1" in exog_vars
    assert "X2" not in exog_vars
    assert "X2_hat" not in exog_vars


@pytest.mark.xfail(strict=True, reason=XFAIL_TRANSFORMED_ENDOGENOUS)
def test_hat_suffix_filtering_with_transformed_endogenous(data: pd.DataFrame) -> None:
    """Formulaic names generated terms after the endogenous term, not its variables."""
    fit = pf.feols("Y ~ X1 + [np.exp(X2) ~ Z1]", data=data)

    exog_terms = {str(term) for term in fit.FixestFormula.exogenous}

    # `np.exp(X2)` generates `np.exp(X2)_hat`, never `X2_hat`.
    assert exog_terms == {"1", "X1"}
    assert fit.FixestFormula.second_stage == "Y ~ 1 + X1 + np.exp(X2)"
    assert "np.exp(X2)" in fit.coef().index


@pytest.mark.xfail(strict=True, reason=XFAIL_TRANSFORMED_ENDOGENOUS)
def test_transformed_endogenous_matches_precomputed_column(data: pd.DataFrame) -> None:
    """Transforming the endogenous variable inline equals transforming it in the data."""
    precomputed = data.assign(exp_X2=np.exp(data["X2"]))

    inline = pf.feols("Y ~ X1 + [np.exp(X2) ~ Z1]", data=data)
    column = pf.feols("Y ~ X1 + [exp_X2 ~ Z1]", data=precomputed)

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

    rhs_spec = fit._model_spec["second_stage"].rhs
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
        list(
            iter_model_spec_categorical_levels(
                malformed_spec, pd.DataFrame({"f1": [1]})
            )
        )


def test_contrasts_state_key_format(data: pd.DataFrame) -> None:
    """i() stores contrast state under __contrasts_<var>__."""
    fit = pf.feols("Y ~ X1 + i(f1, X2)", data=data)

    rhs_spec = fit._model_spec["second_stage"].rhs
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

    fe_spec = fit._model_spec["fe"]
    fe_state = fe_spec.transform_state["__fixed_effect__(f1)"]
    enc_df = fe_state["__fixed_effect_encoding__"]

    assert isinstance(enc_df, pd.DataFrame)
    assert "__fixed_effect_encoding__" in enc_df.columns


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1 + i(f1)",
        "Y ~ X1 + C(f1)",
        "Y ~ X1 | f1",
        "Y ~ X1 | f1^f2",
        # Categorical factors whose levels are *evaluated* rather than read off
        # a column: ModelSpec.factor_variables reports X2, not floor(X2).
        pytest.param(
            "Y ~ X1 + C(np.floor(X2))",
            marks=pytest.mark.xfail(strict=True, reason=XFAIL_TRANSFORMED_CATEGORICAL),
        ),
        pytest.param(
            "Y ~ C(np.floor(X2)):X1",
            marks=pytest.mark.xfail(strict=True, reason=XFAIL_TRANSFORMED_CATEGORICAL),
        ),
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


@pytest.mark.xfail(strict=True, reason=XFAIL_TRANSFORMED_CATEGORICAL)
def test_unseen_level_of_transformed_categorical_is_nan(data: pd.DataFrame) -> None:
    """Only rows whose *evaluated* level is unseen are dropped to NaN."""
    fit = pf.feols("Y ~ C(np.floor(X2))", data=data)

    newdata = data.iloc[:20].copy()
    newdata.loc[newdata.index[0], "X2"] = 1e6
    pred = fit.predict(newdata=newdata)

    assert np.isnan(pred[0])
    assert np.all(np.isfinite(pred[1:]))
