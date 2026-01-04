"""
Tests for the new formula parsing implementation in pyfixest/estimation/formula/parse.py.

This module contains:
- Part 1: Unit tests for internal parsing functions
- Part 2: End-to-end compatibility tests via feols()
- Part 3: Edge case tests
"""

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula.parse import (
    Formula,
    _MultipleEstimation,
    _MultipleEstimationType,
    _parse_multiple_estimation,
    parse,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_data():
    """Generate test data for compatibility tests."""
    return pf.get_data(N=500, seed=12345)


# =============================================================================
# Part 1: Unit Tests for formula/parse.py
# =============================================================================


class TestParseMultipleEstimation:
    """Tests for _parse_multiple_estimation() function."""

    @pytest.mark.parametrize(
        "variables,expected_constant,expected_variable,expected_kind",
        [
            # Basic cases (no multiple estimation)
            (["a", "b", "c"], ["a", "b", "c"], [], None),
            (["X1"], ["X1"], [], None),
            # sw() cases
            (["sw(x,y)"], [], ["x", "y"], _MultipleEstimationType.sw),
            (["a", "sw(x,y)", "d"], ["a", "d"], ["x", "y"], _MultipleEstimationType.sw),
            (["sw(a,b,c)"], [], ["a", "b", "c"], _MultipleEstimationType.sw),
            # csw() cases
            (["csw(x,y)"], [], ["x", "y"], _MultipleEstimationType.csw),
            (
                ["a", "b", "csw(x,y,z)"],
                ["a", "b"],
                ["x", "y", "z"],
                _MultipleEstimationType.csw,
            ),
            # sw0() cases
            (["sw0(x,y)"], [], ["x", "y"], _MultipleEstimationType.sw0),
            (["a", "sw0(x,y)"], ["a"], ["x", "y"], _MultipleEstimationType.sw0),
            # csw0() cases
            (["csw0(x,y,z)"], [], ["x", "y", "z"], _MultipleEstimationType.csw0),
            (
                ["a", "b", "csw0(x,y,z)"],
                ["a", "b"],
                ["x", "y", "z"],
                _MultipleEstimationType.csw0,
            ),
        ],
    )
    def test_parse_multiple_estimation(
        self, variables, expected_constant, expected_variable, expected_kind
    ):
        """Test parsing of multiple estimation syntax."""
        result = _parse_multiple_estimation(variables)

        assert result.constant == expected_constant
        assert result.variable == expected_variable
        assert result.kind == expected_kind


class TestMultipleEstimationSteps:
    """Tests for _MultipleEstimation.steps property."""

    @pytest.mark.parametrize(
        "constant,variable,kind,expected_steps",
        [
            # sw0 cases - sequential with zero step
            (
                ["x", "y"],
                ["a", "b"],
                _MultipleEstimationType.sw0,
                ["x+y", "x+y+a", "x+y+b"],
            ),
            ([], ["a", "b"], _MultipleEstimationType.sw0, ["0", "a", "b"]),
            (["x"], ["a"], _MultipleEstimationType.sw0, ["x", "x+a"]),
            # sw cases - sequential without zero step
            (["x", "y"], ["a", "b"], _MultipleEstimationType.sw, ["x+y+a", "x+y+b"]),
            ([], ["a", "b"], _MultipleEstimationType.sw, ["a", "b"]),
            (["x"], ["a", "b", "c"], _MultipleEstimationType.sw, ["x+a", "x+b", "x+c"]),
            # csw0 cases - cumulative with zero step
            (
                ["x", "y"],
                ["a", "b"],
                _MultipleEstimationType.csw0,
                ["x+y", "x+y+a", "x+y+a+b"],
            ),
            ([], ["a", "b"], _MultipleEstimationType.csw0, ["0", "a", "a+b"]),
            (
                [],
                ["a", "b", "c"],
                _MultipleEstimationType.csw0,
                ["0", "a", "a+b", "a+b+c"],
            ),
            # csw cases - cumulative without zero step
            (
                ["x", "y"],
                ["a", "b"],
                _MultipleEstimationType.csw,
                ["x+y+a", "x+y+a+b"],
            ),
            ([], ["a", "b"], _MultipleEstimationType.csw, ["a", "a+b"]),
            (
                ["x"],
                ["a", "b", "c"],
                _MultipleEstimationType.csw,
                ["x+a", "x+a+b", "x+a+b+c"],
            ),
            # No multiple estimation (kind=None)
            (["x", "y"], [], None, ["x+y"]),
            (["x"], [], None, ["x"]),
        ],
    )
    def test_multiple_estimation_steps(self, constant, variable, kind, expected_steps):
        """Test generation of estimation steps."""
        me = _MultipleEstimation(constant=constant, variable=variable, kind=kind)
        assert me.steps == expected_steps

    def test_is_multiple_property(self):
        """Test is_multiple property."""
        me_single = _MultipleEstimation(constant=["x"], variable=[], kind=None)
        me_multiple = _MultipleEstimation(
            constant=["x"], variable=["a"], kind=_MultipleEstimationType.sw
        )

        assert me_single.is_multiple is False
        assert me_multiple.is_multiple is True


class TestParseFunction:
    """Tests for the main parse() function."""

    @pytest.mark.parametrize(
        "formula,expected_dependent,expected_independent,expected_fe,expected_is_iv",
        [
            # Basic formulas
            ("Y ~ X1", ["Y"], ["X1"], None, False),
            ("Y ~ X1 + X2", ["Y"], ["X1", "X2"], None, False),
            ("Y + Y2 ~ X1", ["Y", "Y2"], ["X1"], None, False),
            # With fixed effects
            ("Y ~ X1 | f1", ["Y"], ["X1"], ["f1"], False),
            ("Y ~ X1 | f1 + f2", ["Y"], ["X1"], ["f1", "f2"], False),
            ("Y ~ X1 + X2 | f1", ["Y"], ["X1", "X2"], ["f1"], False),
            # IV formulas (endogenous var is added to independent)
            ("Y ~ 1 | Z1 ~ X1", ["Y"], ["Z1", "1"], None, True),
            ("Y ~ X1 | Z1 ~ X2", ["Y"], ["Z1", "X1"], None, True),
            ("Y ~ X1 | f1 | Z1 ~ X2", ["Y"], ["Z1", "X1"], ["f1"], True),
            # Edge cases
            ("Y ~ 1", ["Y"], ["1"], None, False),
            ("Y ~ 1 | f1", ["Y"], ["1"], ["f1"], False),
        ],
    )
    def test_parse_basic(
        self,
        formula,
        expected_dependent,
        expected_independent,
        expected_fe,
        expected_is_iv,
    ):
        """Test basic formula parsing."""
        parsed = parse(formula)

        assert parsed.dependent == expected_dependent
        assert parsed.independent.constant == expected_independent
        assert parsed.is_iv == expected_is_iv

        if expected_fe is None:
            assert parsed.fixed_effects is None
        else:
            assert parsed.fixed_effects is not None
            assert parsed.fixed_effects.constant == expected_fe

    def test_parse_with_sort(self):
        """Test sort parameter."""
        parsed_unsorted = parse("Y ~ Z + A + M", sort=False)
        parsed_sorted = parse("Y ~ Z + A + M", sort=True)

        assert parsed_unsorted.independent.constant == ["Z", "A", "M"]
        assert parsed_sorted.independent.constant == ["A", "M", "Z"]

    def test_parse_intercept_parameter(self):
        """Test intercept parameter is passed through."""
        with_intercept = parse("Y ~ X1", intercept=True)
        without_intercept = parse("Y ~ X1", intercept=False)

        assert with_intercept.intercept is True
        assert without_intercept.intercept is False


class TestFormulaDataclass:
    """Tests for the Formula dataclass."""

    def test_fml_basic(self):
        """Test basic formula string generation."""
        f = Formula(dependent="Y", independent="X1+X2")
        assert f.fml == "Y~X1+X2"

    def test_fml_with_fe(self):
        """Test formula with fixed effects."""
        f = Formula(dependent="Y", independent="X1", fixed_effects="f1")
        assert f.fml == "Y~X1|f1"

    def test_fml_with_multiple_fe(self):
        """Test formula with multiple fixed effects."""
        f = Formula(dependent="Y", independent="X1", fixed_effects="f1+f2")
        assert f.fml == "Y~X1|f1+f2"

    def test_fml_with_iv(self):
        """Test formula with instrumental variables."""
        f = Formula(dependent="Y", independent="X1", endogenous="Z1", instruments="X2")
        assert f.fml == "Y~X1|Z1~X2"

    def test_fml_with_iv_and_fe(self):
        """Test formula with IV and fixed effects."""
        f = Formula(
            dependent="Y",
            independent="X1",
            fixed_effects="f1",
            endogenous="Z1",
            instruments="X2",
        )
        assert f.fml == "Y~X1|f1|Z1~X2"

    def test_fml_no_intercept(self):
        """Test formula without intercept."""
        f = Formula(dependent="Y", independent="X1", intercept=False)
        assert f.fml == "Y~X1-1"

    def test_fml_second_stage_basic(self):
        """Test second stage formula generation."""
        f = Formula(dependent="Y", independent="X1+X2")
        assert f.second_stage == "Y~X1+X2"

    def test_fml_second_stage_no_intercept(self):
        """Test second stage formula without intercept."""
        f = Formula(dependent="Y", independent="X1+X2", intercept=False)
        assert f.second_stage == "Y~X1+X2-1"

    def test_fml_first_stage_none_for_non_iv(self):
        """Test first stage is None for non-IV."""
        f = Formula(dependent="Y", independent="X1")
        assert f.first_stage is None

    def test_fml_first_stage_for_iv(self):
        """Test first stage formula for IV."""
        f = Formula(
            dependent="Y",
            independent="Z1+X1",
            endogenous="Z1",
            instruments="X2",
        )
        assert f.first_stage == "Z1~X2+Z1+X1-Z1"


class TestParsedFormulaProperties:
    """Tests for ParsedFormula properties."""

    def test_is_multiple_single_model(self):
        """Test is_multiple for single model."""
        parsed = parse("Y ~ X1")
        assert parsed.is_multiple is False

    def test_is_multiple_multiple_dependents(self):
        """Test is_multiple with multiple dependent variables."""
        parsed = parse("Y + Y2 ~ X1")
        assert parsed.is_multiple is True

    def test_is_multiple_sw_syntax(self):
        """Test is_multiple with sw() syntax."""
        parsed = parse("Y ~ sw(X1, X2)")
        assert parsed.is_multiple is True

    def test_is_multiple_fe_sw_syntax(self):
        """Test is_multiple with sw() in fixed effects."""
        parsed = parse("Y ~ X1 | sw(f1, f2)")
        assert parsed.is_multiple is True

    def test_is_fixed_effects_false(self):
        """Test is_fixed_effects when no FE."""
        parsed = parse("Y ~ X1")
        assert parsed.is_fixed_effects is False

    def test_is_fixed_effects_true(self):
        """Test is_fixed_effects when FE present."""
        parsed = parse("Y ~ X1 | f1")
        assert parsed.is_fixed_effects is True

    def test_is_iv_false(self):
        """Test is_iv for non-IV."""
        parsed = parse("Y ~ X1")
        assert parsed.is_iv is False

    def test_is_iv_true(self):
        """Test is_iv for IV."""
        parsed = parse("Y ~ 1 | Z1 ~ X1")
        assert parsed.is_iv is True


class TestParseErrors:
    """Tests for error handling in parse()."""

    def test_duplicate_multiple_estimation_syntax(self):
        """Test error for duplicate multiple estimation types."""
        with pytest.raises(DuplicateKeyError):
            parse("Y ~ sw(a,b) + csw(c,d)")

    def test_duplicate_in_fixed_effects(self):
        """Test error for duplicate multiple estimation in FE."""
        with pytest.raises(DuplicateKeyError):
            parse("Y ~ X1 | sw(f1,f2) + csw(f3,f4)")

    def test_endogenous_as_covariate(self):
        """Test error when endogenous variable is a covariate."""
        with pytest.raises(EndogVarsAsCovarsError):
            parse("Y ~ Z1 | Z1 ~ X1")

    def test_instruments_as_covariate(self):
        """Test error when instrument is a covariate."""
        with pytest.raises(InstrumentsAsCovarsError):
            parse("Y ~ X1 | Z1 ~ X1")

    def test_underdetermined_iv(self):
        """Test error for underdetermined IV system."""
        with pytest.raises(UnderDeterminedIVError):
            parse("Y ~ 1 | Z1 + Z2 ~ X1")

    def test_multiple_estimation_with_iv(self):
        """Test error for multiple estimation with IV."""
        with pytest.raises(NotImplementedError):
            parse("Y + Y2 ~ 1 | Z1 ~ X1")

    def test_multiple_estimation_fe_with_iv(self):
        """Test error for multiple estimation in FE with IV."""
        with pytest.raises(NotImplementedError):
            parse("Y ~ 1 | sw(f1, f2) | Z1 ~ X1")

    def test_too_many_formula_parts(self):
        """Test error for too many formula parts."""
        with pytest.raises(FormulaSyntaxError):
            parse("Y ~ X1 | f1 | Z1 ~ X2 | extra")

    def test_no_tilde(self):
        """Test error for formula without tilde."""
        with pytest.raises(FormulaSyntaxError):
            parse("Y X1")

    def test_too_many_tildes(self):
        """Test error for formula with too many tildes."""
        # Multiple tildes in main part causes ValueError during unpacking
        with pytest.raises((FormulaSyntaxError, ValueError)):
            parse("Y ~ X1 ~ X2 ~ X3")


# =============================================================================
# Part 2: Multiple Estimation & Structure Tests
# =============================================================================


@pytest.mark.parametrize(
    "formula,expected_n_models",
    [
        ("Y ~ X1", 1),
        ("Y ~ sw(X1, X2)", 2),
        ("Y ~ csw(X1, X2)", 2),
        ("Y ~ sw0(X1, X2)", 3),
        ("Y ~ csw0(X1, X2)", 3),
        ("Y + Y2 ~ X1", 2),
        ("Y ~ X1 | sw(f1, f2)", 2),
        ("Y ~ sw(X1, X2) | csw(f1, f2)", 4),  # 2 x 2
    ],
)
def test_correct_number_of_models(test_data, formula: str, expected_n_models: int):
    """Verify the correct number of models are generated from multiple estimation syntax."""
    fit = pf.feols(formula, data=test_data)

    n_models = len(fit.to_list()) if hasattr(fit, "to_list") else 1

    assert n_models == expected_n_models, (
        f"Expected {expected_n_models} models for '{formula}', got {n_models}"
    )


def test_explicit_no_fe_coefficients_match(test_data):
    """Verify Y ~ X1 | 0 produces same coefficients as Y ~ X1."""
    fit_implicit = pf.feols("Y ~ X1", data=test_data)
    fit_explicit = pf.feols("Y ~ X1 | 0", data=test_data)

    assert np.allclose(fit_implicit.coef().values, fit_explicit.coef().values)
    assert np.allclose(fit_implicit.se().values, fit_explicit.se().values)


def test_explicit_no_fe_iv_coefficients_match(test_data):
    """Verify Y ~ 1 | 0 | Y2 ~ X1 produces same coefficients as Y ~ 1 | Y2 ~ X1."""
    fit_implicit = pf.feols("Y ~ 1 | Y2 ~ X1", data=test_data)
    fit_explicit = pf.feols("Y ~ 1 | 0 | Y2 ~ X1", data=test_data)

    assert np.allclose(fit_implicit.coef().values, fit_explicit.coef().values)
    assert np.allclose(fit_implicit.se().values, fit_explicit.se().values)


# Properties test data
PROPERTY_TEST_FORMULAS = [
    # (formula, is_iv, is_multiple, has_fe)
    ("Y ~ X1", False, False, False),
    ("Y ~ X1 | f1", False, False, True),
    ("Y ~ sw(X1, X2)", False, True, False),
    ("Y + Y2 ~ X1", False, True, False),
    ("Y ~ 1 | Z1 ~ X1", True, False, False),
    ("Y ~ X1 | f1 | Z1 ~ X2", True, False, True),
    ("Y ~ X1 | sw(f1, f2)", False, True, True),
]


@pytest.mark.parametrize(
    "formula,expected_is_iv,expected_is_multiple,expected_has_fe",
    PROPERTY_TEST_FORMULAS,
)
def test_parsed_formula_properties_parametrized(
    formula, expected_is_iv, expected_is_multiple, expected_has_fe
):
    """Test that ParsedFormula properties are correctly set."""
    parsed = parse(formula)

    assert parsed.is_iv == expected_is_iv, f"is_iv mismatch for {formula}"
    assert parsed.is_multiple == expected_is_multiple, (
        f"is_multiple mismatch for {formula}"
    )
    assert parsed.is_fixed_effects == expected_has_fe, (
        f"is_fixed_effects mismatch for {formula}"
    )


# Formulas to test FixestFormulaDict structure
STRUCTURE_TEST_FORMULAS = [
    "Y ~ X1",
    "Y ~ X1 + X2",
    "Y ~ X1 | f1",
    "Y ~ sw(X1, X2)",
    "Y ~ csw(X1, X2)",
    "Y ~ 1 | Z1 ~ X1",
]


@pytest.mark.parametrize("formula", STRUCTURE_TEST_FORMULAS)
def test_fixest_formula_dict_structure(formula: str):
    """Verify FixestFormulaDict has expected structure."""
    parsed = parse(formula)
    fml_dict = parsed.specifications

    # Should be a dict
    assert isinstance(fml_dict, dict)

    # All values should be lists of Formula objects
    for _, formulas in fml_dict.items():
        assert isinstance(formulas, list)
        assert len(formulas) > 0

        for f in formulas:
            # Each Formula should have required attributes
            assert hasattr(f, "dependent")
            assert hasattr(f, "independent")
            assert hasattr(f, "fml")
            assert hasattr(f, "second_stage")
            assert hasattr(f, "first_stage")

            # fml should be a non-empty string
            assert isinstance(f.fml, str)
            assert len(f.fml) > 0


# =============================================================================
# Part 3: Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases that might differ between old and new implementations."""

    def test_empty_independent_with_intercept(self):
        """Test formula with only intercept."""
        parsed = parse("Y ~ 1")
        assert parsed.dependent == ["Y"]
        assert "1" in parsed.independent.constant

    def test_whitespace_handling(self):
        """Test various whitespace patterns."""
        formulas = [
            "Y~X1",
            "Y ~ X1",
            "Y  ~  X1",
            "Y ~ X1|f1",
            "Y ~ X1 | f1",
            "Y  ~  X1  |  f1",
        ]
        for fml in formulas:
            parsed = parse(fml)
            assert parsed.dependent == ["Y"]
            assert "X1" in parsed.independent.constant

    def test_fixed_effects_none_in_dict(self):
        """Test that no fixed effects results in None key in FixestFormulaDict."""
        parsed = parse("Y ~ X1")
        fml_dict = parsed.specifications
        assert None in fml_dict  # No fixed effects should have None key

    def test_fixed_effects_key_in_dict(self):
        """Test that fixed effects are used as keys in FixestFormulaDict."""
        parsed = parse("Y ~ X1 | f1")
        fml_dict = parsed.specifications
        assert "f1" in fml_dict

    def test_sort_parameter_effect(self):
        """Test sort parameter sorts independent variables."""
        parsed_unsorted = parse("Y ~ Z + A + M", sort=False)
        parsed_sorted = parse("Y ~ Z + A + M", sort=True)

        assert parsed_unsorted.independent.constant == ["Z", "A", "M"]
        assert parsed_sorted.independent.constant == ["A", "M", "Z"]

    def test_intercept_parameter_in_formula(self):
        """Test intercept parameter affects Formula generation."""
        with_intercept = parse("Y ~ X1", intercept=True)
        without_intercept = parse("Y ~ X1", intercept=False)

        formula_with = next(iter(with_intercept.specifications.values()))[0]
        formula_without = next(iter(without_intercept.specifications.values()))[0]

        assert formula_with.intercept is True
        assert formula_without.intercept is False
        assert "-1" not in formula_with.fml
        assert "-1" in formula_without.fml

    def test_multiple_dependent_variables(self):
        """Test parsing multiple dependent variables."""
        parsed = parse("Y + Y2 + Y3 ~ X1")
        assert parsed.dependent == ["Y", "Y2", "Y3"]
        assert parsed.is_multiple is True

    def test_multiple_independent_variables(self):
        """Test parsing multiple independent variables."""
        parsed = parse("Y ~ X1 + X2 + X3")
        assert parsed.independent.constant == ["X1", "X2", "X3"]

    def test_complex_formula(self):
        """Test a complex formula with multiple features."""
        parsed = parse("Y ~ X1 + X2 | f1 + f2")
        assert parsed.dependent == ["Y"]
        assert parsed.independent.constant == ["X1", "X2"]
        assert parsed.fixed_effects.constant == ["f1", "f2"]
        assert parsed.is_fixed_effects is True
        assert parsed.is_iv is False
        assert parsed.is_multiple is False

    def test_iv_with_multiple_instruments(self):
        """Test IV with multiple instruments."""
        parsed = parse("Y ~ X1 | Z1 ~ X2 + X3")
        assert parsed.is_iv is True
        assert parsed.endogenous == ["Z1"]
        assert parsed.instruments == ["X2+X3"]  # Joined as single string

    def test_iv_with_fe(self):
        """Test IV formula with fixed effects."""
        parsed = parse("Y ~ X1 | f1 | Z1 ~ X2")
        assert parsed.is_iv is True
        assert parsed.is_fixed_effects is True
        assert parsed.fixed_effects.constant == ["f1"]
        assert parsed.endogenous == ["Z1"]

    def test_explicit_no_fe_syntax(self):
        """Test explicit no fixed effects syntax: Y ~ X1 | 0."""
        parsed_explicit = parse("Y ~ X1 | 0")
        parsed_implicit = parse("Y ~ X1")

        # Both should resolve to None FE in specifications
        specs_explicit = parsed_explicit.specifications
        specs_implicit = parsed_implicit.specifications

        assert list(specs_explicit.keys()) == [None]
        assert list(specs_implicit.keys()) == [None]

        # Formulas should be equivalent
        fml_explicit = specs_explicit[None][0]
        fml_implicit = specs_implicit[None][0]
        assert fml_explicit.fml == fml_implicit.fml
        assert fml_explicit.fixed_effects is None
        assert fml_implicit.fixed_effects is None

    def test_explicit_no_fe_syntax_with_iv(self):
        """Test explicit no fixed effects with IV: Y ~ 1 | 0 | Z1 ~ X1."""
        parsed_explicit = parse("Y ~ 1 | 0 | Z1 ~ X1")
        parsed_implicit = parse("Y ~ 1 | Z1 ~ X1")

        # Both should resolve to None FE in specifications
        specs_explicit = parsed_explicit.specifications
        specs_implicit = parsed_implicit.specifications

        assert list(specs_explicit.keys()) == [None]
        assert list(specs_implicit.keys()) == [None]

        # Both should be IV regressions
        assert parsed_explicit.is_iv is True
        assert parsed_implicit.is_iv is True

        # Formulas should be equivalent
        fml_explicit = specs_explicit[None][0]
        fml_implicit = specs_implicit[None][0]
        assert fml_explicit.fml == fml_implicit.fml
        assert fml_explicit.fixed_effects is None
        assert fml_implicit.fixed_effects is None
        assert fml_explicit.endogenous == fml_implicit.endogenous
        assert fml_explicit.instruments == fml_implicit.instruments
