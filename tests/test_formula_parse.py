"""
Tests for the formula parsing implementation in pyfixest/estimation/formula/parse.py.

This module contains:
- Part 1: Unit tests for Formula.parse() and internal parsing functions
- Part 2: End-to-end compatibility tests via feols()
- Part 3: Edge case tests
"""

import warnings

import formulaic
import numpy as np
import pytest

import pyfixest as pf
from pyfixest.errors import FormulaSyntaxError
from pyfixest.estimation.deprecated.model_matrix_fixest_ import model_matrix_fixest
from pyfixest.estimation.formula.parse import Formula, _expand_all_multiple_estimation
from pyfixest.estimation.formula.utils import (
    _get_position_of_first_parenthesis_pair,
    _preprocess_fixed_effect_interactions,
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


class TestMultipleEstimationExpansion:
    """Tests for multiple estimation expansion."""

    @pytest.mark.parametrize(
        "formula,expected",
        [
            # No multiple estimation
            ("Y ~ X1", ["Y ~ X1"]),
            ("Y ~ X1 + X2", ["Y ~ X1 + X2"]),
            # sw() cases
            ("Y ~ sw(X1, X2)", ["Y ~ X1", "Y ~ X2"]),
            ("Y ~ A + sw(X1, X2)", ["Y ~ A + X1", "Y ~ A + X2"]),
            ("Y ~ sw(X1, X2, X3)", ["Y ~ X1", "Y ~ X2", "Y ~ X3"]),
            # csw() cases
            ("Y ~ csw(X1, X2)", ["Y ~ X1", "Y ~ X1 + X2"]),
            (
                "Y ~ A + csw(X1, X2, X3)",
                [
                    "Y ~ A + X1",
                    "Y ~ A + X1 + X2",
                    "Y ~ A + X1 + X2 + X3",
                ],
            ),
            # sw0() cases
            ("Y ~ sw0(X1, X2)", ["Y ~ 1", "Y ~ X1", "Y ~ X2"]),
            ("Y ~ A + sw0(X1, X2)", ["Y ~ A + 1", "Y ~ A + X1", "Y ~ A + X2"]),
            # csw0() cases
            ("Y ~ csw0(X1, X2)", ["Y ~ 1", "Y ~ X1", "Y ~ X1 + X2"]),
            (
                "Y ~ A + csw0(X1, X2, X3)",
                [
                    "Y ~ A + 1",
                    "Y ~ A + X1",
                    "Y ~ A + X1 + X2",
                    "Y ~ A + X1 + X2 + X3",
                ],
            ),
            # mvsw() cases - all combinations of arguments, with zero step
            (
                "Y ~ mvsw(X1, X2)",
                ["Y ~ 1", "Y ~ X1", "Y ~ X2", "Y ~ X1 + X2"],
            ),
            (
                "Y ~ mvsw(X1, X2, X3)",
                [
                    "Y ~ 1",
                    "Y ~ X1",
                    "Y ~ X2",
                    "Y ~ X3",
                    "Y ~ X1 + X2",
                    "Y ~ X1 + X3",
                    "Y ~ X2 + X3",
                    "Y ~ X1 + X2 + X3",
                ],
            ),
            (
                "Y ~ A + mvsw(X1, X2)",
                ["Y ~ A + 1", "Y ~ A + X1", "Y ~ A + X2", "Y ~ A + X1 + X2"],
            ),
            (
                "Y ~ A + mvsw(X1, X2, X3)",
                [
                    "Y ~ A + 1",
                    "Y ~ A + X1",
                    "Y ~ A + X2",
                    "Y ~ A + X3",
                    "Y ~ A + X1 + X2",
                    "Y ~ A + X1 + X3",
                    "Y ~ A + X2 + X3",
                    "Y ~ A + X1 + X2 + X3",
                ],
            ),
            # mvsw() with single argument
            ("Y ~ mvsw(X1)", ["Y ~ 1", "Y ~ X1"]),
            # mvsw() with fixed effects
            (
                "Y ~ mvsw(X1, X2) | f1",
                ["Y ~ 1 | f1", "Y ~ X1 | f1", "Y ~ X2 | f1", "Y ~ X1 + X2 | f1"],
            ),
            # mvsw() in fixed effects
            (
                "Y ~ X1 | mvsw(f1, f2)",
                ["Y ~ X1 | 1", "Y ~ X1 | f1", "Y ~ X1 | f2", "Y ~ X1 | f1 + f2"],
            ),
            # Multiple estimation with sums of variables
            ("Y ~ sw0(f1, f1+f2)", ["Y ~ 1", "Y ~ f1", "Y ~ f1+f2"]),
            ("Y ~ csw0(f1, f1+f2)", ["Y ~ 1", "Y ~ f1", "Y ~ f1 + f1+f2"]),
            # Fixed effects with multiple estimation
            ("Y ~ X1 | sw(f1, f2)", ["Y ~ X1 | f1", "Y ~ X1 | f2"]),
            # Two operators in the same formula part
            (
                "Y ~ sw(X1, X2) + csw(X3, X4)",
                [
                    "Y ~ X1 + X3",
                    "Y ~ X1 + X3 + X4",
                    "Y ~ X2 + X3",
                    "Y ~ X2 + X3 + X4",
                ],
            ),
            # Three operators in the same formula part
            (
                "Y ~ sw(X1, X2) + csw(X3, X4) + sw0(X5, X6)",
                [
                    "Y ~ X1 + X3 + 1",
                    "Y ~ X1 + X3 + X5",
                    "Y ~ X1 + X3 + X6",
                    "Y ~ X1 + X3 + X4 + 1",
                    "Y ~ X1 + X3 + X4 + X5",
                    "Y ~ X1 + X3 + X4 + X6",
                    "Y ~ X2 + X3 + 1",
                    "Y ~ X2 + X3 + X5",
                    "Y ~ X2 + X3 + X6",
                    "Y ~ X2 + X3 + X4 + 1",
                    "Y ~ X2 + X3 + X4 + X5",
                    "Y ~ X2 + X3 + X4 + X6",
                ],
            ),
            # Multiple estimation in covariates and fixed effects
            (
                "Y ~ sw(X1, X2) | sw(f1, f2)",
                [
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f2",
                    "Y ~ X2 | f1",
                    "Y ~ X2 | f2",
                ],
            ),
            (
                "Y ~ csw(X1, X2) | csw(f1, f2)",
                [
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f1 + f2",
                    "Y ~ X1 + X2 | f1",
                    "Y ~ X1 + X2 | f1 + f2",
                ],
            ),
            # Multiple estimation in dependent vars, covariates, and fixed effects
            (
                "sw(Y1, Y2) ~ sw(X1, X2) | sw(f1, f2)",
                [
                    "Y1 ~ X1 | f1",
                    "Y1 ~ X1 | f2",
                    "Y1 ~ X2 | f1",
                    "Y1 ~ X2 | f2",
                    "Y2 ~ X1 | f1",
                    "Y2 ~ X1 | f2",
                    "Y2 ~ X2 | f1",
                    "Y2 ~ X2 | f2",
                ],
            ),
            # Multiple dep vars with sw in covariates and csw in fixed effects
            (
                "sw(Y1, Y2) ~ sw(X1, X2) | csw(f1, f2)",
                [
                    "Y1 ~ X1 | f1",
                    "Y1 ~ X1 | f1 + f2",
                    "Y1 ~ X2 | f1",
                    "Y1 ~ X2 | f1 + f2",
                    "Y2 ~ X1 | f1",
                    "Y2 ~ X1 | f1 + f2",
                    "Y2 ~ X2 | f1",
                    "Y2 ~ X2 | f1 + f2",
                ],
            ),
            # sw0 in covariates + sw in fixed effects
            (
                "Y ~ sw0(X1, X2) | sw(f1, f2)",
                [
                    "Y ~ 1 | f1",
                    "Y ~ 1 | f2",
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f2",
                    "Y ~ X2 | f1",
                    "Y ~ X2 | f2",
                ],
            ),
            # csw in covariates + csw0 in fixed effects
            (
                "Y ~ csw(X1, X2) | csw0(f1, f2)",
                [
                    "Y ~ X1 | 1",
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f1 + f2",
                    "Y ~ X1 + X2 | 1",
                    "Y ~ X1 + X2 | f1",
                    "Y ~ X1 + X2 | f1 + f2",
                ],
            ),
            # sw(dep vars) + csw(covariates) + sw(fixed effects)
            (
                "sw(Y1, Y2) ~ csw(X1, X2) | sw(f1, f2)",
                [
                    "Y1 ~ X1 | f1",
                    "Y1 ~ X1 | f2",
                    "Y1 ~ X1 + X2 | f1",
                    "Y1 ~ X1 + X2 | f2",
                    "Y2 ~ X1 | f1",
                    "Y2 ~ X1 | f2",
                    "Y2 ~ X1 + X2 | f1",
                    "Y2 ~ X1 + X2 | f2",
                ],
            ),
            # mvsw in covariates + sw in fixed effects
            (
                "Y ~ mvsw(X1, X2) | sw(f1, f2)",
                [
                    "Y ~ 1 | f1",
                    "Y ~ 1 | f2",
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f2",
                    "Y ~ X2 | f1",
                    "Y ~ X2 | f2",
                    "Y ~ X1 + X2 | f1",
                    "Y ~ X1 + X2 | f2",
                ],
            ),
            # mvsw in covariates + csw in fixed effects
            (
                "Y ~ mvsw(X1, X2) | csw(f1, f2)",
                [
                    "Y ~ 1 | f1",
                    "Y ~ 1 | f1 + f2",
                    "Y ~ X1 | f1",
                    "Y ~ X1 | f1 + f2",
                    "Y ~ X2 | f1",
                    "Y ~ X2 | f1 + f2",
                    "Y ~ X1 + X2 | f1",
                    "Y ~ X1 + X2 | f1 + f2",
                ],
            ),
        ],
    )
    def test_expand_all_multiple_estimation(self, formula, expected):
        """Test expansion of multiple estimation syntax."""
        result = list(_expand_all_multiple_estimation(formula))
        assert result == expected


class TestParenthesisPair:
    """Tests for _get_position_of_first_parenthesis_pair."""

    def test_simple_pair(self):
        string = "sw(X1, X2)"
        start, end = _get_position_of_first_parenthesis_pair(string)
        assert string[start:end] == "X1, X2"

    def test_nested_pair_as_first_content_char(self):
        """Content starting with '(' must not terminate at the inner ')'.

        Regression test: the scan previously skipped the first content
        character, so `sw((a+b), c)` returned `"(a+b"` instead of the full
        content of the outer pair.
        """
        string = "sw((a+b), c)"
        start, end = _get_position_of_first_parenthesis_pair(string)
        assert string[start:end] == "(a+b), c"

    def test_unmatched_parenthesis_raises_value_error(self):
        """Unclosed '(' raises ValueError, not IndexError."""
        with pytest.raises(ValueError, match="Unmatched"):
            _get_position_of_first_parenthesis_pair("sw(a, b")
        with pytest.raises(ValueError, match="Unmatched"):
            _get_position_of_first_parenthesis_pair("sw(a, b( | f1)")

    def test_no_parenthesis_raises_value_error(self):
        with pytest.raises(ValueError, match="No parenthesis"):
            _get_position_of_first_parenthesis_pair("sw")

    def test_expansion_with_leading_nested_group(self):
        """sw() arguments wrapped in parentheses expand correctly end to end."""
        result = list(_expand_all_multiple_estimation("Y ~ sw((X1+X2), X3)"))
        assert result == ["Y ~ (X1+X2)", "Y ~ X3"]


class TestFormulaParse:
    """Tests for Formula.parse() and Formula.parse_to_dict()."""

    @pytest.mark.parametrize(
        "formula,expected_count",
        [
            ("Y ~ X1", 1),
            ("Y ~ sw(X1, X2)", 2),
            ("Y ~ csw(X1, X2)", 2),
            ("Y ~ sw0(X1, X2)", 3),
            ("Y ~ csw0(X1, X2)", 3),
            ("Y ~ mvsw(X1, X2)", 4),
            ("Y ~ mvsw(X1, X2, X3)", 8),
            # With fixed effects
            ("Y ~ X1 | f1", 1),
            ("Y ~ sw(X1, X2) | f1", 2),
            ("Y ~ csw(X1, X2) | f1", 2),
            ("Y ~ sw0(X1, X2) | f1", 3),
            ("Y ~ csw0(X1, X2) | f1", 3),
            ("Y ~ mvsw(X1, X2) | f1", 4),
            # Multiple estimation in fixed effects
            ("Y ~ X1 | sw(f1, f2)", 2),
            ("Y ~ X1 | csw(f1, f2)", 2),
            ("Y ~ X1 | sw0(f1, f2)", 3),
            ("Y ~ X1 | csw0(f1, f2)", 3),
            ("Y ~ X1 | mvsw(f1, f2)", 4),
            # Multiple estimation in covariates and fixed effects
            ("Y ~ sw(X1, X2) | sw(f1, f2)", 4),
            ("Y ~ csw(X1, X2) | csw(f1, f2)", 4),
            ("Y ~ sw0(X1, X2) | sw0(f1, f2)", 9),
            ("Y ~ mvsw(X1, X2) | sw(f1, f2)", 8),
            # Multiple estimation in dependent vars, covariates, and fixed effects
            ("Y + Y2 ~ sw(X1, X2) | sw(f1, f2)", 8),
            ("sw(Y1, Y2) ~ csw(X1, X2) | csw(f1, f2)", 8),
            ("Y + Y2 + Y3 ~ sw(X1, X2) | sw(f1, f2)", 12),
        ],
    )
    def test_parse_count(self, formula, expected_count):
        """Test that parse returns the correct number of Formula objects."""
        result = Formula.parse(formula)
        assert len(result) == expected_count

    def test_parse_basic(self):
        """Test parsing a basic formula with no fixed effects or IV."""
        result = Formula.parse("Y ~ X1 + X2")
        assert len(result) == 1
        f = result[0]
        assert f.second_stage == "Y ~ 1 + X1 + X2"
        assert not f.is_fixed_effects
        assert not f.is_instrumental_variable

    def test_parse_preserves_subclass(self):
        """Formula.parse() constructs instances of the calling subclass."""

        class SubFormula(Formula):
            pass

        result = SubFormula.parse("Y ~ X1")
        grouped = SubFormula.parse_to_dict("Y ~ X1")

        assert isinstance(result[0], SubFormula)
        assert isinstance(grouped[None][0], SubFormula)

    def test_parse_with_fe(self):
        """Test parsing a formula with fixed effects."""
        result = Formula.parse("Y ~ X1 | f1")
        assert len(result) == 1
        f = result[0]
        assert f.second_stage == "Y ~ X1"
        assert str(f.fixed_effects) == "f1"

    @pytest.mark.parametrize(
        ("formula", "has_fixed_effects"),
        [("Y ~ X1", False), ("Y ~ X1 | f1", True)],
    )
    def test_legacy_model_matrix_fixed_effects(
        self, test_data, formula, has_fixed_effects
    ):
        """The deprecated ModelMatrix shim handles Formulaic fixed effects."""
        with pytest.warns(FutureWarning, match="model_matrix_fixest"):
            result = model_matrix_fixest(Formula.parse(formula)[0], test_data)

        assert (result["fe"] is not None) is has_fixed_effects

    # def test_parse_iv(self):
    #     result = Formula.parse("Y ~ X1 | f1 | Z1 ~ W1")
    #     assert len(result) == 1
    #     f = result[0]
    #     assert f.second_stage == "Y ~ X1 + Z1"
    #     assert f.fixed_effects == "f1"
    #     assert f.first_stage == "Z1 ~ W1"

    def test_parse_multiple_dependents(self):
        """Y + Y2 ~ X1 is preprocessed to sw(Y, Y2) ~ X1."""
        result = Formula.parse("Y + Y2 ~ X1")
        assert len(result) == 2
        assert result[0].second_stage == "Y ~ 1 + X1"
        assert result[1].second_stage == "Y2 ~ 1 + X1"

    def test_parse_to_dict_groups_by_fe(self):
        """Test parsing of formulas into dictionary."""
        result = Formula.parse_to_dict("Y ~ X1 | sw(f1, f2)")
        assert "f1" in result
        assert "f2" in result
        assert len(result["f1"]) == 1
        assert len(result["f2"]) == 1

    def test_parse_to_dict_no_fe(self):
        """Test parsing of formulas into dictionary without fixed effects."""
        result = Formula.parse_to_dict("Y ~ X1")
        assert None in result
        assert len(result[None]) == 1

    def test_parse_sw_in_fe_and_independent(self):
        """Cross-product: sw in both independent and FE."""
        result = Formula.parse("Y ~ sw(X1, X2) | sw(f1, f2)")
        assert len(result) == 4  # 2 x 2

    def test_parse_multiple_dep_vars_with_sw_and_csw(self):
        """Y1 + Y2 (preprocessed to sw) + sw in covars + csw in FE."""
        result = Formula.parse("Y1 + Y2 ~ sw(X1, X2) | csw(f1, f2)")
        assert len(result) == 8  # 2 dep * 2 covars * 2 FE
        second_stages = [f.second_stage for f in result]
        fixed_effects = [str(f.fixed_effects) for f in result]
        assert second_stages == [
            "Y1 ~ X1",
            "Y1 ~ X1",
            "Y1 ~ X2",
            "Y1 ~ X2",
            "Y2 ~ X1",
            "Y2 ~ X1",
            "Y2 ~ X2",
            "Y2 ~ X2",
        ]
        assert fixed_effects == [
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
        ]

    def test_parse_csw0_in_fe_maps_to_none(self):
        """csw0 in FE produces a '1' zero-step which maps to fixed_effects=None."""
        result = Formula.parse("Y ~ X1 | csw0(f1, f2)")
        assert len(result) == 3
        assert not result[0].is_fixed_effects  # zero step: "1" -> None
        assert str(result[1].fixed_effects) == "f1"
        assert str(result[2].fixed_effects) == "f1 + f2"

    def test_parse_sw0_in_fe_maps_to_none(self):
        """sw0 in FE produces a '1' zero-step which maps to fixed_effects=None."""
        result = Formula.parse("Y ~ X1 | sw0(f1, f2)")
        assert len(result) == 3
        assert not result[0].is_fixed_effects  # zero step: "1" -> None
        assert str(result[1].fixed_effects) == "f1"
        assert str(result[2].fixed_effects) == "f2"

    def test_parse_mvsw_covars_with_csw_fe(self):
        """Mvsw in covariates combined with csw in fixed effects."""
        result = Formula.parse("Y ~ mvsw(X1, X2) | csw(f1, f2)")
        assert len(result) == 8  # 4 mvsw * 2 csw
        second_stages = [f.second_stage for f in result]
        fixed_effects = [str(f.fixed_effects) for f in result]
        assert second_stages == [
            "Y ~ 1",
            "Y ~ 1",
            "Y ~ X1",
            "Y ~ X1",
            "Y ~ X2",
            "Y ~ X2",
            "Y ~ X1 + X2",
            "Y ~ X1 + X2",
        ]
        assert fixed_effects == [
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
            "f1",
            "f1 + f2",
        ]

    def test_parse_to_dict_csw0_fe_groups(self):
        """parse_to_dict should group csw0 FE correctly, with None for zero-step."""
        result = Formula.parse_to_dict("Y ~ X1 | csw0(f1, f2)")
        assert None in result  # zero step
        assert "f1" in result
        assert "f1 + f2" in result


class TestFixedEffectInteractions:
    """Tests for canonical and legacy fixed-effect interaction syntax."""

    def test_canonical_interaction_parses_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            parsed = Formula.parse("Y ~ X1 | f1:f2")[0]

        assert str(parsed._formula.rhs[1]) == "1 + f1:f2"
        assert parsed.formula == "Y ~ X1 | f1:f2"

    def test_legacy_interaction_warns_once_and_is_canonical(self):
        canonical = Formula.parse("Y ~ X1 | f1:f2")[0]

        with pytest.warns(
            DeprecationWarning,
            match=r"Instead of `Y ~ X1 \| f1\^f2` use `Y ~ X1 \| f1:f2`",
        ) as caught:
            legacy = Formula.parse("Y ~ X1 | f1^f2")[0]

        assert len(caught) == 1
        assert legacy._formula.rhs[1] == canonical._formula.rhs[1]
        assert legacy.formula == canonical.formula == "Y ~ X1 | f1:f2"

    @pytest.mark.parametrize(
        "fixed_effects,expected",
        [
            ("f1^f2^f3", "f1:f2:f3"),
            ("f1^f2:f3 + f4^f5", "f1:f2:f3 + f4:f5"),
        ],
    )
    def test_legacy_chained_and_mixed_interactions(self, fixed_effects, expected):
        with pytest.warns(DeprecationWarning) as caught:
            parsed = Formula.parse(f"Y ~ X1 | {fixed_effects}")[0]

        assert len(caught) == 1
        assert set(str(parsed.fixed_effects).split(" + ")) == set(expected.split(" + "))
        assert "^" not in parsed.formula

    def test_rewrite_is_scoped_to_top_level_fixed_effects(self):
        formula = "Y^out ~ (X1 + X2)^2 + [X3 ~ (Z1 + Z2)^2] | I(f1^2) + f2^f3"

        with pytest.warns(DeprecationWarning) as caught:
            translated = _preprocess_fixed_effect_interactions(formula)

        assert len(caught) == 1
        assert translated == (
            "Y^out ~ (X1 + X2)^2 + [X3 ~ (Z1 + Z2)^2] | I(f1^2) + f2:f3"
        )

    def test_formulaic_power_operators_match_outside_fixed_effects(self):
        caret = Formula.parse("Y ~ (X1 + X2)^2 | f1:f2")[0]
        double_star = Formula.parse("Y ~ (X1 + X2)**2 | f1:f2")[0]

        assert caret._formula.rhs[0] == double_star._formula.rhs[0]
        assert str(caret._formula.rhs[0]) == "1 + X1 + X2 + X1:X2"

    def test_nested_fe_and_iv_power_expressions_are_untouched(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            nested_fe = Formula.parse("Y ~ X1 | I(f1^2)")[0]
            iv_power = Formula.parse("Y ~ X1 + [X2 ~ (Z1 + Z2)^2] | f1:f2")[0]

        assert str(nested_fe.fixed_effects) == "I(f1 ^ 2)"
        assert str(iv_power.instruments) == "1 + Z1 + Z2 + Z1:Z2"


class TestValidation:
    """Tests for formula validation / error handling."""

    def test_no_tilde(self):
        """Check minimum number of tildes."""
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y X1")

    def test_too_many_parts(self):
        """Check maximum number of formula parts is not exceeded."""
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y ~ X1 | f1 | Z1 ~ X2 | extra")

    def test_too_many_tildes_in_part(self):
        """Check maximum number of tildes is not exceeded."""
        with pytest.raises(formulaic.errors.FormulaSyntaxError):
            Formula.parse("Y ~ X1 ~ X2 ~ X3")

    def test_three_parts_without_iv(self):
        """Y ~ X | f1 | f2 should error (should be Y ~ X | f1 + f2)."""
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y ~ X1 | f1 | f2")

    def test_three_parts_with_tilde_in_fe(self):
        """Y ~ X | Z ~ W | A ~ B should error (FE part has tilde)."""
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y ~ X | Z ~ W | A ~ B")

    def test_first_part_must_have_tilde(self):
        """Formula must have at least one tilde."""
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y | f1")

    @pytest.mark.parametrize(
        "formula",
        [
            "Y ~ sw((X1, X2))",
            "Y ~ csw((X1, X2))",
            "Y ~ sw0((X1, X2))",
            "Y ~ csw0((X1, X2))",
        ],
    )
    def test_extra_parens_in_multiple_estimation(self, formula):
        """sw((a, b)) should error — extra parens swallow the separator."""
        with pytest.raises(FormulaSyntaxError, match="at least 2 arguments"):
            Formula.parse(formula)


# =============================================================================
# Part 2: End-to-end compatibility tests via feols()
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
        ("Y ~ mvsw(X1, X2)", 4),
        ("Y ~ mvsw(X1, X2, Z1)", 8),
        ("Y ~ mvsw(X1, X2) | f1", 4),
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
    """Verify Y ~ X1 | 1 produces same coefficients as Y ~ X1."""
    fit_implicit = pf.feols("Y ~ X1", data=test_data)
    fit_explicit = pf.feols("Y ~ X1 | 1", data=test_data)

    assert np.allclose(fit_implicit.coef().values, fit_explicit.coef().values)
    assert np.allclose(fit_implicit.se().values, fit_explicit.se().values)


def test_explicit_no_fe_iv_coefficients_match(test_data):
    """Verify Y ~ 1 | 1 | Y2 ~ X1 produces same coefficients as Y ~ 1 | Y2 ~ X1."""
    fit_implicit = pf.feols("Y ~ 1 | Y2 ~ X1", data=test_data)
    fit_explicit = pf.feols("Y ~ 1 | 1 | Y2 ~ X1", data=test_data)

    assert np.allclose(fit_implicit.coef().values, fit_explicit.coef().values)
    assert np.allclose(fit_implicit.se().values, fit_explicit.se().values)


def test_interacted_fixed_effect_matches_manual_interaction(test_data):
    """Interacted fixed effects match an explicitly constructed group."""
    data = test_data.dropna(subset=["Y", "X1", "f1", "f2"]).copy()
    data["f1_f2"] = list(zip(data["f1"], data["f2"], strict=True))

    interacted = pf.feols("Y ~ X1 | f1:f2", data=data, fixef_rm="none")
    manual = pf.feols("Y ~ X1 | f1_f2", data=data, fixef_rm="none")

    np.testing.assert_allclose(interacted.coef(), manual.coef())
    np.testing.assert_allclose(interacted.se(), manual.se())


# =============================================================================
# Part 3: Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases in formula parsing."""

    def test_intercept_only(self):
        """Test intercept only."""
        result = Formula.parse("Y ~ 1")
        assert len(result) == 1
        assert result[0].second_stage == "Y ~ 1"

    def test_no_fe_in_dict(self):
        """No fixed effects results in None key in parse_to_dict."""
        result = Formula.parse_to_dict("Y ~ X1")
        assert None in result

    def test_fe_key_in_dict(self):
        """Fixed effects are used as keys in parse_to_dict."""
        result = Formula.parse_to_dict("Y ~ X1 | f1")
        assert "f1" in result

    def test_multiple_dependent_variables(self):
        """Test multiple independent variables."""
        result = Formula.parse("Y + Y2 + Y3 ~ X1")
        assert len(result) == 3

    @pytest.mark.parametrize(
        "formula,expected_second_stage",
        [
            ("I(Y + Y2) ~ X1", "I(Y + Y2) ~ 1 + X1"),
            ("I(Y + Y2 + Y3) ~ X1", "I(Y + Y2 + Y3) ~ 1 + X1"),
            ("log(Y + Y2) ~ X1", "log(Y + Y2) ~ 1 + X1"),
        ],
    )
    def test_transformed_dependent_with_plus(self, formula, expected_second_stage):
        """A `+` nested in a transform is not a multiple-dependent separator."""
        result = Formula.parse(formula)
        assert len(result) == 1
        assert result[0].second_stage == expected_second_stage

    def test_transformed_dependent_matches_precomputed_column(self, test_data):
        """`I(Y + Y2)` estimates the summed outcome, not two separate models."""
        data = test_data.dropna().copy()
        data["Y_sum"] = data["Y"] + data["Y2"]

        transformed = pf.feols("I(Y + Y2) ~ X1", data)
        precomputed = pf.feols("Y_sum ~ X1", data)

        np.testing.assert_allclose(
            transformed.coef().to_numpy(), precomputed.coef().to_numpy()
        )

    def test_iv_endogenous_in_second_stage(self):
        """Endogenous variable should be added to second_stage covariates."""
        result = Formula.parse("Y ~ X1 | Z1 ~ W1")
        f = result[0]
        assert "Z1" in f.second_stage
        # assert f.first_stage == "Z1 ~ W1"

    def test_iv_transformed_endogenous_in_second_stage(self):
        """A transformed endogenous variable survives the _hat term filtering."""
        f = Formula.parse("Y ~ X1 | log(Z1) ~ W1")[0]
        assert f.second_stage == "Y ~ 1 + X1 + log(Z1)"
        assert f.first_stage.startswith("log(Z1) ~")

    def test_iv_with_fe_endogenous_in_second_stage(self):
        """Endogenous variable should be in second_stage even with FE."""
        result = Formula.parse("Y ~ X1 | f1 | Z1 ~ W1")
        f = result[0]
        assert "Z1" in f.second_stage
        assert str(f.fixed_effects) == "f1"
        assert str(f.first_stage) == "Z1 ~ 1 + W1 + X1"

    def test_explicit_no_fe_syntax(self):
        """Y ~ X1 | 0 and Y ~ X1 should produce equivalent formulas."""
        result_explicit = Formula.parse_to_dict("Y ~ X1 | 0")
        result_implicit = Formula.parse_to_dict("Y ~ X1")

        assert list(result_explicit.keys()) == [None]
        assert list(result_implicit.keys()) == [None]

        f_explicit = result_explicit[None][0]
        f_implicit = result_implicit[None][0]
        assert f_explicit.second_stage == f_implicit.second_stage
        assert not f_explicit.is_fixed_effects
        assert not f_implicit.is_fixed_effects

    def test_explicit_no_fe_with_iv(self):
        """Y ~ 1 | 0 | Z1 ~ X1 and Y ~ 1 | Z1 ~ X1 should be equivalent."""
        result_explicit = Formula.parse_to_dict("Y ~ 1 | 0 | Z1 ~ X1")
        result_implicit = Formula.parse_to_dict("Y ~ 1 | Z1 ~ X1")

        assert list(result_explicit.keys()) == [None]
        assert list(result_implicit.keys()) == [None]

        f_explicit = result_explicit[None][0]
        f_implicit = result_implicit[None][0]
        assert f_explicit.second_stage == f_implicit.second_stage
        assert not f_explicit.is_fixed_effects
        assert not f_implicit.is_fixed_effects
        assert f_explicit.first_stage == f_implicit.first_stage

    def test_formula_roundtrip(self):
        """Parsing a formula and reconstructing it should preserve structure."""
        formulas = [
            "Y ~ X1",
            "Y ~ X1 + X2",
            "Y ~ X1 | f1",
            "Y ~ X1 | f1 + f2",
        ]
        for fml in formulas:
            result = Formula.parse(fml)
            assert len(result) == 1
            # Reconstructed formula should re-parse to the same structure
            reparsed = Formula.parse(result[0].formula)
            assert len(reparsed) == 1
            assert reparsed[0].second_stage == result[0].second_stage
            assert reparsed[0].is_fixed_effects == result[0].is_fixed_effects
            if result[0].is_fixed_effects:
                assert reparsed[0].fixed_effects == result[0].fixed_effects

            assert (
                reparsed[0].is_instrumental_variable
                == result[0].is_instrumental_variable
            )
            if result[0].is_instrumental_variable:
                assert reparsed[0].first_stage == result[0].first_stage
