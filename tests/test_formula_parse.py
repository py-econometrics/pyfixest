"""
Tests for the formula parsing implementation in pyfixest/estimation/formula/parse.py.

This module contains:
- Part 1: Unit tests for Formula.parse() and internal parsing functions
- Part 2: Edge case tests
"""

import re

import pytest

from pyfixest.errors import FormulaSyntaxError
from pyfixest.estimation.formula.parse import Formula, _expand_all_multiple_estimation

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
        ],
    )
    def test_expand_all_multiple_estimation(self, formula, expected):
        """Test expansion of multiple estimation syntax."""
        result = _expand_all_multiple_estimation(formula)
        assert result == expected


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
        assert f.second_stage == "Y ~ X1 + X2"
        assert f.fixed_effects is None
        assert f.first_stage is None

    def test_parse_with_fe(self):
        """Test parsing a formula with fixed effects."""
        result = Formula.parse("Y ~ X1 | f1")
        assert len(result) == 1
        f = result[0]
        assert f.second_stage == "Y ~ X1"
        assert f.fixed_effects == "f1"

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
        assert result[0].second_stage == "Y ~ X1"
        assert result[1].second_stage == "Y2 ~ X1"

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
        with pytest.raises(FormulaSyntaxError):
            Formula.parse("Y ~ X1 ~ X2 ~ X3")

    def test_three_parts_without_iv(self):
        """Y ~ X | f1 | f2 should error (should be Y ~ X | f1 + f2)."""
        with pytest.raises(FormulaSyntaxError, match="Three-part formula"):
            Formula.parse("Y ~ X1 | f1 | f2")

    def test_three_parts_with_tilde_in_fe(self):
        """Y ~ X | Z ~ W | A ~ B should error (FE part has tilde)."""
        with pytest.raises(
            FormulaSyntaxError, match=re.compile("fixed effects.*cannot contain")
        ):
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
# Part 2: Edge Case Tests
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

    def test_iv_endogenous_in_second_stage(self):
        """Endogenous variable should be added to second_stage covariates."""
        result = Formula.parse("Y ~ X1 | Z1 ~ W1")
        f = result[0]
        assert "Z1" in f.second_stage
        # assert f.first_stage == "Z1 ~ W1"

    def test_iv_with_fe_endogenous_in_second_stage(self):
        """Endogenous variable should be in second_stage even with FE."""
        result = Formula.parse("Y ~ X1 | f1 | Z1 ~ W1")
        f = result[0]
        assert "Z1" in f.second_stage
        assert f.fixed_effects == "f1"
        # assert f.first_stage == "Z1 ~ W1"

    def test_explicit_no_fe_syntax(self):
        """Y ~ X1 | 0 and Y ~ X1 should produce equivalent formulas."""
        result_explicit = Formula.parse_to_dict("Y ~ X1 | 0")
        result_implicit = Formula.parse_to_dict("Y ~ X1")

        assert list(result_explicit.keys()) == [None]
        assert list(result_implicit.keys()) == [None]

        f_explicit = result_explicit[None][0]
        f_implicit = result_implicit[None][0]
        assert f_explicit.second_stage == f_implicit.second_stage
        assert f_explicit.fixed_effects is None
        assert f_implicit.fixed_effects is None

    def test_explicit_no_fe_with_iv(self):
        """Y ~ 1 | 0 | Z1 ~ X1 and Y ~ 1 | Z1 ~ X1 should be equivalent."""
        result_explicit = Formula.parse_to_dict("Y ~ 1 | 0 | Z1 ~ X1")
        result_implicit = Formula.parse_to_dict("Y ~ 1 | Z1 ~ X1")

        assert list(result_explicit.keys()) == [None]
        assert list(result_implicit.keys()) == [None]

        f_explicit = result_explicit[None][0]
        f_implicit = result_implicit[None][0]
        assert f_explicit.second_stage == f_implicit.second_stage
        assert f_explicit.fixed_effects is None
        assert f_implicit.fixed_effects is None
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
            assert reparsed[0].fixed_effects == result[0].fixed_effects
            assert reparsed[0].first_stage == result[0].first_stage
