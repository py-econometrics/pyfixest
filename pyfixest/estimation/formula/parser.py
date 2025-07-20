"""
Main formula parser that orchestrates all components.

This module contains the main FixestFormulaParser class that coordinates
the tokenizer, validator, multiple estimation parser, and formula builder
to provide a clean, high-level interface for formula parsing.
"""

from typing import Dict, List

from .core import FixestFormula, FormulaTokenizer
from .multiple_estimation import MultipleEstimationParser
from .validation import FormulaValidator


class FixestFormulaParser:
    """
    Main formula parser with clean architecture and comprehensive error handling.

    This class provides the main entry point for parsing econometric formulas
    with support for multiple estimation syntax, IV models, and fixed effects.
    It orchestrates the various components (tokenizer, validator, multiple
    estimation parser) to provide a clean, high-level interface.

    Parameters
    ----------
    formula : str
        Formula string to parse

    Attributes
    ----------
    original_formula : str
        The original formula string
    tokens : FormulaTokens
        Parsed formula components
    formulas : List[FixestFormula]
        All formula combinations (for multiple estimation)
    formula_dict : Dict[str, List[FixestFormula]]
        Formulas organized by fixed effects
    FixestFormulaDict : Dict[str, List[FixestFormula]]
        Alias for formula_dict (backward compatibility)
    is_multiple_estimation : bool
        Whether this involves multiple estimation
    _is_multiple_estimation : bool
        Alias for is_multiple_estimation (backward compatibility)
    is_iv : bool
        Whether this involves IV estimation

    Examples
    --------
    >>> parser = FixestFormulaParser("y ~ x1 + x2 | firm_id")
    >>> len(parser.formulas)
    1
    >>> parser.formulas[0].has_fixed_effects
    True

    >>> parser = FixestFormulaParser("y ~ sw(x1, x2, x3)")
    >>> len(parser.formulas)
    3
    >>> parser.is_multiple_estimation
    True
    """

    def __init__(self, formula: str):

        self.original_formula = formula.replace(" ", "")

        # Initialize components
        self._tokenizer = FormulaTokenizer()
        self._validator = FormulaValidator()
        self._multiple_estimation_parser = MultipleEstimationParser()

        # Parse and validate formula
        self._parse_and_validate_formula()

        # Generate all formula combinations
        self._generate_all_formulas()

        # Organize for estimation
        self._organize_formulas()

        # Set convenience flags
        self._set_flags()

        # Set backward compatibility attributes
        self._set_backward_compatibility_attributes()

    def _parse_and_validate_formula(self) -> None:
        """Parse and validate the original formula into tokens."""

        try:
            # Pre-validation of formula structure
            self._validator.validate_formula_structure(self.original_formula)

            # Parse into tokens
            self.tokens = self._tokenizer.tokenize(self.original_formula)

            # Validate parsed tokens
            self._validator.validate_tokens(self.tokens)

        except Exception as e:
            raise ValueError(
                f"Failed to parse formula '{self.original_formula}': {str(e)}"
            ) from e

    def _generate_all_formulas(self) -> None:
        """Generate all formula combinations for multiple estimation."""
        # Check for multiple estimation syntax in covariates
        covar_combinations = self._expand_covariates()

        # For now, we only support multiple estimation in covariates
        # But the structure allows for future expansion to other components
        depvar_combinations = self._expand_depvars()
        fixef_combinations = self._expand_fixed_effects()

        # Generate all combinations
        self.formulas = []

        for depvar in depvar_combinations:
            for covars in covar_combinations:
                for fixef in fixef_combinations:
                    formula = FixestFormula(
                        depvar=depvar,
                        covars=covars,
                        fixed_effects=fixef,
                        endogenous=self.tokens.endogenous,
                        instruments=self.tokens.instruments
                    )
                    self.formulas.append(formula)

    def _expand_covariates(self) -> List[List[str]]:
        """Expand covariates that may contain multiple estimation syntax."""
        if not self.tokens.covars:
            return [[]]

        covar_string = " + ".join(self.tokens.covars)
        variables, multiple_type = self._multiple_estimation_parser.parse_variable_list(covar_string)

        if multiple_type is None:
            # No multiple estimation syntax, return variables as-is
            return [variables]
        else:
            # Need to separate base variables from multiple estimation variables
            # This is a simplified approach - in practice, we'd need more sophisticated
            # tracking of which variables came from which part of the original formula

            # For now, assume all variables from multiple estimation syntax are "multiple"
            # and any variables not part of the syntax are "base"
            base_vars = []
            multiple_vars = variables  # Simplified assumption

            return self._multiple_estimation_parser.expand_formula_combinations(
                base_vars, multiple_vars, multiple_type
            )

    def _expand_depvars(self) -> List[str]:
        """Expand dependent variables (currently single variable only)."""
        # For now, we don't support multiple estimation in depvars
        # but this method provides the structure for future expansion
        if len(self.tokens.depvars) != 1:
            raise ValueError("Currently only single dependent variable is supported")
        return self.tokens.depvars

    def _expand_fixed_effects(self) -> List[List[str]]:
        """Expand fixed effects (currently just returns the list)."""
        # For now, we don't support multiple estimation in fixed effects
        # but this method provides the structure for future expansion
        return [self.tokens.fixef]

    def _organize_formulas(self) -> None:
        """
        Organize formulas by fixed effects for efficient estimation.
        Motivation: models with the same set of fixed effects can inherit
        'demeand' variables from previous steps (if the same of missings is)
        shared).
        """
        self.formula_dict: Dict[str, List[FixestFormula]] = {}

        for formula in self.formulas:
            # Create key from fixed effects
            if formula.fixed_effects:
                key = "+".join(sorted(formula.fixed_effects))
            else:
                key = "0"

            if key not in self.formula_dict:
                self.formula_dict[key] = []

            self.formula_dict[key].append(formula)

    def _set_flags(self) -> None:
        """Set convenience flags based on the parsed formulas."""
        self.is_multiple_estimation = len(self.formulas) > 1
        self.is_iv = any(f.is_iv for f in self.formulas)

        # Check for unsupported combinations
        if self.is_multiple_estimation and self.is_iv:
            raise NotImplementedError(
                "Multiple estimation is currently not supported with IV models"
            )

    def _set_backward_compatibility_attributes(self) -> None:
        """Set attributes for backward compatibility with existing code."""
        # FixestMulti expects these specific attributes
        self.FixestFormulaDict = self.formula_dict
        self._is_multiple_estimation = self.is_multiple_estimation

    def set_fixest_multi_flag(self) -> None:
        """
        Set a flag to indicate whether multiple estimations are being performed.

        This method is for backward compatibility with the existing FixestMulti class.
        It sets the _is_multiple_estimation flag based on the number of formulas.
        """
        self._is_multiple_estimation = self.is_multiple_estimation

        # Backward compatibility: check for unsupported IV + multiple estimation
        if self._is_multiple_estimation and self.is_iv:
            raise NotImplementedError(
                """
                Multiple Estimations is currently not supported with IV.
                This is mostly due to insufficient testing and will be possible
                with a future release of PyFixest.
                """
            )

    def get_formula_summary(self) -> Dict[str, any]:
        """
        Get a summary of the parsed formulas.

        Returns
        -------
        Dict[str, any]
            Summary information about the parsed formulas
        """
        return {
            "original_formula": self.original_formula,
            "num_formulas": len(self.formulas),
            "is_multiple_estimation": self.is_multiple_estimation,
            "is_iv": self.is_iv,
            "has_fixed_effects": any(f.has_fixed_effects for f in self.formulas),
            "dependent_variables": self.tokens.depvars,
            "unique_covariates": list(set().union(*[f.covars for f in self.formulas])),
            "fixed_effects": self.tokens.fixef,
            "endogenous_variables": self.tokens.endogenous,
            "instruments": self.tokens.instruments,
        }

    def get_validation_report(self) -> Dict[str, any]:
        """
        Get a detailed validation report.

        Returns
        -------
        Dict[str, any]
            Validation report for the parsed formulas
        """
        validation_summary = self._validator.get_validation_summary(self.tokens)

        return {
            "formula": self.original_formula,
            "validation_summary": validation_summary,
            "all_checks_passed": all(validation_summary.values()),
            "num_formulas_generated": len(self.formulas),
        }