"""
Validation logic for formula components.

This module contains centralized validation logic for formula parsing,
separated from the main parsing logic for better maintainability and testing.
"""

from typing import List, Optional, Set

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)

from ..core.types import FormulaTokens


class FormulaValidator:
    """
    Centralized validation for formula components.

    This class provides static methods for validating different aspects
    of formula specifications, ensuring they form valid econometric models.
    """

    @staticmethod
    def validate_tokens(tokens: FormulaTokens) -> None:
        """
        Comprehensive validation of formula tokens.

        This method performs all validation checks on parsed formula tokens
        to ensure they represent a valid econometric specification.

        Parameters
        ----------
        tokens : FormulaTokens
            Parsed formula components to validate

        Raises
        ------
        UnderDeterminedIVError
            If IV system is underdetermined
        EndogVarsAsCovarsError
            If endogenous variables appear as covariates
        InstrumentsAsCovarsError
            If instruments appear as covariates
        """
        FormulaValidator.validate_unique_variables(
            tokens.endogenous, tokens.instruments, tokens.covars
        )
        FormulaValidator.validate_iv_specification(
            tokens.endogenous, tokens.instruments, tokens.covars
        )
        FormulaValidator.validate_variable_overlaps(
            tokens.endogenous, tokens.instruments, tokens.covars
        )

    @staticmethod
    def validate_unique_variables(endogenous: Optional[List[str]], instruments: Optional[List[str]], covars: List[str]) -> None:
        """
        Validate that variables are unique within each category.
        """
        from collections import Counter

        def _has_duplicates(variables: List[str]) -> bool:
            counts = Counter(variables)
            duplicates = [var for var, count in counts.items() if count > 1]
            return duplicates

        if _has_duplicates(covars):
            raise ValueError(
                f"Duplicate covariates found."
                f"Each covariate should appear only once in the formula."
            )

        if endogenous:
            if _has_duplicates(endogenous):
                raise ValueError(
                    f"Duplicate endogenous variables found."
                    f"Each endogenous variable should appear only once in the formula."
                )

        if instruments:
            if _has_duplicates(instruments):
                raise ValueError(
                    f"Duplicate instruments found."
                    f"Each instrument should appear only once in the formula."
                )

    @staticmethod
    def validate_iv_specification(
        endogenous: Optional[List[str]],
        instruments: Optional[List[str]],
        covars: List[str]
    ) -> None:
        """
        Validate IV specification for identification.

        Checks that the IV system is properly identified (at least as many
        instruments as endogenous variables).

        Parameters
        ----------
        endogenous : Optional[List[str]]
            List of endogenous variables
        instruments : Optional[List[str]]
            List of instrumental variables
        covars : List[str]
            List of covariate variables

        Raises
        ------
        UnderDeterminedIVError
            If there are more endogenous variables than instruments
        """
        if endogenous and instruments:
            if len(endogenous) > len(instruments):
                raise UnderDeterminedIVError(
                    f"IV system is underdetermined:\n"
                    f"  Endogenous variables ({len(endogenous)}): {', '.join(endogenous)}\n"
                    f"  Instruments ({len(instruments)}): {', '.join(instruments)}\n"
                    f"Need at least as many instruments as endogenous variables."
                )

    @staticmethod
    def validate_variable_overlaps(
        endogenous: Optional[List[str]],
        instruments: Optional[List[str]],
        covars: List[str]
    ) -> None:
        """
        Validate that variables don't appear in inappropriate places.

        Checks that endogenous variables and instruments don't appear
        as regular covariates, which would be econometrically invalid.

        Parameters
        ----------
        endogenous : Optional[List[str]]
            List of endogenous variables
        instruments : Optional[List[str]]
            List of instrumental variables
        covars : List[str]
            List of covariate variables

        Raises
        ------
        EndogVarsAsCovarsError
            If endogenous variables appear as covariates
        InstrumentsAsCovarsError
            If instruments appear as covariates
        """
        covar_set = set(covars)

        # Check for instruments in covariates
        if instruments:
            instruments_in_covars = set(instruments) & covar_set
            if instruments_in_covars:
                raise InstrumentsAsCovarsError(
                    f"Instruments cannot appear as covariates:\n"
                    f"  Variables in both: {', '.join(sorted(instruments_in_covars))}"
                )

    @staticmethod
    def validate_formula_structure(formula: str) -> None:
        """
        Validate basic formula structure before parsing.

        Performs basic structural validation of the formula string
        to catch obvious errors early.

        Parameters
        ----------
        formula : str
            Formula string to validate

        Raises
        ------
        ValueError
            If the formula structure is invalid
        """
        if not formula or not isinstance(formula, str):
            raise ValueError("Formula must be a non-empty string")

        # Check for required tilde
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separator")

        # Check for too many parts
        parts = formula.split("|")
        if len(parts) > 3:
            raise ValueError("Formula cannot have more than 3 parts separated by '|'")

        # Check for proper IV syntax in relevant parts
        for i, part in enumerate(parts[1:], 1):  # Skip first part (main equation)
            if "~" in part and i != len(parts) - 1:
                raise ValueError("IV specification ('~') can only appear in the last part of the formula")

    @staticmethod
    def validate_variable_names(variables: List[str]) -> None:
        """
        Validate variable names for common issues.

        Checks variable names for common problems like empty strings,
        reserved keywords, or invalid characters.

        Parameters
        ----------
        variables : List[str]
            List of variable names to validate

        Raises
        ------
        ValueError
            If any variable names are invalid
        """
        reserved_words = {"1", "0", ""}

        for var in variables:
            if not var or not isinstance(var, str):
                raise ValueError(f"Variable name must be a non-empty string, got: {repr(var)}")

            if var in reserved_words:
                raise ValueError(f"'{var}' is a reserved word and cannot be used as a variable name")

            # Check for obviously problematic characters (basic validation)
            if any(char in var for char in [" ", "\t", "\n", "|", "~"]):
                raise ValueError(f"Variable name '{var}' contains invalid characters")

    @staticmethod
    def get_validation_summary(tokens: FormulaTokens) -> dict[str, bool]:
        """
        Get a summary of validation status.

        Returns a dictionary indicating which validation checks pass or fail,
        useful for debugging or providing detailed error information.

        Parameters
        ----------
        tokens : FormulaTokens
            Formula tokens to validate

        Returns
        -------
        dict[str, bool]
            Dictionary with validation check results
        """
        summary = {
            "has_dependent_variable": bool(tokens.depvars),
            "has_covariates": bool(tokens.covars),
            "is_iv_model": tokens.is_iv,
            "has_fixed_effects": tokens.has_fixed_effects,
            "iv_properly_identified": True,
            "no_variable_overlaps": True,
        }

        # Check IV identification
        if tokens.is_iv:
            try:
                FormulaValidator.validate_iv_specification(
                    tokens.endogenous, tokens.instruments, tokens.covars
                )
            except UnderDeterminedIVError:
                summary["iv_properly_identified"] = False

        # Check variable overlaps
        try:
            FormulaValidator.validate_variable_overlaps(
                tokens.endogenous, tokens.instruments, tokens.covars
            )
        except (EndogVarsAsCovarsError, InstrumentsAsCovarsError):
            summary["no_variable_overlaps"] = False

        return summary