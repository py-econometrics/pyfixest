"""
Formula tokenizer for parsing formula strings into structured components.

This module is responsible for the low-level parsing of formula strings
into their constituent parts (dependent variables, covariates, fixed effects, etc.)
"""

from typing import List, Tuple

from .types import FormulaTokens


class FormulaTokenizer:
    """
    Responsible for breaking formula strings into structured components.

    This class handles the low-level parsing of formula strings into their
    constituent parts. It focuses purely on tokenization without performing
    validation or business logic.
    """

    @staticmethod
    def tokenize(formula: str) -> FormulaTokens:
        """
        Parse formula string into structured components.

        This method takes a formula string and breaks it down into its constituent
        parts: dependent variables, covariates, fixed effects, endogenous variables,
        and instruments.

        Parameters
        ----------
        formula : str
            Formula string in format "depvar ~ covars | fixef | endog ~ instruments"

        Returns
        -------
        FormulaTokens
            Structured representation of formula components

        Raises
        ------
        ValueError
            If the formula string is malformed or empty

        Examples
        --------
        >>> tokens = FormulaTokenizer.tokenize("y ~ x1 + x2 | fe1")
        >>> tokens.depvars
        ['y']
        >>> tokens.covars
        ['x1', 'x2']
        >>> tokens.fixef
        ['fe1']
        """
        if not formula or not isinstance(formula, str):
            raise ValueError("Formula must be a non-empty string")

        # Clean up the formula string
        formula = "".join(formula.split())

        try:
            return FormulaTokenizer._parse_formula_parts(formula)
        except Exception as e:
            raise ValueError(f"Invalid formula syntax: '{formula}'") from e

    @staticmethod
    def _parse_formula_parts(formula: str) -> FormulaTokens:
        """
        Parse the different parts of the formula string.

        This method handles the actual parsing logic, splitting the formula
        into its components and handling different formula structures.

        Parameters
        ----------
        formula : str
            Cleaned formula string

        Returns
        -------
        FormulaTokens
            Parsed formula components
        """
        # Split the formula string into its components

        parts = formula.split("|")

        if len(parts) < 1:
            raise ValueError("Formula must contain at least dependent and independent variables")

        # Parse main equation: depvar ~ covars
        if "~" not in parts[0]:
            raise ValueError("Formula must contain '~' separator")

        depvar_part, covar_part = parts[0].split("~", 1)
        depvars = [v.strip() for v in depvar_part.split("+") if v.strip()]
        covars = [v.strip() for v in covar_part.split("+") if v.strip() and v != "1"]

        # Initialize optional components
        fixef: List[str] = []
        endogenous = None
        instruments = None

        # Parse remaining parts based on formula structure
        if len(parts) == 2:
            # Could be fixed effects or IV specification
            if "~" in parts[1]:
                # IV specification: endog ~ instruments
                endogenous, instruments = FormulaTokenizer._parse_iv_part(parts[1])
                # Add endogenous variables to covariates (following R fixest convention)
                covars = endogenous + covars
            else:
                # Fixed effects
                fixef = [v.strip() for v in parts[1].split("+") if v.strip() and v != "0"]

        elif len(parts) == 3:
            # Both fixed effects and IV: depvar ~ covars | fixef | endog ~ instruments
            fixef = [v.strip() for v in parts[1].split("+") if v.strip() and v != "0"]
            endogenous, instruments = FormulaTokenizer._parse_iv_part(parts[2])
            # Add endogenous variables to covariates
            covars = endogenous + covars

        elif len(parts) > 3:
            raise ValueError("Formula cannot have more than 3 parts separated by '|'")

        return FormulaTokens(
            depvars=depvars,
            covars=covars,
            fixef=fixef,
            endogenous=endogenous,
            instruments=instruments
        )

    @staticmethod
    def _parse_iv_part(iv_part: str) -> Tuple[List[str], List[str]]:
        """
        Parse the IV part of the formula: endog ~ instruments.

        Parameters
        ----------
        iv_part : str
            The IV specification part of the formula

        Returns
        -------
        Tuple[List[str], List[str]]
            Tuple of (endogenous_variables, instruments)

        Raises
        ------
        ValueError
            If the IV specification is malformed
        """
        if "~" not in iv_part:
            raise ValueError("IV specification must contain '~'")

        endog_part, instrument_part = iv_part.split("~", 1)

        endogenous = [v.strip() for v in endog_part.split("+") if v.strip()]
        instruments = [v.strip() for v in instrument_part.split("+") if v.strip()]

        if not endogenous:
            raise ValueError("Must specify at least one endogenous variable")
        if not instruments:
            raise ValueError("Must specify at least one instrument")

        return endogenous, instruments