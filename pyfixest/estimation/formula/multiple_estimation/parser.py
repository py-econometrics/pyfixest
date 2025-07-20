"""
Parser for multiple estimation syntax.

This module handles the parsing and expansion of multiple estimation syntax
like sw(), csw(), sw0(), and csw0() into individual formula combinations.
"""

import re
from typing import List, Optional, Tuple

from pyfixest.errors import DuplicateKeyError

from ..core.types import MultipleEstimationType
from .strategies import StrategyFactory


class MultipleEstimationParser:
    """
    Handles parsing of multiple estimation syntax like sw(), csw(), etc.

    This class is responsible for identifying and expanding multiple estimation
    syntax into individual formula combinations. It uses the strategy pattern
    to handle different types of multiple estimation.

    Attributes
    ----------
    strategies : dict
        Dictionary mapping estimation types to their strategy implementations
    """

    def __init__(self):
        """Initialize the parser with available strategies."""
        self.strategies = StrategyFactory.get_all_strategies()

    def parse_variable_list(self, variable_string: str) -> Tuple[List[str], Optional[MultipleEstimationType]]:
        """
        Parse a variable string and identify multiple estimation syntax.

        This method parses a string containing variable names and identifies
        any multiple estimation syntax present. It extracts both the variables
        and the type of multiple estimation being used.

        Parameters
        ----------
        variable_string : str
            String like "x1 + sw(x2, x3)" or "x1 + x2"

        Returns
        -------
        Tuple[List[str], Optional[MultipleEstimationType]]
            Tuple of (variables, multiple_estimation_type).
            multiple_estimation_type is None if no multiple estimation syntax is found.

        Raises
        ------
        DuplicateKeyError
            If multiple different multiple estimation syntax types are used

        Examples
        --------
        >>> parser = MultipleEstimationParser()
        >>> vars, me_type = parser.parse_variable_list("x1 + x2")
        >>> vars
        ['x1', 'x2']
        >>> me_type is None
        True

        >>> vars, me_type = parser.parse_variable_list("x1 + sw(x2, x3)")
        >>> 'x1' in vars and 'x2' in vars and 'x3' in vars
        True
        >>> me_type
        <MultipleEstimationType.STEPWISE: 'sw'>
        """
        variables = []
        multiple_type = None

        for var_part in variable_string.split("+"):
            var_part = var_part.strip()

            # Check for multiple estimation syntax
            parsed_vars, me_type = self._parse_multiple_estimation_syntax(var_part)

            if me_type is not None:
                if multiple_type is not None:
                    raise DuplicateKeyError(
                        f"Multiple estimation syntax can only be used once. "
                        f"Found both {multiple_type.value} and {me_type.value}"
                    )
                multiple_type = me_type
                variables.extend(parsed_vars)
            else:
                variables.append(var_part)

        return variables, multiple_type

    def _parse_multiple_estimation_syntax(self, var_string: str) -> Tuple[List[str], Optional[MultipleEstimationType]]:
        """
        Parse individual variable string for multiple estimation syntax.

        This method checks a single variable string for multiple estimation
        patterns and extracts the variables and syntax type.

        Parameters
        ----------
        var_string : str
            Individual variable string to parse

        Returns
        -------
        Tuple[List[str], Optional[MultipleEstimationType]]
            Tuple of (variables, estimation_type)
        """
        # Define patterns in order of precedence (more specific first)
        patterns = {
            r'sw0\((.*?)\)': MultipleEstimationType.STEPWISE_ZERO,
            r'csw0\((.*?)\)': MultipleEstimationType.CUMULATIVE_STEPWISE_ZERO,
            r'sw\((.*?)\)': MultipleEstimationType.STEPWISE,
            r'csw\((.*?)\)': MultipleEstimationType.CUMULATIVE_STEPWISE,
        }

        for pattern, me_type in patterns.items():
            match = re.search(pattern, var_string)
            if match:
                variable_list = [v.strip() for v in match.group(1).split(',')]
                return variable_list, me_type

        return [var_string], None

    def expand_formula_combinations(
        self,
        base_variables: List[str],
        multiple_variables: List[str],
        multiple_type: MultipleEstimationType
    ) -> List[List[str]]:
        """
        Expand multiple estimation syntax into all formula combinations.

        This method combines base variables (that appear in all models) with
        variables subject to multiple estimation to create all possible
        formula combinations.

        Parameters
        ----------
        base_variables : List[str]
            Variables that appear in all models
        multiple_variables : List[str]
            Variables subject to multiple estimation
        multiple_type : MultipleEstimationType
            Type of multiple estimation

        Returns
        -------
        List[List[str]]
            All variable combinations for the different models

        Examples
        --------
        >>> parser = MultipleEstimationParser()
        >>> combinations = parser.expand_formula_combinations(
        ...     ["x1"], ["x2", "x3"], MultipleEstimationType.CUMULATIVE_STEPWISE
        ... )
        >>> combinations
        [['x1', 'x2'], ['x1', 'x2', 'x3']]
        """
        strategy = self.strategies[multiple_type]
        include_zero = multiple_type in [
            MultipleEstimationType.STEPWISE_ZERO,
            MultipleEstimationType.CUMULATIVE_STEPWISE_ZERO
        ]

        variable_combinations = strategy.expand_variables(multiple_variables, include_zero)

        # Combine with base variables
        result = []
        for var_combo in variable_combinations:
            combined = base_variables + var_combo
            # Remove empty strings and duplicates while preserving order
            final_vars = []
            seen = set()
            for var in combined:
                if var and var not in seen:
                    final_vars.append(var)
                    seen.add(var)
            result.append(final_vars)

        return result

    def has_multiple_estimation_syntax(self, variable_string: str) -> bool:
        """
        Check if a variable string contains multiple estimation syntax.

        Parameters
        ----------
        variable_string : str
            String to check for multiple estimation syntax

        Returns
        -------
        bool
            True if multiple estimation syntax is found
        """
        _, me_type = self.parse_variable_list(variable_string)
        return me_type is not None

    def get_supported_syntax(self) -> List[str]:
        """
        Get list of supported multiple estimation syntax patterns.

        Returns
        -------
        List[str]
            List of supported syntax patterns
        """
        return ["sw()", "sw0()", "csw()", "csw0()"]