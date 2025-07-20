"""
Individual formula representation and manipulation.

This module contains the FixestFormula class that represents a single,
well-defined econometric model specification without multiple estimation syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FixestFormula:
    """
    Immutable representation of a single regression formula.

    This class represents a single, well-defined econometric model specification
    without any multiple estimation syntax. It provides methods for generating
    formula strings and handling first/second stage formulas for IV models.

    Parameters
    ----------
    depvar : str
        The dependent variable
    covars : List[str]
        List of covariate variables
    fixed_effects : List[str]
        List of fixed effect variables
    endogenous : Optional[List[str]]
        List of endogenous variables (for IV models)
    instruments : Optional[List[str]]
        List of instrumental variables (for IV models)

    Attributes
    ----------
    depvar : str
        The dependent variable
    covars : List[str]
        List of covariate variables
    fixed_effects : List[str]
        List of fixed effect variables
    endogenous : Optional[List[str]]
        List of endogenous variables (for IV models)
    instruments : Optional[List[str]]
        List of instrumental variables (for IV models)
    """

    depvar: str
    covars: List[str]
    fixed_effects: List[str] = field(default_factory=list)
    endogenous: Optional[List[str]] = None
    instruments: Optional[List[str]] = None

    def __post_init__(self):
        """Validate formula after initialization."""
        if not self.depvar:
            raise ValueError("Dependent variable cannot be empty")

        # Convert fixed_effects "0" to empty list for consistency
        if self.fixed_effects == ["0"]:
            self.fixed_effects = []

    @property
    def fml(self) -> str:
        """
        Generate clean Wilkinson formula string.

        Returns
        -------
        str
            A properly formatted formula string without spaces
        """
        return self._build_formula_string()

    @property
    def is_iv(self) -> bool:
        """
        Check if this is an IV specification.

        Returns
        -------
        bool
            True if this formula includes endogenous variables and instruments
        """
        return self.endogenous is not None and self.instruments is not None

    @property
    def has_fixed_effects(self) -> bool:
        """
        Check if formula includes fixed effects.

        Returns
        -------
        bool
            True if this formula includes fixed effects
        """
        return bool(self.fixed_effects)

    @property
    def _fval(self) -> str:
        """
        Get fixed effects as a string for backward compatibility.

        This property provides backward compatibility with existing code
        that expects the _fval attribute.

        Returns
        -------
        str
            Fixed effects as a "+"-separated string, or "0" if no fixed effects
        """
        if self.fixed_effects:
            return "+".join(self.fixed_effects)
        else:
            return "0"

    def _build_formula_string(self) -> str:
        """
        Build clean, consistent formula string.

        This method constructs a properly formatted formula string following
        the Wilkinson notation with appropriate separators.

        Returns
        -------
        str
            Formatted formula string
        """
        # Main part: depvar ~ covars
        covar_str = " + ".join(self.covars) if self.covars else "1"
        formula = f"{self.depvar} ~ {covar_str}"

        # Add fixed effects if present
        if self.fixed_effects:
            fe_str = " + ".join(self.fixed_effects)
            formula += f"|{fe_str}"

        # Add IV specification if present
        if self.is_iv:
            endog_str = "+".join(self.endogenous)  # type: ignore
            inst_str = "+".join(self.instruments)  # type: ignore
            formula += f"|{endog_str}~{inst_str}"

        return formula.replace(" ", "")  # Remove spaces for consistency

    def get_first_and_second_stage_fml(self) -> Tuple[str, Optional[str]]:
        """
        Generate first and second stage formulas for IV estimation.

        This method creates the appropriate formulas for two-stage least squares
        estimation. For non-IV models, only the second stage formula is returned.

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing (second_stage_formula, first_stage_formula).
            The first_stage_formula is None for non-IV models.

        Examples
        --------
        For OLS model:
        >>> formula = FixestFormula(depvar="y", covars=["x1", "x2"])
        >>> second, first = formula.get_first_and_second_stage_fml()
        >>> second
        'y ~ x1 + x2 + 1'
        >>> first is None
        True

        For IV model:
        >>> formula = FixestFormula(
        ...     depvar="y",
        ...     covars=["endog", "x1"],
        ...     endogenous=["endog"],
        ...     instruments=["z1", "z2"]
        ... )
        >>> second, first = formula.get_first_and_second_stage_fml()
        >>> second
        'y ~ endog + x1 + 1'
        >>> first
        'endog ~ z1 + z2 + x1 + 1'
        """
        # Second stage: depvar ~ covars + 1
        covar_str = " + ".join(self.covars) if self.covars else ""
        second_stage = f"{self.depvar} ~ {covar_str} + 1"

        # First stage only for IV models
        first_stage = None
        if self.is_iv:
            # First stage: endogvar ~ instruments + other_covars + 1
            # Remove endogenous variables from covariates for first stage
            other_covars = [c for c in self.covars if c not in (self.endogenous or [])]
            endog_str = " + ".join(self.endogenous)  # type: ignore
            inst_str = " + ".join(self.instruments)  # type: ignore
            other_str = " + ".join(other_covars) if other_covars else ""

            first_stage = f"{endog_str} ~ {inst_str}"
            if other_str:
                first_stage += f" + {other_str}"
            first_stage += " + 1"

        return second_stage, first_stage

    @property
    def fml_second_stage(self) -> str:
        """
        Get the second stage formula string.

        Returns
        -------
        str
            Second stage formula string
        """
        second_stage, _ = self.get_first_and_second_stage_fml()
        return second_stage

    @property
    def fml_first_stage(self) -> Optional[str]:
        """
        Get the first stage formula string.

        Returns
        -------
        Optional[str]
            First stage formula string, None for non-IV models
        """
        _, first_stage = self.get_first_and_second_stage_fml()
        return first_stage

    def get_covariate_names(self) -> List[str]:
        """
        Get list of all covariate names in the formula.

        Returns
        -------
        List[str]
            List of covariate variable names
        """
        return self.covars.copy()

    def get_fixed_effect_names(self) -> List[str]:
        """
        Get list of all fixed effect names in the formula.

        Returns
        -------
        List[str]
            List of fixed effect variable names
        """
        return self.fixed_effects.copy()

    def get_endogenous_names(self) -> List[str]:
        """
        Get list of endogenous variable names (if any).

        Returns
        -------
        List[str]
            List of endogenous variable names, empty list if not an IV model
        """
        return self.endogenous.copy() if self.endogenous else []

    def get_instrument_names(self) -> List[str]:
        """
        Get list of instrumental variable names (if any).

        Returns
        -------
        List[str]
            List of instrumental variable names, empty list if not an IV model
        """
        return self.instruments.copy() if self.instruments else []