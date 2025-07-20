"""
Core types and data structures for formula parsing.

This module contains the fundamental data types used throughout the formula
parsing system, including enums and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)


class FormulaTokenType(Enum):
    """Types of formula components."""
    DEPVAR = "depvar"
    COVAR = "covar"
    FIXEF = "fixef"
    ENDOG = "endog"
    INSTRUMENT = "instrument"


class MultipleEstimationType(Enum):
    """Types of multiple estimation syntax."""
    STEPWISE = "sw"
    STEPWISE_ZERO = "sw0"
    CUMULATIVE_STEPWISE = "csw"
    CUMULATIVE_STEPWISE_ZERO = "csw0"


@dataclass
class FormulaTokens:
    """
    Clean data structure for parsed formula components.

    This class represents the parsed components of a formula string,
    providing a structured way to access different parts of the formula
    along with validation methods.

    Attributes
    ----------
    depvars : List[str]
        Dependent variables
    covars : List[str]
        Covariate variables
    fixef : List[str]
        Fixed effect variables
    endogenous : Optional[List[str]]
        Endogenous variables (for IV)
    instruments : Optional[List[str]]
        Instrumental variables (for IV)
    """
    depvars: List[str]
    covars: List[str]
    fixef: List[str] = field(default_factory=list)
    endogenous: Optional[List[str]] = None
    instruments: Optional[List[str]] = None

    def __post_init__(self):
        """Validate formula components after initialization."""
        self.validate()

    @property
    def is_iv(self) -> bool:
        """Check if this represents an IV specification."""
        return self.endogenous is not None and self.instruments is not None

    @property
    def has_fixed_effects(self) -> bool:
        """Check if formula includes fixed effects."""
        return bool(self.fixef)

    def validate(self) -> None:
        """
        Validate formula components for logical consistency.

        This method performs comprehensive validation of the formula components
        to ensure they form a valid econometric specification.

        Note: Endogenous variables are expected to appear in the covariates list
        following R fixest convention, so we only check that instruments don't
        appear as covariates.

        Raises
        ------
        UnderDeterminedIVError
            If there are more endogenous variables than instruments
        InstrumentsAsCovarsError
            If instruments appear as covariates
        """
        # Validate IV specification
        if self.endogenous and self.instruments:
            if len(self.endogenous) > len(self.instruments):
                raise UnderDeterminedIVError(
                    f"IV system is underdetermined:\n"
                    f"  Endogenous variables ({len(self.endogenous)}): {', '.join(self.endogenous)}\n"
                    f"  Instruments ({len(self.instruments)}): {', '.join(self.instruments)}\n"
                    f"Need at least as many instruments as endogenous variables."
                )

        # Check for instruments in covariates (endogenous vars in covariates is expected)
        if self.instruments:
            instruments_in_covars = set(self.instruments) & set(self.covars)
            if instruments_in_covars:
                raise InstrumentsAsCovarsError(
                    f"Instruments cannot appear as covariates:\n"
                    f"  Variables in both: {', '.join(instruments_in_covars)}"
                )