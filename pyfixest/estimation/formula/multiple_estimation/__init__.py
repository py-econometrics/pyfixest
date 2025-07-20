"""
Multiple estimation components.

This submodule handles the parsing and expansion of multiple estimation syntax
like sw(), csw(), sw0(), and csw0().
"""

from .parser import MultipleEstimationParser
from .strategies import (
    CumulativeStepwiseStrategy,
    MultipleEstimationStrategy,
    StepwiseStrategy,
    StrategyFactory,
)

__all__ = [
    "MultipleEstimationParser",
    "MultipleEstimationStrategy",
    "StepwiseStrategy",
    "CumulativeStepwiseStrategy",
    "StrategyFactory",
]