"""
Formula parsing submodule for PyFixest.

This submodule provides a clean, well-architected formula parser with
comprehensive error handling, multiple estimation support, and clear
separation of concerns.

Main Components
---------------
- FixestFormulaParser: Main entry point for formula parsing
- FixestFormula: Individual formula representation
- FormulaTokenizer: Low-level formula string parsing
- FormulaValidator: Centralized validation logic
- MultipleEstimationParser: Multiple estimation syntax handling

Examples
--------
>>> from pyfixest.estimation.formula import FixestFormulaParser
>>> parser = FixestFormulaParser("y ~ x1 + x2 | firm_id")
>>> parser.formulas[0].fml
'y~x1+x2|firm_id'
"""

# Main public API
from .parser import FixestFormulaParser

# Core components (for advanced users)
from .core import FixestFormula, FormulaTokenizer, FormulaTokens

# Validation (for testing and debugging)
from .validation import FormulaValidator

# Multiple estimation (for advanced use cases)
from .multiple_estimation import MultipleEstimationParser

# Backward compatibility alias
FixestFormulaParser2 = FixestFormulaParser

__all__ = [
    # Main public interface
    "FixestFormulaParser",
    "FixestFormulaParser2",  # Backward compatibility

    # Core components
    "FixestFormula",
    "FormulaTokenizer",
    "FormulaTokens",

    # Advanced components
    "FormulaValidator",
    "MultipleEstimationParser",
]

# Version info for this submodule
__version__ = "2.0.0"
__author__ = "PyFixest Development Team"