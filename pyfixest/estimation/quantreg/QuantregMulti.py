"""Backward-compatible shim for QuantregMulti.

Prefer importing from `pyfixest.estimation.quantreg.quantreg_multi`.
"""

from pyfixest.estimation.quantreg.quantreg_multi import QuantregMulti

__all__ = ["QuantregMulti"]
