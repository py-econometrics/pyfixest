"""Backward-compatible shim for FixestMulti.

Prefer importing from `pyfixest.estimation.fixest_multi`.
"""

from pyfixest.estimation.fixest_multi import FixestMulti

__all__ = ["FixestMulti"]
