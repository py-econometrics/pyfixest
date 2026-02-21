"""Centralized compatibility exports for legacy `pyfixest.estimation` modules."""

from pyfixest.estimation.compat.feiv import Feiv
from pyfixest.estimation.compat.feols import (
    Feols,
    _check_vcov_input,
    _deparse_vcov_input,
)
from pyfixest.estimation.compat.fepois import Fepois
from pyfixest.estimation.compat.fixest_multi import FixestMulti
from pyfixest.estimation.compat.quantreg_multi import QuantregMulti

__all__ = [
    "Feiv",
    "Feols",
    "Fepois",
    "FixestMulti",
    "QuantregMulti",
    "_check_vcov_input",
    "_deparse_vcov_input",
]

