"""Backward-compatible shim. Real module at pyfixest.estimation.models.feols_."""

from pyfixest.estimation.models.feols_ import *  # noqa: F403
from pyfixest.estimation.models.feols_ import (  # noqa: F401
    Feols,
    _check_vcov_input,
    _deparse_vcov_input,
)
