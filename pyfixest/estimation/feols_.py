"""Backward-compatible shim. Real module at pyfixest.estimation.compat.feols."""

from pyfixest.estimation.compat.feols import *  # noqa: F403
from pyfixest.estimation.compat.feols import (  # noqa: F401
    Feols,
    _check_vcov_input,
    _deparse_vcov_input,
)
