"""Backward-compatible shim. Real module at pyfixest.estimation.models.feols_."""

from pyfixest.estimation.models.feols_ import *  # noqa: F403
from pyfixest.estimation.models.feols_ import Feols  # noqa: F401
from pyfixest.estimation.internals.vcov_utils import (  # noqa: F401
    _check_vcov_input,
    _deparse_vcov_input,
)
