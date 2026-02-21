"""Backward-compatible shim for the historical `pyfixest.estimation.Feols` module."""

from pyfixest.estimation.models.feols_ import *  # noqa: F403
from pyfixest.estimation.models.feols_ import Feols  # noqa: F401

