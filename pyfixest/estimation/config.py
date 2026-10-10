from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.literals import (
    EstimationMethod,
    QuantregMultiOptions,
)
from pyfixest.estimation.internals.model_state import EstimationOptions, VcovSpec


@dataclass(frozen=True)
class QuantileProcess:
    """The quantiles one ``quantreg()`` call fits jointly, and the algorithm.

    Each child fit carries its own quantile in its options.
    """

    quantiles: list[float]
    multi_method: QuantregMultiOptions


@dataclass(frozen=True)
class EstimationConfig:
    """Everything a user asked for in one call to `feols()`, `feglm()`, etc.

    The API function checks the user's arguments and stores the cleaned-up
    values here; the runner then fits models from this record alone.

    Attributes
    ----------
    method
        Which kind of model to fit, e.g. "feols" or "fepois".
    data, formulas
        The user's data and parsed, expanded single-model formulas.
    options
        The estimation settings. Every fitted model keeps a copy.
    vcov
        The parsed variance-covariance choice.
    split, fsplit
        Optional variable to fit the model separately by group.
    quantile_process
        For `quantreg()` with several quantiles only.
    """

    method: EstimationMethod
    data: Any
    formulas: tuple[Formula, ...]
    options: EstimationOptions
    vcov: VcovSpec
    split: str | None = None
    fsplit: str | None = None
    quantile_process: QuantileProcess | None = None
