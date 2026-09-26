from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyfixest.estimation.internals.literals import (
    EstimationMethod,
    QuantregMultiOptions,
)
from pyfixest.estimation.internals.model_state import EstimationOptions, VcovSpec


@dataclass(frozen=True)
class QuantileProcess:
    """The quantiles one ``quantreg()`` call fits jointly, and the algorithm.

    Each child fit carries its own quantile in its options; the fan-out itself
    is not an option of any single fit.
    """

    quantiles: list[float]
    multi_method: QuantregMultiOptions


@dataclass(frozen=True)
class EstimationConfig:
    """Immutable record of what one call of a public estimation function requests.

    The API function validates its arguments and builds the typed values:
    `options` is the value every fitted model publishes, and `vcov` is the
    parsed covariance estimator. The remaining fields say which model class
    to dispatch to and how the call expands into several models.
    """

    method: EstimationMethod
    data: Any
    fml: str
    options: EstimationOptions
    vcov: VcovSpec
    split: str | None = None
    fsplit: str | None = None
    # only for the joint fit of several quantiles ("quantreg_multi")
    quantile_process: QuantileProcess | None = None
