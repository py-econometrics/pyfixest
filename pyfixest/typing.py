"""Public type aliases for pyfixest inputs, options, and fitted results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from narwhals.typing import IntoDataFrame

if TYPE_CHECKING:
    from pyfixest.estimation.FixestMulti_ import FixestMulti
    from pyfixest.estimation.models.feglm_ import Feglm
    from pyfixest.estimation.models.feiv_ import Feiv
    from pyfixest.estimation.models.feols_ import Feols
    from pyfixest.estimation.models.fepois_ import Fepois
    from pyfixest.estimation.quantreg.quantreg_ import Quantreg

DataFrameType: TypeAlias = IntoDataFrame

RegressionVcovType: TypeAlias = Literal[
    "iid", "hetero", "HC1", "HC2", "HC3", "NW", "DK"
]
QuantregVcovType: TypeAlias = Literal["iid", "hetero", "HC1", "HC2", "HC3", "nid"]
VcovKwargs: TypeAlias = dict[str, str | int]
SscConfig: TypeAlias = dict[str, str | bool]

PlotBackend: TypeAlias = Literal["matplotlib", "lets_plot"]
EventStudyEstimator: TypeAlias = Literal["twfe", "did2s", "saturated"]
WeightsType: TypeAlias = Literal["aweights", "fweights"]
GlmFamily: TypeAlias = Literal["gaussian", "logit", "probit", "poisson"]

if TYPE_CHECKING:
    ModelResult: TypeAlias = Feols | Fepois | Feiv | Feglm | Quantreg
    ModelInput: TypeAlias = ModelResult | FixestMulti | list[ModelResult]
else:
    # Keep importing this lightweight typing module from importing estimation
    # implementations at runtime. Static type checkers evaluate the branch above.
    ModelResult: TypeAlias = Any
    ModelInput: TypeAlias = Any

__all__ = [
    "DataFrameType",
    "EventStudyEstimator",
    "GlmFamily",
    "ModelInput",
    "ModelResult",
    "PlotBackend",
    "QuantregVcovType",
    "RegressionVcovType",
    "SscConfig",
    "VcovKwargs",
    "WeightsType",
]
