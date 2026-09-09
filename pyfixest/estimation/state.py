"""Public components of fitted estimator state.

Formula tables are detached on access; published numerical arrays are read-only.
See the [fitted-state guide](/how-to/fitted-state.qmd) for domains and retention.
"""

from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.internals.model_state import (
    GlmWorkingState,
    ObservationWeights,
    WithinIvData,
    WithinLinearData,
)

__all__ = [
    "GlmWorkingState",
    "ModelMatrix",
    "ObservationWeights",
    "WithinIvData",
    "WithinLinearData",
]
