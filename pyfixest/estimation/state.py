"""Public components of fitted estimator state.

Components expose fitted state for inspection. Mutating their contents is
unsupported and may invalidate results.
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
