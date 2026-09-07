from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.estimation.internals.literals import WeightsTypeOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationWeights:
    """Canonical observation weights retained by a fitted model.

    ``values`` are always user-scale weights; ``None`` means no weights.

    Parameters
    ----------
    values : NDArray[np.float64] or None
        Flat, user-scale observation weights. ``None`` for an unweighted fit.
    kind : {"aweights", "fweights"} or None
        Weight type. ``None`` for an unweighted fit.
    n_rows : int
        Number of physical rows used for estimation.
    n_effective : int or float
        Effective observation count: ``n_rows`` for unweighted fits and
        analytic weights, and ``sum(values)`` for frequency weights.
    """

    values: NDArray[np.float64] | None
    kind: WeightsTypeOptions | None
    n_rows: int
    n_effective: int | float

    def __post_init__(self) -> None:
        # `unweighted()` and `from_values()` are the only constructors used by
        # the estimators; these two guards catch direct misconstruction.
        if self.values is not None and self.kind is None:
            raise ValueError("Weighted observations must declare a weight kind.")
        if self.values is not None and len(self.values) != self.n_rows:
            raise ValueError("Observation weights must contain one value per row.")

    @classmethod
    def unweighted(cls, *, n_rows: int) -> ObservationWeights:
        """Construct the representation of an unweighted fit."""
        return cls(
            values=None,
            kind=None,
            n_rows=n_rows,
            n_effective=n_rows,
        )

    @classmethod
    def from_values(
        cls,
        weights: NDArray[np.float64],
        *,
        kind: WeightsTypeOptions,
    ) -> ObservationWeights:
        """Construct canonical weighted state from user-scale weights."""
        observation_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        n_rows = len(observation_weights)
        n_effective = (
            n_rows if kind == "aweights" else float(np.sum(observation_weights))
        )
        return cls(
            values=observation_weights,
            kind=kind,
            n_rows=n_rows,
            n_effective=n_effective,
        )

    @property
    def is_weighted(self) -> bool:
        """Whether this state contains user-supplied observation weights."""
        return self.values is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinLinearData:
    """Linear-model arrays after within transformation, in original units.

    These arrays have not been multiplied by square-root observation weights.
    For IV models, ``design`` is the full structural regressor matrix and may
    include endogenous regressors. ``instruments`` is the full instrument
    matrix, including exogenous regressors that instrument themselves.
    """

    response: NDArray[np.float64]
    design: NDArray[np.float64]
    instruments: NDArray[np.float64] | None = None
    endogenous: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinIvData(WithinLinearData):
    """Within-scale IV arrays whose instrument and endogenous roles are present."""

    instruments: NDArray[np.float64]
    endogenous: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class GlmWorkingState:
    """Final GLM IRLS state in within scale.

    ``working_weights`` are the final IRLS weights themselves.  Square-root
    weighted arrays are solver-local temporaries and deliberately absent.
    """

    working_response_within: NDArray[np.float64]
    design_within: NDArray[np.float64]
    working_weights: NDArray[np.float64]
    eta: NDArray[np.float64]
    mu: NDArray[np.float64]
    response_residuals: NDArray[np.float64]
    working_residuals: NDArray[np.float64]
