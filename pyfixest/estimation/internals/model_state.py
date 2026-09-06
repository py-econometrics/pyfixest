from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.estimation.internals.literals import WeightsTypeOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationWeights:
    """Canonical observation weights retained by a fitted model.

    ``values`` are always the user-scale weights, never their square roots or
    an estimator-specific working weight.  ``None`` is the explicit unweighted
    fast path: callers must not allocate a vector of ones merely to represent
    an unweighted fit.

    Parameters
    ----------
    values : NDArray[np.float64] or None
        Flat, user-scale observation weights. ``None`` for an unweighted fit.
    kind : {"aweights", "fweights"} or None
        Weight semantics. ``None`` for an unweighted fit.
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
        if self.n_rows < 0:
            raise ValueError("n_rows must be non-negative.")

        if self.values is None:
            if self.kind is not None:
                raise ValueError("Unweighted observations cannot have a weight kind.")
            if self.n_effective != self.n_rows:
                raise ValueError(
                    "Unweighted observations must have n_effective equal to n_rows."
                )
            return

        if self.kind is None:
            raise ValueError("Weighted observations must declare a weight kind.")
        if self.kind not in ("aweights", "fweights"):
            raise ValueError("Weight kind must be 'aweights' or 'fweights'.")
        if self.values.ndim != 1:
            raise ValueError("Observation weight values must be a flat array.")
        if len(self.values) != self.n_rows:
            raise ValueError("Observation weights must contain one value per row.")

        expected_n = (
            self.n_rows if self.kind == "aweights" else float(np.sum(self.values))
        )
        if self.n_effective != expected_n:
            raise ValueError("n_effective must match the observation-weight semantics.")

    @classmethod
    def unweighted(cls, *, n_rows: int) -> ObservationWeights:
        """Construct the allocation-free representation of an unweighted fit."""
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
