from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.estimation.internals.literals import WeightsTypeOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationWeights:
    """Canonical observation weights retained by a fitted model.

    ``values`` are always user-scale weights; ``None`` means no weights.
    Observation counts, including the frequency-weight sum, live in
    ``SampleInfo``.

    Parameters
    ----------
    values : NDArray[np.float64] or None
        Flat, user-scale observation weights. ``None`` for an unweighted fit.
    weights_type : {"aweights", "fweights"} or None
        Weight type. ``None`` for an unweighted fit.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1", pf.get_data(), weights="weights")
    fit.observation_weights.values[:3]
    ```
    """

    values: NDArray[np.float64] | None
    weights_type: WeightsTypeOptions | None

    def __post_init__(self) -> None:
        # `unweighted()` and `from_values()` are the only constructors used by
        # the estimators; this guard catches direct misconstruction.
        if self.values is not None and self.weights_type is None:
            raise ValueError("Weighted observations must declare a `weights_type`.")

    @classmethod
    def unweighted(cls) -> ObservationWeights:
        """Construct the representation of an unweighted fit."""
        return cls(values=None, weights_type=None)

    @classmethod
    def from_values(
        cls,
        weights: NDArray[np.float64],
        *,
        weights_type: WeightsTypeOptions,
    ) -> ObservationWeights:
        """Construct canonical weighted state from user-scale weights."""
        observation_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        return cls(values=observation_weights, weights_type=weights_type)

    @property
    def is_weighted(self) -> bool:
        """Whether this state contains user-supplied observation weights."""
        return self.values is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class DroppedRowCounts:
    """Rows dropped from the fitted sample, counted by the stage that dropped them.

    Each stage counts only the rows it newly dropped from the rows that survived
    the earlier stages, so the counts sum to the number of dropped rows and
    never double-count. Stages run in field order: formula missing-value
    handling, the nonfinite filter, singleton fixed-effect removal, and GLM
    separation. A sample split selects each child's input rows before these
    stages and is not a drop.

    Parameters
    ----------
    missing : int
        Rows with a missing value in any formula variable.
    nonfinite : int
        Rows with an infinite value in a materialized column.
    singleton : int
        Rows removed as singleton fixed-effect levels (``fixef_rm="singleton"``).
    separation : int
        Rows removed by the GLM separation check.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.sample.dropped_by_stage
    ```
    """

    missing: int = 0
    nonfinite: int = 0
    singleton: int = 0
    separation: int = 0

    @property
    def total(self) -> int:
        """Number of dropped rows over all stages."""
        return self.missing + self.nonfinite + self.singleton + self.separation


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleInfo:
    """The row sample a model was fitted on.

    The estimator builds it with ``from_weights`` from the dropped-row
    bookkeeping of its ``ModelMatrix`` and its observation weights. The
    estimation functions discard the index of the user's data, so
    ``dropped_positions`` count from zero in the frame the estimator
    received, after any sample split; the demeaning cache keys a row sample
    by the same set. It is scalar-sized apart from that set and survives
    every storage option.

    Parameters
    ----------
    dropped_positions : frozenset[int]
        Positions of the rows dropped by any filtering stage.
    n_rows : int
        Number of physical fitted rows.
    n_obs : int or float
        Number of observations as fixest's ``nobs``: ``n_rows`` for
        unweighted fits and analytic weights, and the weight sum for
        frequency weights.
    dropped_by_stage : DroppedRowCounts
        Dropped rows by filtering stage; their total equals
        ``len(dropped_positions)``.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.sample.n_rows, fit.sample.n_obs, fit.sample.dropped_by_stage.missing
    ```
    """

    dropped_positions: frozenset[int]
    n_rows: int
    n_obs: int | float
    dropped_by_stage: DroppedRowCounts

    def __post_init__(self) -> None:
        if self.dropped_by_stage.total != len(self.dropped_positions):
            raise ValueError(
                "Dropped-row counts must sum to the number of dropped positions."
            )

    @classmethod
    def from_weights(
        cls,
        *,
        n_rows: int,
        dropped_positions: frozenset[int],
        dropped_by_stage: DroppedRowCounts,
        weights: ObservationWeights,
    ) -> SampleInfo:
        """Describe the fitted sample from dropped-row bookkeeping and observation weights.

        fixest counts a frequency weight as that many repeated rows, so
        ``n_obs`` is the weight sum; otherwise it is the row count.
        """
        if weights.values is not None and len(weights.values) != n_rows:
            raise ValueError("Observation weights must contain one value per row.")
        n_obs: int | float = n_rows
        if weights.weights_type == "fweights":
            assert weights.values is not None
            n_obs = float(weights.values.sum())
        return cls(
            dropped_positions=dropped_positions,
            n_rows=n_rows,
            n_obs=n_obs,
            dropped_by_stage=dropped_by_stage,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinLinearData:
    """Response and regressors after demeaning by the fixed effects.

    Without fixed effects, the arrays retain their original values. When
    observation weights are supplied, demeaning uses those weights.
    These arrays have not been multiplied by square-root observation weights.

    Parameters
    ----------
    response : NDArray[np.float64]
        Demeaned response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Demeaned regressors after removing collinear columns,
        shape (n_rows, n_coefficients).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.within_data.design.shape
    ```
    """

    response: NDArray[np.float64]
    design: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class WithinIvData(WithinLinearData):
    """Response, regressors, and instruments after demeaning by the fixed effects.

    ``design`` is the full structural regressor matrix, including the
    endogenous regressors. ``instruments`` is the full instrument matrix,
    including exogenous regressors that instrument themselves.
    Every array is demeaned using the observation weights when supplied, but
    is not multiplied by their square roots. Without fixed effects, the
    arrays retain their original values.

    Parameters
    ----------
    response : NDArray[np.float64]
        Demeaned response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Demeaned regressors, including endogenous regressors, after removing
        collinear columns, shape (n_rows, n_coefficients).
    instruments : NDArray[np.float64]
        Demeaned instruments, including exogenous regressors, after removing
        collinear columns, shape (n_rows, n_instruments).
    endogenous : NDArray[np.float64]
        Demeaned endogenous regressor, shape (n_rows, 1).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", pf.get_data())
    fit.within_data.instruments.shape
    ```
    """

    instruments: NDArray[np.float64]
    endogenous: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class GlmWorkingState:
    """Final GLM IRLS state in within scale.

    ``working_weights`` are the IRLS weights of the last iteration,
    ``W_i = w_i / (g'(mu_i)^2 V(mu_i))``, where ``w_i`` is the user-supplied
    observation weight (one when unweighted). They therefore already contain
    the observation weights, which live separately and unchanged in
    ``ObservationWeights``; nothing downstream multiplies by ``w`` again. For
    the Gaussian family ``W`` equals ``w``. Square-root weighted arrays are
    solver-local temporaries and deliberately absent.

    Parameters
    ----------
    working_response_within : NDArray[np.float64]
        Final within-scale working response, shape (n_rows,).
    design_within : NDArray[np.float64]
        Demeaned working regressors after removing collinear columns,
        shape (n_rows, n_coefficients).
    working_weights : NDArray[np.float64]
        Final IRLS weights including observation weights, shape (n_rows,).
    eta : NDArray[np.float64]
        Linear predictor including any offset, shape (n_rows,).
    mu : NDArray[np.float64]
        Response mean, shape (n_rows,).
    response_residuals : NDArray[np.float64]
        Observed response minus its mean, shape (n_rows,).
    working_residuals : NDArray[np.float64]
        Within working response minus design times coefficients, shape (n_rows,).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feglm("Y ~ X1", pf.get_data(), family="gaussian")
    fit.working_state.mu[:3]
    ```
    """

    working_response_within: NDArray[np.float64]
    design_within: NDArray[np.float64]
    working_weights: NDArray[np.float64]
    eta: NDArray[np.float64]
    mu: NDArray[np.float64]
    response_residuals: NDArray[np.float64]
    working_residuals: NDArray[np.float64]
