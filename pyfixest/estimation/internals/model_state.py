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
    weights_type : {"aweights", "fweights"} or None
        Weight type. ``None`` for an unweighted fit.
    n_rows : int
        Number of physical rows used for estimation.
    n_effective : int or float
        Effective observation count: ``n_rows`` for unweighted fits and
        analytic weights, and ``sum(values)`` for frequency weights.

    Notes
    -----
    Arrays are exposed for inspection. Mutating their contents is unsupported
    and may invalidate fitted results.
    See [fitted state](/how-to/fitted-state.qmd).

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
    n_rows: int
    n_effective: int | float

    def __post_init__(self) -> None:
        # `unweighted()` and `from_values()` are the only constructors used by
        # the estimators; these two guards catch direct misconstruction.
        if self.values is not None and self.weights_type is None:
            raise ValueError("Weighted observations must declare a `weights_type`.")
        if self.values is not None and len(self.values) != self.n_rows:
            raise ValueError("Observation weights must contain one value per row.")

    @classmethod
    def unweighted(cls, *, n_rows: int) -> ObservationWeights:
        """Construct the representation of an unweighted fit."""
        return cls(
            values=None,
            weights_type=None,
            n_rows=n_rows,
            n_effective=n_rows,
        )

    @classmethod
    def from_values(
        cls,
        weights: NDArray[np.float64],
        *,
        weights_type: WeightsTypeOptions,
    ) -> ObservationWeights:
        """Construct canonical weighted state from user-scale weights."""
        observation_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        n_rows = len(observation_weights)
        n_effective = (
            n_rows if weights_type == "aweights" else float(np.sum(observation_weights))
        )
        return cls(
            values=observation_weights,
            weights_type=weights_type,
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
    Arrays are exposed for inspection. Mutating their contents is unsupported
    and may invalidate fitted results.
    See [fitted state](/how-to/fitted-state.qmd).

    Parameters
    ----------
    response : NDArray[np.float64]
        Within-scale response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Selected within-scale structural design, shape (n_rows, n_coefficients).

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
    """Within-scale IV arrays with instrument and endogenous roles.

    ``design`` is the full structural regressor matrix, including the
    endogenous regressors. ``instruments`` is the full instrument matrix,
    including exogenous regressors that instrument themselves.
    See [fitted state](/how-to/fitted-state.qmd).

    Parameters
    ----------
    response : NDArray[np.float64]
        Within-scale structural response, shape (n_rows, 1).
    design : NDArray[np.float64]
        Selected structural design, shape (n_rows, n_coefficients).
    instruments : NDArray[np.float64]
        Selected full instrument matrix, shape (n_rows, n_instruments).
    endogenous : NDArray[np.float64]
        Endogenous regressor, shape (n_rows, 1).

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
    solver-local temporaries and deliberately absent. Arrays are exposed for
    inspection; mutating their contents may invalidate fitted results and is
    unsupported.
    See [fitted state](/how-to/fitted-state.qmd).

    Parameters
    ----------
    working_response_within : NDArray[np.float64]
        Final within-scale working response, shape (n_rows,).
    design_within : NDArray[np.float64]
        Selected within-scale working design, shape (n_rows, n_coefficients).
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
