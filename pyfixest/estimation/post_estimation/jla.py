"""Regression-specific post-estimation quantities calculated using the Johnson-Lindenstrauss approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyfixest.core import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.internals.jla import (
    LinearOperator,
    RandomizedDiagonalResult,
    approximate_diagonal,
)


@dataclass(frozen=True, slots=True)
class FixedEffectResidualProjection:
    """Apply the fixed-effect residual maker M_D."""

    fixed_effects: NDArray[np.int32]
    weights: NDArray[np.float64]
    demeaner: AnyDemeaner
    preconditioner: Preconditioner | None

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_observations, number_observations)``."""
        number_observations = self.fixed_effects.shape[0]
        return (number_observations, number_observations)

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the fixed-effect residual projection in working coordinates.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Column vectors in weighted working coordinates, with shape
            ``(number_observations, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Fixed-effect residualised column vectors in weighted working
            coordinates.

        Raises
        ------
        ValueError
            If the fixed-effect residualization does not converge.
        """
        square_root_weights = np.sqrt(self.weights)[:, None]
        residualised, converged, _ = self.demeaner.demean(
            x=right_hand_side / square_root_weights,
            flist=self.fixed_effects,
            weights=self.weights,
            cached_preconditioner=self.preconditioner,
        )
        if not converged:
            raise ValueError(
                f"Fixed-effect residualisation failed after {self.demeaner.fixef_maxiter} iterations"
            )
        return square_root_weights * residualised


@dataclass(frozen=True, slots=True)
class RegressionProjectionOperator:
    """Apply the complete regression projection in working coordinates."""

    fixed_effect_residual_projection: LinearOperator | None
    covariates: NDArray[np.float64]
    gramian: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_observations, number_observations)``."""
        number_observations = self.covariates.shape[0]
        return (number_observations, number_observations)

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the complete regression projection in working coordinates.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Column vectors in weighted working coordinates, with shape
            ``(number_observations, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Projection of the column vectors onto the combined fixed-effect
            and covariate column space.
        """
        residualised = (
            right_hand_side
            if self.fixed_effect_residual_projection is None
            else self.fixed_effect_residual_projection.apply(right_hand_side)
        )
        if self.covariates.shape[1] > 0:
            coefficients = np.linalg.solve(
                self.gramian, self.covariates.T @ residualised
            )
            residualised = residualised - self.covariates @ coefficients

        return right_hand_side - residualised


def leverage(
    *,
    regression_projection: RegressionProjectionOperator,
    number_probes: int,
    random_number_generator: np.random.Generator,
    compute_monte_carlo_standard_errors: bool = False,
) -> RandomizedDiagonalResult:
    """Approximate regression leverage values using random projections.

    For an orthogonal regression projection ``H``,

    ``diag(H) = diag(H @ H.T)``.

    Therefore, leverage values can be estimated from the squared row norms of
    random embeddings of ``H``. This is the leverage approximation used by
    [Kline, Saggio, and Sølvsten
    (2020)](https://doi.org/10.3982/ECTA16410), based on the Rademacher
    random-projection construction of [Achlioptas
    (2003)](https://doi.org/10.1016/S0022-0000(03)00025-4).

    Parameters
    ----------
    regression_projection : RegressionProjectionOperator
        Operator applying the complete regression projection in weighted
        working coordinates.
    number_probes : int
        Number of independent Rademacher probes.
    random_number_generator : np.random.Generator
        Random-number generator used to construct the probes.
    compute_monte_carlo_standard_errors : bool, optional
        Whether to estimate Monte Carlo standard errors.

    Returns
    -------
    RandomizedDiagonalResult
        Approximate leverage values and optional Monte Carlo standard errors.
    """
    return approximate_diagonal(
        left=regression_projection,
        number_probes=number_probes,
        random_number_generator=random_number_generator,
        compute_monte_carlo_standard_errors=compute_monte_carlo_standard_errors,
    )


def influence(
    *,
    residuals: NDArray[np.float64],
    leverage: RandomizedDiagonalResult,
) -> RandomizedDiagonalResult:
    """Calculate each observation's leave-one-out influence in an OLS model.

    For observation ``i``, the returned value is

    ``fitted_i - fitted_i_without_i = leverage_i * residual_i / (1 - leverage_i)``.

    The Monte Carlo standard errors use a first-order propagation of the
    algorithmic uncertainty in the randomized leverage approximation. They do
    not describe sampling uncertainty. See [Belsley, Kuh, and Welsch
    (1980)](https://doi.org/10.1002/0471725153) for regression deletion
    diagnostics.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        OLS residuals on the scale of the dependent variable, with shape
        ``(number_observations,)``.
    leverage : RandomizedDiagonalResult
        Leverage estimates and optional Monte Carlo standard errors.

    Returns
    -------
    RandomizedDiagonalResult
        Leave-one-out influence values and optional Monte Carlo standard
        errors.

    Raises
    ------
    ValueError
        If any estimated leverage is greater than or equal to one, in which
        case deleting the corresponding observation does not yield a defined
        fitted value.
    """
    denominator = 1 - leverage.estimate
    if np.any(denominator <= 0):
        raise ValueError(
            "Leave-one-out influence is undefined for observations with "
            "leverage greater than or equal to one."
        )

    estimate = residuals * leverage.estimate / denominator
    monte_carlo_standard_errors = (
        None
        if leverage.monte_carlo_standard_errors is None
        else np.abs(residuals) * leverage.monte_carlo_standard_errors / denominator**2
    )
    return RandomizedDiagonalResult(
        estimate=estimate,
        monte_carlo_standard_errors=monte_carlo_standard_errors,
        number_probes=leverage.number_probes,
    )
