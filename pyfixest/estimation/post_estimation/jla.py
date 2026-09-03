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


@dataclass(frozen=True, slots=True)
class DiagonalScoreCovarianceFactor:
    """Apply a diagonal score covariance factor."""

    number_observations: int
    scale: float | NDArray[np.float64] = 1.0

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale)
        if scale.ndim > 0 and scale.shape != (self.number_observations,):
            raise ValueError(
                f"scale must be a scalar or have shape ({self.number_observations},)."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_observations, number_scores)``."""
        return (self.number_observations, self.number_observations)

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale observation-level score perturbations.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Probe vectors with shape ``(number_observations, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Scaled score perturbations.
        """
        scale = np.asarray(self.scale, dtype=np.float64)
        return (
            float(scale) * right_hand_side
            if scale.ndim == 0
            else scale[:, None] * right_hand_side
        )


@dataclass(frozen=True, slots=True)
class ClusterScoreCovarianceFactor:
    """Expand cluster-level perturbations into observation-level scores."""

    score_scale: NDArray[np.float64]
    cluster_indices: NDArray[np.intp]
    number_clusters: int

    def __post_init__(self) -> None:
        if self.score_scale.ndim != 1:
            raise ValueError("score_scale must be one-dimensional.")
        number_observations = self.score_scale.shape[0]
        if self.cluster_indices.shape != (number_observations,):
            raise ValueError(
                "cluster_indices must have the same length as score_scale."
            )
        if self.number_clusters < 1:
            raise ValueError("number_clusters must be positive.")
        if np.any(self.cluster_indices < 0) or np.any(
            self.cluster_indices >= self.number_clusters
        ):
            raise ValueError(
                "cluster_indices must be between zero and number_clusters - 1."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_observations, number_clusters)``."""
        return (self.score_scale.shape[0], self.number_clusters)

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map cluster-level probes to scaled observation-level scores.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Probe vectors with shape ``(number_clusters, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Scaled score perturbations with one row per observation.
        """
        return self.score_scale[:, None] * right_hand_side[self.cluster_indices]


@dataclass(frozen=True, slots=True)
class IdentityOperator:
    """Return vectors without transforming them."""

    size: int

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("size must be non-negative.")

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(size, size)``."""
        return (self.size, self.size)

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one or more column vectors unchanged.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Column vectors with shape ``(size, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            The input column vectors.
        """
        return right_hand_side


@dataclass(frozen=True, slots=True)
class PredictionAveragingOperator:
    """Average observation-level quantities within groups."""

    group_indices: NDArray[np.intp]

    def __post_init__(self) -> None:
        if self.group_indices.ndim != 1:
            raise ValueError("group_indices must be one-dimensional.")
        if self.group_indices.size == 0:
            raise ValueError("group_indices must contain at least one observation.")
        if not np.issubdtype(self.group_indices.dtype, np.integer):
            raise ValueError("group_indices must contain integers.")
        if np.any(self.group_indices < 0):
            raise ValueError("group_indices must be non-negative.")
        if np.unique(self.group_indices).shape[0] != self.number_groups:
            raise ValueError("group_indices must be contiguous and zero-based.")

    @property
    def number_groups(self) -> int:
        """Number of groups."""
        return int(self.group_indices.max()) + 1

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_groups, number_observations)``."""
        return (self.number_groups, self.group_indices.shape[0])

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Average rows of one or more column vectors within groups.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Observation-level vectors with shape
            ``(number_observations, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Group averages with shape ``(number_groups, number_vectors)``.
        """
        group_sums = np.zeros(
            (self.number_groups, right_hand_side.shape[1]), dtype=np.float64
        )
        np.add.at(group_sums, self.group_indices, right_hand_side)
        group_counts = np.bincount(
            self.group_indices, minlength=self.number_groups
        ).astype(np.float64)
        return group_sums / group_counts[:, None]


@dataclass(frozen=True, slots=True)
class PredictionJacobian:
    """Compose the operators defining a prediction-derived quantity.

    The represented operator is ``prediction_to_quantity @``
    ``score_to_prediction @ score_covariance_factor``.
    """

    score_covariance_factor: LinearOperator
    score_to_prediction: LinearOperator
    prediction_to_quantity: LinearOperator

    def __post_init__(self) -> None:
        if self.score_to_prediction.shape[1] != self.score_covariance_factor.shape[0]:
            raise ValueError(
                "score_to_prediction input size must equal "
                "score_covariance_factor output size."
            )
        if self.prediction_to_quantity.shape[1] != self.score_to_prediction.shape[0]:
            raise ValueError(
                "prediction_to_quantity input size must equal "
                "score_to_prediction output size."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(number_predictions, number_scores)``."""
        return (
            self.prediction_to_quantity.shape[0],
            self.score_covariance_factor.shape[1],
        )

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map score perturbations to perturbations of fitted values.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Probe vectors with shape ``(number_scores, number_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Perturbations of the in-sample fitted values.
        """
        score_perturbations = self.score_covariance_factor.apply(right_hand_side)
        prediction_perturbations = self.score_to_prediction.apply(score_perturbations)
        return self.prediction_to_quantity.apply(prediction_perturbations)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """In-sample prediction estimates and statistical uncertainty.

    Attributes
    ----------
    estimate : NDArray[np.float64]
        Fitted values with shape ``(number_observations,)``.
    variance : RandomizedDiagonalResult
        JLA prediction-variance approximation and its numerical uncertainty.
    group_keys : tuple[tuple[object, ...], ...] | None
        Group keys corresponding to aggregated estimates. ``None`` for
        observation-level or overall-average predictions.
    """

    estimate: NDArray[np.float64]
    variance: RandomizedDiagonalResult
    group_keys: tuple[tuple[object, ...], ...] | None = None

    @property
    def standard_error(self) -> NDArray[np.float64]:
        """Calculate statistical standard errors of the predictions.

        Returns
        -------
        NDArray[np.float64]
            Prediction standard errors.
        """
        return np.sqrt(self.variance.estimate)


def prediction_uncertainty(
    *,
    fitted_values: NDArray[np.float64],
    prediction_jacobian: PredictionJacobian,
    number_probes: int,
    random_number_generator: np.random.Generator,
    group_keys: tuple[tuple[object, ...], ...] | None = None,
) -> PredictionResult:
    """Approximate prediction uncertainty from a prediction Jacobian.

    For a covariance-scaled prediction Jacobian ``J``, the prediction
    variances are ``diag(J @ J.T)``.

    Parameters
    ----------
    fitted_values : NDArray[np.float64]
        Prediction estimates.
    prediction_jacobian : PredictionJacobian
        Operator mapping standardized score perturbations to prediction
        perturbations.
    number_probes : int
        Number of independent Rademacher probes.
    random_number_generator : np.random.Generator
        Random-number generator used to construct the probes.
    group_keys : tuple[tuple[object, ...], ...] | None, optional
        Group keys corresponding to aggregated predictions.

    Returns
    -------
    PredictionResult
        Prediction estimates and their JLA variance approximation.

    Raises
    ------
    ValueError
        If fewer than two probes are requested or the Jacobian output size
        differs from the number of predictions.
    """
    if prediction_jacobian.shape[0] != fitted_values.shape[0]:
        raise ValueError(
            "prediction_jacobian output size must equal the number of predictions."
        )
    variance_approximation = approximate_diagonal(
        left=prediction_jacobian,
        number_probes=number_probes,
        random_number_generator=random_number_generator,
        compute_monte_carlo_standard_errors=True,
    )
    return PredictionResult(
        estimate=fitted_values,
        variance=variance_approximation,
        group_keys=group_keys,
    )


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
