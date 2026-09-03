from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import pyfixest as pf
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
from pyfixest.estimation.internals.jla import RandomizedDiagonalResult
from pyfixest.estimation.post_estimation import jla
from pyfixest.estimation.post_estimation.jla import (
    ClusterScoreCovarianceFactor,
    DiagonalScoreCovarianceFactor,
    FixedEffectResidualProjection,
    IdentityOperator,
    PredictionAveragingOperator,
    PredictionJacobian,
    PredictionResult,
    RegressionProjectionOperator,
    influence,
    leverage,
    prediction_uncertainty,
)

FixedEffectDesign = tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]


def _averaging_matrix(
    data: pd.DataFrame,
    average: bool | tuple[str, ...],
) -> tuple[NDArray[np.float64], tuple[tuple[object, ...], ...] | None]:
    number_observations = data.shape[0]
    if average is False:
        return np.eye(number_observations), None
    if average is True:
        return np.full((1, number_observations), 1 / number_observations), None

    group_index = pd.MultiIndex.from_frame(data.loc[:, list(average)])
    group_indices, unique_groups = pd.factorize(group_index, sort=False)
    group_membership = (
        group_indices[None, :] == np.arange(len(unique_groups))[:, None]
    ).astype(np.float64)
    averaging_matrix = group_membership / group_membership.sum(axis=1, keepdims=True)
    group_keys = tuple(tuple(key) for key in unique_groups.tolist())
    return averaging_matrix, group_keys


def _regression_projection(
    fixed_effect_design: FixedEffectDesign,
) -> tuple[RegressionProjectionOperator, NDArray[np.float64]]:
    fixed_effects, weights, dummy_matrix = fixed_effect_design
    square_root_weights = np.sqrt(weights)
    raw_covariates = np.column_stack(
        [
            np.linspace(-1.0, 2.0, fixed_effects.shape[0]),
            np.array([0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 8.0, 6.0, 7.0, 11.0, 9.0, 10.0]),
        ]
    )
    fixed_effect_projection = FixedEffectResidualProjection(
        fixed_effects=fixed_effects,
        weights=weights,
        demeaner=MapDemeaner(fixef_tol=1e-12),
        preconditioner=None,
    )
    transformed_covariates = fixed_effect_projection.apply(
        square_root_weights[:, None] * raw_covariates
    )
    projection = RegressionProjectionOperator(
        fixed_effect_residual_projection=fixed_effect_projection,
        covariates=transformed_covariates,
        gramian=transformed_covariates.T @ transformed_covariates,
    )
    weighted_design_matrix = square_root_weights[:, None] * np.column_stack(
        [dummy_matrix, raw_covariates]
    )
    return projection, weighted_design_matrix


@pytest.fixture
def fixed_effect_design() -> FixedEffectDesign:
    fixed_effects = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 0],
            [1, 1],
            [1, 2],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
        dtype=np.int32,
    )
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.75, 2.5, 1.25, 0.8, 1.8, 2.2, 1.1])
    dummy_matrix = np.column_stack(
        [
            fixed_effects[:, factor][:, None] == np.arange(3)
            for factor in range(fixed_effects.shape[1])
        ]
    ).astype(np.float64)
    return fixed_effects, weights, dummy_matrix


def test_fixed_effect_residual_projection_matches_weighted_dummy_projection(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    fixed_effects, weights, dummy_matrix = fixed_effect_design
    right_hand_side = np.arange(36, dtype=np.float64).reshape(12, 3) / 7
    square_root_weights = np.sqrt(weights)
    weighted_dummy_matrix = square_root_weights[:, None] * dummy_matrix

    projection = FixedEffectResidualProjection(
        fixed_effects=fixed_effects,
        weights=weights,
        demeaner=MapDemeaner(fixef_tol=1e-12),
        preconditioner=None,
    )

    actual = projection.apply(right_hand_side)
    expected = (
        right_hand_side
        - weighted_dummy_matrix
        @ np.linalg.pinv(weighted_dummy_matrix)
        @ right_hand_side
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Fixed-effect residual projection differs from dense reference.",
    )
    assert projection.shape == (12, 12)


def test_regression_projection_matches_weighted_design_projection(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, weighted_design_matrix = _regression_projection(fixed_effect_design)
    right_hand_side = np.column_stack(
        [
            np.linspace(-2.0, 1.0, projection.shape[0]),
            np.cos(np.arange(projection.shape[0])),
        ]
    )

    actual = projection.apply(right_hand_side)
    expected = (
        weighted_design_matrix
        @ np.linalg.pinv(weighted_design_matrix)
        @ right_hand_side
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Regression projection differs from dense weighted-design reference.",
    )
    assert projection.shape == (12, 12)


def test_leverage_approximates_dense_weighted_projection_diagonal(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, weighted_design_matrix = _regression_projection(fixed_effect_design)

    result = leverage(
        regression_projection=projection,
        number_probes=10_000,
        random_number_generator=np.random.default_rng(921),
        compute_monte_carlo_standard_errors=True,
    )

    dense_projection = weighted_design_matrix @ np.linalg.pinv(weighted_design_matrix)
    expected = np.diag(dense_projection)
    assert result.monte_carlo_standard_errors is not None
    assert result.monte_carlo_standard_errors.shape == expected.shape
    assert result.number_probes == 10_000
    np.testing.assert_array_less(
        np.abs(result.estimate - expected),
        5 * result.monte_carlo_standard_errors + 1e-10,
        err_msg=("Randomized leverage error exceeds five Monte Carlo standard errors."),
    )


def test_prediction_uncertainty_approximates_dense_projection_variance(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, weighted_design_matrix = _regression_projection(fixed_effect_design)
    fitted_values = np.linspace(-1.0, 1.0, projection.shape[0])
    residual_variance = 2.5

    result = prediction_uncertainty(
        fitted_values=fitted_values,
        prediction_jacobian=PredictionJacobian(
            score_covariance_factor=DiagonalScoreCovarianceFactor(
                number_observations=projection.shape[1],
                scale=np.sqrt(residual_variance),
            ),
            score_to_prediction=projection,
            prediction_to_quantity=IdentityOperator(size=projection.shape[0]),
        ),
        number_probes=10_000,
        random_number_generator=np.random.default_rng(741),
    )

    dense_projection = weighted_design_matrix @ np.linalg.pinv(weighted_design_matrix)
    expected_variance = residual_variance * np.diag(dense_projection)
    assert isinstance(result, PredictionResult)
    assert result.variance.number_probes == 10_000
    assert result.variance.monte_carlo_standard_errors is not None
    np.testing.assert_array_equal(result.estimate, fitted_values)
    np.testing.assert_array_less(
        np.abs(result.standard_error**2 - expected_variance),
        5 * result.variance.monte_carlo_standard_errors + 1e-10,
        err_msg=("Prediction variance error exceeds five Monte Carlo standard errors."),
    )


def test_prediction_jacobian_composes_score_covariance_factor(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, _ = _regression_projection(fixed_effect_design)
    score_scale = np.linspace(0.5, 1.5, projection.shape[1])
    right_hand_side = np.arange(projection.shape[1] * 2, dtype=np.float64).reshape(
        projection.shape[1], 2
    )
    jacobian = PredictionJacobian(
        score_covariance_factor=DiagonalScoreCovarianceFactor(
            number_observations=projection.shape[1],
            scale=score_scale,
        ),
        score_to_prediction=projection,
        prediction_to_quantity=IdentityOperator(size=projection.shape[0]),
    )

    expected = projection.apply(score_scale[:, None] * right_hand_side)
    np.testing.assert_allclose(jacobian.apply(right_hand_side), expected)


def test_identity_operator_returns_vectors_unchanged() -> None:
    right_hand_side = np.arange(8, dtype=np.float64).reshape(4, 2)
    identity = IdentityOperator(size=4)

    assert identity.shape == (4, 4)
    assert identity.apply(right_hand_side) is right_hand_side

    with pytest.raises(ValueError, match="size must be non-negative"):
        IdentityOperator(size=-1)


def test_prediction_jacobian_applies_averaging_operator(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, _ = _regression_projection(fixed_effect_design)
    group_indices = np.arange(projection.shape[0], dtype=np.intp) % 3
    averaging = PredictionAveragingOperator(
        group_indices=group_indices,
    )
    jacobian = PredictionJacobian(
        score_covariance_factor=DiagonalScoreCovarianceFactor(
            number_observations=projection.shape[1],
        ),
        score_to_prediction=projection,
        prediction_to_quantity=averaging,
    )
    right_hand_side = np.arange(projection.shape[1] * 2, dtype=np.float64).reshape(
        projection.shape[1], 2
    )

    expected = averaging.apply(projection.apply(right_hand_side))
    np.testing.assert_allclose(jacobian.apply(right_hand_side), expected)
    assert jacobian.shape == (3, projection.shape[1])


@pytest.mark.parametrize(
    ("group_indices", "message"),
    [
        pytest.param(
            np.ones((3, 1), dtype=np.intp),
            "group_indices must be one-dimensional",
            id="dimensions",
        ),
        pytest.param(
            np.array([], dtype=np.intp),
            "group_indices must contain at least one observation",
            id="empty",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            "group_indices must contain integers",
            id="dtype",
        ),
        pytest.param(
            np.array([-1, 0], dtype=np.intp),
            "group_indices must be non-negative",
            id="negative",
        ),
        pytest.param(
            np.array([0, 2, 2], dtype=np.intp),
            "group_indices must be contiguous and zero-based",
            id="noncontiguous",
        ),
    ],
)
def test_prediction_averaging_operator_rejects_invalid_inputs(
    group_indices: NDArray[np.intp] | NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictionAveragingOperator(
            group_indices=group_indices,
        )


def test_cluster_score_covariance_factor_expands_cluster_probes(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, _ = _regression_projection(fixed_effect_design)
    number_observations = projection.shape[1]
    cluster_indices = np.arange(number_observations, dtype=np.intp) % 3
    score_scale = np.linspace(-1.0, 1.0, number_observations)
    factor = ClusterScoreCovarianceFactor(
        score_scale=score_scale,
        cluster_indices=cluster_indices,
        number_clusters=3,
    )
    jacobian = PredictionJacobian(
        score_covariance_factor=factor,
        score_to_prediction=projection,
        prediction_to_quantity=IdentityOperator(size=projection.shape[0]),
    )
    cluster_probes = np.arange(6, dtype=np.float64).reshape(3, 2)

    expected_scores = score_scale[:, None] * cluster_probes[cluster_indices]
    np.testing.assert_array_equal(factor.apply(cluster_probes), expected_scores)
    np.testing.assert_allclose(
        jacobian.apply(cluster_probes), projection.apply(expected_scores)
    )


@pytest.mark.parametrize(
    ("score_scale", "cluster_indices", "number_clusters", "message"),
    [
        pytest.param(
            np.ones((3, 1)),
            np.arange(3, dtype=np.intp),
            3,
            "score_scale must be one-dimensional",
            id="score_scale",
        ),
        pytest.param(
            np.ones(3),
            np.arange(2, dtype=np.intp),
            3,
            "same length as score_scale",
            id="cluster_length",
        ),
        pytest.param(
            np.ones(3),
            np.zeros(3, dtype=np.intp),
            0,
            "number_clusters must be positive",
            id="number_clusters",
        ),
        pytest.param(
            np.ones(3),
            np.array([0, 1, 3], dtype=np.intp),
            3,
            "between zero and number_clusters - 1",
            id="cluster_range",
        ),
    ],
)
def test_cluster_score_covariance_factor_rejects_invalid_inputs(
    score_scale: NDArray[np.float64],
    cluster_indices: NDArray[np.intp],
    number_clusters: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClusterScoreCovarianceFactor(
            score_scale=score_scale,
            cluster_indices=cluster_indices,
            number_clusters=number_clusters,
        )


def test_prediction_jacobian_rejects_incompatible_operator_dimensions(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, _ = _regression_projection(fixed_effect_design)

    with pytest.raises(ValueError, match="scale must be a scalar or have shape"):
        DiagonalScoreCovarianceFactor(
            number_observations=projection.shape[1],
            scale=np.ones(2),
        )
    with pytest.raises(ValueError, match="input size must equal"):
        PredictionJacobian(
            score_covariance_factor=DiagonalScoreCovarianceFactor(
                number_observations=projection.shape[1] + 1,
            ),
            score_to_prediction=projection,
            prediction_to_quantity=IdentityOperator(size=projection.shape[0]),
        )
    with pytest.raises(ValueError, match="prediction_to_quantity input size"):
        PredictionJacobian(
            score_covariance_factor=DiagonalScoreCovarianceFactor(
                number_observations=projection.shape[1],
            ),
            score_to_prediction=projection,
            prediction_to_quantity=PredictionAveragingOperator(
                group_indices=np.zeros(projection.shape[0] - 1, dtype=np.intp),
            ),
        )


def test_prediction_uncertainty_rejects_invalid_jacobian(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    projection, _ = _regression_projection(fixed_effect_design)
    prediction_jacobian = PredictionJacobian(
        score_covariance_factor=DiagonalScoreCovarianceFactor(
            number_observations=projection.shape[1],
        ),
        score_to_prediction=projection,
        prediction_to_quantity=IdentityOperator(size=projection.shape[0]),
    )

    with pytest.raises(ValueError, match="output size"):
        prediction_uncertainty(
            fitted_values=np.ones(projection.shape[0] - 1),
            prediction_jacobian=prediction_jacobian,
            number_probes=2,
            random_number_generator=np.random.default_rng(1),
        )


def test_influence_transforms_leverage_and_uncertainty() -> None:
    residuals = np.array([2.0, -1.0, 0.5])
    leverage_result = RandomizedDiagonalResult(
        estimate=np.array([0.2, 0.5, 0.8]),
        monte_carlo_standard_errors=np.array([0.01, 0.02, 0.03]),
        number_probes=250,
    )

    result = influence(
        residuals=residuals,
        leverage=leverage_result,
    )

    denominator = 1 - leverage_result.estimate
    expected = residuals * leverage_result.estimate / denominator
    expected_standard_errors = (
        np.abs(residuals) * leverage_result.monte_carlo_standard_errors / denominator**2
    )
    assert isinstance(result, RandomizedDiagonalResult)
    assert result.number_probes == 250
    np.testing.assert_allclose(result.estimate, expected)
    np.testing.assert_allclose(
        result.monte_carlo_standard_errors,
        expected_standard_errors,
    )


@pytest.mark.parametrize(
    ("formula", "fixed_effect_columns", "include_covariate"),
    [
        pytest.param("y ~ x", [], True, id="no_fixed_effects"),
        pytest.param("y ~ x | f1", ["f1"], True, id="one_fixed_effect"),
        pytest.param(
            "y ~ 1 | f1 + f2",
            ["f1", "f2"],
            False,
            id="fixed_effects_only",
        ),
    ],
)
def test_feols_leverage_matches_dense_projection_across_designs(
    fixed_effect_design: FixedEffectDesign,
    formula: str,
    fixed_effect_columns: list[str],
    include_covariate: bool,
) -> None:
    fixed_effects, _, _ = fixed_effect_design
    number_observations = fixed_effects.shape[0]
    covariate = np.linspace(-1.0, 2.0, number_observations)
    data = pd.DataFrame(
        {
            "y": 0.5 * covariate + np.sin(np.arange(number_observations)),
            "x": covariate,
            "f1": fixed_effects[:, 0],
            "f2": fixed_effects[:, 1],
        }
    )
    fit = pf.feols(formula, data=data)

    result = fit.leverage(
        number_probes=10_000,
        seed=8192,
        compute_monte_carlo_standard_errors=True,
    )

    design_blocks = []
    if fixed_effect_columns:
        for column in fixed_effect_columns:
            levels = np.unique(data[column])
            design_blocks.append(
                (data[column].to_numpy()[:, None] == levels).astype(np.float64)
            )
    else:
        design_blocks.append(np.ones((number_observations, 1)))
    if include_covariate:
        design_blocks.append(covariate[:, None])
    design_matrix = np.column_stack(design_blocks)
    dense_projection = design_matrix @ np.linalg.pinv(design_matrix)
    expected = np.diag(dense_projection)
    assert result.monte_carlo_standard_errors is not None
    np.testing.assert_array_less(
        np.abs(result.estimate - expected),
        5 * result.monte_carlo_standard_errors + 1e-10,
        err_msg=("Feols leverage error exceeds five Monte Carlo standard errors."),
    )


@pytest.mark.parametrize(
    ("formula", "fixed_effect_columns", "include_covariate", "average"),
    [
        pytest.param("y ~ x | f1", ["f1"], True, False, id="one_fixed_effect"),
        pytest.param(
            "y ~ x | f1 + f2",
            ["f1", "f2"],
            True,
            False,
            id="two_fixed_effects",
        ),
        pytest.param(
            "y ~ x | f1 + f2",
            ["f1", "f2"],
            True,
            True,
            id="overall_average",
        ),
        pytest.param(
            "y ~ x | f1 + f2",
            ["f1", "f2"],
            True,
            ("f1", "f2"),
            id="group_averages",
        ),
    ],
)
def test_feols_predictions_matches_dense_dummy_covariance(
    fixed_effect_design: FixedEffectDesign,
    formula: str,
    fixed_effect_columns: list[str],
    include_covariate: bool,
    average: bool | tuple[str, ...],
) -> None:
    fixed_effects, _, _ = fixed_effect_design
    number_observations = fixed_effects.shape[0]
    covariate = np.linspace(-1.0, 2.0, number_observations)
    data = pd.DataFrame(
        {
            "y": 0.5 * covariate + np.sin(np.arange(number_observations)),
            "x": covariate,
            "f1": fixed_effects[:, 0],
            "f2": fixed_effects[:, 1],
        }
    )
    fit = pf.feols(formula, data=data, vcov="iid")

    result = fit.predictions(average=average, number_probes=20_000, seed=7721)

    design_blocks = [
        (data[column].to_numpy()[:, None] == np.unique(data[column])).astype(np.float64)
        for column in fixed_effect_columns
    ]
    if include_covariate:
        design_blocks.append(covariate[:, None])
    design_matrix = np.column_stack(design_blocks)
    dense_projection = design_matrix @ np.linalg.pinv(design_matrix)
    averaging_matrix, expected_group_keys = _averaging_matrix(data, average)
    ssc = float(fit._ssc[0])
    residual_variance = ssc * np.sum(fit._u_hat**2) / (fit._N - 1)
    expected_variance = residual_variance * np.diag(
        averaging_matrix @ dense_projection @ averaging_matrix.T
    )
    expected_estimate = averaging_matrix @ fit.predict()

    assert result.variance.monte_carlo_standard_errors is not None
    assert result.group_keys == expected_group_keys
    np.testing.assert_allclose(result.estimate, expected_estimate, rtol=0, atol=1e-14)
    np.testing.assert_array_less(
        np.abs(result.standard_error**2 - expected_variance),
        5 * result.variance.monte_carlo_standard_errors + 1e-10,
        err_msg=(
            "Feols prediction variance error exceeds five Monte Carlo standard errors."
        ),
    )


def test_feols_predictions_without_fixed_effects_uses_jla() -> None:
    data = pf.get_data().dropna()
    fit = pf.feols("Y ~ X1 + X2", data=data, vcov="iid")

    link = fit.predictions(type="link", number_probes=20_000, seed=99)
    response = fit.predictions(type="response", number_probes=20_000, seed=99)
    expected_variance = np.einsum("ij,jk,ik->i", fit._X, fit._vcov, fit._X)

    np.testing.assert_array_equal(link.estimate, fit.predict())
    np.testing.assert_array_equal(response.estimate, link.estimate)
    np.testing.assert_array_equal(response.standard_error, link.standard_error)
    assert link.variance.monte_carlo_standard_errors is not None
    assert link.variance.number_probes == 20_000
    np.testing.assert_array_less(
        np.abs(link.standard_error**2 - expected_variance),
        5 * link.variance.monte_carlo_standard_errors + 1e-10,
        err_msg=("Prediction variance error exceeds five Monte Carlo standard errors."),
    )


@pytest.mark.parametrize(
    "average",
    [
        pytest.param(False, id="unit"),
        pytest.param(True, id="overall_average"),
        pytest.param(("f1",), id="group_averages"),
    ],
)
def test_feols_crv1_predictions_match_dense_dummy_covariance(
    average: bool | tuple[str, ...],
) -> None:
    number_observations = 72
    observation = np.arange(number_observations)
    data = pd.DataFrame(
        {
            "y": np.cos(observation / 5) + observation / 100,
            "x": np.sin(observation / 7) + observation / 50,
            "f1": observation % 4,
            "f2": (observation // 4) % 6,
            "g1": (observation // 3) % 8,
        }
    )
    fit = pf.feols(
        "y ~ x | f1 + f2",
        data=data,
        vcov={"CRV1": "g1"},
    )

    result = fit.predictions(average=average, number_probes=20_000, seed=2871)

    design_matrix = np.column_stack(
        [
            (data["f1"].to_numpy()[:, None] == np.unique(data["f1"])).astype(
                np.float64
            ),
            (data["f2"].to_numpy()[:, None] == np.unique(data["f2"])).astype(
                np.float64
            ),
            data["x"].to_numpy(),
        ]
    )
    dense_projection = design_matrix @ np.linalg.pinv(design_matrix)
    averaging_matrix, expected_group_keys = _averaging_matrix(data, average)
    cluster_values = data["g1"].to_numpy()
    cluster_membership = (cluster_values[:, None] == np.unique(cluster_values)).astype(
        np.float64
    )
    component_jacobian = (
        averaging_matrix @ dense_projection @ (fit._u_hat[:, None] * cluster_membership)
    )
    expected_variance = float(np.asarray(fit._ssc).reshape(-1)[0]) * np.sum(
        component_jacobian**2, axis=1
    )
    expected_estimate = averaging_matrix @ fit.predict()

    assert result.variance.monte_carlo_standard_errors is not None
    assert result.group_keys == expected_group_keys
    np.testing.assert_allclose(result.estimate, expected_estimate, rtol=0, atol=1e-14)
    np.testing.assert_array_less(
        np.abs(result.variance.estimate - expected_variance),
        5 * result.variance.monte_carlo_standard_errors + 1e-10,
        err_msg=(
            "CRV1 prediction variance error exceeds five Monte Carlo standard errors."
        ),
    )


def test_feols_predictions_supports_store_data_false_and_cached_preconditioner(
    fixed_effect_design: FixedEffectDesign,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_effects, _, _ = fixed_effect_design
    observation = np.arange(fixed_effects.shape[0])
    data = pd.DataFrame(
        {
            "y": np.cos(observation / 3),
            "x": np.sin(observation / 4),
            "f1": fixed_effects[:, 0],
            "f2": fixed_effects[:, 1],
        }
    )
    fit = pf.feols(
        "y ~ x | f1 + f2",
        data=data,
        vcov={"CRV1": "f1"},
        store_data=False,
        demeaner=LsmrDemeaner(
            fixef_atol=1e-12,
            fixef_btol=1e-12,
            preconditioner="additive",
        ),
    )
    captured_jacobians: list[PredictionJacobian] = []
    original_prediction_uncertainty = jla.prediction_uncertainty

    def capture_projection(**kwargs):
        captured_jacobians.append(kwargs["prediction_jacobian"])
        return original_prediction_uncertainty(**kwargs)

    monkeypatch.setattr(jla, "prediction_uncertainty", capture_projection)
    first = fit.predictions(number_probes=100, seed=823)
    second = fit.predictions(number_probes=100, seed=823)
    averaged = fit.predictions(average=True, number_probes=100, seed=823)

    score_to_prediction = captured_jacobians[0].score_to_prediction
    assert isinstance(score_to_prediction, RegressionProjectionOperator)
    fixed_effect_projection = score_to_prediction.fixed_effect_residual_projection
    assert isinstance(fixed_effect_projection, FixedEffectResidualProjection)
    assert fixed_effect_projection.preconditioner is fit.preconditioner
    assert isinstance(captured_jacobians[0].prediction_to_quantity, IdentityOperator)
    assert isinstance(
        captured_jacobians[2].prediction_to_quantity,
        PredictionAveragingOperator,
    )
    np.testing.assert_array_equal(first.estimate, second.estimate)
    np.testing.assert_allclose(averaged.estimate, [np.mean(first.estimate)])
    np.testing.assert_array_equal(first.standard_error, second.standard_error)
    np.testing.assert_array_equal(
        first.variance.monte_carlo_standard_errors,
        second.variance.monte_carlo_standard_errors,
    )


@pytest.mark.parametrize("preconditioner", ["additive", "diagonal"])
def test_feols_leverage_uses_fitted_projection_and_cached_preconditioner(
    fixed_effect_design: FixedEffectDesign,
    monkeypatch: pytest.MonkeyPatch,
    preconditioner: str,
) -> None:
    fixed_effects, weights, dummy_matrix = fixed_effect_design
    number_observations = fixed_effects.shape[0]
    covariate = np.linspace(-1.0, 2.0, number_observations)
    data = pd.DataFrame(
        {
            "y": 0.5 * covariate + np.sin(np.arange(number_observations)),
            "x": covariate,
            "f1": fixed_effects[:, 0],
            "f2": fixed_effects[:, 1],
            "weights": weights,
        }
    )
    fit = pf.feols(
        "y ~ x | f1 + f2",
        data=data,
        weights="weights",
        store_data=False,
        demeaner=LsmrDemeaner(
            fixef_atol=1e-12,
            fixef_btol=1e-12,
            preconditioner=preconditioner,  # type: ignore[arg-type]
        ),
    )

    captured_projections: list[RegressionProjectionOperator] = []
    original_leverage = jla.leverage

    def capture_projection(**kwargs):
        captured_projections.append(kwargs["regression_projection"])
        return original_leverage(**kwargs)

    monkeypatch.setattr(jla, "leverage", capture_projection)
    result = fit.leverage(
        number_probes=10_000,
        seed=128,
        compute_monte_carlo_standard_errors=True,
    )

    projection = captured_projections[0]
    assert isinstance(
        projection.fixed_effect_residual_projection,
        FixedEffectResidualProjection,
    )
    assert (
        projection.fixed_effect_residual_projection.preconditioner is fit.preconditioner
    )

    square_root_weights = np.sqrt(weights)
    weighted_design_matrix = square_root_weights[:, None] * np.column_stack(
        [dummy_matrix, covariate]
    )
    dense_projection = weighted_design_matrix @ np.linalg.pinv(weighted_design_matrix)
    expected = np.diag(dense_projection)
    assert result.monte_carlo_standard_errors is not None
    assert result.monte_carlo_standard_errors.shape == expected.shape
    assert result.number_probes == 10_000
    np.testing.assert_array_less(
        np.abs(result.estimate - expected),
        5 * result.monte_carlo_standard_errors + 1e-10,
        err_msg=("Feols leverage error exceeds five Monte Carlo standard errors."),
    )
    first = fit.leverage()
    second = fit.leverage()
    assert first.monte_carlo_standard_errors is None
    assert first.number_probes == 100
    np.testing.assert_array_equal(first.estimate, second.estimate)


def test_feols_influence_matches_deleted_refits(
    fixed_effect_design: FixedEffectDesign,
) -> None:
    fixed_effects, weights, _ = fixed_effect_design
    number_observations = fixed_effects.shape[0]
    covariate = np.linspace(-1.0, 2.0, number_observations)
    data = pd.DataFrame(
        {
            "y": 0.5 * covariate + np.sin(np.arange(number_observations)),
            "x": covariate,
            "f1": fixed_effects[:, 0],
            "f2": fixed_effects[:, 1],
            "weights": weights,
        }
    )
    fit = pf.feols(
        "y ~ x | f1 + f2",
        data=data,
        weights="weights",
    )

    result = fit.influence(
        number_probes=20_000,
        seed=114,
        compute_monte_carlo_standard_errors=True,
    )

    fitted = fit.predict()
    expected = np.empty(number_observations)
    for observation in range(number_observations):
        leave_one_out_fit = pf.feols(
            "y ~ x | f1 + f2",
            data=data.drop(index=observation),
            weights="weights",
        )
        leave_one_out_prediction = leave_one_out_fit.predict(
            newdata=data.iloc[[observation]]
        )[0]
        expected[observation] = fitted[observation] - leave_one_out_prediction

    assert result.monte_carlo_standard_errors is not None
    np.testing.assert_array_less(
        np.abs(result.estimate - expected),
        5 * result.monte_carlo_standard_errors + 1e-10,
        err_msg=(
            "Randomized leave-one-out influence error exceeds five Monte Carlo "
            "standard errors."
        ),
    )


def test_feols_leverage_rejects_unsupported_models_and_inputs() -> None:
    data = pf.get_data().dropna()
    ols_fit = pf.feols("Y ~ X1", data=data)
    lean_fit = pf.feols("Y ~ X1 | f1", data=data, lean=True)
    iv_fit = pf.feols("Y ~ 1 + [X1 ~ Z1] | f1", data=data)
    glm_fit = pf.feglm("Y ~ X1", data=data, family="gaussian")
    frequency_weight_fit = pf.feols(
        "Y ~ X1",
        data=data.assign(frequency_weights=2),
        weights="frequency_weights",
        weights_type="fweights",
    )
    with pytest.raises(ValueError, match="unavailable for lean models"):
        lean_fit.leverage(number_probes=2)
    with pytest.raises(NotImplementedError, match="not implemented for IV models"):
        iv_fit.leverage(number_probes=2)
    with pytest.raises(NotImplementedError, match="only for OLS models"):
        glm_fit.leverage(number_probes=2)
    with pytest.raises(NotImplementedError, match="frequency weights"):
        frequency_weight_fit.leverage(number_probes=2)
    with pytest.raises(ValueError, match="number_probes must be greater than one"):
        ols_fit.leverage(number_probes=1)
    with pytest.raises(ValueError, match="unavailable for lean models"):
        lean_fit.influence(number_probes=2)


def test_feols_predictions_rejects_unsupported_models_and_inputs() -> None:
    data = pf.get_data().dropna()
    lean_fit = pf.feols("Y ~ X1 | f1", data=data, lean=True)
    iv_fit = pf.feols("Y ~ 1 + [X1 ~ Z1] | f1", data=data)
    glm_fit = pf.feglm("Y ~ X1", data=data, family="gaussian")
    weighted_fit = pf.feols(
        "Y ~ X1 | f1",
        data=data.assign(regression_weights=2.0),
        weights="regression_weights",
    )
    hetero_fit = pf.feols("Y ~ X1 | f1", data=data, vcov="hetero")
    crv3_fit = pf.feols("Y ~ X1", data=data, vcov={"CRV3": "f1"})
    multiway_crv1_fit = pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1 + f2"})
    iid_fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")
    no_data_fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid", store_data=False)
    missing_group_fit = pf.feols(
        "Y ~ X1 | f1",
        data=data.assign(
            group_with_missing=np.where(np.arange(len(data)) == 0, np.nan, 1.0)
        ),
        vcov="iid",
    )
    fixed_effect_only_fit = pf.feols("Y ~ 1 | f1", data=data, vcov="iid")

    with pytest.raises(ValueError, match="unavailable for lean models"):
        lean_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="not implemented for IV models"):
        iv_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="only for OLS models"):
        glm_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="regression weights"):
        weighted_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="only IID and CRV1 inference"):
        hetero_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="only IID and CRV1 inference"):
        crv3_fit.predictions(number_probes=2)
    with pytest.raises(NotImplementedError, match="only one-way CRV1 inference"):
        multiway_crv1_fit.predictions(number_probes=2)
    with pytest.raises(
        NotImplementedError, match="without estimated covariate coefficients"
    ):
        fixed_effect_only_fit.predictions(number_probes=2)
    with pytest.raises(ValueError, match="Invalid argument"):
        iid_fit.predictions(type="invalid", number_probes=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="number_probes must be greater than one"):
        iid_fit.predictions(number_probes=1)
    with pytest.raises(ValueError, match="non-empty tuple of column names"):
        iid_fit.predictions(average=(), number_probes=2)
    with pytest.raises(ValueError, match="non-empty tuple of column names"):
        iid_fit.predictions(average="f1", number_probes=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Grouping columns must be unique"):
        iid_fit.predictions(average=("f1", "f1"), number_probes=2)
    with pytest.raises(ValueError, match="require stored data"):
        no_data_fit.predictions(average=("f1",), number_probes=2)
    with pytest.raises(ValueError, match="Grouping columns not found"):
        iid_fit.predictions(average=("unknown",), number_probes=2)
    with pytest.raises(ValueError, match="must not contain missing values"):
        missing_group_fit.predictions(average=("group_with_missing",), number_probes=2)


def test_influence_rejects_unit_leverage() -> None:
    leverage_result = RandomizedDiagonalResult(
        estimate=np.array([0.5, 1.0]),
        monte_carlo_standard_errors=None,
        number_probes=100,
    )

    with pytest.raises(
        ValueError,
        match="leverage greater than or equal to one",
    ):
        influence(
            residuals=np.array([1.0, 1.0]),
            leverage=leverage_result,
        )


def test_leverage_is_available_on_fetched_multiple_estimation_model() -> None:
    multiple_fit = pf.feols(
        "Y ~ sw(X1, X2) | f1",
        data=pf.get_data().dropna(),
    )
    fit = multiple_fit.fetch_model(0, print_fml=False)

    result = fit.leverage(number_probes=16)
    influence_result = fit.influence(number_probes=16)
    prediction_result = fit.predictions(number_probes=16)

    assert result.estimate.shape == (fit._N_rows,)
    assert influence_result.estimate.shape == (fit._N_rows,)
    assert prediction_result.estimate.shape == (fit._N_rows,)


def test_fixed_effect_residual_projection_raises_on_nonconvergence() -> None:
    random_number_generator = np.random.default_rng(42)
    number_observations = 100
    fixed_effects = random_number_generator.integers(
        0,
        10,
        size=(number_observations, 2),
        dtype=np.int32,
    )
    projection = FixedEffectResidualProjection(
        fixed_effects=fixed_effects,
        weights=np.ones(number_observations),
        demeaner=MapDemeaner(fixef_maxiter=1),
        preconditioner=None,
    )

    with pytest.raises(
        ValueError,
        match="Fixed-effect residualisation failed after 1 iterations",
    ):
        projection.apply(random_number_generator.normal(size=(number_observations, 2)))
