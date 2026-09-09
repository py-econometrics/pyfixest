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
    FixedEffectResidualProjection,
    RegressionProjectionOperator,
    influence,
    leverage,
)

FixedEffectDesign = tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]


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

    assert result.estimate.shape == (fit._N_rows,)
    assert influence_result.estimate.shape == (fit._N_rows,)


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
