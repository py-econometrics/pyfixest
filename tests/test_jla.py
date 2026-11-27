from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from pyfixest.estimation.internals.jla import (
    approximate_diagonal,
    sketch,
)


@dataclass(frozen=True, slots=True)
class DenseOperator:
    matrix: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.matrix.shape[0], self.matrix.shape[1])

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.matrix @ right_hand_side


def test_one_column_matrix_product_matches_dense_diagonal() -> None:
    left_matrix = np.array([[1.0], [2.0], [-1.0]])
    right_matrix = np.array([[3.0], [-2.0], [4.0]])

    result = approximate_diagonal(
        left=DenseOperator(left_matrix),
        right=DenseOperator(right_matrix),
        number_probes=7,
        random_number_generator=np.random.default_rng(42),
        compute_monte_carlo_standard_errors=True,
    )

    expected = np.diag(left_matrix @ right_matrix.T)

    np.testing.assert_array_equal(result.estimate, expected)
    np.testing.assert_array_equal(
        result.monte_carlo_standard_errors,
        np.zeros_like(expected),
    )
    assert result.number_probes == 7


def test_one_column_symmetric_matrix_product_matches_dense_diagonal() -> None:
    left_matrix = np.array([[1.0], [2.0], [-1.0]])

    result = approximate_diagonal(
        left=DenseOperator(left_matrix),
        number_probes=7,
        random_number_generator=np.random.default_rng(42),
        compute_monte_carlo_standard_errors=True,
    )

    expected = np.diag(left_matrix @ left_matrix.T)

    np.testing.assert_array_equal(result.estimate, expected)
    np.testing.assert_array_equal(
        result.monte_carlo_standard_errors,
        np.zeros_like(expected),
    )
    assert result.number_probes == 7


def test_randomized_matrix_product_matches_dense_diagonal() -> None:
    left_matrix = np.array(
        [
            [1.0, 0.5, -1.0],
            [2.0, -0.5, 0.25],
            [-1.5, 1.0, 0.75],
        ]
    )
    right_matrix = np.array(
        [
            [0.25, 1.0, 2.0],
            [-1.0, 0.5, 0.75],
            [2.0, -0.25, 1.0],
        ]
    )

    result = approximate_diagonal(
        left=DenseOperator(left_matrix),
        right=DenseOperator(right_matrix),
        number_probes=10_000,
        random_number_generator=np.random.default_rng(921),
        compute_monte_carlo_standard_errors=True,
    )

    expected = np.diag(left_matrix @ right_matrix.T)
    assert result.monte_carlo_standard_errors is not None

    np.testing.assert_array_less(
        np.abs(result.estimate - expected),
        5 * result.monte_carlo_standard_errors,
        err_msg=(
            "Randomized matrix-product diagonal error exceeds five "
            "Monte Carlo standard errors."
        ),
    )


def test_sketch_approximates_dense_gram_matrix() -> None:
    matrix = np.array(
        [
            [1.0, 0.5, -1.0],
            [2.0, -0.5, 0.25],
            [-1.5, 1.0, 0.75],
        ]
    )

    result = sketch(
        operator=DenseOperator(matrix),
        number_probes=10_000,
        random_number_generator=np.random.default_rng(921),
    )

    actual = result.embedding @ result.embedding.T
    expected = matrix @ matrix.T

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.01,
        atol=0.01,
        err_msg="Randomized sketch does not approximate the dense Gram matrix.",
    )
    assert result.number_probes == 10_000


@pytest.mark.parametrize("number_probes", [0, 1])
def test_rejects_insufficient_number_probes(number_probes: int) -> None:
    operator = DenseOperator(np.eye(2))

    with pytest.raises(
        ValueError,
        match="number_probes must be greater than one",
    ):
        approximate_diagonal(
            left=operator,
            number_probes=number_probes,
            random_number_generator=np.random.default_rng(42),
        )


@pytest.mark.parametrize("right_shape", [(2, 3), (3, 2)])
def test_rejects_incompatible_operator_shapes(
    right_shape: tuple[int, int],
) -> None:
    with pytest.raises(
        ValueError,
        match="left and right must have the same shape",
    ):
        approximate_diagonal(
            left=DenseOperator(np.ones((2, 2))),
            right=DenseOperator(np.ones(right_shape)),
            number_probes=2,
            random_number_generator=np.random.default_rng(42),
        )
