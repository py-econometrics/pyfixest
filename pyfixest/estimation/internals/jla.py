from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class LinearOperator(Protocol):
    """Linear transformation supporting a matrix-valued right-hand side.

    An operator with shape ``(m, n)`` maps an array with shape
    ``(n, number_of_vectors)`` to an array with shape
    ``(m, number_of_vectors)``. Implementations may use dense, sparse, or
    implicit representations.
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Operator dimensions as ``(output_size, input_size)``."""
        ...

    def apply(self, right_hand_side: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the operator to one or more column vectors.

        Parameters
        ----------
        right_hand_side : NDArray[np.float64]
            Column vectors with shape ``(input_size, number_of_vectors)``.

        Returns
        -------
        NDArray[np.float64]
            Transformed vectors with shape
            ``(output_size, number_of_vectors)``.
        """
        ...


@dataclass(frozen=True, slots=True)
class RandomizedSketch:
    """Normalized Achlioptas random embeddings of a linear operator's rows.

    Attributes
    ----------
    embedding : NDArray[np.float64]
        Johnson--Lindenstrauss row embeddings with shape
        ``(output_size, number_probes)``. For an operator ``G`` and Rademacher
        probes ``Z``, this contains ``G @ Z / sqrt(number_probes)``.
    """

    embedding: NDArray[np.float64]

    @property
    def number_probes(self) -> int:
        """Number of random projections represented by the sketch."""
        return self.embedding.shape[1]


@dataclass(frozen=True, slots=True)
class RandomizedDiagonalResult:
    """Randomized diagonal estimates and algorithmic uncertainty.

    Attributes
    ----------
    estimate : NDArray[np.float64]
        Estimated diagonal with shape ``(output_size,)``.
    monte_carlo_standard_errors : NDArray[np.float64] | None
        Monte Carlo standard errors with shape ``(output_size,)``. ``None``
        when standard errors were not requested.
    number_probes : int
        Number of independent random probes used by the approximation.
    """

    estimate: NDArray[np.float64]
    monte_carlo_standard_errors: NDArray[np.float64] | None
    number_probes: int


def sketch(
    *,
    operator: LinearOperator,
    number_probes: int,
    random_number_generator: np.random.Generator,
) -> RandomizedSketch:
    """Create normalized random embeddings of an operator's rows.

    The embedding uses the Rademacher random-projection construction of
    [Achlioptas
    (2003)](https://doi.org/10.1016/S0022-0000(03)00025-4).

    Parameters
    ----------
    operator : LinearOperator
        Operator whose rows are to be embedded.
    number_probes : int
        Number of independent Rademacher probes. Must be greater than one.
    random_number_generator : np.random.Generator
        Random-number generator used to construct the probes.

    Returns
    -------
    RandomizedSketch
        Normalized row embeddings of the operator.

    Raises
    ------
    ValueError
      If fewer than two probes are requested.
    """
    _validate_number_probes(number_probes=number_probes)
    probe_vectors = _draw_rademacher_probes(
        random_number_generator=random_number_generator,
        input_size=operator.shape[1],
        number_probes=number_probes,
    )
    embedding = operator.apply(probe_vectors) / np.sqrt(number_probes)
    return RandomizedSketch(embedding=embedding)


def approximate_diagonal(
    *,
    left: LinearOperator,
    right: LinearOperator | None = None,
    number_probes: int,
    random_number_generator: np.random.Generator,
    compute_monte_carlo_standard_errors: bool = False,
) -> RandomizedDiagonalResult:
    """Approximate ``diag(L @ R.T)`` using random row embeddings.

    This uses the Rademacher random-projection construction of [Achlioptas
    (2003)](https://doi.org/10.1016/S0022-0000(03)00025-4) to approximate row
    inner products.

    Parameters
    ----------
    left : LinearOperator
        Operator representing ``L``.
    right : LinearOperator | None, optional
        Operator representing ``R``. It must have the same input and output
        dimensions as ``left``. ``None`` denotes ``R = L``.
    number_probes : int
        Number of independent Rademacher probes. Must be greater than one.
    random_number_generator : np.random.Generator
        Random-number generator used to construct the probes.
    compute_monte_carlo_standard_errors : bool, optional
        Whether to compute Monte Carlo standard errors from the variation
        across probes. Defaults to ``False``.

    Returns
    -------
    RandomizedDiagonalResult
        Diagonal estimates and optional Monte Carlo standard errors.

    Raises
    ------
    ValueError
        If dimensions are incompatible or numerical arguments are invalid.
    """
    _validate_inputs(left=left, right=right, number_probes=number_probes)
    probe_vectors = _draw_rademacher_probes(
        random_number_generator=random_number_generator,
        input_size=left.shape[1],
        number_probes=number_probes,
    )
    left_values = left.apply(probe_vectors)
    right_values = left_values if right is None else right.apply(probe_vectors)
    probe_estimates = left_values * right_values
    estimate = np.mean(probe_estimates, axis=1)
    monte_carlo_standard_errors = (
        np.std(probe_estimates, axis=1, ddof=1) / np.sqrt(number_probes)
        if compute_monte_carlo_standard_errors
        else None
    )
    return RandomizedDiagonalResult(
        estimate=estimate,
        monte_carlo_standard_errors=monte_carlo_standard_errors,
        number_probes=number_probes,
    )


def _draw_rademacher_probes(
    random_number_generator: np.random.Generator,
    input_size: int,
    number_probes: int,
) -> NDArray[np.float64]:
    draws = random_number_generator.integers(
        low=0, high=2, size=(number_probes, input_size), dtype=np.int8
    )
    return (2.0 * draws.astype(np.float64) - 1.0).T


def _validate_number_probes(*, number_probes: int) -> None:
    if number_probes < 2:
        raise ValueError("number_probes must be greater than one.")


def _validate_inputs(
    left: LinearOperator,
    right: LinearOperator | None,
    number_probes: int,
) -> None:
    _validate_number_probes(number_probes=number_probes)
    if right is not None and left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")
