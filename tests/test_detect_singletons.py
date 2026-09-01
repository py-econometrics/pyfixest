import numpy as np
import pytest

from pyfixest.core.detect_singletons import detect_singletons


def test_frequency_counts_prevent_false_singletons_and_still_cascade():
    """Frequency totals, not physical row counts, define expanded singletons."""
    fixed_effects = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [2, 1],
        ]
    )
    weights = np.array([2, 1, 1, 1])

    np.testing.assert_array_equal(
        detect_singletons(fixed_effects, frequency_weights=weights),
        np.array([False, True, True, True]),
    )


@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, 0.0, -1.0])
def test_frequency_counts_must_be_finite_and_strictly_positive(invalid_weight):
    """Direct callers receive the same weight-domain validation as estimators."""
    fixed_effects = np.array([[0], [0]])
    weights = np.array([1.0, invalid_weight])

    with pytest.raises(ValueError, match="finite and strictly positive"):
        detect_singletons(fixed_effects, frequency_weights=weights)


input1 = np.array([[0, 2, 1], [0, 2, 1], [0, 1, 3], [0, 1, 2], [0, 1, 2]])
solution1 = np.array([False, False, True, False, False])

input2 = np.array([[0, 2, 1], [0, 2, 1], [3, 1, 2], [0, 1, 1], [0, 1, 2]])
solution2 = np.array([False, False, True, True, True])

input3 = np.array([[0, 2, 1], [0, 2, 1], [0, 1, 1], [0, 1, 2], [0, 1, 2]])
solution3 = np.array([False, False, False, False, False])


@pytest.mark.parametrize(
    argnames="input, solution",
    argvalues=[(input1, solution1), (input2, solution2), (input3, solution3)],
)
@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons],
    ids=["rust"],
)
def test_correctness(input, solution, detection_function):
    assert np.array_equal(detection_function(input), solution)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons],
    ids=["rust"],
)
def test_single_column(detection_function):
    """Test with a single fixed effect column."""
    input_data = np.array([[0], [0], [1], [2], [2]])
    expected = np.array([False, False, True, False, False])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons],
    ids=["rust"],
)
def test_all_singletons(detection_function):
    """Test when all observations are singletons."""
    input_data = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    expected = np.array([True, True, True, True])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons],
    ids=["rust"],
)
def test_no_singletons(detection_function):
    """Test when there are no singletons."""
    input_data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    expected = np.array([False, False, False, False])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons],
    ids=["rust"],
)
def test_large_input(detection_function):
    """Test with a larger input to check performance and correctness."""
    rng = np.random.default_rng(42)
    N = 10000
    input_data = np.column_stack(
        [
            rng.integers(0, N // 10, N),
            rng.integers(0, N // 5, N),
            rng.integers(0, N // 2, N),
        ]
    )

    # For large input, we compare against the Numba implementation as reference
    reference = detect_singletons(input_data)
    result = detection_function(input_data)

    assert np.array_equal(result, reference)
    assert len(result) == N
    assert result.dtype == np.bool_
