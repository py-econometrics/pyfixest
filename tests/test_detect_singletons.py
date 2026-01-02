import numpy as np
import pytest

from pyfixest.core.detect_singletons import detect_singletons as detect_singletons_rust
from pyfixest.estimation.detect_singletons_ import (
    detect_singletons as detect_singletons_numba,
)
from pyfixest.estimation.jax.detect_singletons_jax import detect_singletons_jax

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
    argvalues=[detect_singletons_rust, detect_singletons_numba, detect_singletons_jax],
    ids=["rust", "numba", "jax"],
)
def test_correctness(input, solution, detection_function):
    assert np.array_equal(detection_function(input), solution)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons_rust, detect_singletons_numba, detect_singletons_jax],
    ids=["rust", "numba", "jax"],
)
def test_single_column(detection_function):
    """Test with a single fixed effect column."""
    input_data = np.array([[0], [0], [1], [2], [2]])
    expected = np.array([False, False, True, False, False])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons_rust, detect_singletons_numba, detect_singletons_jax],
    ids=["rust", "numba", "jax"],
)
def test_all_singletons(detection_function):
    """Test when all observations are singletons."""
    input_data = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    expected = np.array([True, True, True, True])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons_rust, detect_singletons_numba, detect_singletons_jax],
    ids=["rust", "numba", "jax"],
)
def test_no_singletons(detection_function):
    """Test when there are no singletons."""
    input_data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    expected = np.array([False, False, False, False])
    result = detection_function(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    argnames="detection_function",
    argvalues=[detect_singletons_rust, detect_singletons_numba, detect_singletons_jax],
    ids=["rust", "numba", "jax"],
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
    reference = detect_singletons_numba(input_data)
    result = detection_function(input_data)

    assert np.array_equal(result, reference)
    assert len(result) == N
    assert result.dtype == np.bool_


# Tests specific to the Rust wrapper's Python preprocessing logic


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rust_wrapper_rejects_float_dtypes(dtype):
    """Test that the Rust wrapper raises TypeError for float dtypes."""
    input_data = np.array([[0, 1], [0, 1], [1, 2]], dtype=dtype)
    with pytest.raises(TypeError, match="Fixed effects must be integers"):
        detect_singletons_rust(input_data)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]
)
def test_rust_wrapper_accepts_integer_dtypes(dtype):
    """Test that the Rust wrapper accepts all integer dtypes."""
    input_data = np.array([[0, 1], [0, 1], [1, 2], [1, 2]], dtype=dtype)
    expected = np.array([False, False, False, False])
    result = detect_singletons_rust(input_data)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_rust_wrapper_handles_memory_layout(order):
    """Test that the Rust wrapper handles both C and F memory layouts."""
    input_data = np.array(
        [[0, 2, 1], [0, 2, 1], [0, 1, 3], [0, 1, 2], [0, 1, 2]],
        dtype=np.int64,
        order=order,
    )
    expected = np.array([False, False, True, False, False])
    result = detect_singletons_rust(input_data)
    assert np.array_equal(result, expected)
