import numpy as np
import pytest

from pyfixest.estimation.detect_singletons_ import (
    detect_singletons,
    detect_singletons_jax,
)

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
    argvalues=[detect_singletons, detect_singletons_jax],
    ids=["numba", "jax"],
)
def test_correctness(input, solution, detection_function):
    assert np.array_equal(detection_function(input), solution)
