import numpy as np

from pyfixest.detect_singletons import detect_singletons

input1 = np.array([[0, 2, 1], [0, 2, 1], [0, 1, 3], [0, 1, 2], [0, 1, 2]])
solution1 = np.array([False, False, True, False, False])

input2 = np.array([[0, 2, 1], [0, 2, 1], [3, 1, 2], [0, 1, 1], [0, 1, 2]])
solution2 = np.array([False, False, True, True, True])

input3 = np.array([[0, 2, 1], [0, 2, 1], [0, 1, 1], [0, 1, 2], [0, 1, 2]])
solution3 = np.array([False, False, False, False, False])


def test_correctness():
    assert np.array_equal(detect_singletons(input1), solution1)
    assert np.array_equal(detect_singletons(input2), solution2)
    assert np.array_equal(detect_singletons(input3), solution3)
