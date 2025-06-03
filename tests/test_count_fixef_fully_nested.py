import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyfixest.core import count_fixef_fully_nested_all

# Import your module here - adjust import path as needed
from pyfixest.estimation.numba.nested_fixef_nb import (
    _count_fixef_fully_nested_all as count_fixef_fully_nested_all_nb,
)


@pytest.mark.parametrize(
    "fn", [count_fixef_fully_nested_all, count_fixef_fully_nested_all_nb]
)
def test_count_fixef_fully_nested_basic(benchmark, fn):
    """Basic test for count_fixef_fully_nested_all_rs function."""
    # Setup test data
    all_fe = np.array(["fe1", "fe2", "cluster1"])
    cluster_names = np.array(["cluster1"])

    # Fixed effects data where fe1 is nested in cluster1
    fe_data = np.array(
        [
            [1, 5, 0],  # row 1 - fe1=1, fe2=5, cluster1=0
            [1, 6, 1],  # row 2 - fe1=1, fe2=6, cluster1=1
            [2, 5, 0],  # row 3 - fe1=2, fe2=5, cluster1=0
            [2, 6, 1],  # row 4 - fe1=2, fe2=6, cluster1=1
        ],
        dtype=np.uintp,
    )

    # Cluster data - one column for the "cluster1" variable
    cluster_data = np.array(
        [
            [10],  # fe1=1 always maps to cluster=10
            [10],
            [20],  # fe1=2 always maps to cluster=20
            [20],
        ],
        dtype=np.uintp,
    )

    # Call function
    mask, count = benchmark(fn, all_fe, cluster_names, cluster_data, fe_data)

    # Expected results
    expected_mask = np.array(
        [True, False, True]
    )  # fe1 is nested, fe2 is not, cluster1 is a cluster
    expected_count = 2

    # Assertions
    assert_array_equal(np.array(mask), expected_mask)
    assert count == expected_count
