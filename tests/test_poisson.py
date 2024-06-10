import contextlib
import os

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation.estimation import fepois
from pyfixest.utils.set_rpy2_path import update_r_paths

update_r_paths()

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


def test_separation():
    """Test separation detection."""
    example1 = pd.DataFrame.from_dict(
        {
            "Y": [0, 0, 0, 1, 2, 3],
            "fe1": ["a", "a", "b", "b", "b", "c"],
            "fe2": ["c", "c", "d", "d", "d", "e"],
            "X": np.random.normal(0, 1, 6),
        }
    )

    with pytest.warns(
        UserWarning, match="2 observations removed because of separation."
    ):
        mod = fepois("Y ~ X  | fe1", data=example1, vcov="hetero", method=["fe"])  # noqa: F841

    example2 = pd.DataFrame.from_dict(
        {
            "Y": [0, 0, 0, 1, 2, 3],
            "X1": [2, -1, 0, 0, 5, 6],
            "X2": [-1, 2, 0, 0, -10, -12],
        }
    )
    with pytest.warns(
        UserWarning, match="1 observations removed because of separation."
    ):
        mod = fepois("Y ~ X1 + X2", data=example2, vcov="hetero", method=["ir"])  # noqa: F841
