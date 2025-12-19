import pytest
import pandas as pd
import numpy as np
from pyfixest.estimation import feols
from pyfixest.utils.utils import get_data

def test_sensitivity_stats():
    # Setup
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    fit.vcov("iid")
    sens = fit.sensitivity_analysis()

    # Case 1: Specific variable (Scalar output)
    stats_single = sens.sensitivity_stats(X="X1")
    assert isinstance(stats_single, dict)
    assert isinstance(stats_single["estimate"], (float, np.float64))
    assert "rv_q" in stats_single

    # Case 2: All variables (Vector output)
    stats_all = sens.sensitivity_stats(X=None)
    assert isinstance(stats_all, dict)
    # Check that it returns arrays for multiple coefficients
    assert isinstance(stats_all["estimate"], np.ndarray)
    assert len(stats_all["estimate"]) == len(fit._coefnames)
    assert stats_all["rv_q"].shape == stats_all["estimate"].shape