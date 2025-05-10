import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pyfixest as pf

from pyfixest.estimation import feols

# Import R packages
quantreg = importr("quantreg")
stats = importr("stats")

# Enable pandas conversion
pandas2ri.activate()

@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1",
        "Y ~ X1 + X2",
    ],
)
@pytest.mark.parametrize(
    "vcov",
    [
        "iid",
    ],
)
@pytest.mark.parametrize("data", [pf.get_data()])
@pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])

def test_quantreg_vs_r(data, fml, vcov, quantile):
    """
    Test that pyfixest's quantreg implementation equals R's quantreg implementation.
    """
    # Create test data
    np.random.seed(123)


    # Fit model in pyfixest
    fit_py = pf.quantreg(fml, data=data, vcov=vcov, quantile=quantile)

    # Fit model in R
    r_data = pandas2ri.py2rpy(data)
    r_formula = ro.Formula(fml)

    # Fit R model
    fit_r = quantreg.rq(r_formula, data=r_data, tau=quantile)

    # Compare coefficients
    py_coef = fit_py.coef().to_numpy()
    r_coef = np.array(fit_r.rx2("coefficients"))
    np.testing.assert_allclose(py_coef, r_coef, rtol=1e-5)