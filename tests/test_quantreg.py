import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf

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
    #    "iid",
        "nid",
    ],
)
@pytest.mark.parametrize("data", [pf.get_data()])
@pytest.mark.parametrize("quantile", [0.35, 0.5, 0.9])
@pytest.mark.parametrize("method", ["fn"])
def test_quantreg_vs_r(data, fml, vcov, quantile, method):
    "Test that pyfixest's quantreg implementation equals R's quantreg implementation."
    # Fit model in pyfixest
    fit_py = pf.quantreg(fml, data=data, vcov=vcov, quantile=quantile, method=method)

    # Fit model in R
    r_data = pandas2ri.py2rpy(data)
    r_formula = ro.Formula(fml)

    # Fit R model
    fit_r = quantreg.rq(r_formula, data=r_data, tau=quantile, method=method)

    # Compare coefficients
    py_coef = fit_py.coef().to_numpy()
    r_coef = np.array(fit_r.rx2("coefficients"))
    np.testing.assert_allclose(py_coef, r_coef, rtol=2e-03, atol=2e-03)

    # compare standard errors
    py_se = fit_py.se().to_numpy()
    r_summ = ro.r["summary"](fit_r, se=vcov)
    coeff_mat = r_summ.rx2("coefficients")
    r_se = np.array(coeff_mat)[:, 1]
    np.testing.assert_allclose(py_se, r_se, rtol=2e-03, atol=2e-03)


def test_qplot():
    data = pf.get_data(N=1000)
    fit1 = pf.quantreg("Y ~ X1 + X2", data=data, quantile=0.5, method="fn")
    fit2 = pf.quantreg("Y ~ X1 + X2", data=data, quantile=0.9, method="fn")

    pf.qplot([fit1, fit2])
