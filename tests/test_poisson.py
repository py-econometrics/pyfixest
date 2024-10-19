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
    y = np.array([0, 0, 0, 1, 2, 3])
    df1 = np.array(["a", "a", "b", "b", "b", "c"])
    df2 = np.array(["c", "c", "d", "d", "d", "e"])
    x = np.random.normal(0, 1, 6)

    df = pd.DataFrame({"Y": y, "fe1": df1, "fe2": df2, "x": x})

    with pytest.warns(
        UserWarning, match="2 observations removed because of separation."
    ):
        mod = fepois("Y ~ x  | fe1", data=df, vcov="hetero")  # noqa: F841


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
def test_against_fixest(fml):
    data = pf.get_data(model="Fepois")
    iwls_tol = 1e-12

    # vcov = "hetero"
    vcov = "hetero"
    fit = pf.fepois(fml, data=data, vcov=vcov, iwls_tol=iwls_tol)
    fit_r = fixest.fepois(ro.Formula(fml), data=data, vcov=vcov, glm_tol=iwls_tol)

    np.testing.assert_allclose(
        fit_r.rx2("irls_weights").reshape(-1, 1), fit._weights, atol=1e-08, rtol=1e-07
    )
    np.testing.assert_allclose(
        fit_r.rx2("linear.predictors").reshape(-1, 1),
        fit._Xbeta,
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("scores").reshape(-1, 1),
        fit._scores.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )

    np.testing.assert_allclose(
        fit_r.rx2("hessian"), fit._hessian, atol=1e-08, rtol=1e-07
    )

    np.testing.assert_allclose(
        fit_r.rx2("deviance"), fit.deviance, atol=1e-08, rtol=1e-07
    )
