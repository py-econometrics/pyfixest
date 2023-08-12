import pytest
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.utils import get_data, get_poisson_data

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture
def data():
    data = get_poisson_data(seed=6574)
    data = data.dropna()
    data["X3"] = pd.Categorical(data.X3.astype(str))
    data["X4"] = pd.Categorical(data.X4.astype(str))

    return data


def test_internally(data):
    """
    Test predict() method internally.
    Currently only for OLS.
    """

    pyfixest = pf.Fixest(data=data).feols(fml="Y~csw(X1, X2) | X3", vcov="iid")

    mod = pyfixest.fetch_model("0")
    mod.fixef()
    original_prediction = mod.predict()
    updated_prediction = mod.predict(data=mod._data)
    np.allclose(original_prediction, updated_prediction)

    # now expect error with updated predicted being a subset of data
    with pytest.raises(ValueError):
        updated_prediction = mod.predict(data=data.iloc[0:100, :])
        np.allclose(original_prediction, updated_prediction)


@pytest.mark.parametrize(
    "fml",
    [
        "Y~ X1 | X3",
        "Y~ X1 | X3 + X4",
        # "Y~ X1 | X3^X4",
    ],
)
def test_vs_fixest(data, fml):
    """
    Test predict and resid methods against fixest.
    """

    pyfixest = pf.Fixest(data=data).feols(fml=fml, vcov="HC1")
    pyfixest2 = pf.Fixest(data=data).fepois(fml=fml, vcov="HC1")

    feols_mod = pyfixest.fetch_model("0")
    feols_mod.fixef()

    fepois_mod = pyfixest2.fetch_model("0")
    # fepois_mod.fixef()

    # fixest estimation
    r_fixest_ols = fixest.feols(
        ro.Formula(fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
        se="hetero",
    )

    r_fixest_pois = fixest.fepois(
        ro.Formula(fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
        se="hetero",
    )

    # test OLS fit
    if not np.allclose(feols_mod.coef().values, r_fixest_ols.rx2("coefficients")):
        raise ValueError("Coefficients are not equal")

    # test Poisson fit
    if not np.allclose(fepois_mod.coef(), r_fixest_pois.rx2("coefficients")):
        raise ValueError("Coefficients are not equal")

    # test sumFE for OLS
    if not np.allclose(feols_mod.sumFE, r_fixest_ols.rx2("sumFE")):
        raise ValueError("sumFE for OLS are not equal")

    # test sumFE for Poisson
    # if not np.allclose(
    #    fepois_mod.sumFE,
    #    r_fixest_pois.rx2("sumFE")
    # ):
    #    raise ValueError("sumFE for Poisson are not equal")

    # test predict for OLS
    if not np.allclose(feols_mod.predict(), r_fixest_ols.rx2("fitted.values")):
        raise ValueError("Predictions for OLS are not equal")

    # test predict for Poisson
    if not np.allclose(fepois_mod.predict(), r_fixest_pois.rx2("fitted.values")):
        raise ValueError("Predictions for Poisson are not equal")

    # test resid for OLS
    if not np.allclose(feols_mod.resid(), r_fixest_ols.rx2("residuals")):
        raise ValueError("Residuals for OLS are not equal")

    # test resid for Poisson
    # if not np.allclose(
    #    fepois_mod.resid(),
    #    r_fixest_pois.rx2("residuals")
    # ):
    #    raise ValueError("Residuals for Poisson are not equal")
