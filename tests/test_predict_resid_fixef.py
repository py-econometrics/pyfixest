import pytest
import numpy as np
import pandas as pd
from pyfixest.utils import get_data
from pyfixest.estimation import feols, fepois
from pyfixest.exceptions import NotImplementedError

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture
def data():
    data = get_data(seed=65714, model = "Fepois")
    data = data.dropna()

    return data


def test_internally(data):
    """
    Test predict() method internally.
    Currently only for OLS.
    """


    # predict via feols, without fixed effect
    fit = feols(fml="Y~csw(X1, X2)", data=data, vcov="iid")
    mod = fit.fetch_model(0)
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)

    # predict via feols, with fixef effect
    fit = feols(fml="Y~csw(X1, X2) | f1", data=data, vcov="iid")
    mod = fit.fetch_model(0)
    mod.fixef()
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)

    # now expect error with updated predicted being a subset of data
    with pytest.raises(ValueError):
        updated_prediction = mod.predict(newdata=data.iloc[0:100, :])
        np.allclose(original_prediction, updated_prediction)



    # fepois NotImplementedError(s)
    fit = fepois(fml="Y~X1*X2", data=data, vcov="hetero")
    with pytest.raises(NotImplementedError):
        fit.predict(newdata=fit._data)



    # fepois with fixed effect
    fit = fepois(fml="Y~X1*X2 | f1", data=data, vcov="hetero")
    with pytest.raises(NotImplementedError):
        fit.predict()




@pytest.mark.parametrize(
    "fml",
    [
        "Y~ X1 | f1",
        "Y~ X1 | f1 + f2",
        # "Y~ X1 | X3^X4",
    ],
)
def test_vs_fixest(data, fml):
    """
    Test predict and resid methods against fixest.
    """

    feols_mod = feols(fml=fml, data=data, vcov="HC1")
    fepois_mod = fepois(fml=fml, data=data, vcov="HC1")

    data2 = data[1:500]

    feols_mod.fixef()

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
    if not np.allclose(feols_mod._sumFE, r_fixest_ols.rx2("sumFE")):
        raise ValueError("sumFE for OLS are not equal")

    # test sumFE for Poisson
    # if not np.allclose(
    #    fepois_mod._sumFE,
    #    r_fixest_pois.rx2("sumFE")
    # ):
    #    raise ValueError("sumFE for Poisson are not equal")

    # test predict for OLS
    if not np.allclose(feols_mod.predict(), r_fixest_ols.rx2("fitted.values")):
        raise ValueError("Predictions for OLS are not equal")

    # test predict for Poisson
    #if not np.allclose(fepois_mod.predict(), r_fixest_pois.rx2("fitted.values")):
    #    raise ValueError("Predictions for Poisson are not equal")

    # test on new data - OLS.
    if not np.allclose(feols_mod.predict(newdata = data2), stats.predict(r_fixest_ols, newdata = data2)):
        raise ValueError("Predictions for OLS are not equal")

    # test predict for Poisson
    #if not np.allclose(fepois_mod.predict(data = data2), stats.predict(r_fixest_pois, newdata = data2)):
    #    raise ValueError("Predictions for Poisson are not equal")

    # test resid for OLS
    if not np.allclose(feols_mod.resid(), r_fixest_ols.rx2("residuals")):
        raise ValueError("Residuals for OLS are not equal")

    # test resid for Poisson
    # if not np.allclose(
    #    fepois_mod.resid(),
    #    r_fixest_pois.rx2("residuals")
    # ):
    #    raise ValueError("Residuals for Poisson are not equal")
