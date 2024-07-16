import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

from pyfixest.errors import NotImplementedError
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.utils.dev_utils import _extract_variable_level
from pyfixest.utils.utils import get_data

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture
def data():
    data = get_data(seed=65714, model="Fepois")
    data = data.dropna()

    return data


def test_internally(data):
    """
    Test predict() method internally.

    Notes
    -----
    Currently only for OLS.
    """
    # predict via feols, without fixed effect
    fit = feols(fml="Y~csw(X1, X2)", data=data, vcov="iid")
    mod = fit.fetch_model(0)
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)
    assert mod._data.shape[0] == original_prediction.shape[0]
    assert mod._data.shape[0] == updated_prediction.shape[0]

    # predict via feols, with fixef effect
    fit = feols(fml="Y~csw(X1, X2) | f1", data=data, vcov="iid")
    mod = fit.fetch_model(0)
    mod.fixef()
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)
    assert mod._data.shape[0] == original_prediction.shape[0]
    assert mod._data.shape[0] == updated_prediction.shape[0]

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
    """Test predict and resid methods against fixest."""
    feols_mod = feols(fml=fml, data=data, vcov="HC1")
    fepois_mod = fepois(fml=fml, data=data, vcov="HC1")

    data2 = data.copy()[1:500]

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

    if not np.allclose(len(feols_mod.predict()), len(stats.predict(r_fixest_ols))):
        raise ValueError("Predictions for OLS are not the same length")
    # test predict for Poisson
    # if not np.allclose(fepois_mod.predict(), r_fixest_pois.rx2("fitted.values")):
    #    raise ValueError("Predictions for Poisson are not equal")

    # test on new data - OLS.
    if not np.allclose(
        feols_mod.predict(newdata=data2), stats.predict(r_fixest_ols, newdata=data2)
    ):
        raise ValueError("Predictions for OLS are not equal")

    if not np.allclose(
        len(feols_mod.predict(newdata=data2)),
        len(stats.predict(r_fixest_ols, newdata=data2)),
    ):
        raise ValueError("Predictions for OLS are not of the same length.")

    # test predict for Poisson
    # if not np.allclose(fepois_mod.predict(data = data2), stats.predict(r_fixest_pois, newdata = data2)):  # noqa: W505
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

    # test with missing fixed effects


def test_predict_nas():
    # tests to fix #246: https://github.com/py-econometrics/pyfixest/issues/246

    # NaNs in depvar, covar and fixed effects
    data = get_data()

    # test 1
    fml = "Y ~ X1 + X2 | f1"
    fit = feols(fml, data=data)
    res = fit.predict(newdata=data)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=data)
    np.testing.assert_allclose(res, res_r)
    assert data.shape[0] == len(res)
    assert len(res) == len(res_r)

    # test 2
    newdata = data.copy()[0:200]
    newdata.loc[199, "f1"] = np.nan

    fml = "Y ~ X1 + X2 | f1"
    fit = feols(fml, data=data)
    res = fit.predict(newdata=newdata)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=newdata)
    np.testing.assert_allclose(res, res_r)
    assert newdata.shape[0] == len(res)
    assert len(res) == len(res_r)

    newdata.loc[198, "Y"] = np.nan
    res = fit.predict(newdata=newdata)
    res_r = stats.predict(fit_r, newdata=newdata)
    np.testing.assert_allclose(res, res_r)
    assert newdata.shape[0] == len(res)
    assert len(res) == len(res_r)

    # test 3
    fml = "Y ~ X1 + X2 + f1| f1 "
    fit = feols(fml, data=data)
    res = fit.predict(newdata=data)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=data)
    np.testing.assert_allclose(res, res_r)
    assert data.shape[0] == len(res)
    assert len(res) == len(res_r)


@pytest.mark.parametrize(
    "fml",
    [
        "Y~ X1 | f1",
        "Y~ X1 | f1 + f2",
        # "Y~ X1 | X3^X4",
    ],
)
def test_new_fixef_level(data, fml):
    data2 = data.copy()[1:500]

    feols_mod = feols(fml=fml, data=data, vcov="HC1")
    # fixest estimation
    r_fixest_ols = fixest.feols(
        ro.Formula(fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
        se="hetero",
    )

    updated_prediction_py = feols_mod.predict(newdata=data2)
    updated_prediction_r = stats.predict(r_fixest_ols, newdata=data2)

    if not np.allclose(updated_prediction_py, updated_prediction_r):
        raise ValueError("Updated predictions are not equal")


def test_categorical_covariate_predict():
    """Test if predict handles missing levels in covariate correctly."""
    rng = np.random.default_rng(12345)
    df = pd.DataFrame(
        {
            "y": rng.normal(0, 1, 1000),
            "x": rng.choice(range(124), size=1000, replace=True),
        }
    )

    df_sub = df.query("x == 1 or x == 2 or x == 3").copy()

    py_fit = feols("y ~ C(x, contr.treatment(base=1))", df)
    py_predict = py_fit.predict(df_sub)

    r_predict = np.array(
        [
            -0.14351887,
            -0.14351887,
            -0.04064215,
            -0.04064215,
            -0.04064215,
            0.02801946,
            -0.04064215,
            0.02801946,
            0.02801946,
            0.02801946,
            -0.04064215,
            0.02801946,
            0.02801946,
            0.02801946,
            0.02801946,
            -0.04064215,
            -0.14351887,
            -0.04064215,
            0.02801946,
            0.02801946,
            -0.04064215,
            0.02801946,
            -0.14351887,
            -0.04064215,
            -0.04064215,
            0.02801946,
            0.02801946,
            -0.14351887,
            0.02801946,
            -0.04064215,
            -0.14351887,
            0.02801946,
            -0.14351887,
            0.02801946,
        ]
    )

    np.testing.assert_allclose(py_predict, r_predict, rtol=1e-08, atol=1e-08)


def test_extract_variable_level():
    """Verify the correct extracation of lists, floats, and integers."""
    var = "C(SHOPPER_PLATFORM)[T.['ios', 'android']]"
    assert _extract_variable_level(var) == ("C(SHOPPER_PLATFORM)", "['ios', 'android']")
    var = "C(f3)[T.1.0]"
    assert _extract_variable_level(var) == ("C(f3)", "1.0")
    var = "C(f4)[T.1]"
    assert _extract_variable_level(var) == ("C(f4)", "1")
