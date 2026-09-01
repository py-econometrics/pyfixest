import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture
def data():
    data = pf.get_data(seed=6534714, model="Fepois")
    data = data.dropna()

    return data


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1",
        "Y~X1 |f1",
        "Y ~ X1 | f1 + f2",
        "Y ~ 1 | f1",
        "Y ~ X1*X2",
        "Y ~ X1*X2 | f1",
    ],
)
@pytest.mark.parametrize("weights", [None, "weights"])
def test_ols_prediction_internally(data, fml, weights):
    """
    Test predict() method internally.

    Notes
    -----
    Currently only for OLS.
    """
    # predict via pf.feols, without fixed effect
    mod = pf.feols(fml=fml, data=data, vcov="iid", weights=weights)
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)
    assert mod._data.shape[0] == original_prediction.shape[0]
    assert mod._data.shape[0] == updated_prediction.shape[0]

    # now expect error with updated predicted being a subset of data
    updated_prediction2 = mod.predict(newdata=data.iloc[0:100, :])
    assert len(updated_prediction2) != len(updated_prediction), (
        "Arrays have the same length"
    )


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y~X1 |f1", "Y ~ X1 | f1 + f2"])
@pytest.mark.parametrize("weights", ["weights"])
def test_poisson_prediction_internally(data, weights, fml):
    mod = pf.fepois(fml=fml, data=data, vcov="hetero", weights=weights)
    original_prediction = mod.predict()
    updated_prediction = mod.predict(newdata=mod._data)
    np.allclose(original_prediction, updated_prediction)
    assert mod._data.shape[0] == original_prediction.shape[0]
    assert mod._data.shape[0] == updated_prediction.shape[0]

    # now expect error with updated predicted being a subset of data
    updated_prediction2 = mod.predict(newdata=data.iloc[0:100, :])
    assert len(updated_prediction2) != len(updated_prediction), (
        "Arrays have the same length"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y~ X1 | f1",
        "Y~ X1 | f1 + f2",
        "Y~ X1 | f1:f2",
    ],
)
def test_vs_fixest(data, fml):
    """Test predict and resid methods against fixest."""
    feols_mod = pf.feols(fml=fml, data=data, vcov="HC1")
    fepois_mod = pf.fepois(fml=fml, data=data, vcov="HC1")

    data2 = data.copy()[1:500]

    feols_mod.fixef(atol=1e-12, btol=1e-12)
    fepois_mod.fixef(atol=1e-12, btol=1e-12)

    # fixest estimation
    r_fml = fml.replace("f1:f2", "f1^f2")
    r_fixest_ols = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        se="hetero",
    )

    r_fixest_pois = fixest.fepois(
        ro.Formula(r_fml),
        data=data,
        se="hetero",
    )

    # test OLS fit
    if not np.allclose(feols_mod.coef().values, r_fixest_ols.rx2("coefficients")):
        raise ValueError("Coefficients are not equal")

    if not (stats.nobs(r_fixest_ols)[0] == feols_mod._N):
        raise ValueError("The Number of Observations does not match.")

    # test Poisson fit
    if not np.allclose(fepois_mod.coef(), r_fixest_pois.rx2("coefficients")):
        raise ValueError("Coefficients are not equal")

    # test sumFE for OLS
    if not np.allclose(feols_mod._sumFE, r_fixest_ols.rx2("sumFE")):
        raise ValueError("sumFE for OLS are not equal")

    # test sumFE for Poisson
    if not np.allclose(fepois_mod._sumFE, r_fixest_pois.rx2("sumFE"), atol=1e-07):
        raise ValueError("sumFE for Poisson are not equal")

    # test predict for OLS
    if not np.allclose(
        feols_mod.predict()[0:5], r_fixest_ols.rx2("fitted.values")[0:5]
    ):
        raise ValueError("Predictions for OLS are not equal")

    if not np.allclose(len(feols_mod.predict()), len(stats.predict(r_fixest_ols))):
        raise ValueError("Predictions for OLS are not the same length")

    if not np.allclose(
        fepois_mod.predict(type="response"), r_fixest_pois.rx2("fitted.values")
    ):
        raise ValueError("Predictions for Poisson are not equal")

    # test on new data - OLS.
    if not np.allclose(
        feols_mod.predict(newdata=data2)[0:5],
        stats.predict(r_fixest_ols, newdata=data2)[0:5],
        equal_nan=True,
    ):
        raise ValueError("Predictions for OLS are not equal with newdata.")

    if not np.allclose(
        len(feols_mod.predict(newdata=data2)),
        len(stats.predict(r_fixest_ols, newdata=data2)),
    ):
        raise ValueError("Predictions for OLS are not of the same length.")

    # test predict for Poisson
    if not np.allclose(
        fepois_mod.predict(newdata=data2, type="link")[11:16],
        stats.predict(r_fixest_pois, newdata=data2, type="link")[11:16],
        atol=1e-07,
        equal_nan=True,
    ):
        raise ValueError("Predictions for Poisson are not equal")

    # test resid for OLS
    if not np.allclose(feols_mod.resid()[20:25], r_fixest_ols.rx2("residuals")[20:25]):
        raise ValueError("Residuals for OLS are not equal")

    # test resid for Poisson
    if not np.allclose(fepois_mod.resid(), r_fixest_pois.rx2("residuals")):
        raise ValueError("Residuals for Poisson are not equal")

    # test fepois predict on newdata with an offset
    data_off = data.copy()
    data_off["off"] = np.log(np.random.default_rng(0).uniform(0.5, 3.0, len(data_off)))
    fepois_mod_off = pf.fepois(fml=fml, data=data_off, offset="off")
    r_fixest_pois_off = fixest.fepois(
        ro.Formula(r_fml), data=data_off, offset=ro.Formula("~off")
    )
    data2_off = data_off.copy()[1:500]
    if not np.allclose(
        fepois_mod_off.predict(newdata=data2_off, type="link")[11:16],
        stats.predict(r_fixest_pois_off, newdata=data2_off, type="link")[11:16],
        atol=1e-05,
        equal_nan=True,
    ):
        raise ValueError(
            "Predictions for Poisson with offset are not equal for type 'link'."
        )

    if not np.allclose(
        fepois_mod_off.predict(newdata=data2_off, type="response")[11:16],
        stats.predict(r_fixest_pois_off, newdata=data2_off, type="response")[11:16],
        atol=1e-05,
        equal_nan=True,
    ):
        raise ValueError(
            "Predictions for Poisson with offset are not equal for type 'response'."
        )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("weights_name", "weights_type"),
    [("weights", "aweights"), ("fweights", "fweights")],
)
def test_weighted_fixef_is_on_response_scale(data, weights_name, weights_type):
    """Weighted fixed effects and predictions stay in response units."""
    group_size = data.groupby("f1")["f1"].transform("size")
    weighted_data = data.loc[group_size > 1].copy().reset_index(drop=True)
    weighted_data["fweights"] = np.arange(len(weighted_data)) % 4 + 1

    fit = pf.feols(
        "Y ~ X1 | f1",
        data=weighted_data,
        weights=weights_name,
        weights_type=weights_type,
    )
    fixed_effects = fit.fixef(atol=1e-12, btol=1e-12)

    fit_r = fixest.feols(
        ro.Formula("Y ~ X1 | f1"),
        data=weighted_data,
        weights=ro.Formula(f"~{weights_name}"),
    )
    ro.globalenv[".pyfixest_weighted_fixef_fit"] = fit_r
    fixed_effects_r = ro.r["fixef"](fit_r).rx2("f1")
    fixed_effect_levels_r = np.asarray(
        ro.r("names(fixef(.pyfixest_weighted_fixef_fit)$f1)"), dtype=float
    )

    response_scale_fixed_effect = fit.predict() - weighted_data[
        "X1"
    ].to_numpy() * fit.coef().xs("X1")
    np.testing.assert_allclose(
        fit._sumFE,
        response_scale_fixed_effect,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit._sumFE,
        np.asarray(fit_r.rx2("sumFE")),
        rtol=1e-8,
        atol=1e-8,
    )

    fixed_effects_by_level = fixed_effects.set_index(
        fixed_effects["level"].astype(float)
    )["coefficient"].sort_index()
    fixed_effects_r_by_level = pd.Series(
        np.asarray(fixed_effects_r),
        index=fixed_effect_levels_r,
    ).sort_index()
    np.testing.assert_allclose(
        fixed_effects_by_level,
        fixed_effects_r_by_level,
        rtol=1e-8,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        fit.predict(),
        np.asarray(fit_r.rx2("fitted.values")),
        rtol=1e-8,
        atol=1e-8,
    )
    newdata = weighted_data.iloc[:100]
    np.testing.assert_allclose(
        fit.predict(newdata=newdata),
        np.asarray(stats.predict(fit_r, newdata=newdata)),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.against_r_core
def test_predict_nas():
    # tests to fix #246: https://github.com/py-econometrics/pyfixest/issues/246

    # NaNs in depvar, covar and fixed effects
    data = pf.get_data()

    # test 1
    fml = "Y ~ X1 + X2 | f1"
    fit = pf.feols(fml, data=data)
    res = fit.predict(newdata=data)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=data)
    np.testing.assert_allclose(res, res_r, atol=1e-05, rtol=1e-05)
    assert data.shape[0] == len(res)
    assert len(res) == len(res_r)

    # test 2
    newdata = data.copy()[0:200]
    newdata.loc[199, "f1"] = np.nan

    fml = "Y ~ X1 + X2 | f1"
    fit = pf.feols(fml, data=data)
    res = fit.predict(newdata=newdata)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=newdata)
    np.testing.assert_allclose(res, res_r, atol=1e-05, rtol=1e-05)
    assert newdata.shape[0] == len(res)
    assert len(res) == len(res_r)

    newdata.loc[198, "Y"] = np.nan
    res = fit.predict(newdata=newdata)
    res_r = stats.predict(fit_r, newdata=newdata)
    np.testing.assert_allclose(res, res_r, atol=1e-05, rtol=1e-05)
    assert newdata.shape[0] == len(res)
    assert len(res) == len(res_r)

    # test 3
    fml = "Y ~ X1 + X2 | f1 "
    fit = pf.feols(fml, data=data)
    res = fit.predict(newdata=data)
    fit_r = fixest.feols(ro.Formula(fml), data=data)
    res_r = stats.predict(fit_r, newdata=data)
    np.testing.assert_allclose(res, res_r, atol=1e-05, rtol=1e-05)
    assert data.shape[0] == len(res)
    assert len(res) == len(res_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        "Y~ X1 | f1",
        "Y~ X1 | f1 + f2",
        "Y~ X1 | f1:f2",
    ],
)
def test_new_fixef_level(data, fml):
    data2 = data.copy()[1:500]

    feols_mod = pf.feols(fml=fml, data=data, vcov="HC1")
    # fixest estimation
    r_fml = fml.replace("f1:f2", "f1^f2")
    r_fixest_ols = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        se="hetero",
    )

    updated_prediction_py = feols_mod.predict(newdata=data2)
    updated_prediction_r = stats.predict(r_fixest_ols, newdata=data2)

    if not np.allclose(updated_prediction_py[10:15], updated_prediction_r[10:15]):
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

    py_fit = pf.feols("y ~ C(x, contr.treatment(base=1))", df)
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


def test_specific_categorical_prediction():
    """Test prediction with a specific categorical case."""
    test_df = pd.DataFrame(
        {"y": [2, 3, 4, 5], "x": [1, 1, 2, 3], "f": ["a", "b", "a", "a"]}
    )
    test_model = pf.feols("y ~ x + C(f)", data=test_df)
    prediction = test_model.predict(newdata=pd.DataFrame({"x": [1], "f": ["b"]}))
    expected_prediction = 3
    np.testing.assert_almost_equal(prediction[0], expected_prediction, decimal=3)


def _lspline(series: pd.Series, knots: list[float]) -> np.array:
    """Generate a linear spline design matrix for the input series based on knots."""
    vector = series.values
    columns = []

    for i, knot in enumerate(knots):
        column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
        columns.append(column)
        vector = vector - column

    # Add the remainder as the last column
    columns.append(vector)

    # Combine columns into a design matrix
    return np.column_stack(columns)


def test_context_capture_with_out_of_sample_predict():
    data = pf.get_data(N=2000)

    spline_split = _lspline(data["X2"], [0, 1])
    data["X2_0"] = spline_split[:, 0]
    data["X2_0_1"] = spline_split[:, 1]
    data["X2_1"] = spline_split[:, 2]

    # Split data to training and testing
    data_train = data.iloc[:1000]
    data_test = data.iloc[1000:]

    explicit_fit = pf.feols("Y ~ X2_0 + X2_0_1 + X2_1 | f1 + f2", data=data_train)
    context_captured_fit = pf.feols(
        "Y ~ _lspline(X2,[0,1]) | f1 + f2", data=data_train, context=0
    )
    context_captured_fit_map = pf.feols(
        "Y ~ _lspline(X2,[0,1]) | f1 + f2",
        data=data_train,
        context={"_lspline": _lspline},
    )

    for context_fit in [context_captured_fit, context_captured_fit_map]:
        np.testing.assert_almost_equal(
            context_fit.predict(data_test), explicit_fit.predict(data_test), decimal=3
        )
