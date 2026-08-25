import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf

broom = importr("broom")
fixest = importr("fixest")
quantreg_r = importr("quantreg")
stats = importr("stats")

FEOLS_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 | f1 + f2"]
FEPOIS_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 | f1"]
FEGLM_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 | f1"]
QUANTREG_FORMULAS = ["Y ~ X1", "Y ~ X1 + X2"]
INFERENCE_TYPES = [
    pytest.param("iid", id="iid"),
    pytest.param("hetero", id="hetero"),
    pytest.param({"CRV1": "group_id"}, id="CRV1"),
]
# Match the comprehensive fixest matrix for every named coefficient.
COEFFICIENT_RTOL = 1e-8
COEFFICIENT_ATOL = 1e-8


@pytest.fixture(scope="module")
def linear_data():
    return pf.get_data(N=500, seed=76540251, model="Feols").dropna()


@pytest.fixture(scope="module")
def count_data():
    data = pf.get_data(N=500, seed=7651, model="Fepois").dropna()
    rng = np.random.default_rng(20260825)
    latent = 0.5 + 0.8 * data["X1"] - 0.4 * data["X2"]
    probability = 1 / (1 + np.exp(-latent))
    data["Y_bin"] = rng.binomial(1, probability)
    return data


@pytest.fixture(scope="module")
def quantile_data():
    rng = np.random.default_rng(3993)
    x1 = rng.normal(size=800)
    x2 = rng.normal(size=800)
    y = 1 + 2 * x1 + 3 * x2 + rng.normal(size=800)
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2})


def _r_coefficient_names(r_fit) -> list[str]:
    ro.globalenv[".pyfixest_fast_fit"] = r_fit
    names = ro.r("names(coef(.pyfixest_fast_fit))")
    return ["Intercept" if name == "(Intercept)" else str(name) for name in names]


def _r_tidy_fixest(r_fit) -> pd.DataFrame:
    tidy = pd.DataFrame(broom.tidy_fixest(r_fit, conf_int=ro.BoolVector([True]))).T
    tidy.columns = [
        "term",
        "estimate",
        "std.error",
        "statistic",
        "p.value",
        "conf.low",
        "conf.high",
    ]
    tidy["term"] = tidy["term"].replace({"(Intercept)": "Intercept"})
    numeric_columns = [column for column in tidy.columns if column != "term"]
    tidy[numeric_columns] = tidy[numeric_columns].astype(np.float64)
    return tidy.set_index("term")


def _r_inference(inference):
    if isinstance(inference, dict):
        return ro.Formula(f"~{inference['CRV1']}")
    return inference


def _r_ssc(k_adj, G_adj):
    return fixest.ssc(k_adj, "nonnested", False, G_adj, "min", "min")


def _compare_fixest_fit(
    py_fit,
    r_fit,
    data,
    *,
    inference_rtol,
    inference_atol,
    residual_rtol,
    residual_atol,
    prediction_rtol,
    prediction_atol,
    prediction_types,
):
    """Compare the same core fit and inference outputs as test_vs_fixest.py."""
    r_names = _r_coefficient_names(r_fit)
    r_tidy = _r_tidy_fixest(r_fit).loc[r_names]
    py_positions = [py_fit.coef().index.get_loc(name) for name in r_names]

    assert set(py_fit.coef().index) == set(r_names), (
        "estimated coefficient names differ from R fixest"
    )
    np.testing.assert_allclose(
        py_fit.coef().loc[r_names],
        r_tidy["estimate"],
        rtol=COEFFICIENT_RTOL,
        atol=COEFFICIENT_ATOL,
        err_msg="coefficient estimates differ from R fixest",
    )
    np.testing.assert_allclose(
        py_fit._vcov[np.ix_(py_positions, py_positions)],
        np.asarray(stats.vcov(r_fit)),
        rtol=inference_rtol,
        atol=inference_atol,
        err_msg="covariance matrix differs from R fixest",
    )
    np.testing.assert_allclose(
        py_fit.se().loc[r_names],
        r_tidy["std.error"],
        rtol=inference_rtol,
        atol=inference_atol,
        err_msg="standard errors differ from R fixest",
    )
    np.testing.assert_allclose(
        py_fit.tstat().loc[r_names],
        r_tidy["statistic"],
        rtol=inference_rtol,
        atol=inference_atol,
        err_msg="test statistics differ from R fixest",
    )
    np.testing.assert_allclose(
        py_fit.pvalue().loc[r_names],
        r_tidy["p.value"],
        rtol=inference_rtol,
        atol=inference_atol,
        err_msg="p-values differ from R fixest",
    )
    np.testing.assert_allclose(
        py_fit.confint().loc[r_names],
        r_tidy[["conf.low", "conf.high"]],
        rtol=inference_rtol,
        atol=inference_atol,
        err_msg="confidence intervals differ from R fixest",
    )

    ro.globalenv[".pyfixest_fast_fit"] = r_fit
    assert int(py_fit._df_k) == int(
        ro.r('attr(.pyfixest_fast_fit$cov.scaled, "df.K")')[0]
    ), "model degrees of freedom differ from R fixest"
    assert int(py_fit._df_t) == int(
        ro.r('attr(.pyfixest_fast_fit$cov.scaled, "df.t")')[0]
    ), "inference degrees of freedom differ from R fixest"
    assert int(stats.nobs(r_fit)[0]) == py_fit._N, (
        "observation count differs from R fixest"
    )

    np.testing.assert_allclose(
        py_fit.resid()[:5],
        np.asarray(stats.residuals(r_fit))[:5],
        rtol=residual_rtol,
        atol=residual_atol,
        err_msg="first five residuals differ from R fixest",
    )
    for prediction_type in prediction_types:
        np.testing.assert_allclose(
            py_fit.predict(newdata=data.iloc[:5], type=prediction_type),
            stats.predict(r_fit, newdata=data.iloc[:5], type=prediction_type),
            rtol=prediction_rtol,
            atol=prediction_atol,
            err_msg=f"first five {prediction_type} predictions differ from R fixest",
        )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", FEOLS_FORMULAS)
@pytest.mark.parametrize("inference", INFERENCE_TYPES)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("k_adj", [True])
@pytest.mark.parametrize("G_adj", [True])
def test_feols_fast_against_fixest(linear_data, fml, inference, weights, k_adj, G_adj):
    py_fit = pf.feols(
        fml=fml,
        data=linear_data,
        vcov=inference,
        weights=weights,
        ssc=pf.ssc(k_adj=k_adj, G_adj=G_adj),
    )
    r_kwargs = {
        "data": linear_data,
        "vcov": _r_inference(inference),
        "ssc": _r_ssc(k_adj, G_adj),
    }
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    r_fit = fixest.feols(ro.Formula(fml), **r_kwargs)

    _compare_fixest_fit(
        py_fit,
        r_fit,
        linear_data,
        inference_rtol=1e-7,
        inference_atol=1e-8,
        # Recovered fixed effects follow different iterative paths.
        residual_rtol=1e-6,
        residual_atol=1e-8,
        prediction_rtol=1e-6,
        prediction_atol=1e-8,
        prediction_types=("link",),
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", FEPOIS_FORMULAS)
@pytest.mark.parametrize("inference", INFERENCE_TYPES)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("k_adj", [True])
@pytest.mark.parametrize("G_adj", [True])
def test_fepois_fast_against_fixest(count_data, fml, inference, weights, k_adj, G_adj):
    py_fit = pf.fepois(
        fml=fml,
        data=count_data,
        vcov=inference,
        weights=weights,
        ssc=pf.ssc(k_adj=k_adj, G_adj=G_adj),
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )
    r_kwargs = {
        "data": count_data,
        "vcov": _r_inference(inference),
        "ssc": _r_ssc(k_adj, G_adj),
        "glm_tol": 1e-10,
        "glm_iter": 100,
    }
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    r_fit = fixest.fepois(ro.Formula(fml), **r_kwargs)

    _compare_fixest_fit(
        py_fit,
        r_fit,
        count_data,
        inference_rtol=1e-4,
        inference_atol=1e-6,
        # IRLS stopping and fixed-effect recovery agree less tightly than coefs.
        residual_rtol=1e-4,
        residual_atol=1e-6,
        prediction_rtol=1e-4,
        prediction_atol=1e-6,
        prediction_types=("link", "response"),
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", FEGLM_FORMULAS)
@pytest.mark.parametrize("inference", INFERENCE_TYPES)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
@pytest.mark.parametrize("k_adj", [True])
@pytest.mark.parametrize("G_adj", [True])
def test_feglm_fast_against_fixest(
    count_data, fml, inference, weights, family, k_adj, G_adj
):
    py_fml = fml.replace("Y", "Y_bin", 1) if family in ("logit", "probit") else fml
    py_fit = pf.feglm(
        fml=py_fml,
        data=count_data,
        family=family,
        vcov=inference,
        weights=weights,
        ssc=pf.ssc(k_adj=k_adj, G_adj=G_adj),
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )
    r_family = {
        "logit": stats.binomial(link="logit"),
        "probit": stats.binomial(link="probit"),
        "poisson": stats.poisson(),
    }[family]
    r_kwargs = {
        "data": count_data,
        "family": r_family,
        "vcov": _r_inference(inference),
        "ssc": _r_ssc(k_adj, G_adj),
        "glm_tol": 1e-10,
        "glm_iter": 100,
    }
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    r_fit = fixest.feglm(ro.Formula(py_fml), **r_kwargs)

    # IRLS-derived inference and fitted values need family-specific tolerances;
    # named coefficients remain subject to the 1e-8 contract above.
    tolerance = 1e-4 if family == "poisson" else 1e-5
    _compare_fixest_fit(
        py_fit,
        r_fit,
        count_data,
        inference_rtol=tolerance,
        inference_atol=1e-6,
        residual_rtol=tolerance,
        residual_atol=1e-6,
        prediction_rtol=tolerance,
        prediction_atol=1e-6,
        prediction_types=("link", "response"),
    )


@pytest.mark.against_r_core
def test_feglm_gaussian_reference_behavior(linear_data):
    """Lock in pyfixest's documented Gaussian-GLM compatibility decision."""
    fml = "Y ~ X1 + X2"
    py_ssc = pf.ssc(k_adj=True, G_adj=True)
    r_ssc = _r_ssc(k_adj=True, G_adj=True)
    py_glm = pf.feglm(
        fml=fml,
        data=linear_data,
        family="gaussian",
        vcov="iid",
        ssc=py_ssc,
        iwls_tol=1e-10,
    )
    py_ols = pf.feols(fml=fml, data=linear_data, vcov="iid", ssc=py_ssc)
    r_lm = stats.lm(ro.Formula(fml), data=linear_data)
    r_glm = stats.glm(ro.Formula(fml), data=linear_data, family=stats.gaussian())
    r_feols = fixest.feols(ro.Formula(fml), data=linear_data, vcov="iid", ssc=r_ssc)
    r_feglm = fixest.feglm(
        ro.Formula(fml),
        data=linear_data,
        family=stats.gaussian(),
        vcov="iid",
        ssc=r_ssc,
    )

    pd.testing.assert_frame_equal(py_glm.tidy(), py_ols.tidy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        py_glm._vcov,
        py_ols._vcov,
        rtol=0,
        atol=1e-10,
        err_msg="pyfixest Gaussian GLM and OLS covariance matrices differ",
    )
    _compare_fixest_fit(
        py_glm,
        r_feols,
        linear_data,
        inference_rtol=1e-8,
        inference_atol=1e-8,
        residual_rtol=1e-8,
        residual_atol=1e-8,
        prediction_rtol=1e-8,
        prediction_atol=1e-8,
        prediction_types=("link", "response"),
    )

    for r_fit in (r_lm, r_glm):
        np.testing.assert_allclose(
            py_glm.coef(),
            np.asarray(stats.coef(r_fit)),
            rtol=1e-8,
            atol=1e-8,
            err_msg="Gaussian-GLM coefficients differ from base R",
        )
        np.testing.assert_allclose(
            py_glm._vcov,
            np.asarray(stats.vcov(r_fit)),
            rtol=1e-8,
            atol=1e-8,
            err_msg="Gaussian-GLM covariance differs from base R",
        )
        assert py_glm._df_t == int(stats.df_residual(r_fit)[0]), (
            "Gaussian-GLM residual degrees of freedom differ from base R"
        )

    np.testing.assert_allclose(
        py_glm.coef(),
        np.asarray(stats.coef(r_feglm)),
        rtol=1e-8,
        atol=1e-8,
        err_msg="Gaussian-GLM coefficients differ from R fixest::feglm",
    )
    assert not np.allclose(
        py_glm._vcov, np.asarray(stats.vcov(r_feglm)), rtol=1e-8, atol=1e-8
    ), "expected fixest::feglm covariance divergence was not observed"


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", QUANTREG_FORMULAS)
@pytest.mark.parametrize("quantile", [0.02, 0.35, 0.5, 0.9])
@pytest.mark.parametrize("method", ["fn", "pfn"])
def test_quantreg_fast_against_r_quantreg(quantile_data, fml, quantile, method):
    py_fit = pf.quantreg(
        fml=fml,
        data=quantile_data,
        quantile=quantile,
        method=method,
        vcov="nid",
        tol=1e-6,
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=83838,
    )
    r_fit = quantreg_r.rq(
        ro.Formula(fml),
        data=quantile_data,
        tau=quantile,
        method=method,
        eps=1e-6,
    )
    r_names = _r_coefficient_names(r_fit)
    assert set(py_fit.coef().index) == set(r_names), (
        "estimated coefficient names differ from R quantreg"
    )

    # Mirror the solver-specific contract in tests/test_quantreg.py.
    np.testing.assert_allclose(
        py_fit.coef().loc[r_names],
        np.asarray(r_fit.rx2("coefficients")),
        rtol=1e-3,
        atol=1e-6,
        err_msg="quantile-regression coefficients differ from R quantreg",
    )
    r_summary = ro.r["summary"](r_fit, se="nid")
    np.testing.assert_allclose(
        py_fit.se().loc[r_names],
        np.asarray(r_summary.rx2("coefficients"))[:, 1],
        rtol=1e-3,
        atol=1e-6,
        err_msg="quantile-regression standard errors differ from R quantreg",
    )
    if method == "fn":
        py_residuals = py_fit.resid()
        r_residuals = np.asarray(r_fit.rx2("residuals"))
        np.testing.assert_allclose(
            py_residuals[:5],
            r_residuals[:5],
            rtol=1e-3,
            atol=1e-8,
            err_msg="first five residuals differ from R quantreg",
        )
        r_objective = np.sum(np.abs(r_residuals) * (quantile - (r_residuals < 0)))
        np.testing.assert_allclose(
            py_fit.objective_value,
            r_objective,
            rtol=1e-6,
            atol=1e-8,
            err_msg="quantile-regression objective differs from R quantreg",
        )
