import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf

fixest = importr("fixest")
quantreg_r = importr("quantreg")
stats = importr("stats")

FIXEST_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 | f1 + f2"]
POISSON_FORMULAS = ["Y ~ X1 + X2", "Y ~ X1 + X2 | f1"]
GLM_FORMULAS = ["Y_bin ~ X1 + X2", "Y_bin ~ X1 + X2 | f1"]
QUANTREG_FORMULAS = ["Y ~ X1", "Y ~ X1 + X2"]
VCOV_TYPES = ["iid", "hetero"]

R_SSC = fixest.ssc(True, "nonnested", False, True, "min", "min")
PY_SSC = pf.ssc(k_adj=True, G_adj=True)


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


def _compare_fixest_fit(
    py_fit,
    r_fit,
    data,
    *,
    rtol,
    atol,
    prediction_type,
    prediction_rtol,
    prediction_atol,
):
    r_names = _r_coefficient_names(r_fit)
    py_coef = py_fit.coef()
    assert set(py_coef.index) == set(r_names)

    py_positions = [py_coef.index.get_loc(name) for name in r_names]
    np.testing.assert_allclose(
        py_coef.loc[r_names].to_numpy(),
        np.asarray(stats.coef(r_fit)),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        py_fit._vcov[np.ix_(py_positions, py_positions)],
        np.asarray(stats.vcov(r_fit)),
        rtol=rtol,
        atol=atol,
    )
    assert int(stats.nobs(r_fit)[0]) == py_fit._N
    np.testing.assert_allclose(
        py_fit.predict(newdata=data.iloc[:5], type=prediction_type),
        stats.predict(r_fit, newdata=data.iloc[:5], type=prediction_type),
        rtol=prediction_rtol,
        atol=prediction_atol,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", FIXEST_FORMULAS)
@pytest.mark.parametrize("vcov", VCOV_TYPES)
@pytest.mark.parametrize("weights", [None, "weights"])
def test_feols_fast_against_fixest(linear_data, fml, vcov, weights):
    py_fit = pf.feols(
        fml=fml,
        data=linear_data,
        vcov=vcov,
        weights=weights,
        ssc=PY_SSC,
    )
    r_kwargs = {"data": linear_data, "vcov": vcov, "ssc": R_SSC}
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    r_fit = fixest.feols(ro.Formula(fml), **r_kwargs)

    _compare_fixest_fit(
        py_fit,
        r_fit,
        linear_data,
        rtol=1e-8,
        atol=1e-8,
        prediction_type="link",
        # Recovered fixed effects use different iterative paths.
        prediction_rtol=1e-5,
        prediction_atol=1e-5,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", POISSON_FORMULAS)
@pytest.mark.parametrize("vcov", VCOV_TYPES)
@pytest.mark.parametrize("weights", [None, "weights"])
def test_fepois_fast_against_fixest(count_data, fml, vcov, weights):
    py_fit = pf.fepois(
        fml=fml,
        data=count_data,
        vcov=vcov,
        weights=weights,
        ssc=PY_SSC,
        iwls_tol=1e-10,
    )
    r_kwargs = {"data": count_data, "vcov": vcov, "ssc": R_SSC}
    if weights is not None:
        r_kwargs["weights"] = ro.Formula(f"~{weights}")
    r_fit = fixest.fepois(ro.Formula(fml), **r_kwargs)

    _compare_fixest_fit(
        py_fit,
        r_fit,
        count_data,
        # IRLS stopping and fixed-effect recovery differ slightly across packages.
        rtol=1e-4,
        atol=1e-6,
        prediction_type="response",
        prediction_rtol=1e-4,
        prediction_atol=1e-5,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", GLM_FORMULAS)
@pytest.mark.parametrize("vcov", VCOV_TYPES)
@pytest.mark.parametrize("family", ["logit", "probit"])
def test_feglm_fast_against_fixest(count_data, fml, vcov, family):
    py_fit = pf.feglm(
        fml=fml,
        data=count_data,
        family=family,
        vcov=vcov,
        ssc=PY_SSC,
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )
    r_fit = fixest.feglm(
        ro.Formula(fml),
        data=count_data,
        family=stats.binomial(link=family),
        vcov=vcov,
        ssc=R_SSC,
        glm_tol=1e-10,
        glm_iter=100,
    )

    _compare_fixest_fit(
        py_fit,
        r_fit,
        count_data,
        # Binary GLM IRLS paths agree less tightly than closed-form OLS.
        rtol=1e-5,
        atol=1e-7,
        prediction_type="response",
        prediction_rtol=1e-4,
        prediction_atol=1e-5,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", QUANTREG_FORMULAS)
@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
def test_quantreg_fast_against_r_quantreg(quantile_data, fml, quantile):
    py_fit = pf.quantreg(
        fml=fml,
        data=quantile_data,
        quantile=quantile,
        method="fn",
        vcov="nid",
        tol=1e-6,
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=83838,
    )
    r_fit = quantreg_r.rq(
        ro.Formula(fml),
        data=quantile_data,
        tau=quantile,
        method="fn",
        eps=1e-6,
    )
    r_names = _r_coefficient_names(r_fit)
    assert set(py_fit.coef().index) == set(r_names)

    np.testing.assert_allclose(
        py_fit.coef().loc[r_names].to_numpy(),
        np.asarray(r_fit.rx2("coefficients")),
        # Both implementations stop at a 1e-6 optimization tolerance.
        rtol=1e-3,
        atol=1e-6,
    )
    r_summary = ro.r["summary"](r_fit, se="nid")
    np.testing.assert_allclose(
        py_fit.se().loc[r_names].to_numpy(),
        np.asarray(r_summary.rx2("coefficients"))[:, 1],
        rtol=1e-3,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        py_fit.resid()[:5],
        np.asarray(r_fit.rx2("residuals"))[:5],
        rtol=1e-3,
        atol=1e-8,
    )
