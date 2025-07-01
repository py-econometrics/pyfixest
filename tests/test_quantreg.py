import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
import statsmodels.formula.api as smf
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf

# Import R packages
quantreg = importr("quantreg")
stats = importr("stats")

# Enable pandas conversion
pandas2ri.activate()


@pytest.fixture
def stata_results_crv():
    """
    Results from Stata's qreg2 package.
    For code on how the Stata results were generated, see
    this issue: https://github.com/py-econometrics/pyfixest/issues/923.
    """
    return pd.DataFrame(
        {
            "fml": ["Y~X1", "Y~X1", "Y~X1", "Y~X1+X2", "Y~X1+X2", "Y~X1+X2"],
            "quantile": [0.35, 0.50, 0.95, 0.35, 0.50, 0.95],
            "Intercept": [
                -0.1282921,
                0.919176,
                6.036114,
                0.3199207,
                1.071671,
                4.030318,
            ],
            "coef_X1": [1.692162, 1.730726, 1.695720, 1.653593, 1.592271, 1.664043],
            "coef_X2": [np.nan, np.nan, np.nan, 0.8902029, 0.8690366, 0.8852484],
            "se_Intercept": [
                0.1939684,
                0.216606,
                0.3964476,
                0.189286,
                0.1869863,
                0.2984174,
            ],
            "se_X1": [0.0992838, 0.1270346, 0.1482422, 0.0347237, 0.0486158, 0.1144752],
            "se_X2": [np.nan, np.nan, np.nan, 0.0210688, 0.0274879, 0.0278543],
        }
    )


@pytest.mark.against_r_core
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
        "nid",
    ],
)
@pytest.mark.parametrize("data", [pf.get_data(N=5_000, seed=3131)])
@pytest.mark.parametrize("quantile", [0.02, 0.35, 0.5, 0.9])
@pytest.mark.parametrize("method", ["fn", "pfn"])
def test_quantreg_vs_r(data, fml, vcov, quantile, method):
    """
    Test that pyfixest's quantreg implementation equals R's quantreg implementation.
    Only tests nid errors as Powell sandwich errors are not implemented in quantreg.
    Tested below against statsmodels.
    """
    # Fit model in pyfixest

    rng = np.random.default_rng(3993)
    data["Y"] = 1 + 2 * data["X1"] + rng.normal(size=len(data))
    data["Y"] = data["Y"] + 3 * data["X2"] if "X2" in fml else data["Y"]

    tol = 1e-6

    fit_py = pf.quantreg(
        fml,
        data=data,
        vcov=vcov,
        quantile=quantile,
        method=method,
        tol=tol,
        ssc=pf.ssc(adj=False, cluster_adj=False),
    )

    # Fit model in R
    r_data = pandas2ri.py2rpy(data)
    r_formula = ro.Formula(fml)

    # Fit R model
    fit_r = quantreg.rq(r_formula, data=r_data, tau=quantile, method=method, eps=tol)

    # Compare coefficients
    py_coef = fit_py.coef().to_numpy()
    r_coef = np.array(fit_r.rx2("coefficients"))
    np.testing.assert_allclose(py_coef, r_coef, rtol=1e-08, atol=1e-08)

    py_se = fit_py.se().to_numpy()
    r_summ = ro.r["summary"](fit_r, se=vcov)

    coeff_mat = r_summ.rx2("coefficients")
    r_se = np.array(coeff_mat)[:, 1]
    np.testing.assert_allclose(py_se, r_se, rtol=1e-07, atol=1e-07)

    if method == "fn":
        # no residuals for pfn?
        # compare residuals
        py_resid = fit_py.resid()
        r_resid = np.array(fit_r.rx2("residuals"))
        np.testing.assert_allclose(py_resid[:5], r_resid[:5], rtol=1e-03, atol=1e-08)

        # compare objective function
        def total_loss(resid, quantile):
            return np.sum(np.abs(resid) * (quantile - (resid < 0)))

        # py_loss = total_loss(py_resid, quantile)
        py_loss = fit_py.objective_value
        r_loss = total_loss(r_resid, quantile)
        np.testing.assert_allclose(py_loss, r_loss, rtol=1e-06, atol=1e-08)


@pytest.mark.against_r_core
def test_qplot():
    data = pf.get_data(N=1000)
    fit1 = pf.quantreg("Y ~ X1 + X2", data=data, quantile=0.5, method="fn")
    fit2 = pf.quantreg("Y ~ X1 + X2", data=data, quantile=0.9, method="fn")

    pf.qplot([fit1, fit2])


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1+X2"])
@pytest.mark.parametrize("data", [pf.get_data(seed=12).dropna()])
@pytest.mark.parametrize("quantile", [0.35, 0.5, 0.95])
def test_quantreg_crv(data, fml, quantile, stata_results_crv):
    "Test quantreg's CRV errors vs Stata's qreg2."

    def expected(q, cols):
        row = stata_results_crv[
            (stata_results_crv["fml"] == fml) & (stata_results_crv["quantile"] == q)
        ]
        return row[cols].to_numpy().ravel()

    fit = pf.quantreg(
        fml,
        data=data,
        vcov={"CRV1": "f1"},
        quantile=quantile,
        ssc=pf.ssc(adj=False, cluster_adj=False),
    )

    coef = fit.coef().to_numpy()
    se = fit.se().to_numpy()

    coef_cols = ["Intercept", "coef_X1"] + (["coef_X2"] if "X2" in fml else [])
    se_cols = ["se_Intercept", "se_X1"] + (["se_X2"] if "X2" in fml else [])

    exp_coef = expected(quantile, coef_cols)
    exp_se = expected(quantile, se_cols)

    np.testing.assert_allclose(coef, exp_coef, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(se, exp_se, rtol=1e-6, atol=1e-6)


def get_data2(N, seed):
    "Generate data for testing."
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 2))
    Y = 1 + 2 * X[:, 0] + 3 * X[:, 1] - 2 * X[:, 1] ** 2 + rng.normal(size=N)
    f1 = rng.choice(range(10), size=N)
    return pd.DataFrame({"Y": Y, "X1": X[:, 0], "X2": X[:, 1], "f1": f1})


@pytest.mark.against_r_core
@pytest.mark.parametrize("data", [get_data2(N=1000, seed=2141233)])
@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 + X2"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", "nid", {"CRV1": "f1"}])
@pytest.mark.parametrize("method", ["fn", "pfn"])
@pytest.mark.parametrize("multi_method", ["cfm1", "cfm2"])
def test_quantreg_multiple_quantiles(data, fml, vcov, method, multi_method):
    "Test that multiple quantile syntax via QuantregMulti produces the same results as the single quantile syntax."
    quantiles = list(np.linspace(0.05, 0.95, 10))
    seed = 1231

    fit_single = [
        pf.quantreg(fml, data=data, quantile=q, method=method, vcov=vcov, seed=seed)
        for q in quantiles
    ]
    fit_multi = pf.quantreg(
        fml,
        data=data,
        quantile=quantiles,
        vcov=vcov,
        seed=seed,
        method=method,
        multi_method="cfm1",
    )

    for q in range(len(quantiles)):
        # test coefficients
        single_coef = fit_single[q].coef().to_numpy()
        multi_coef = fit_multi.fetch_model(q).coef().to_numpy()

        np.testing.assert_allclose(
            single_coef,
            multi_coef,
            rtol=1e-06,  # is this too low?
            atol=1e-06,  # is this too low?
            err_msg=f"Quantile: {quantiles[q]} with method: {method} and multi_method: {multi_method}",
        )

        # test standard errors
        single_se = fit_single[q].se().to_numpy()
        multi_se = fit_multi.fetch_model(q).se().to_numpy()

        np.testing.assert_allclose(
            single_se,
            multi_se,
            rtol=1e-06,  # is this too low?
            atol=1e-06,  # is this too low?
            err_msg=f"Quantile: {quantiles[q]}",
        )


@pytest.mark.against_r_core
def test_pfn_seed():
    "Test that calling method = 'pfn' on the same seed leads to identical results."
    data = pf.get_data(N=100, seed=3131).dropna()

    fml = "Y ~ X1"
    method = "pfn"
    seed = 7272712
    fit1 = pf.quantreg(fml=fml, data=data, method=method, seed=seed)
    fit2 = pf.quantreg(fml=fml, data=data, method=method, seed=seed)

    fit1_coef = fit1.coef()
    fit2_coef = fit2.coef()

    np.testing.assert_allclose(
        fit1_coef,
        fit2_coef,
        rtol=1e-09,
        atol=1e-09,
    )


@pytest.mark.against_r_core
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
        "hetero",
    ],
)
@pytest.mark.parametrize("data", [pf.get_data(N=100_000, seed=4242)])
@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("method", ["fn"])
def test_quantreg_vs_statsmodels(data, fml, vcov, quantile, method):
    """
    Test that pyfixest's quantreg implementation equals statsmodels' quantreg implementation.
    Used to verify correctness of iid and hetero standard errors.
    Note: minor differences because pf uses uniform kernel, while statsmodels uses a epanechnikov kernel,
    plus the fact that pf uses a interior point solver while statsmodels uses IWLS.
    """
    rng = np.random.default_rng(3993)
    data["Y"] = 1 + 2 * data["X1"] + rng.normal(size=len(data))
    data["Y"] = data["Y"] + 3 * data["X2"] if "X2" in fml else data["Y"]

    fit_py = pf.quantreg(
        fml,
        data=data,
        vcov=vcov,
        quantile=quantile,
        method=method,
        ssc=pf.ssc(adj=False, cluster_adj=False),
    )

    fit_sm = smf.quantreg(fml, data=data).fit(q=quantile)

    py_coef = fit_py.coef().to_numpy()
    sm_coef = fit_sm.params.to_numpy()
    np.testing.assert_allclose(py_coef, sm_coef, rtol=0.01, atol=1e-03)

    py_se = fit_py.se().to_numpy()
    if vcov == "iid":
        fit_sm_iid = smf.quantreg(fml, data=data).fit(
            q=quantile, vcov="iid", kernel="cos", bandwidth="hsheather"
        )
        sm_se = fit_sm_iid.bse.to_numpy()
        np.testing.assert_allclose(py_se, sm_se, rtol=0.03, atol=1e-03)
    else:
        fit_sm_robust = smf.quantreg(fml, data=data).fit(
            q=quantile, vcov="robust", kernel="cos", bandwidth="hsheather"
        )
        sm_se = fit_sm_robust.bse.to_numpy()
        np.testing.assert_allclose(py_se, sm_se, rtol=0.03, atol=1e-03)
