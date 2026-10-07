import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
import statsmodels.formula.api as smf
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation.quantreg.vcov_ import vcov_hetero_qreg

# Import R packages
quantreg = importr("quantreg")
stats = importr("stats")


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
    Tests nid errors; heteroskedastic kernel-sandwich errors are tested separately.
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
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=83838,
    )

    # Fit model in R
    r_data = pandas2ri.py2rpy(data)
    r_formula = ro.Formula(fml)

    # Fit R model
    fit_r = quantreg.rq(r_formula, data=r_data, tau=quantile, method=method, eps=tol)

    # Compare coefficients
    py_coef = fit_py.coef().to_numpy()
    r_coef = np.array(fit_r.rx2("coefficients"))
    np.testing.assert_allclose(py_coef, r_coef, rtol=1e-03, atol=1e-06)

    py_se = fit_py.se().to_numpy()
    r_summ = ro.r["summary"](fit_r, se=vcov)

    coeff_mat = r_summ.rx2("coefficients")
    r_se = np.array(coeff_mat)[:, 1]
    np.testing.assert_allclose(py_se, r_se, rtol=1e-03, atol=1e-06)

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
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=9389323,
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


def get_heteroskedastic_quantreg_data(N=2_000):
    "Generate the deterministic heteroskedastic data from issue #1744."
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    x2 = rng.uniform(size=N)
    y = 1 + x + (1 + 2 * np.abs(x)) * rng.normal(size=N)
    return pd.DataFrame({"y": y, "x": x, "x2": x2})


@pytest.mark.against_r_core
@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.9])
@pytest.mark.parametrize("method", ["fn", "pfn"])
def test_quantreg_hetero_vs_r_kernel_sandwich(quantile, method):
    "Test heteroskedastic kernel-sandwich errors against R quantreg."
    data = get_heteroskedastic_quantreg_data()
    ssc = pf.ssc(k_adj=False, G_adj=False)
    tol = 1e-8

    fit_py = pf.quantreg(
        "y ~ x + x2",
        data=data,
        quantile=quantile,
        vcov="hetero",
        method=method,
        tol=tol,
        ssc=ssc,
        seed=83838,
    )

    r_data = pandas2ri.py2rpy(data)
    fit_r = quantreg.rq(
        ro.Formula("y ~ x + x2"),
        data=r_data,
        tau=quantile,
        method=method,
        eps=tol,
    )
    with ro.default_converter.context():
        r_summary = ro.r["summary"](fit_r, se="ker")
        r_coefficients = r_summary.rx2("coefficients")
        r_names = list(ro.r["dimnames"](r_coefficients)[0])
        r_n_obs = int(ro.r["nrow"](fit_r.rx2("model"))[0])
    r_names = ["Intercept" if name == "(Intercept)" else name for name in r_names]
    r_se = pd.Series(np.asarray(r_coefficients)[:, 1], index=r_names)
    py_se = fit_py.se()

    assert fit_py.sample_info.n_obs == r_n_obs, (
        "PyFixest and R quantreg must retain the same observations."
    )
    assert list(py_se.index) == r_names, (
        "PyFixest and R quantreg must report the same named coefficients."
    )
    # The pfn solver showed the largest relative discrepancy in design review
    # (1.8e-9), so 1e-8 covers solver stopping error without masking drift.
    np.testing.assert_allclose(
        py_se.to_numpy(),
        r_se.to_numpy(),
        rtol=1e-8,
        atol=1e-10,
        err_msg=f"hetero SE mismatch at tau={quantile} with method={method}",
    )


def test_quantreg_hetero_materially_differs_from_iid():
    "Test that heteroskedastic inference no longer collapses to IID inference."
    data = get_heteroskedastic_quantreg_data()
    ssc = pf.ssc(k_adj=False, G_adj=False)

    iid_se = pf.quantreg(
        "y ~ x + x2", data=data, quantile=0.5, vcov="iid", ssc=ssc
    ).se()
    hetero_se = pf.quantreg(
        "y ~ x + x2", data=data, quantile=0.5, vcov="hetero", ssc=ssc
    ).se()

    assert hetero_se["x"] > 1.5 * iid_se["x"], (
        "The heteroskedastic slope SE must materially exceed its IID counterpart."
    )


@pytest.mark.against_r_core
def test_quantreg_hetero_halves_boundary_bandwidth():
    "Test the R-compatible boundary bandwidth at an extreme quantile."
    data = get_heteroskedastic_quantreg_data(N=60)
    ssc = pf.ssc(k_adj=False, G_adj=False)
    quantile = 0.05

    fit_py = pf.quantreg(
        "y ~ x + x2",
        data=data,
        quantile=quantile,
        vcov="hetero",
        method="fn",
        tol=1e-8,
        ssc=ssc,
    )
    fit_r = quantreg.rq(
        ro.Formula("y ~ x + x2"),
        data=pandas2ri.py2rpy(data),
        tau=quantile,
        method="fn",
        eps=1e-8,
    )
    with ro.default_converter.context():
        r_coefficients = ro.r["summary"](fit_r, se="ker").rx2("coefficients")
        r_names = list(ro.r["dimnames"](r_coefficients)[0])
        r_n_obs = int(ro.r["nrow"](fit_r.rx2("model"))[0])
    r_names = ["Intercept" if name == "(Intercept)" else name for name in r_names]

    assert fit_py.sample_info.n_obs == r_n_obs, (
        "PyFixest and R quantreg must retain the same boundary-case observations."
    )
    assert list(fit_py.se().index) == r_names, (
        "PyFixest and R quantreg must report the same boundary-case coefficients."
    )
    np.testing.assert_allclose(
        fit_py.se().to_numpy(),
        np.asarray(r_coefficients)[:, 1],
        rtol=1e-8,
        atol=1e-10,
        err_msg="hetero SE mismatch after boundary-bandwidth halving",
    )


@pytest.mark.parametrize("method", ["fn", "pfn"])
def test_quantreg_hetero_rejects_zero_residual_bandwidth(method):
    "Test that a constant outcome reports a degenerate kernel bandwidth."
    data = pd.DataFrame({"y": np.ones(200), "x": np.linspace(-1, 1, 200)})

    with pytest.raises(ValueError, match="kernel residual bandwidth"):
        pf.quantreg(
            "y ~ x",
            data=data,
            quantile=0.5,
            vcov="hetero",
            method=method,
            seed=83838,
        )


def test_vcov_hetero_rejects_deterministic_zero_residual_bandwidth():
    "Test that zero residuals deterministically reject a zero kernel bandwidth."
    n_obs = 20
    X = np.column_stack((np.ones(n_obs), np.linspace(-1, 1, n_obs)))

    with pytest.raises(ValueError, match="kernel residual bandwidth"):
        vcov_hetero_qreg(
            X=X,
            Y=np.zeros(n_obs),
            u_hat=np.zeros(n_obs),
            q=0.5,
            N=n_obs,
        )


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
    Note: minor differences because pyfixest's hetero route uses R quantreg's
    Gaussian kernel sandwich, while statsmodels uses an Epanechnikov kernel,
    plus the fact that pyfixest uses an interior point solver while statsmodels uses IWLS.
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
        ssc=pf.ssc(k_adj=False, G_adj=False),
        seed=383838,
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
