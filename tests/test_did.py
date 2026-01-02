from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.did.estimation import did2s as did2s_pyfixest
from pyfixest.did.estimation import event_study, lpdid
from pyfixest.utils.check_r_install import check_r_install

pandas2ri.activate()
# Core Packages
stats = importr("stats")
broom = importr("broom")
fixest = importr("fixest")
# Extended Packages
if import_check := check_r_install("did2s", strict=False):
    did2s = importr("did2s")
# Check mpdata avaialibility
MPDATA_LOC = "C:/Users/alexa/Documents/pyfixest-zalando-talk/mpdta.csv"
mpdata_check = Path(MPDATA_LOC).is_file()


@pytest.fixture
def data():
    rng = np.random.default_rng(1243)
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    df_het["weights"] = rng.uniform(0, 10, size=len(df_het))
    df_het["X"] = rng.normal(size=len(df_het))

    return df_het


@pytest.mark.against_r_extended
def test_event_study(data):
    """Test the event_study() function."""
    fit_did2s = event_study(
        data=data,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        estimator="did2s",
    )

    run_did2s_r = False
    if run_did2s_r:
        fit_did2s_r = did2s.did2s(
            data=data,
            yname="dep_var",
            first_stage=ro.Formula("~ 0 | state + year"),
            second_stage=ro.Formula("~ i(treat, ref = FALSE)"),
            treatment="treat",
            cluster_var="state",
        )

        r_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
        r_df = pd.DataFrame(r_df).T
    else:
        r_df = {
            0: ["treat::TRUE"],
            1: [2.152215],
            2: [0.047607],
            3: [45.20833],
            4: [0.0],
            5: [2.058905],
            6: [2.245524],
        }
        r_df = pd.DataFrame(r_df)

    np.testing.assert_allclose(fit_did2s.coef(), r_df[1], atol=1e-05, rtol=1e-05)
    np.testing.assert_allclose(fit_did2s.se(), float(r_df[2]), atol=1e-05, rtol=1e-05)


@pytest.mark.against_r_extended
@pytest.mark.parametrize("weights", [None, "weights"])
def test_did2s(data, weights):
    """Test the did2s() function."""
    run_r = False
    if run_r:
        _get_r_did2s_results(data, weights)

    r_results = pd.read_csv(
        f"tests/data/all_did2s_dfs{'_weights' if weights is not None else ''}.csv",
        index_col=[0],
    )

    py_args = {
        "data": data,
        "yname": "dep_var",
        "first_stage": "~ 0 | state + year",
        "second_stage": "~ treat",
        "treatment": "treat",
        "cluster": "state",
    }

    if weights:
        py_args["weights"] = "weights"

    rng = np.random.default_rng(12345)
    data["X"] = rng.normal(size=len(data))

    # Model 1: ATT, no covariates
    fit_did2s_py1 = did2s_pyfixest(**py_args)
    fit_did2s_r1 = pd.DataFrame(r_results.xs("model1")).T

    np.testing.assert_allclose(
        np.asarray(fit_did2s_py1.coef(), dtype=np.float64),
        np.asarray(fit_did2s_r1.iloc[:, 2], dtype=np.float64),
        atol=1e-05,
        rtol=1e-05,
    )

    np.testing.assert_allclose(
        fit_did2s_py1.se(), float(fit_did2s_r1.iloc[:, 3]), atol=1e-05, rtol=1e-05
    )

    # Model 2
    py_args["second_stage"] = "~i(rel_year, ref = -1.0)"

    fit_py2 = did2s_pyfixest(**py_args)
    fit_did2s_r2 = r_results.xs("model2")

    np.testing.assert_allclose(
        np.asarray(fit_py2.coef(), dtype=np.float64),
        np.asarray(fit_did2s_r2.iloc[:, 2], dtype=np.float64),
        atol=1e-05,
        rtol=1e-05,
    )
    np.testing.assert_allclose(
        fit_py2.se(),
        fit_did2s_r2.iloc[:, 3].values.astype(float),
        atol=1e-05,
        rtol=1e-05,
    )

    # Model 3
    py_args["first_stage"] = "~ X | state + year"

    fit_py3 = did2s_pyfixest(**py_args)
    fit_did2s_r3 = r_results.xs("model3")

    np.testing.assert_allclose(
        np.asarray(fit_py3.coef(), dtype=np.float64),
        np.asarray(fit_did2s_r3.iloc[:, 2], dtype=np.float64),
        atol=1e-05,
        rtol=1e-05,
    )

    np.testing.assert_allclose(
        fit_py3.se(),
        fit_did2s_r3.iloc[:, 3].values.astype(float),
        atol=1e-05,
        rtol=1e-05,
    )

    # binary non boolean treatment variable, just check that it runs
    data["treat"] = data["treat"].astype(int)
    did2s_pyfixest(
        data,
        yname="dep_var",
        first_stage="~ X | state + year",
        second_stage="~ X + i(rel_year, ref = -1.0)",
        treatment="treat",
        cluster="state",
        weights="weights",
    )


def test_errors(data):
    # test expected errors: treatment

    # boolean strings cannot be converted
    data["treat"] = data["treat"].astype(str)
    with pytest.raises(ValueError):
        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
        )

    rng = np.random.default_rng(12)
    data["treat2"] = rng.choice([0, 1, 2], size=len(data))
    with pytest.raises(ValueError):
        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
        )

    data["treat3"] = rng.choice([0], size=len(data))
    with pytest.raises(ValueError):
        fit = did2s_pyfixest(  # noqa: F841
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
        )


def test_lpdid():
    """Test the lpdid estimator."""
    # test vs stata
    data = pd.read_stata("pyfixest/did/data/lpdidtestdata1.dta")
    data = data.astype(np.float64)

    fit = lpdid(
        data,
        yname="Y",
        idname="unit",
        tname="time",
        gname="event_date",
        att=False,
        pre_window=5,
        post_window=10,
    )
    coefs = fit._coeftable["Estimate"].values
    N = fit._coeftable["N"].values

    # values obtained from Stata
    np.testing.assert_allclose(coefs[0], -0.042566, rtol=1e-05)
    np.testing.assert_allclose(coefs[-1], 72.635834, rtol=1e-05)
    np.testing.assert_allclose(N[0], 40662)
    np.testing.assert_allclose(N[-1], 28709)

    fit = lpdid(
        data,
        yname="Y",
        idname="unit",
        tname="time",
        gname="event_date",
        att=True,
        pre_window=5,
        post_window=10,
    )

    coefs = fit._coeftable["Estimate"].values
    N = fit._coeftable["N"].values
    np.testing.assert_allclose(coefs[0], 31.79438, rtol=1e-05)
    np.testing.assert_allclose(N, 28709)

    # test vs R

    data = pd.read_csv("pyfixest/did/data/df_het.csv")

    rng = np.random.default_rng(1231)
    data["X"] = rng.normal(size=len(data))

    data.drop("treat", axis=1)
    data.drop("rel_year", axis=1)

    fit = lpdid(
        data=data, yname="dep_var", idname="unit", tname="year", gname="g", att=False
    )
    coefs = fit._coeftable["Estimate"].values

    # values obtained from R package lpdid
    # library(lpdid)
    # library(did2s)
    # data(data) # could also just load data from pyfixest/did/data/data.csv
    # data$rel_year <- ifelse(data$rel_year == Inf, -9999, data$rel_year)
    # fit <- lpdid(data, window = c(-20, 20), y = "dep_var",
    #          unit_index = "unit", time_index = "year",
    #          rel_time = "rel_year")
    # fit$coeftable$Estimate

    np.testing.assert_allclose(coefs[0], -0.073055295)
    np.testing.assert_allclose(coefs[-1], 2.911501018)

    fit.iplot()
    fit.tidy()


@pytest.mark.against_r_core
@pytest.mark.parametrize("unit", ["unit"])
@pytest.mark.parametrize("cluster", ["unit", "unit2"])
def test_fully_interacted(unit, cluster):
    df_multi_cohort = pd.read_csv(
        resources.files("pyfixest.did.data").joinpath("df_het.csv")
    )
    if cluster == "unit2":
        rng = np.random.default_rng(21)
        df_multi_cohort["unit2"] = rng.choice(range(100), size=len(df_multi_cohort))

    saturated_py = pf.event_study(
        data=df_multi_cohort,
        yname="dep_var",
        idname=unit,
        tname="year",
        gname="g",
        estimator="saturated",
        cluster=cluster,
    )
    saturated_py.test_treatment_heterogeneity()

    saturated_r = fixest.feols(
        ro.Formula("dep_var ~ 1 + sunab(g, year, no_agg = TRUE) | unit + year"),
        data=df_multi_cohort,
        vcov=ro.Formula(f"~{cluster}"),
    )

    r_tidy = pd.DataFrame(broom.tidy_fixest(saturated_r)).T
    r_est = r_tidy.iloc[:, 1].astype(float).values
    r_se = r_tidy.iloc[:, 2].astype(float).values

    py_est = np.asarray(saturated_py.coef(), dtype=float)
    py_se = np.asarray(saturated_py.se(), dtype=float)

    np.testing.assert_allclose(
        np.sort(r_est)[:5],
        np.sort(py_est)[:5],
        err_msg="R and Python *estimates* do not match for fully interacted model",
    )
    np.testing.assert_allclose(
        np.sort(r_se)[:5],
        np.sort(py_se)[:5],
        err_msg="R and Python *standard errors* do not match for fully interacted model",
    )

    # ---------------------------------------------------------
    # Test aggregated coefficients (no_agg=False in sunab)
    # ---------------------------------------------------------
    py_agg_tidy = saturated_py.aggregate()

    saturated_agg_r = fixest.feols(
        ro.Formula("dep_var ~ 1 + sunab(g, year, no_agg = FALSE) | unit + year"),
        data=df_multi_cohort,
        vcov=ro.Formula(f"~{cluster}"),
    )

    r_agg_tidy = pd.DataFrame(broom.tidy_fixest(saturated_agg_r)).T

    r_agg_est = r_agg_tidy.iloc[:, 1].astype(float).values
    r_agg_se = r_agg_tidy.loc[:, 2].astype(float).values

    py_agg_est = py_agg_tidy["Estimate"].astype(float).values
    py_agg_se = py_agg_tidy["Std. Error"].astype(float).values

    np.testing.assert_allclose(
        r_agg_est,
        py_agg_est,
        err_msg="R and Python *aggregated coefs* do not match",
    )
    np.testing.assert_allclose(
        r_agg_se,
        py_agg_se,
        err_msg="R and Python *aggregated SEs* do not match",
    )


@pytest.mark.against_r_core
@pytest.mark.skipif(
    mpdata_check is False,
    reason="mpdata not available online as csv, only run test locally.",
)
@pytest.mark.parametrize("mpdata_path", [MPDATA_LOC])
def test_fully_interacted_mpdata(mpdata_path):
    mpdata = pd.read_csv(mpdata_path)
    mpdata["first_treat"] = mpdata["first.treat"]

    fit_saturated = pf.event_study(
        data=mpdata,
        yname="lemp",
        idname="countyreal",
        tname="year",
        gname="first_treat",
        estimator="saturated",
    )
    fit_saturated.test_treatment_heterogeneity()

    r_model = fixest.feols(
        ro.Formula(
            "lemp ~ 1 + sunab(first.treat, year, no_agg=TRUE) | countyreal + year"
        ),
        data=mpdata,
    )

    r_tidy = pd.DataFrame(broom.tidy_fixest(r_model)).T
    r_est = r_tidy.iloc[:, 1].astype(float).values
    r_se = r_tidy.iloc[:, 2].astype(float).values

    py_est = np.asarray(fit_saturated.coef(), dtype=float)
    py_se = np.asarray(fit_saturated.se(), dtype=float)

    np.testing.assert_allclose(
        np.sort(r_est)[:5],
        np.sort(py_est)[:5],
        err_msg="R and Python *estimates* do not match for mpdata (fully interacted model)",
    )
    np.testing.assert_allclose(
        np.sort(r_se)[:5],
        np.sort(py_se)[:5],
        err_msg="R and Python *standard errors* do not match for mpdata (fully interacted model)",
    )

    py_agg_tidy = fit_saturated.aggregate()

    r_agg_model = fixest.feols(
        ro.Formula(
            "lemp ~ 1 + sunab(first.treat, year, no_agg=FALSE) | countyreal + year"
        ),
        data=mpdata,
    )
    r_agg_tidy = pd.DataFrame(broom.tidy_fixest(r_agg_model)).T
    r_agg_est = r_agg_tidy.iloc[:, 1].astype(float).values
    r_agg_se = r_agg_tidy.iloc[:, 2].astype(float).values

    py_agg_est = py_agg_tidy["Estimate"].astype(float).values
    py_agg_se = py_agg_tidy["Std. Error"].astype(float).values

    np.testing.assert_allclose(
        r_agg_est,
        py_agg_est,
        err_msg="R and Python *aggregated coefs* do not match for mpdata",
    )
    np.testing.assert_allclose(
        r_agg_se,
        py_agg_se,
        err_msg="R and Python *aggregated SEs* do not match for mpdata",
    )

    # run plotting
    fit_saturated.iplot()
    fit_saturated.iplot_aggregate()


def _get_r_did2s_results(data, weights):
    """Test the did2s() function."""
    all_did2s_dict = {}

    r_args = {
        "data": data,
        "yname": "dep_var",
        "first_stage": ro.Formula("~ 0 | state + year"),
        "second_stage": ro.Formula("~ i(treat, ref = FALSE)"),
        "treatment": "treat",
        "cluster_var": "state",
    }

    if weights:
        r_args["weights"] = "weights"

    rng = np.random.default_rng(12345)
    data["X"] = rng.normal(size=len(data))

    # Step 1
    fit_did2s_r = did2s.did2s(**r_args)
    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T
    all_did2s_dict["model1"] = did2s_df

    # Step

    # Step 2
    r_args["second_stage"] = ro.Formula("~ i(rel_year, ref = c(-1))")
    fit_r = did2s.did2s(**r_args)
    did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T
    all_did2s_dict["model2"] = did2s_df

    # Step 3
    r_args["first_stage"] = ro.Formula("~ X | state + year")
    fit_r = did2s.did2s(**r_args)
    did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T
    all_did2s_dict["model3"] = did2s_df

    all_dfs = pd.concat(all_did2s_dict)

    all_dfs.to_csv(
        f"tests/data/all_did2s_dfs{'_weights' if weights is not None else ''}.csv",
        index=True,
    )

    # Return combined DataFrame if needed
    return all_dfs
