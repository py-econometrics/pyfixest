import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from formulaic.errors import FactorEvaluationError
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

from pyfixest.did.estimation import did2s as did2s_pyfixest
from pyfixest.did.estimation import event_study, lpdid

pandas2ri.activate()
did2s = importr("did2s")
stats = importr("stats")
broom = importr("broom")


@pytest.fixture
def data():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    return df_het


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

    fit_did2s_r = did2s.did2s(
        data=data,
        yname="dep_var",
        first_stage=ro.Formula("~ 0 | state + year"),
        second_stage=ro.Formula("~ i(treat, ref = FALSE)"),
        treatment="treat",
        cluster_var="state",
    )

    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T

    if True:
        np.testing.assert_allclose(
            fit_did2s.coef(), stats.coef(fit_did2s_r), atol=1e-05, rtol=1e-05
        )
        np.testing.assert_allclose(
            fit_did2s.se(), float(did2s_df[2]), atol=1e-05, rtol=1e-05
        )


def test_did2s(data):
    """Test the did2s() function."""
    rng = np.random.default_rng(12345)
    data["X"] = rng.normal(size=len(data))

    # ATT, no covariates
    fit_did2s = did2s_pyfixest(
        data=data,
        yname="dep_var",
        first_stage="~ 0 | state + year",
        second_stage="~ treat",
        treatment="treat",
        cluster="state",
    )

    fit_did2s_r = did2s.did2s(
        data=data,
        yname="dep_var",
        first_stage=ro.Formula("~ 0 | state + year"),
        second_stage=ro.Formula("~ i(treat, ref = FALSE)"),
        treatment="treat",
        cluster_var="state",
    )

    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T

    np.testing.assert_allclose(
        fit_did2s.coef(), stats.coef(fit_did2s_r), atol=1e-05, rtol=1e-05
    )
    np.testing.assert_allclose(
        fit_did2s.se(), float(did2s_df[2]), atol=1e-05, rtol=1e-05
    )

    if True:
        # ATT, event study

        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ 0 | state + year",
            second_stage="~i(rel_year, ref = -1.0)",
            treatment="treat",
            cluster="state",
        )

        fit_r = did2s.did2s(
            data=data,
            yname="dep_var",
            first_stage=ro.Formula("~ 0 | state + year"),
            second_stage=ro.Formula("~ i(rel_year, ref = c(-1))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(
            fit.coef(), stats.coef(fit_r), atol=1e-05, rtol=1e-05
        )
        np.testing.assert_allclose(
            fit.se(), did2s_df[2].values.astype(float), atol=1e-05, rtol=1e-05
        )

    if True:
        # test event study with covariate in first stage
        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~i(rel_year, ref = -1.0)",
            treatment="treat",
            cluster="state",
        )

        fit_r = did2s.did2s(
            data=data,
            yname="dep_var",
            first_stage=ro.Formula("~ X | state + year"),
            second_stage=ro.Formula("~ i(rel_year, ref = c(-1))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(
            fit.coef(), stats.coef(fit_r), atol=1e-05, rtol=1e-05
        )
        np.testing.assert_allclose(
            fit.se(), did2s_df[2].values.astype(float), atol=1e-05, rtol=1e-05
        )

    if True:
        # test event study with covariate in first stage and second stage
        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year, ref = -1.0)",
            treatment="treat",
            cluster="state",
        )

        fit_r = did2s.did2s(
            data=data,
            yname="dep_var",
            first_stage=ro.Formula("~ X | state + year"),
            second_stage=ro.Formula("~ X + i(rel_year, ref = c(-1))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(
            fit.coef(), stats.coef(fit_r), atol=1e-05, rtol=1e-05
        )
        np.testing.assert_allclose(
            fit.se(), did2s_df[2].values.astype(float), atol=1e-05, rtol=1e-05
        )

    if True:
        # binary non boolean treatment variable, just check that it runs
        data["treat"] = data["treat"].astype(int)
        fit = did2s_pyfixest(
            data,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year, ref = -1.0)",
            treatment="treat",
            cluster="state",
        )


def test_errors(data):
    # test expected errors: treatment

    # boolean strings cannot be converted
    data["treat"] = data["treat"].astype(str)
    with pytest.raises(FactorEvaluationError):
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
    with pytest.raises(FactorEvaluationError):
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
