from pyfixest.experimental.did import event_study
from pyfixest.experimental.did import did2s as did2s_pyfixest
import pandas as pd
import numpy as np
import pytest

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
did2s = importr("did2s")
stats = importr("stats")
broom = importr("broom")


def test_event_study():
    """
    Test the event_study() function.
    """

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")

    fit_did2s = event_study(
        data=df_het,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        estimator="did2s",
    )

    fit_did2s_r = did2s.did2s(
        data=df_het,
        yname="dep_var",
        first_stage=ro.Formula("~ 0 | state + year"),
        second_stage=ro.Formula("~ i(treat, ref = FALSE)"),
        treatment="treat",
        cluster_var="state",
    )

    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T

    if True:
        np.testing.assert_allclose(fit_did2s.coef(), stats.coef(fit_did2s_r))
        np.testing.assert_allclose(fit_did2s.se(), float(did2s_df[2]))


def test_did2s():
    """
    Test the did2s() function.
    """

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")
    df_het["X"] = np.random.normal(size=len(df_het))

    # ATT, no covariates
    fit_did2s = did2s_pyfixest(
        data=df_het,
        yname="dep_var",
        first_stage="~ 0 | state + year",
        second_stage="~ 0 + treat",
        treatment="treat",
        cluster="state",
    )

    fit_did2s_r = did2s.did2s(
        data=df_het,
        yname="dep_var",
        first_stage=ro.Formula("~ 0 | state + year"),
        second_stage=ro.Formula("~ i(treat, ref = FALSE)"),
        treatment="treat",
        cluster_var="state",
    )

    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T

    np.testing.assert_allclose(fit_did2s.coef(), stats.coef(fit_did2s_r))
    np.testing.assert_allclose(fit_did2s.se(), float(did2s_df[2]))

    if True:
        # ATT, event study

        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ 0 | state + year",
            second_stage="~i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )

        fit_r = did2s.did2s(
            data=df_het,
            yname="dep_var",
            first_stage=ro.Formula("~ 0 | state + year"),
            second_stage=ro.Formula("~ i(rel_year, ref = c(-1, Inf))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(fit.coef(), stats.coef(fit_r))
        np.testing.assert_allclose(fit.se(), did2s_df[2].values.astype(float))

    if True:
        # test event study with covariate in first stage
        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )

        fit_r = did2s.did2s(
            data=df_het,
            yname="dep_var",
            first_stage=ro.Formula("~ X | state + year"),
            second_stage=ro.Formula("~ i(rel_year, ref = c(-1, Inf))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(fit.coef(), stats.coef(fit_r))
        np.testing.assert_allclose(fit.se(), did2s_df[2].values.astype(float))

    if True:
        # test event study with covariate in first stage and second stage
        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )

        fit_r = did2s.did2s(
            data=df_het,
            yname="dep_var",
            first_stage=ro.Formula("~ X | state + year"),
            second_stage=ro.Formula("~ X + i(rel_year, ref = c(-1, Inf))"),
            treatment="treat",
            cluster_var="state",
        )

        did2s_df = broom.tidy_fixest(fit_r, conf_int=ro.BoolVector([True]))
        did2s_df = pd.DataFrame(did2s_df).T

        np.testing.assert_allclose(fit.coef(), stats.coef(fit_r))
        np.testing.assert_allclose(fit.se(), did2s_df[2].values.astype(float))

    if True:
        # binary non boolean treatment variable, just check that it runs
        df_het["treat"] = df_het["treat"].astype(int)
        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )


def test_errors():
    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")

    # test expected errors: treatment

    # boolean strings cannot be converted
    df_het["treat"] = df_het["treat"].astype(str)
    with pytest.raises(ValueError):
        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )

    df_het["treat2"] = np.random.choice([0, 1, 2], size=len(df_het))
    with pytest.raises(ValueError):
        fit = did2s_pyfixest(
            df_het,
            yname="dep_var",
            first_stage="~ X | state + year",
            second_stage="~ X + i(rel_year)",
            treatment="treat",
            cluster="state",
            i_ref1=[-1.0, np.inf],
        )
