from pyfixest.experimental.did import event_study
import pandas as pd
import numpy as np

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
did2s = importr("did2s")
stats = importr("stats")
broom = importr("broom")


def test_event_study():

    df_het = pd.read_csv("pyfixest/experimental/data/df_het.csv")

    fit_did2s = event_study(
        data = df_het,
        yname = "dep_var",
        idname= "state",
        tname = "year",
        gname = "g",
        estimator = "did2s"
    )

    fit_did2s_r = did2s.did2s(
        data = df_het,
        yname = "dep_var", first_stage = ro.Formula("~ 0 | state + year"),
        second_stage = ro.Formula("~ i(treat, ref = FALSE)"), treatment = "treat",
        cluster_var = "state"
    )
    did2s_df = broom.tidy_fixest(fit_did2s_r, conf_int=ro.BoolVector([True]))
    did2s_df = pd.DataFrame(did2s_df).T

    if True:
        np.testing.assert_allclose(
            fit_did2s.coef(), stats.coef(fit_did2s_r)
        )
        np.testing.assert_allclose(
            fit_did2s.se(), float(did2s_df[2])
        )




