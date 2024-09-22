import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.did.estimation import did2s, event_study


@pytest.fixture
def data():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")
    return df_het


def test_event_study_twfe(data):
    twfe = event_study(
        data=data,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        att=True,
        estimator="twfe",
    )

    twfe_feols = pf.feols("dep_var ~ treat | state + year", data=data)

    assert np.allclose(
        twfe.coef().values, twfe_feols.coef().values
    ), "TWFE coefficients are not the same."
    assert np.allclose(
        twfe.se().values, twfe_feols.se().values
    ), "TWFE standard errors are not the same."
    assert np.allclose(
        twfe.pvalue().values, twfe_feols.pvalue().values
    ), "TWFE p-values are not the same."
    assert np.allclose(
        twfe.confint().values, twfe_feols.confint().values
    ), "TWFE confidence intervals are not the same."

    twfe = event_study(
        data=data,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        att=False,
        estimator="twfe",
    )

    twfe_feols = pf.feols("dep_var ~ i(treat, ref = -1) | state + year", data=data)

    assert np.allclose(
        twfe.coef().values, twfe_feols.coef().values
    ), "TWFE coefficients are not the same."
    assert np.allclose(
        twfe.se().values, twfe_feols.se().values
    ), "TWFE standard errors are not the same."
    assert np.allclose(
        twfe.pvalue().values, twfe_feols.pvalue().values
    ), "TWFE p-values are not the same."
    # assert np.allclose(
    #    twfe.confint().values, twfe_feols.confint().values
    # ), "TWFE confidence intervals are not the same."


def test_event_study_did2s(data):
    event_study_did2s = event_study(
        data=data,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        att=True,
        estimator="did2s",
    )

    fit_did2s = did2s(
        data=data,
        yname="dep_var",
        first_stage="~ 0 | state + year",
        second_stage="~treat",
        treatment="treat",
        cluster="state",
    )

    assert np.allclose(
        event_study_did2s.coef().values, fit_did2s.coef().values
    ), "DID2S coefficients are not the same."
    assert np.allclose(
        event_study_did2s.se().values, fit_did2s.se().values
    ), "DID2S standard errors are not the same."
    assert np.allclose(
        event_study_did2s.pvalue().values, fit_did2s.pvalue().values
    ), "DID2S p-values are not the same."
    assert np.allclose(
        event_study_did2s.confint().values, fit_did2s.confint().values
    ), "DID2S confidence intervals are not the same."
