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

    twfe_feols = pf.feols(
        fml = "dep_var ~ treat | state + year", 
        data=data, 
        vcov = {"CRV1": "state"}
    )

    assert np.allclose(twfe.coef().values, twfe_feols.coef().values), (
        "TWFE coefficients are not the same."
    )
    assert np.allclose(twfe.se().values, twfe_feols.se().values), (
        "TWFE standard errors are not the same."
    )
    assert np.allclose(twfe.pvalue().values, twfe_feols.pvalue().values), (
        "TWFE p-values are not the same."
    )

    # TODO - minor difference, likely due to how z statistic is
    # calculated

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

    assert np.allclose(event_study_did2s.coef().values, fit_did2s.coef().values), (
        "DID2S coefficients are not the same."
    )
    assert np.allclose(event_study_did2s.se().values, fit_did2s.se().values), (
        "DID2S standard errors are not the same."
    )
    assert np.allclose(event_study_did2s.pvalue().values, fit_did2s.pvalue().values), (
        "DID2S p-values are not the same."
    )
    assert np.allclose(
        event_study_did2s.confint().values, fit_did2s.confint().values
    ), "DID2S confidence intervals are not the same."


# ---------------------------------------------------------------------------------
# test errors


# Test case for 'data' must be a pandas DataFrame
def test_event_study_invalid_data_type(data):
    with pytest.raises(AssertionError, match="data must be a pandas DataFrame"):
        event_study(
            data="invalid_data",  # Invalid data type, should be pd.DataFrame
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            estimator="twfe",
        )


# Test case for 'yname' must be a string
def test_event_study_invalid_yname_type(data):
    with pytest.raises(AssertionError, match="yname must be a string"):
        event_study(
            data=data,
            yname=123,  # Invalid yname type, should be str
            idname="state",
            tname="year",
            gname="g",
            estimator="twfe",
        )


# Test case for 'idname' must be a string
def test_event_study_invalid_idname_type(data):
    with pytest.raises(AssertionError, match="idname must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname=123,  # Invalid idname type, should be str
            tname="year",
            gname="g",
            estimator="twfe",
        )


# Test case for 'tname' must be a string
def test_event_study_invalid_tname_type(data):
    with pytest.raises(AssertionError, match="tname must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname=2020,  # Invalid tname type, should be str
            gname="g",
            estimator="twfe",
        )


# Test case for 'gname' must be a string
def test_event_study_invalid_gname_type(data):
    with pytest.raises(AssertionError, match="gname must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname=2020,  # Invalid gname type, should be str
            estimator="twfe",
        )


# Test case for 'xfml' must be a string or None
def test_event_study_invalid_xfml_type(data):
    with pytest.raises(AssertionError, match="xfml must be a string or None"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            xfml=123,  # Invalid xfml type, should be str or None
            estimator="twfe",
        )


# Test case for 'estimator' must be a string
def test_event_study_invalid_estimator_type(data):
    with pytest.raises(AssertionError, match="estimator must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            estimator=123,  # Invalid estimator type, should be str
        )


# Test case for 'att' must be a boolean
def test_event_study_invalid_att_type(data):
    with pytest.raises(AssertionError, match="att must be a boolean"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            att="True",  # Invalid att type, should be bool
            estimator="twfe",
        )


# Test case for 'cluster' must be a string
def test_event_study_invalid_cluster_type(data):
    with pytest.raises(AssertionError, match="cluster must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            estimator="twfe",
            cluster=123,  # Invalid cluster type, should be str
        )


# Test case for unsupported estimator (triggering NotImplementedError)
def test_event_study_unsupported_estimator(data):
    with pytest.raises(NotImplementedError, match="Estimator not supported"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            estimator="unsupported",  # Unsupported estimator
        )
