import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import pyfixest as pf
from pyfixest.did.estimation import did2s, event_study
from pyfixest.did.twfe import TWFE


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
        fml="dep_var ~ treat | state + year", data=data, vcov={"CRV1": "state"}
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


@pytest.mark.parametrize("dtype", ["int16", "uint32", "Int64"])
@pytest.mark.parametrize("never_treated", [False, True])
def test_event_study_integer_time_dtypes(dtype, never_treated):
    unit = np.repeat(np.arange(12), 5)
    year = np.tile(np.arange(2000, 2005), 12)
    cohort = 2001 + unit % 3
    if never_treated:
        cohort[unit < 4] = 0
    treated = (year >= cohort) & (cohort > 0)
    data = pd.DataFrame(
        {
            "unit": unit,
            "year": year,
            "g": cohort,
            "y": 2 * treated + unit / 10 + (year - 2000) / 5 + np.sin(unit + year),
        }
    ).astype({"year": dtype, "g": dtype})
    original = data.copy()
    fit = event_study(
        data, yname="y", idname="unit", tname="year", gname="g", estimator="twfe"
    )

    # An explicit dummy-variable OLS design is independent of DID preprocessing.
    design = pd.DataFrame(
        np.column_stack(
            [
                np.ones(len(data)),
                treated,
                pd.get_dummies(unit, drop_first=True),
                pd.get_dummies(year, drop_first=True),
            ]
        ).astype(float)
    ).rename(columns={1: "is_treated"})
    reference = sm.OLS(data["y"], design).fit(
        cov_type="cluster", cov_kwds={"groups": unit, "use_correction": False}
    )
    # Match fixest's small-sample convention: unit effects are nested in the
    # clusters, so only the treatment coefficient and time effects count in K.
    n = len(data)
    groups = np.unique(unit).size
    k = 1 + np.unique(year).size
    correction = (n - 1) / (n - k) * groups / (groups - 1)
    # Both methods solve the same small, well-conditioned least-squares problem.
    np.testing.assert_allclose(
        fit.coef().loc["is_treated"],
        reference.params.loc["is_treated"],
        rtol=1e-10,
        atol=1e-12,
        err_msg="Integer storage dtype must preserve the TWFE treatment coefficient",
    )
    np.testing.assert_allclose(
        fit.se().loc["is_treated"],
        reference.bse.loc["is_treated"] * np.sqrt(correction),
        rtol=1e-10,
        atol=1e-12,
        err_msg="Integer storage dtype must preserve the unit-clustered standard error",
    )

    model = TWFE(data, yname="y", idname="unit", tname="year", gname="g")
    expected_relative_time = np.where(cohort > 0, year - cohort, np.inf)
    np.testing.assert_array_equal(
        model._data["rel_time"],
        expected_relative_time,
        err_msg="Pre-treatment relative periods must be negative for unsigned inputs",
    )
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("column", ["year", "g"])
def test_event_study_nullable_time_missing(column):
    data = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "g": [2001, 2001, 0, 0],
            "y": [1, 2, 3, 4],
        }
    ).astype({column: "Int64"})
    data.loc[0, column] = pd.NA
    with pytest.raises(
        ValueError, match=f"The variable {column} must not contain missing values"
    ):
        event_study(
            data, yname="y", idname="unit", tname="year", gname="g", estimator="twfe"
        )


@pytest.mark.parametrize("column", ["year", "g"])
def test_event_study_unsigned_time_out_of_range(column):
    data = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "g": [2001, 2001, 0, 0],
            "y": [1, 2, 3, 4],
        }
    ).astype({column: "uint64"})
    data.loc[0, column] = np.uint64(2**63)
    with pytest.raises(
        ValueError, match=f"The variable {column} must fit in a signed 64-bit integer"
    ):
        event_study(
            data, yname="y", idname="unit", tname="year", gname="g", estimator="twfe"
        )


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
