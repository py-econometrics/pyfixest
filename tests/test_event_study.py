import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.did.estimation import did2s, event_study
from pyfixest.errors import VcovTypeNotSupportedError


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
    assert twfe.model.method == "twfe"

    # TODO - minor difference, likely due to how z statistic is
    # calculated

    # assert np.allclose(
    #    twfe.confint().values, twfe_feols.confint().values
    # ), "TWFE confidence intervals are not the same."


@pytest.mark.parametrize("xfml", [None, "C(band)"])
def test_event_study_did2s(data, xfml):
    if xfml is not None:
        data = data.assign(band=pd.Categorical((data["unit"] + data["year"]) % 3))
    event_study_did2s = event_study(
        data=data,
        yname="dep_var",
        idname="state",
        tname="year",
        gname="g",
        att=True,
        estimator="did2s",
        xfml=xfml,
    )

    fit_did2s = did2s(
        data=data,
        yname="dep_var",
        first_stage=f"~ {xfml or '0'} | state + year",
        second_stage=f"~ treat + {xfml}" if xfml is not None else "~ treat",
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

    for fit in (event_study_did2s, fit_did2s):
        if xfml is not None:
            assert [
                name for name in fit.coef().index if name.startswith("C(band)")
            ] == [
                "C(band)[T.1]",
                "C(band)[T.2]",
            ]
        covariance = fit.variance_covariance
        inference = fit.tidy().copy()
        for vcov in (
            {"CRV1": "state"},
            {"CRV1": "unit"},
            {"CRV3": "state"},
            "hetero",
            "iid",
        ):
            with pytest.raises(
                VcovTypeNotSupportedError,
                match=r"vcov\(\) is not supported for 'did2s' fits: "
                r"fit.capabilities.covariance_update is False\.",
            ):
                fit.vcov(vcov)
            assert fit.variance_covariance is covariance
            pd.testing.assert_frame_equal(fit.tidy(), inference)


@pytest.mark.parametrize("weights", [None, "weights"])
def test_did2s_formula_encodings(data, weights):
    """Interacted fixed effects match an explicit combined column."""
    rng = np.random.default_rng(1243)
    data = data.assign(
        weights=rng.uniform(0, 10, size=len(data)),
        X=rng.normal(size=len(data)),
        region=data["state"] % 3,
        year_region=data["year"] * 10 + data["state"] % 3,
    )
    kwargs = dict(
        data=data,
        yname="dep_var",
        second_stage="~ i(treat)",
        treatment="treat",
        cluster="state",
        weights=weights,
    )
    expected = did2s(first_stage="~ X | unit + year_region", **kwargs)
    for interaction in ("year^region", "year:region"):
        fit = did2s(first_stage=f"~ X | unit + {interaction}", **kwargs)
        np.testing.assert_allclose(fit.coef(), expected.coef(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            fit.variance_covariance.vcov,
            expected.variance_covariance.vcov,
            rtol=1e-8,
            atol=1e-10,
        )


@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("stage", ["first_stage", "second_stage"])
def test_did2s_quoted_reference(data, weights, stage):
    """Quoted references in either stage match an explicit relabeling."""
    rng = np.random.default_rng(1243)
    data = data.assign(
        weights=rng.uniform(0, 10, size=len(data)),
        X=rng.normal(size=len(data)),
        treat_f=data["treat"].astype(float),
    )
    stages = dict(first_stage="~ X | unit + year", second_stage="~ i(treat)")
    if stage == "first_stage":
        # Vary within units and years so the categorical control is not absorbed.
        data["group"] = np.where(data["X"] > 0, "Group 3", "Group 2")
        stages[stage] = "~ X + i(group, ref='Group 3') | unit + year"
    else:
        # A colliding stripped label also catches silently changed references.
        data["group"] = data["group"].replace({"Group 1": "Group3"})
        stages[stage] = "~ i(group, treat_f, ref='Group 3')"
    kwargs = dict(yname="dep_var", treatment="treat", cluster="state", weights=weights)
    fit = did2s(data=data, **stages, **kwargs)
    renamed = data.assign(group=data["group"].replace({"Group 3": "reference"}))
    expected = did2s(
        data=renamed,
        **{
            key: value.replace("'Group 3'", "'reference'")
            for key, value in stages.items()
        },
        **kwargs,
    )
    assert list(fit.coef().index) == list(expected.coef().index)
    np.testing.assert_allclose(
        fit.coef(),
        expected.coef(),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Relabeling the quoted reference changed DID2S coefficients",
    )
    np.testing.assert_allclose(
        fit.se(),
        expected.se(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="Relabeling the quoted reference changed DID2S standard errors",
    )


# ---------------------------------------------------------------------------------
# test errors


# Test case for 'data' must be a pandas DataFrame
def test_event_study_invalid_data_type(data):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
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
    with pytest.raises(TypeError, match="yname must be a string"):
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
    with pytest.raises(TypeError, match="idname must be a string"):
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
    with pytest.raises(TypeError, match="tname must be a string"):
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
    with pytest.raises(TypeError, match="gname must be a string"):
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
    with pytest.raises(TypeError, match="xfml must be a string or None"):
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
    with pytest.raises(TypeError, match="estimator must be a string"):
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
    with pytest.raises(TypeError, match="att must be a boolean"):
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
    with pytest.raises(TypeError, match="cluster must be a string"):
        event_study(
            data=data,
            yname="dep_var",
            idname="state",
            tname="year",
            gname="g",
            estimator="twfe",
            cluster=123,  # Invalid cluster type, should be str
        )


@pytest.mark.parametrize(
    "stages,match",
    [
        ({"first_stage": "0 | state + year"}, "first_stage must start with '~'"),
        ({"second_stage": "i(treat)"}, "second_stage must start with '~'"),
        *[
            ({"second_stage": stage}, "must not remove the intercept")
            for stage in (
                "~ i(treat) + 0",
                "~ 0 + i(treat)",
                "~ i(treat) - 1",
                "~ -1 + i(treat)",
            )
        ],
        ({"first_stage": "~ X"}, "First stage formula must contain fixed effects"),
        (
            {"second_stage": "~ i(treat) | unit"},
            "Second stage formula must not contain fixed effects",
        ),
        (
            {"first_stage": "~ sw(X, weights) | unit + year"},
            "first_stage must specify a single model",
        ),
        (
            {"second_stage": "~ sw(X, weights)"},
            "second_stage must specify a single model",
        ),
        (
            {"first_stage": "~ [X ~ weights] | unit + year"},
            "first_stage must not contain instrumental variables",
        ),
        (
            {"second_stage": "~ [X ~ weights]"},
            "second_stage must not contain instrumental variables",
        ),
    ],
)
def test_did2s_formula_validation(data, stages, match):
    with pytest.raises(ValueError, match=match):
        did2s(
            data,
            yname="dep_var",
            **{
                "first_stage": "~ 0 | state + year",
                "second_stage": "~ i(treat)",
                **stages,
            },
            treatment="treat",
            cluster="state",
        )


def test_lpdid_invalid_xfml_type(data):
    with pytest.raises(TypeError, match="xfml must be a string or None"):
        pf.lpdid(
            data, yname="dep_var", idname="unit", tname="year", gname="g", xfml=123
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
