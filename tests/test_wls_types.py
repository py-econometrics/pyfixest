import numpy as np
import pytest

import pyfixest as pf


def _assert_fit_equal(fit1, fit2, vcov_types, rtol=1e-5):
    """Assert that two fits have identical coefficients, vcov, and SEs."""
    assert fit1._N == fit2._N, "Number of observations is not the same."

    np.testing.assert_allclose(
        fit1.coef().values, fit2.coef().values, rtol=rtol, err_msg="Coefficients differ"
    )

    for vcov_type in vcov_types:
        fit1_vcov = fit1.vcov(vcov_type)
        fit2_vcov = fit2.vcov(vcov_type)

        np.testing.assert_allclose(
            fit1_vcov._vcov,
            fit2_vcov._vcov,
            rtol=rtol,
            err_msg=f"Vcov differs for {vcov_type}",
        )
        np.testing.assert_allclose(
            fit1_vcov.se().values,
            fit2_vcov.se().values,
            rtol=rtol,
            err_msg=f"SEs differ for {vcov_type}",
        )


@pytest.mark.parametrize(
    "fml,cols,vcov_types",
    [
        # Without fixed effects - test hetero vcov types
        ("Y ~ X1", ["Y", "X1"], ["iid", "HC1", "HC2", "HC3"]),
        # Without fixed effects - test CRV (need to include cluster var in aggregation)
        ("Y ~ X1", ["Y", "X1", "f1"], [{"CRV1": "f1"}, {"CRV3": "f1"}]),
        # With fixed effects - HC2/HC3 not supported
        (
            "Y ~ X1 | f1",
            ["Y", "X1", "f1"],
            ["iid", "HC1", {"CRV1": "f1"}, {"CRV3": "f1"}],
        ),
    ],
)
def test_fweights_ols(fml, cols, vcov_types):
    """Test that fweights are correctly implemented for OLS models."""
    data = pf.get_data(model="Fepois")

    # Drop rows with NaN in columns used for aggregation to ensure same N
    data = data.dropna(subset=cols)

    data_agg = (
        data[cols].groupby(cols).size().reset_index().rename(columns={0: "count"})
    )

    fit_raw = pf.feols(fml, data=data, vcov="iid")
    fit_agg = pf.feols(
        fml,
        data=data_agg,
        weights="count",
        weights_type="fweights",
        vcov="iid",
    )

    _assert_fit_equal(fit_raw, fit_agg, vcov_types)


@pytest.mark.skip(reason="Poisson fweights has a separate bug - see issue #367")
@pytest.mark.parametrize(
    "fml,fe_col",
    [
        ("Y ~ X1", None),
        ("Y ~ X1 | f1", "f1"),
    ],
)
def test_fweights_poisson(fml, fe_col):
    """Test that fweights are correctly implemented for Poisson models."""
    data = pf.get_data(model="Fepois")

    cols = ["Y", "X1"]
    if fe_col:
        cols.append(fe_col)

    data_agg = (
        data[cols].groupby(cols).size().reset_index().rename(columns={0: "count"})
    )

    fit_raw = pf.fepois(fml, data=data, vcov="iid")
    fit_agg = pf.fepois(
        fml,
        data=data_agg,
        weights="count",
        weights_type="fweights",
        vcov="iid",
    )

    # Poisson only supports HC1 for hetero-robust SEs
    vcov_types = ["iid", "hetero"]
    _assert_fit_equal(fit_raw, fit_agg, vcov_types)


@pytest.mark.skip(reason="Not implemented yet.")
def test_fweights_iv():
    data = pf.get_data()
    data2_w = (
        data[["Y", "X1", "Z1"]]
        .groupby(["Y", "X1", "Z1"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    fit1 = pf.feols("Y ~ 1 | X1 ~ Z1", data=data)
    fit2 = pf.feols(
        "Y ~ 1 | X1 ~ Z1", data=data2_w, weights="count", weights_type="fweights"
    )
    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)

    data3_w = (
        data[["Y", "X1", "Z1", "f1"]]
        .groupby(["Y", "X1", "Z1", "f1"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    fit3 = pf.feols("Y ~ 1 | f1 | X1 ~ Z1 ", data=data.dropna(), vcov={"CRV1": "f1"})
    fit4 = pf.feols(
        "Y ~ 1 | f1 | X1 ~ Z1",
        data=data3_w.dropna(),
        weights="count",
        weights_type="fweights",
        vcov={"CRV1": "f1"},
    )
    np.testing.assert_allclose(fit3.tidy().values, fit4.tidy().values)


def test_aweights():
    data = pf.get_data()
    data["weights"] = np.ones(data.shape[0])

    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1", data=data, weights_type="aweights")
    fit3 = pf.feols("Y ~ X1", data=data, weights="weights", weights_type="aweights")

    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)
    np.testing.assert_allclose(fit1.tidy().values, fit3.tidy().values)
