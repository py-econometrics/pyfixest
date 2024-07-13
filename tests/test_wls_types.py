import numpy as np
import pytest

import pyfixest as pf


def test_fweights_ols():
    "Test that the fweights are correctly implemented for OLS models."
    data = pf.get_data()
    data2_w = (
        data[["Y", "X1"]]
        .groupby(["Y", "X1"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    data3_w = (
        data[["Y", "X1", "f1"]]
        .groupby(["Y", "X1", "f1"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1", data=data2_w, weights="count", weights_type="fweights")
    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)
    np.testing.assert_allclose(fit1.vcov("HC2")._vcov, fit2.vcov("HC2")._vcov)
    np.testing.assert_allclose(fit1.vcov("HC3")._vcov, fit2.vcov("HC3")._vcov)

    fit3 = pf.feols("Y ~ X1 | f1", data=data)
    fit4 = pf.feols(
        "Y ~ X1 | f1", data=data3_w, weights="count", weights_type="fweights"
    )
    np.testing.assert_allclose(fit3.tidy().values, fit4.tidy().values)

    np.testing.assert_allclose(
        fit3.vcov({"CRV3": "f1"})._vcov, fit4.vcov({"CRV3": "f1"})._vcov
    )


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
