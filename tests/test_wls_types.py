import numpy as np
import pytest

import pyfixest as pf


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "f1"}])
def test_fweights_ols_equivalence(fml, vcov):
    "Test that frequency weights yield same result as long data for various vcov types."
    data = pf.get_data(model="Fepois").dropna()

    # Compressed versions
    group_vars = (
        ["Y", "X1", "f1"] if "f1" in fml or vcov == {"CRV1": "f1"} else ["Y", "X1"]
    )

    data_w = (
        data[group_vars]
        .groupby(group_vars)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    # Fit models
    fit_long = pf.feols(fml, data=data, vcov=vcov)
    fit_compressed = pf.feols(
        fml, data=data_w, vcov=vcov, weights="count", weights_type="fweights"
    )

    # Assert point estimates match
    np.testing.assert_allclose(
        fit_long.tidy().values, fit_compressed.tidy().values, rtol=1e-6, atol=1e-8
    )

    # Optional: also check N is the same
    assert fit_long._N == fit_compressed._N, "Sample size mismatch"


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
