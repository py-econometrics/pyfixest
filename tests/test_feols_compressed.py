import numpy as np
import pytest

import pyfixest as pf
from pyfixest.utils.dgps import get_sharkfin

ATOL = 1e-6
RTOL_BOOT = 1.05


@pytest.fixture
def data():
    rng = np.random.default_rng(123)
    data = get_sharkfin()
    data["X"] = rng.normal(size=data.shape[0])
    return data


fmls = [
    "Y ~ treat",
    "Y ~ treat + unit",
    "Y ~ treat + unit + year",
    # with continuous variable
    "Y ~ treat + X",
    # with fixed effect
    "Y ~ treat | unit",
    "Y ~ treat | unit + year",
    "Y ~ treat + X | unit",
    "Y ~ treat + X | unit + year",
]


@pytest.mark.parametrize("fml", fmls)
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "treat"}])
@pytest.mark.parametrize(
    "ssc", [pf.ssc(adj=True, cluster_adj=True), pf.ssc(adj=False, cluster_adj=False)]
)
@pytest.mark.parametrize("dropna", [False, True])
def test_feols_compressed(data, fml, vcov, ssc, dropna):
    fit = pf.feols(fml=fml, data=data.dropna() if dropna else data, vcov=vcov, ssc=ssc)

    if fit._is_clustered:
        clustervar_in_model = (
            fit._clustervar[0] in fit._fixef.split("+")
            or fit._clustervar[0] in fit._coefnames
        )
        if not clustervar_in_model:
            pytest.skip("Currently only testing for cluster fixed effects.")

    fit_c = pf.feols(
        fml=fml,
        data=data.dropna() if dropna else data,
        vcov=vcov,
        use_compression=True,
        ssc=ssc,
        reps=500,
    )

    assert np.all(
        fit.coef().xs("treat") - fit_c.coef().xs("treat") < ATOL
    ), "Error in coef"

    if vcov in ["iid", "hetero"]:
        assert np.all(
            fit.se().xs("treat") - fit_c.se().xs("treat") < ATOL
        ), "Error in se"
        assert np.all(
            fit.pvalue().xs("treat") - fit_c.pvalue().xs("treat") < ATOL
        ), "Error in pvalue"
    else:
        assert np.all(
            fit.se().xs("treat") / fit_c.se().xs("treat") < RTOL_BOOT
        ), "Error in se"
        assert np.all(
            fit.pvalue().xs("treat") / fit_c.pvalue().xs("treat") < RTOL_BOOT
        ), "Error in pvalue"


def test_identical_seed():
    data = pf.get_data()

    # same seed -> identical results
    fit1 = pf.feols("Y ~ f1", data=data, use_compression=True, seed=123)
    fit2 = pf.feols("Y ~ f1", data=data, use_compression=True, seed=123)

    assert np.allclose(fit1.coef().xs("f1"), fit2.coef().xs("f1")), "Error in seed"
    assert np.allclose(fit1.se().xs("f1"), fit2.se().xs("f1")), "Error in seed"
    assert np.allclose(fit1.pvalue().xs("f1"), fit2.pvalue().xs("f1")), "Error in seed"


def test_different_seed():
    data = pf.get_data()

    # different seed, high bootstrap iter -> similar but not identical results
    fit3 = pf.feols("Y ~ f1", data=data, use_compression=True, seed=123, reps=1000)
    fit4 = pf.feols("Y ~ f1", data=data, use_compression=True, seed=125, reps=1000)

    assert np.allclose(fit3.coef().xs("f1"), fit4.coef().xs("f1")), "Error in seed"
    assert np.all(fit3.se().xs("f1") / fit4.se().xs("f1") < RTOL_BOOT), "Error in se"
    assert np.all(
        fit3.pvalue().xs("f1") / fit4.pvalue().xs("f1") < RTOL_BOOT
    ), "Error in pvalue"
