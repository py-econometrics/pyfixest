import numpy as np
import pytest

import pyfixest as pf

ATOL = 1e-6
RTOL_BOOT = 1.05

@pytest.fixture
def data():
    return pf.get_data()


fmls = [

    "Y ~ f1",
    "Y ~ f1 + f2",
    "Y2 ~ f1 + f2 + f3",

    # with continuous variable
    "Y ~ X1 + f1",

    # with fixed effect
    "Y ~ f1 | f2",
    "Y ~ f1 + f3 | f2",
    "Y ~ X1 + f1 | f2",

]


@pytest.mark.parametrize("fml", fmls)
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1":"f1"}])
@pytest.mark.parametrize(
    "ssc", [pf.ssc(adj=True, cluster_adj=True), pf.ssc(adj=False, cluster_adj=False)]
)
@pytest.mark.parametrize("dropna", [False, True])
def test_feols_compressed(data, fml, vcov, ssc, dropna):
    fit = pf.feols(fml=fml, data=data.dropna() if dropna else data, vcov=vcov, ssc=ssc)

    fit_c = pf.feols(
        fml=fml,
        data=data.dropna() if dropna else data,
        vcov=vcov,
        use_compression=True,
        ssc=ssc,
        reps = 500
    )

    assert np.all(fit.coef().xs("f1") - fit_c.coef().xs("f1") < ATOL), "Error in coef"

    if vcov in ["iid", "hetero"]:
        assert np.all(fit.se().xs("f1") - fit_c.se().xs("f1") < ATOL), "Error in se"
        assert np.all(
            fit.pvalue().xs("f1") - fit_c.pvalue().xs("f1") < ATOL
        ), "Error in pvalue"
    else:
        assert np.all(fit.se().xs("f1") / fit_c.se().xs("f1") < RTOL_BOOT), "Error in se"
        assert np.all(
            fit.pvalue().xs("f1") / fit_c.pvalue().xs("f1") < RTOL_BOOT
        ), "Error in pvalue"



def test_identical_seed():
    data = pf.get_data()

    # same seed -> identical results
    fit1 = pf.feols("Y ~ f1", data = data, use_compression=True, seed = 123)
    fit2 = pf.feols("Y ~ f1", data = data, use_compression=True, seed = 123)

    assert np.allclose(fit1.coef().xs("f1"), fit2.coef().xs("f1")), "Error in seed"
    assert np.allclose(fit1.se().xs("f1"), fit2.se().xs("f1")), "Error in seed"
    assert np.allclose(fit1.pvalue().xs("f1"), fit2.pvalue().xs("f1")), "Error in seed"

def test_different_seed():
    data = pf.get_data()

    # different seed, high bootstrap iter -> similar but not identical results
    fit3 = pf.feols("Y ~ f1", data = data, use_compression=True, seed = 123, reps = 1000)
    fit4 = pf.feols("Y ~ f1", data = data, use_compression=True, seed = 125, reps = 1000)

    assert np.allclose(fit3.coef().xs("f1"), fit4.coef().xs("f1")), "Error in seed"
    assert np.all(fit3.se().xs("f1") / fit4.se().xs("f1") < RTOL_BOOT), "Error in se"
    assert np.all(fit3.pvalue().xs("f1") / fit4.pvalue().xs("f1") < RTOL_BOOT), "Error in pvalue"
