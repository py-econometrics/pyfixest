import numpy as np
import pytest

import pyfixest as pf
from pyfixest.utils.dgps import get_sharkfin

ATOL = 1e-6
RTOL_BOOT = 1.05


def check_absolute_diff(x1, x2, tol, msg=None):
    msg = "" if msg is None else msg
    assert np.all(np.abs(x1 - x2) < tol), msg


@pytest.fixture
def data():
    rng = np.random.default_rng(123)
    data = get_sharkfin(1000)
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
    # special syntax
    # "Y ~ treat + i(year)",
    # "Y ~ treat + i(year, ref = 1)",
    # "Y ~ treat + i(year, ref = 1) | unit",
    "Y ~ treat + poly(year, 2)",
]


@pytest.mark.parametrize("fml", fmls)
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "treat"}])
@pytest.mark.parametrize(
    "ssc", [pf.ssc(adj=True, cluster_adj=True), pf.ssc(adj=False, cluster_adj=False)]
)
@pytest.mark.parametrize("dropna", [False, True])
def test_feols_compressed(data, fml, vcov, ssc, dropna):
    data_copy = data.copy()
    if dropna:
        data_copy.loc[0, "Y"] = np.nan
        data_copy.loc[1, "year"] = np.nan

    fit = pf.feols(fml=fml, data=data_copy, vcov=vcov, ssc=ssc)

    if fit._is_clustered:
        clustervar_in_model = (
            fit._clustervar[0] in fit._fixef.split("+")
            or fit._clustervar[0] in fit._coefnames
        )
        if not clustervar_in_model:
            pytest.skip("Currently only testing for cluster fixed effects.")

    if fit._has_fixef:
        with pytest.raises(NotImplementedError):
            fit_c = pf.feols(
                fml=fml,
                data=data_copy,
                vcov=vcov,
                use_compression=True,
                ssc=ssc,
                reps=500,
            )
    else:
        fit_c = pf.feols(
            fml=fml, data=data_copy, vcov=vcov, use_compression=True, ssc=ssc, reps=500
        )

        assert fit._N == fit_c._N, "Error in N"

        check_absolute_diff(
            x1=fit.coef().xs("treat"),
            x2=fit_c.coef().xs("treat"),
            tol=ATOL,
            msg="Error in coef",
        )

        if True:
            if vcov in ["iid", "hetero"]:
                check_absolute_diff(
                    x1=fit.se().xs("treat"),
                    x2=fit_c.se().xs("treat"),
                    tol=ATOL,
                    msg="Error in se",
                )

                check_absolute_diff(
                    x1=fit.pvalue().xs("treat"),
                    x2=fit_c.pvalue().xs("treat"),
                    tol=ATOL,
                    msg="Error in pvalue",
                )

            else:
                assert np.all(
                    np.abs(fit.se().xs("treat") / fit_c.se().xs("treat")) < RTOL_BOOT
                ), "Error in se"
                assert np.all(
                    np.abs(fit.pvalue().xs("treat") / fit_c.pvalue().xs("treat"))
                    < RTOL_BOOT
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
    assert np.all(
        np.abs(fit3.se().xs("f1") / fit4.se().xs("f1")) < RTOL_BOOT
    ), "Error in se"
    assert np.all(
        np.abs(fit3.pvalue().xs("f1") / fit4.pvalue().xs("f1")) < RTOL_BOOT
    ), "Error in pvalue"
