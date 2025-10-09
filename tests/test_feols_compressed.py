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
@pytest.mark.parametrize("vcov", ["iid", "hetero"])
@pytest.mark.parametrize(
    "ssc", [pf.ssc(adj=True, G_adj=True), pf.ssc(adj=False, G_adj=False)]
)
@pytest.mark.parametrize("dropna", [False])
def test_feols_compressed(data, fml, vcov, ssc, dropna):
    """
    Test FeolsCompressed.

    We test equivalence of coeffients, standard errors, and p-values between Feols and FeolsCompressed.
    We trigger the following errors:
    - If fixef are specified, we trigger Mundlak, which is only supported with bootstrap inference.
    - If the vcov is not iid or hetero, we relax the criteria for the standard errors and p-values.
    - We trigger an error to check that when cluster robust variance is used, the cluster variable is in the model.
    """
    data_copy = data.copy()
    if dropna:
        data_copy.loc[0, "Y"] = np.nan
        data_copy.loc[1, "year"] = np.nan

    feols_args = dict(
        fml=fml,
        data=data_copy,
        vcov=vcov,
        ssc=ssc,
    )

    fit = pf.feols(**feols_args)

    if fit._is_clustered:
        clustervar_in_model = (
            fit._clustervar[0] in fit._fixef.split("+")
            or fit._clustervar[0] in fit._coefnames
        )
        if not clustervar_in_model:
            pytest.skip("Currently only testing for cluster fixed effects.")

    feols_args["use_compression"] = True
    feols_args["reps"] = 500
    feols_args["seed"] = 23

    fit_c = None

    try:
        fit_c = pf.feols(**feols_args)
    except NotImplementedError:
        if fit._has_fixef:
            if len(fit._fixef.split("+")) > 2:
                pytest.raises(NotImplementedError)
            if fit._vcov_type != "CRV":
                pytest.raises(NotImplementedError)
        else:
            raise

    if fit_c is not None:
        assert fit._N == fit_c._N, "Error in N"
        assert fit_c._has_weights, "Compressed regression should have weights"
        if fit_c._has_fixef:
            assert fit_c._use_mundlak, "fixef estimaton should be based on Mundlak"

        check_absolute_diff(
            x1=fit.coef().xs("treat"),
            x2=fit_c.coef().xs("treat"),
            tol=ATOL,
            msg="Error in coef",
        )

        # test predict method
        # assert fit.predict()[0:5] == fit_c.predict()[0:5], "Error in predict"
        # data_predict = data.iloc[200:294]
        # assert fit.predict(newdata = data_predict) == fit_c.predict(newdata = data_predict), "Error in predict"

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
            # relaxed criteria for bootstrap inference (need not be identical)
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
    assert np.all(np.abs(fit3.se().xs("f1") / fit4.se().xs("f1")) < RTOL_BOOT), (
        "Error in se"
    )
    assert np.all(
        np.abs(fit3.pvalue().xs("f1") / fit4.pvalue().xs("f1")) < RTOL_BOOT
    ), "Error in pvalue"
