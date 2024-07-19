import re

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation.estimation import feols
from pyfixest.utils.utils import get_data, ssc

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

# note: tolerances are lowered below for
# fepois inference as it is not as precise as feols
# effective tolerances for fepois are 1e-04 and 1e-03
# (the latter only for CRV inferece)
rtol = 1e-06
atol = 1e-06

iwls_maxiter = 25
iwls_tol = 1e-08

ols_fmls = [
    ("Y~X1"),
    ("Y~X1+X2"),
    ("Y~X1|f2"),
    ("Y~X1|f2+f3"),
    ("Y ~ X1 + exp(X2)"),
    ("Y ~ X1 + C(f1)"),
    ("Y ~ X1 + i(f1, ref = 1)"),
    ("Y ~ X1 + C(f1)"),
    ("Y ~ X1 + i(f2, ref = 2.0)"),
    ("Y ~ X1 + C(f1) + C(f2)"),
    ("Y ~ X1 + C(f1) | f2"),
    ("Y ~ X1 + i(f1, ref = 3.0) | f2"),
    ("Y ~ X1 + C(f1) | f2 + f3"),
    ("Y ~ X1 + i(f1, ref = 1) | f2 + f3"),
    ("Y ~ X1 + i(f1) + i(f2)"),
    ("Y ~ X1 + i(f1, ref = 1) + i(f2, ref = 2)"),
    # ("Y ~ X1 + C(f1):C(fe2)"),                  # currently does not work as C():C() translation not implemented # noqa: W505
    # ("Y ~ X1 + C(f1):C(fe2) | f3"),             # currently does not work as C():C() translation not implemented # noqa: W505
    ("Y ~ X1 + X2:f1"),
    ("Y ~ X1 + X2:f1 | f3"),
    ("Y ~ X1 + X2:f1 | f3 + f1"),
    # ("log(Y) ~ X1:X2 | f3 + f1"),               # currently, causes big problems for Fepois (takes a long time) # noqa: W505
    # ("log(Y) ~ log(X1):X2 | f3 + f1"),          # currently, causes big problems for Fepois (takes a long time) # noqa: W505
    # ("Y ~  X2 + exp(X1) | f3 + f1"),            # currently, causes big problems for Fepois (takes a long time) # noqa: W505
    ("Y ~ X1 + i(f1,X2)"),
    ("Y ~ X1 + i(f1,X2) + i(f2, X2)"),
    ("Y ~ X1 + i(f1,X2, ref =1) + i(f2)"),
    ("Y ~ X1 + i(f1,X2, ref =1) + i(f2, X1, ref =2)"),
    ("Y ~ X1 + i(f2,X2)"),
    ("Y ~ X1 + i(f1,X2) | f2"),
    ("Y ~ X1 + i(f1,X2) | f2 + f3"),
    ("Y ~ X1 + i(f1,X2, ref=1.0)"),
    ("Y ~ X1 + i(f2,X2, ref=2.0)"),
    ("Y ~ X1 + i(f1,X2, ref=3.0) | f2"),
    ("Y ~ X1 + i(f1,X2, ref=4.0) | f2 + f3"),
    # ("Y ~ C(f1):X2"),                          # currently does not work as C():X translation not implemented # noqa: W505
    # ("Y ~ C(f1):C(f2)"),                       # currently does not work
    ("Y ~ X1 + I(X2 ** 2)"),
    ("Y ~ X1 + I(X1 ** 2) + I(X2**4)"),
    ("Y ~ X1*X2"),
    ("Y ~ X1*X2 | f1+f2"),
    # ("Y ~ X1/X2"),                             # currently does not work as X1/X2 translation not implemented # noqa: W505
    # ("Y ~ X1/X2 | f1+f2"),                     # currently does not work as X1/X2 translation not implemented # noqa: W505
    ("Y ~ X1 + poly(X2, 2) | f1"),
]

ols_but_not_poisson_fml = [
    ("log(Y) ~ X1"),
    ("Y~X1|f2^f3"),
    ("Y~X1|f1 + f2^f3"),
    ("Y~X1|f2^f3^f1"),
]

iv_fmls = [
    # IV starts here
    ("Y ~ 1 | X1 ~ Z1"),
    "Y ~  X2 | X1 ~ Z1",
    "Y ~ X2 + C(f1) | X1 ~ Z1",
    "Y2 ~ 1 | X1 ~ Z1",
    "Y2 ~ X2 | X1 ~ Z1",
    "Y2 ~ X2 + C(f1) | X1 ~ Z1",
    "log(Y) ~ 1 | X1 ~ Z1",
    "log(Y) ~ X2 | X1 ~ Z1",
    "log(Y) ~ X2 + C(f1) | X1 ~ Z1",
    "Y ~ 1 | f1 | X1 ~ Z1",
    "Y ~ 1 | f1 + f2 | X1 ~ Z1",
    "Y ~ 1 | f1^f2 | X1 ~ Z1",
    "Y ~  X2| f1 | X1 ~ Z1",
    ## tests of overidentified models
    "Y ~ 1 | X1 ~ Z1 + Z2",
    "Y ~ X2 | X1 ~ Z1 + Z2",
    "Y ~ X2 + C(f1) | X1 ~ Z1 + Z2",
    "Y ~ 1 | f1 | X1 ~ Z1 + Z2",
    "Y2 ~ 1 | f1 + f2 | X1 ~ Z1 + Z2",
    "Y2 ~  X2| f2 | X1 ~ Z1 + Z2",
]


rng = np.random.default_rng(8760985)


def check_absolute_diff(x1, x2, tol, msg=None):
    msg = "" if msg is None else msg
    assert np.all(np.abs(x1 - x2) < tol), msg


@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("seed", [76540251])
@pytest.mark.parametrize("beta_type", ["2"])
@pytest.mark.parametrize("error_type", ["2"])
@pytest.mark.parametrize("dropna", [False])
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "group_id"}])
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("f3_type", ["str", "object", "int", "categorical", "float"])
@pytest.mark.parametrize("fml", ols_fmls + ols_but_not_poisson_fml)
@pytest.mark.parametrize("adj", [False, True])
@pytest.mark.parametrize("cluster_adj", [False, True])
def test_single_fit_feols(
    N,
    seed,
    beta_type,
    error_type,
    dropna,
    inference,
    weights,
    f3_type,
    fml,
    adj,
    cluster_adj,
):
    if cluster_adj and inference in ["iid", "hetero"]:
        pytest.skip(
            "Cluster adjustment only works with cluster inference. Nothing to test here."
        )
    if "f3" not in fml:
        pytest.skip("No f3 in formula. Nothing to test here.")

    ssc_ = ssc(adj=adj, cluster_adj=cluster_adj)

    data = get_data(
        N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Feols"
    )

    if dropna:
        data = data.dropna()

    # long story, but categories need to be strings to be converted to R factors,
    # this then produces 'nan' values in the pd.DataFrame ...
    data[data == "nan"] = np.nan

    # test fixed effects that are not floats, but ints or categoricals, etc

    data = _convert_f3(data, f3_type)

    data_r = get_data_r(fml, data)
    r_fml = _c_to_as_factor(fml)

    r_inference = _get_r_inference(inference)

    mod = pf.feols(fml=fml, data=data, vcov=inference, weights=weights, ssc=ssc_)
    if weights is not None:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=r_inference,
            data=data_r,
            ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
            weights=ro.Formula("~" + weights),
        )
    else:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=r_inference,
            data=data_r,
            ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
        )

    py_coef = mod.coef().xs("X1")
    py_se = mod.se().xs("X1")
    py_pval = mod.pvalue().xs("X1")
    py_tstat = mod.tstat().xs("X1")
    py_confint = mod.confint().xs("X1").values
    py_nobs = mod._N
    py_vcov = mod._vcov[0, 0]

    py_resid = mod._u_hat.flatten()  # noqa: F841
    # TODO: test residuals

    df_X1 = _get_r_df(r_fixest)

    r_coef = df_X1["estimate"]
    r_se = df_X1["std.error"]
    r_pval = df_X1["p.value"]
    r_tstat = df_X1["statistic"]
    r_confint = df_X1[["conf.low", "conf.high"]].values.astype(np.float64)
    r_nobs = int(stats.nobs(r_fixest)[0])
    r_resid = r_fixest.rx2("working_residuals")  # noqa: F841
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_nobs, r_nobs, 1e-08, "py_nobs != r_nobs")
    check_absolute_diff(py_coef, r_coef, 1e-08, "py_coef != r_coef")
    check_absolute_diff(py_vcov, r_vcov, 1e-05, "py_vcov != r_vcov")
    check_absolute_diff(py_se, r_se, 1e-06, "py_se != r_se")
    check_absolute_diff(py_pval, r_pval, 1e-06, "py_pval != r_pval")
    check_absolute_diff(py_tstat, r_tstat, 1e-06, "py_tstat != r_tstat")
    check_absolute_diff(py_confint, r_confint, 1e-05, "py_confint != r_confint")

    if not weights:
        py_r2 = mod._r2
        py_r2_within = mod._r2_within
        # py_adj_r2 = mod._adj_r2
        # py_adj_r2_within = mod._adj_r2_within
        r_r = fixest.r2(r_fixest)
        r_r2 = r_r[1]
        r_r2_within = r_r[5]

        check_absolute_diff(py_r2, r_r2, 1e-06, "py_r2 != r_r2")
        if not np.isnan(py_r2_within):
            check_absolute_diff(
                py_r2_within, r_r2_within, 1e-06, "py_r2_within != r_r2_within"
            )


@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("seed", [76540251])
@pytest.mark.parametrize("beta_type", ["2"])
@pytest.mark.parametrize("error_type", ["2"])
@pytest.mark.parametrize("dropna", [False])
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "group_id"}])
@pytest.mark.parametrize("f3_type", ["str", "object", "int", "categorical", "float"])
@pytest.mark.parametrize("fml", ols_fmls)
@pytest.mark.parametrize("adj", [False, True])
@pytest.mark.parametrize("cluster_adj", [False, True])
def test_single_fit_fepois(
    N, seed, beta_type, error_type, dropna, inference, f3_type, fml, adj, cluster_adj
):
    if cluster_adj and inference in ["iid", "hetero"]:
        pytest.skip(
            "Cluster adjustment only works with cluster inference. Nothing to test here."
        )
    if "f3" not in fml:
        pytest.skip("No f3 in formula. Nothing to test here.")

    ssc_ = ssc(adj=adj, cluster_adj=cluster_adj)

    data = get_data(
        N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Fepois"
    )

    if dropna:
        data = data.dropna()

    # long story, but categories need to be strings to be converted to R factors,
    # this then produces 'nan' values in the pd.DataFrame ...
    data[data == "nan"] = np.nan

    # test fixed effects that are not floats, but ints or categoricals, etc
    data = _convert_f3(data, f3_type)

    data_r = get_data_r(fml, data)
    r_fml = _c_to_as_factor(fml)
    r_inference = _get_r_inference(inference)

    mod = pf.fepois(fml=fml, data=data, vcov=inference, ssc=ssc_)
    r_fixest = fixest.fepois(
        ro.Formula(r_fml),
        vcov=r_inference,
        data=data_r,
        ssc=fixest.ssc(adj, "none", cluster_adj, "min", "min", False),
    )

    py_coef = mod.coef().xs("X1")
    py_se = mod.se().xs("X1")
    py_pval = mod.pvalue().xs("X1")
    py_tstat = mod.tstat().xs("X1")
    py_confint = mod.confint().xs("X1").values
    py_nobs = mod._N
    py_vcov = mod._vcov[0, 0]
    py_deviance = mod.deviance

    py_resid = mod._u_hat.flatten()  # noqa: F841
    # TODO: test residuals

    df_X1 = _get_r_df(r_fixest)

    r_coef = df_X1["estimate"]
    r_se = df_X1["std.error"]
    r_pval = df_X1["p.value"]
    r_tstat = df_X1["statistic"]
    r_confint = df_X1[["conf.low", "conf.high"]].values.astype(np.float64)
    r_nobs = int(stats.nobs(r_fixest)[0])
    r_resid = r_fixest.rx2("working_residuals")  # noqa: F841
    r_vcov = stats.vcov(r_fixest)[0, 0]
    r_deviance = r_fixest.rx2("deviance")

    check_absolute_diff(py_nobs, r_nobs, 1e-08, "py_nobs != r_nobs")
    check_absolute_diff(py_coef, r_coef, 1e-08, "py_coef != r_coef")
    check_absolute_diff(py_vcov, r_vcov, 1e-05, "py_vcov != r_vcov")
    check_absolute_diff(py_se, r_se, 1e-04, "py_se != r_se")
    check_absolute_diff(py_pval, r_pval, 1e-04, "py_pval != r_pval")
    check_absolute_diff(py_tstat, r_tstat, 1e-04, "py_tstat != r_tstat")
    check_absolute_diff(py_confint, r_confint, 1e-04, "py_confint != r_confint")
    check_absolute_diff(py_deviance, r_deviance, 1e-06, "py_deviance != r_deviance")


@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("seed", [76540251])
@pytest.mark.parametrize("beta_type", ["2"])
@pytest.mark.parametrize("error_type", ["2"])
@pytest.mark.parametrize("dropna", [False])
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "group_id"}])
@pytest.mark.parametrize("f3_type", ["str", "object", "int", "categorical", "float"])
@pytest.mark.parametrize("fml", iv_fmls)
@pytest.mark.parametrize("adj", [False, True])
@pytest.mark.parametrize("cluster_adj", [False, True])
def test_single_fit_iv(
    N,
    seed,
    beta_type,
    error_type,
    dropna,
    inference,
    weights,
    f3_type,
    fml,
    adj,
    cluster_adj,
):
    if cluster_adj and inference in ["iid", "hetero"]:
        pytest.skip(
            "Cluster adjustment only works with cluster inference. Nothing to test here."
        )

    if "f3" not in fml:
        pytest.skip("No f3 in formula. Nothing to test here.")

    ssc_ = ssc(adj=adj, cluster_adj=cluster_adj)

    data = get_data(
        N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Fepois"
    )

    if dropna:
        data = data.dropna()

    # long story, but categories need to be strings to be converted to R factors,
    # this then produces 'nan' values in the pd.DataFrame ...
    data[data == "nan"] = np.nan

    # test fixed effects that are not floats, but ints or categoricals, etc
    data = _convert_f3(data, f3_type)

    data_r = get_data_r(fml, data)
    r_fml = _c_to_as_factor(fml)
    r_inference = _get_r_inference(inference)

    mod = pf.feols(fml=fml, data=data, vcov=inference, ssc=ssc_)
    if weights is not None:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=r_inference,
            data=data_r,
            ssc=fixest.ssc(True, "none", True, "min", "min", False),
            weights=ro.Formula("~" + weights),
        )
    else:
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=r_inference,
            data=data_r,
            ssc=fixest.ssc(True, "none", True, "min", "min", False),
        )

    py_coef = mod.coef().xs("X1")
    py_se = mod.se().xs("X1")
    py_pval = mod.pvalue().xs("X1")
    py_tstat = mod.tstat().xs("X1")
    py_confint = mod.confint().xs("X1").values
    py_nobs = mod._N
    py_vcov = mod._vcov[0, 0]

    py_resid = mod._u_hat.flatten()  # noqa: F841
    # TODO: test residuals

    df_X1 = _get_r_df(r_fixest)

    r_coef = df_X1["estimate"]
    r_se = df_X1["std.error"]
    r_pval = df_X1["p.value"]
    r_tstat = df_X1["statistic"]
    r_confint = df_X1[["conf.low", "conf.high"]].values.astype(np.float64)
    r_nobs = int(stats.nobs(r_fixest)[0])
    r_resid = r_fixest.rx2("working_residuals")  # noqa: F841
    r_vcov = stats.vcov(r_fixest)[0, 0]

    check_absolute_diff(py_nobs, r_nobs, 1e-08, "py_nobs != r_nobs")
    check_absolute_diff(py_coef, r_coef, 1e-08, "py_coef != r_coef")
    check_absolute_diff(py_vcov, r_vcov, 1e-05, "py_vcov != r_vcov")
    check_absolute_diff(py_se, r_se, 1e-04, "py_se != r_se")
    check_absolute_diff(py_pval, r_pval, 1e-04, "py_pval != r_pval")
    check_absolute_diff(py_tstat, r_tstat, 1e-04, "py_tstat != r_tstat")
    check_absolute_diff(py_confint, r_confint, 1e-04, "py_confint != r_confint")


@pytest.mark.parametrize("N", [100])
@pytest.mark.parametrize("seed", [17021])
@pytest.mark.parametrize("beta_type", ["1"])
@pytest.mark.parametrize("error_type", ["3"])
@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize(
    "fml_multi",
    [
        ("Y ~X1"),
        ("Y ~X1+X2"),
        ("Y~X1|f1"),
        ("Y~X1|f1+f2"),
        ("Y~X2|f2+f3"),
        ("Y~ sw(X1, X2)"),
        ("Y~ sw(X1, X2) |f1 "),
        ("Y~ csw(X1, X2)"),
        ("Y~ csw(X1, X2) | f2"),
        ("Y~ I(X1**2) + csw(f1,f2)"),
        ("Y~ X1 + csw(f1, f2) | f3"),
        ("Y~ X1 + csw0(X2, f3)"),
        ("Y~ csw0(X2, f3) + X2"),
        ("Y~ X1 + csw0(X2, f3) + X2"),
        ("Y ~ X1 + csw0(f1, f2) | f3"),
        ("Y ~ X1 | csw0(f1,f2)"),
        ("Y ~ X1 + sw(X2, f1, f2)"),
        ("Y ~ csw(X1, X2, f3)"),
        # ("Y ~ X2 + csw0(, X2, X2)"),
        ("Y ~ sw(X1, X2) | csw0(f1,f2,f3)"),
        ("Y ~ C(f2):X2 + sw0(X1, f3)"),
        ("Y + Y2 ~X1"),
        ("Y + log(Y2) ~X1+X2"),
        ("Y + Y2 ~X1|f1"),
        ("Y + Y2 ~X1|f1+f2"),
        ("Y + Y2 ~X2|f2+f3"),
        ("Y + Y2 ~ sw(X1, X2)"),
        ("Y + Y2 ~ sw(X1, X2) |f1 "),
        ("Y + Y2 ~ csw(X1, X2)"),
        ("Y + Y2 ~ csw(X1, X2) | f2"),
        ("Y + Y2 ~ I(X1**2) + csw(f1,f2)"),
        ("Y + Y2 ~ X1 + csw(f1, f2) | f3"),
        ("Y + Y2 ~ X1 + csw0(X2, f3)"),
        ("Y + Y2 ~ X1 + csw0(f1, f2) | f3"),
        ("Y + Y2 ~ X1 | csw0(f1,f2)"),
        ("Y + log(Y2) ~ sw(X1, X2) | csw0(f1,f2,f3)"),
        ("Y ~ C(f2):X2 + sw0(X1, f3)"),
        # ("Y ~ i(f1,X2) | csw0(f2)"),
        # ("Y ~ i(f1,X2) | sw0(f2)"),
        # ("Y ~ i(f1,X2) | csw(f2, f3)"),
        # ("Y ~ i(f1,X2) | sw(f2, f3)"),
        # ("Y ~ i(f1,X2, ref = -5) | sw(f2, f3)"),
        # ("Y ~ i(f1,X2, ref = -8) | csw(f2, f3)"),
    ],
)
def test_multi_fit(N, seed, beta_type, error_type, dropna, fml_multi):
    """Test pyfixest against fixest_multi objects."""
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)
    data[data == "nan"] = np.nan

    if dropna:
        data = data.dropna()

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_multi)

    try:
        pyfixest = feols(fml=fml_multi, data=data)
    except ValueError as e:
        if "is not of type 'O' or 'category'" in str(e):
            data["f1"] = pd.Categorical(data.f1.astype(str))
            data["f2"] = pd.Categorical(data.f2.astype(str))
            data["f3"] = pd.Categorical(data.f3.astype(str))
            data[data == "nan"] = np.nan
            pyfixest = feols(fml=fml_multi, data=data)
        else:
            raise ValueError("Code fails with an uninformative error message.")

    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    for x, _ in range(0):
        mod = pyfixest.fetch_model(x)
        py_coef = mod.coef().values
        py_se = mod.se().values

        # sort py_coef, py_se
        py_coef, py_se = (np.sort(x) for x in [py_coef, py_se])

        fixest_object = r_fixest.rx2(x + 1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        # fixest_coef = stats.coef(r_fixest)
        # fixest_se = fixest.se(r_fixest)

        # sort fixest_coef, fixest_se
        fixest_coef, fixest_se = (np.sort(x) for x in [fixest_coef, fixest_se])

        np.testing.assert_allclose(
            py_coef, fixest_coef, rtol=rtol, atol=atol, err_msg="Coefs are not equal."
        )
        np.testing.assert_allclose(
            py_se, fixest_se, rtol=rtol, atol=atol, err_msg="SEs are not equal."
        )


def test_twoway_clustering():
    data = get_data(N=1000, seed=17021, beta_type="1", error_type="1").dropna()

    cluster_adj_options = [True]
    cluster_df_options = ["min", "conventional"]

    for cluster_adj in cluster_adj_options:
        for cluster_df in cluster_df_options:
            fit1 = feols(
                "Y ~ X1 + X2 ",
                data=data,
                vcov={"CRV1": "f1 +f2"},
                ssc=ssc(cluster_adj=cluster_adj, cluster_df=cluster_df),
            )
            fit2 = feols(  # noqa: F841
                "Y ~ X1 + X2 ",
                data=data,
                vcov={"CRV3": " f1+f2"},
                ssc=ssc(cluster_adj=cluster_adj, cluster_df=cluster_df),
            )

            feols_fit1 = fixest.feols(
                ro.Formula("Y ~ X1 + X2"),
                data=data,
                cluster=ro.Formula("~f1+f2"),
                ssc=fixest.ssc(True, "none", cluster_adj, cluster_df, "min", False),
            )

            # test vcov's
            np.testing.assert_allclose(
                fit1._vcov,
                stats.vcov(feols_fit1),
                rtol=1e-04,
                atol=1e-04,
                err_msg=f"CRV1-vcov: cluster_adj = {cluster_adj}, cluster_df = {cluster_df}",
            )

            # now test se's
            np.testing.assert_allclose(
                fit1.se(),
                fixest.se(feols_fit1),
                rtol=1e-04,
                atol=1e-04,
                err_msg=f"CRV1-se: cluster_adj = {cluster_adj}, cluster_df = {cluster_df}",
            )

            # now test pvalues
            np.testing.assert_allclose(
                fit1.pvalue(),
                fixest.pvalue(feols_fit1),
                rtol=1e-04,
                atol=1e-04,
                err_msg=f"CRV1-pvalue: cluster_adj = {cluster_adj}, cluster_df = {cluster_df}",
            )


def test_wls_na():
    """Special tests for WLS and NA values."""
    data = get_data()
    data = data.dropna()

    # case 1: NA in weights
    data["weights"].iloc[0] = np.nan

    fit_py = feols("Y ~ X1", data=data, weights="weights")
    fit_r = fixest.feols(
        ro.Formula("Y ~ X1"),
        data=data,
        weights=ro.Formula("~ weights"),
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    np.testing.assert_allclose(
        fit_py.coef(),
        stats.coef(fit_r),
        rtol=1e-04,
        atol=1e-04,
        err_msg="WLS: Coefs are not equal.",
    )

    # case 2: NA in weights and X1
    data["X1"].iloc[0] = np.nan
    fit_py = feols("Y ~ X1", data=data, weights="weights")
    fit_r = fixest.feols(
        ro.Formula("Y ~ X1"),
        data=data,
        weights=ro.Formula("~ weights"),
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )
    np.testing.assert_allclose(
        fit_py.coef(),
        stats.coef(fit_r),
        rtol=1e-04,
        atol=1e-04,
        err_msg="WLS: Coefs are not equal.",
    )

    # case 3: more NAs in X1:
    data["X1"].iloc[0:10] = np.nan
    fit_py = feols("Y ~ X1", data=data, weights="weights")
    fit_r = fixest.feols(
        ro.Formula("Y ~ X1"),
        data=data,
        weights=ro.Formula("~ weights"),
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )
    np.testing.assert_allclose(
        fit_py.coef(),
        stats.coef(fit_r),
        rtol=1e-04,
        atol=1e-04,
        err_msg="WLS: Coefs are not equal.",
    )


def _py_fml_to_r_fml(py_fml):
    """
    Covernt pyfixest formula.

    pyfixest multiple estimation fml syntax to fixest multiple depvar
    syntax converter,
    i.e. 'Y1 + X2 ~ X' -> 'c(Y1, Y2) ~ X'
    """
    py_fml = py_fml.replace(" ", "").replace("C(", "as.factor(")

    fml2 = py_fml.split("|")

    fml_split = fml2[0].split("~")
    depvars = fml_split[0]
    depvars = f"c({','.join(depvars.split('+'))})"

    if len(fml2) == 1:
        return f"{depvars}~{fml_split[1]}"
    elif len(fml2) == 2:
        return f"{depvars}~{fml_split[1]}|{fml2[1]}"
    else:
        return f"{depvars}~fml_split{1} | {'|'.join(fml2[1:])}"


def _c_to_as_factor(py_fml):
    """Transform formulaic C-syntax for categorical variables into R's as.factor."""
    # Define a regular expression pattern to match "C(variable)"
    pattern = r"C\((.*?)\)"

    # Define the replacement string
    replacement = r"factor(\1, exclude = NA)"

    # Use re.sub() to perform the replacement
    r_fml = re.sub(pattern, replacement, py_fml)

    return r_fml


def get_data_r(fml, data):
    # small intermezzo, as rpy2 does not drop NAs from factors automatically
    # note that fixes does this correctly
    # this currently does not yet work for C() interactions

    vars = fml.split("~")[1].split("|")[0].split("+")

    factor_vars = []
    for var in vars:
        if "C(" in var:
            var = var.replace(" ", "")
            var = var[2:-1]
            factor_vars.append(var)

    # if factor_vars is not empty
    data_r = data[~data[factor_vars].isna().any(axis=1)] if factor_vars else data

    return data_r


@pytest.mark.parametrize(
    "fml",
    [
        # ("dep_var ~ treat"),
        # ("dep_var ~ treat + unit"),
        ("dep_var ~ treat | unit"),
        ("dep_var ~ treat + unit | year"),
        ("dep_var ~ treat | year + unit"),
    ],
)
@pytest.mark.parametrize("data", [(pd.read_csv("pyfixest/did/data/df_het.csv"))])
@pytest.mark.skip("Wald tests will be released with pyfixest 0.14.0.")
def test_wald_test(fml, data):
    fit1 = feols(fml, data)
    fit1.wald_test()

    fit_r = fixest.feols(
        ro.Formula(fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    wald_r = fixest.wald(fit_r)
    wald_stat_r = wald_r[0]
    wald_pval_r = wald_r[1]  # noqa: F841

    np.testing.assert_allclose(fit1._f_statistic, wald_stat_r)
    # np.testing.assert_allclose(fit1._f_statistic_pvalue, wald_pval_r)


def test_singleton_dropping():
    data = get_data()
    # create a singleton fixed effect
    data["f1"].iloc[data.shape[0] - 1] = 999999

    fit_py = feols("Y ~ X1 | f1", data=data, fixef_rm="singleton")
    fit_py2 = feols("Y ~ X1 | f1", data=data, fixef_rm="none")
    fit_r = fixest.feols(
        ro.Formula("Y ~ X1 | f1"),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
        fixef_rm="singleton",
    )

    # test that coefficients match
    coef_py = fit_py.coef().values
    coef_py2 = fit_py2.coef().values
    coef_r = stats.coef(fit_r)

    np.testing.assert_allclose(
        coef_py,
        coef_py2,
        err_msg="singleton dropping leads to different coefficients",
    )
    np.testing.assert_allclose(
        coef_py,
        coef_r,
        rtol=1e-06,
        atol=1e-06,
        err_msg="Coefficients do not match.",
    )

    # test that number of observations match
    nobs_py = fit_py._N
    nobs_r = stats.nobs(fit_r)
    np.testing.assert_allclose(
        nobs_py,
        nobs_r,
        err_msg="Number of observations do not match.",
    )

    # test that standard errors match
    se_py = fit_py.se().values  # noqa: F841
    se_r = fixest.se(fit_r)  # noqa: F841
    # np.testing.assert_allclose(
    #    se_py, se_r, rtol=1e-04, atol=1e-04, err_msg="Standard errors do not match."
    # )


def _convert_f3(data, f3_type):
    """Convert f3 to the desired type."""
    if f3_type == "categorical":
        data["f3"] = pd.Categorical(data["f3"])
    elif f3_type == "int":
        data["f3"] = data["f3"].astype(np.int32)
    elif f3_type == "str":
        data["f3"] = data["f3"].astype(str)
    elif f3_type == "object":
        data["f3"] = data["f3"].astype(object)
    elif f3_type == "float":
        data["f3"] = data["f3"].astype(float)
    else:
        pass
    return data


def _get_r_inference(inference):
    return (
        ro.Formula("~" + inference["CRV1"])
        if isinstance(inference, dict)
        else inference
    )


def _get_r_df(r_fixest):
    fixest_df = broom.tidy_fixest(r_fixest, conf_int=ro.BoolVector([True]))
    df_r = pd.DataFrame(fixest_df).T
    df_r.columns = [
        "term",
        "estimate",
        "std.error",
        "statistic",
        "p.value",
        "conf.low",
        "conf.high",
    ]

    df_X1 = df_r.set_index("term").xs("X1")  # only test for X1

    return df_X1
