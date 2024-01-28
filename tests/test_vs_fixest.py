import pytest
import re
import warnings
import numpy as np
import pandas as pd
from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data, ssc
from pyfixest.exceptions import NotImplementedError

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

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

rng = np.random.default_rng(8760985)


@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("seed", [76540251])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("model", ["Feols"])
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "group_id"}])
@pytest.mark.parametrize("weights", ["weights"])
@pytest.mark.parametrize(
    "fml",
    [
        ("Y~X1"),
        ("Y~X1+X2"),
        ("Y~X1|f2"),
        ("Y~X1|f2+f3"),
        ("log(Y) ~ X1"),
        ("Y ~ X1 + exp(X2)"),
        ("Y ~ X1 + C(f1)"),
        ("Y ~ X1 + C(f2)"),
        ("Y ~ X1 + C(f1) + C(f2)"),
        ("Y ~ X1 + C(f1) | f2"),
        ("Y ~ X1 + C(f1) | f2 + f3"),
        # ("Y ~ X1 + C(f1):C(fe2)"),            # currently does not work as C():C() translation not implemented
        # ("Y ~ X1 + C(f1):C(fe2) | f3"),       # currently does not work as C():C() translation not implemented
        ("Y~X1|f2^f3"),
        ("Y~X1|f1 + f2^f3"),
        # ("Y~X1|f2^f3^f1"),
        ("Y ~ X1 + X2:f1"),
        ("Y ~ X1 + X2:f1 | f3"),
        ("Y ~ X1 + X2:f1 | f3 + f1"),
        # ("log(Y) ~ X1:X2 | f3 + f1"),               # currently, causes big problems for Fepois (takes a long time)
        # ("log(Y) ~ log(X1):X2 | f3 + f1"),          # currently, causes big problems for Fepois (takes a long time)
        # ("Y ~  X2 + exp(X1) | f3 + f1"),            # currently, causes big problems for Fepois (takes a long time)
        ("Y ~ X1 + i(f1,X2)"),  # temporarily non-supported feature
        ("Y ~ X1 + i(f2,X2)"),  # temporarily non-supported feature
        ("Y ~ X1 + i(f1,X2) | f2"),  # temporarily non-supported feature
        ("Y ~ X1 + i(f1,X2) | f2 + f3"),  # temporarily non-supported feature
        # ("Y ~ i(f1,X2, ref='1.0')"),               # currently does not work
        # ("Y ~ i(f2,X2, ref='2.0')"),               # currently does not work
        # ("Y ~ i(f1,X2, ref='3.0') | f2"),          # currently does not work
        # ("Y ~ i(f1,X2, ref='4.0') | f2 + f3"),     # currently does not work
        ("Y ~ X1 + C(f1)"),
        ("Y ~ X1 + C(f1) + C(f2)"),
        # ("Y ~ C(f1):X2"),                          # currently does not work as C():X translation not implemented
        # ("Y ~ C(f1):C(f2)"),                       # currently does not work
        ("Y ~ X1 + C(f1) | f2"),
        ("Y ~ X1 + I(X2 ** 2)"),
        ("Y ~ X1 + I(X1 ** 2) + I(X2**4)"),
        ("Y ~ X1*X2"),
        ("Y ~ X1*X2 | f1+f2"),
        # ("Y ~ X1/X2"),                             # currently does not work as X1/X2 translation not implemented
        # ("Y ~ X1/X2 | f1+f2"),                     # currently does not work as X1/X2 translation not implemented
        # ("Y ~ X1 + poly(X2, 2) | f1"),             # bug in formulaic in case of NAs in X1, X2
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
        # tests of overidentified models
        "Y ~ 1 | X1 ~ Z1 + Z2",
        "Y ~ X2 | X1 ~ Z1 + Z2",
        "Y ~ X2 + C(f1) | X1 ~ Z1 + Z2",
        "Y ~ 1 | f1 | X1 ~ Z1 + Z2",
        "Y2 ~ 1 | f1 + f2 | X1 ~ Z1 + Z2",
        "Y2 ~  X2| f2 | X1 ~ Z1 + Z2",
    ],
)
def test_single_fit(N, seed, beta_type, error_type, dropna, model, inference, weights, fml):
    """
    test pyfixest against fixest via rpy2 (OLS, IV, Poisson)

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    """

    inference_inflation_factor = 1.0

    if model == "Feols":
        data = get_data(
            N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Feols"
        )
    else:
        data = get_data(
            N=N, seed=seed, beta_type=beta_type, error_type=error_type, model="Fepois"
        )

    # long story, but categories need to be strings to be converted to R factors,
    # this then produces 'nan' values in the pd.DataFrame ...
    data[data == "nan"] = np.nan

    if dropna:
        data = data.dropna()

    data_r = get_data_r(fml, data)

    # convert py expressions to R expressions
    r_fml = _c_to_as_factor(fml)
    if isinstance(inference, dict):
        r_inference = ro.Formula("~" + inference["CRV1"])
    else:
        r_inference = inference

    # iid errors
    try:
        with warnings.catch_warnings():
            # ignore run time warnings (likely due to large Y values)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pyfixest = feols(fml=fml, data=data, vcov=inference)
    except ValueError as e:
        if "is not of type 'O' or 'category'" in str(e):
            data["f1"] = pd.Categorical(data.f1.astype(str))
            data["f2"] = pd.Categorical(data.f2.astype(str))
            data["f3"] = pd.Categorical(data.f3.astype(str))
        else:
            raise e

    if model == "Feols":
        pyfixest = feols(fml=fml, data=data, vcov=inference, weights = weights)
        if weights is not None:
            r_fixest = fixest.feols(
                ro.Formula(r_fml),
                vcov=r_inference,
                data=data_r,
                ssc=fixest.ssc(True, "none", True, "min", "min", False),
                weights = ro.Formula("~" + weights)
            )
        else:
            r_fixest = fixest.feols(
                ro.Formula(r_fml),
                vcov=r_inference,
                data=data_r,
                ssc=fixest.ssc(True, "none", True, "min", "min", False),
            )

        run_test = True

    else:
        # check if IV - don not run IV formulas for Poisson

        with warnings.catch_warnings():
            # ignore run time warnings (likely due to large Y values)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            iv_check = feols(fml=fml, data=data, vcov="iid")

        # if inference == "iid":
        #    return pytest.skip("Poisson does not support iid inference")

        if iv_check._is_iv:
            is_iv = True
            run_test = False
        else:
            is_iv = False
            run_test = True

            # if formula does not contain "i(" or "C(", add, separation:
            if "i(" not in fml and "C(" not in fml:
                where_zeros = np.where(data["Y"] == 0)[
                    0
                ]  # because np.where evaluates to a tuple
                # draw three random indices
                # idx = rng.choice(where_zeros, 3, True)
                idx = np.array([10, 11, 12])
                data.loc[idx[0], "f1"] = np.max(data["f1"]) + 1
                data.loc[idx[1], "f2"] = np.max(data["f2"]) + 1
                data.loc[idx[2], "f3"] = np.max(data["f3"]) + 1

            if "i(" in fml:
                pytest.skip("Don't test interactions for Poisson.")

            if "^" in fml:
                pytest.skip("Don't test '^' for Poisson.")

            # relax tolerance for Poisson regression - effective rtol and atol of
            # 5e-05
            inference_inflation_factor = 100

            try:
                pyfixest = fepois(fml=fml, data=data, vcov=inference)
            except NotImplementedError as exception:
                if "inference is not supported" in str(exception):
                    return pytest.skip(
                        "'iid' inference is not supported for Poisson regression."
                    )
                raise
            except ValueError as exception:
                if "dependent variable must be a weakly positive" in str(exception):
                    return pytest.skip(
                        "Poisson model requires strictly positive dependent variable."
                    )
                raise
            except RuntimeError as exception:
                if "Failed to converge after 1000000 iterations." in str(exception):
                    return pytest.skip(
                        "Maximum number of PyHDFE iterations reached. Nothing I can do here."
                    )
                raise

            r_fixest = fixest.fepois(
                ro.Formula(r_fml),
                vcov=r_inference,
                data=data_r,
                ssc=fixest.ssc(True, "none", True, "min", "min", False),
                glm_iter=iwls_maxiter,
                glm_tol=iwls_tol,
            )

            py_nobs = pyfixest._N
            r_nobs = stats.nobs(r_fixest)

    if run_test:
        # get coefficients, standard errors, p-values, t-statistics, confidence intervals

        mod = pyfixest

        py_coef = mod.coef().xs("X1")
        py_se = mod.se().xs("X1")
        py_pval = mod.pvalue().xs("X1")
        py_tstat = mod.tstat().xs("X1")
        py_confint = mod.confint().xs("X1").values
        py_nobs = mod._N
        py_resid = mod._u_hat.flatten()
        # TODO: test residuals

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

        if mod._is_iv:
            df_X1 = df_r.set_index("term").xs("fit_X1")  # only test for X1
        else:
            df_X1 = df_r.set_index("term").xs("X1")  # only test for X1

        r_coef = df_X1["estimate"]
        r_se = df_X1["std.error"]
        r_pval = df_X1["p.value"]
        r_tstat = df_X1["statistic"]
        r_confint = df_X1[["conf.low", "conf.high"]].values.astype(np.float64)
        r_nobs = stats.nobs(r_fixest)
        r_resid = r_fixest.rx2("working_residuals")

        np.testing.assert_allclose(
            py_coef, r_coef, rtol=rtol, atol=atol, err_msg="py_coef != r_coef"
        )

        # np.testing.assert_allclose(
        #    py_resid,
        #    r_resid,
        #    rtol = 1e-04,
        #    atol = 1e-04,
        #    err_msg = "py_resid != r_resid"
        # )

        np.testing.assert_allclose(
            py_se,
            r_se,
            rtol=rtol * inference_inflation_factor,
            atol=atol * inference_inflation_factor,
            err_msg=f"py_se != r_se for {inference} errors.",
        )

        np.testing.assert_allclose(
            py_pval,
            r_pval,
            rtol=rtol * inference_inflation_factor,
            atol=atol * inference_inflation_factor,
            err_msg=f"py_pval != r_pval for {inference} errors.",
        )

        np.testing.assert_allclose(
            py_tstat,
            r_tstat,
            rtol=rtol * inference_inflation_factor,
            atol=atol * inference_inflation_factor,
            err_msg=f"py_tstat != r_tstat for {inference} errors",
        )

        np.testing.assert_allclose(
            py_confint,
            r_confint,
            rtol=rtol * inference_inflation_factor,
            atol=atol * inference_inflation_factor,
            err_msg=f"py_confint != r_confint for {inference} errors",
        )

        np.testing.assert_allclose(
            py_nobs, r_nobs, rtol=rtol, atol=atol, err_msg="py_nobs != r_nobs"
        )

        if model == "Feols":

            if not mod._is_iv:
                py_r2 = mod._r2
                py_r2_within = mod._r2_within
                py_adj_r2 = mod._adj_r2
                py_adj_r2_within = mod._adj_r2_within

                r_r = fixest.r2(r_fixest)
                # unadjusted
                np.testing.assert_allclose(
                    py_r2, r_r[1], rtol=rtol, atol=atol, err_msg="py_r2 != r_r"
                )
                np.testing.assert_allclose(
                    py_r2_within,
                    r_r[5],
                    rtol=rtol,
                    atol=atol,
                    err_msg="py_r2_within != r_r",
                )

                if False:
                    # adjusted
                    np.testing.assert_allclose(
                        py_adj_r2,
                        r_r[2],
                        rtol=rtol,
                        atol=atol,
                        err_msg="py_adj_r2 != r_r",
                    )
                    np.testing.assert_allclose(
                        py_adj_r2_within,
                        r_r[6],
                        rtol=rtol,
                        atol=atol,
                        err_msg="py_adj_r2_within != r_r",
                    )

        if model == "Fepois":
            r_deviance = r_fixest.rx2("deviance")
            py_deviance = mod.deviance
            np.testing.assert_allclose(
                py_deviance,
                r_deviance,
                rtol=rtol,
                atol=atol,
                err_msg="py_deviance != r_deviance",
            )


@pytest.mark.parametrize("N", [100])
@pytest.mark.parametrize("seed", [17021])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
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
    """
    test pyfixest against fixest_multi objects
    """

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
        py_coef, py_se = [np.sort(x) for x in [py_coef, py_se]]

        fixest_object = r_fixest.rx2(x + 1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        # fixest_coef = stats.coef(r_fixest)
        # fixest_se = fixest.se(r_fixest)

        # sort fixest_coef, fixest_se
        fixest_coef, fixest_se = [np.sort(x) for x in [fixest_coef, fixest_se]]

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
            fit2 = feols(
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


def _py_fml_to_r_fml(py_fml):
    """
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
    """
    transform formulaic C-syntax for categorical variables into R's as.factor
    """
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
    if factor_vars:
        data_r = data[~data[factor_vars].isna().any(axis=1)]
    else:
        data_r = data

    return data_r


@pytest.mark.skip("Currently not supported.")
def test_i_interaction():
    """
    Test that interaction syntax via the `i()` operator works as in fixest
    """

    data = get_data(N=1000, seed=17021, beta_type="1", error_type="1").dropna()

    fit1 = feols("Y ~ i(f1, X2)", data=data)
    fit2 = feols("Y ~ X1 + i(f1, X2) | f2", data=data)
    # fit3 = feols("Y ~ X1 + i(f1, X2) | f2", data=data, i_ref1=1.0)
    # fit4 = feols("Y ~ X1 + i(f1, X2) | f2", data=data, i_ref1=[2.0])
    # fit5 = feols("Y ~ X1 + i(f1, X2) | f2", data=data, i_ref1=[2.0, 3.0])

    fit1_r = fixest.feols(
        ro.Formula("Y ~ i(f1, X2)"),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )
    fit2_r = fixest.feols(
        ro.Formula("Y ~ X1 + i(f1, X2) | f2"),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )
    # fit3_r = fixest.feols(
    #    ro.Formula("Y ~ X1 + i(f1, X2, ref = 1.0) | f2"),
    #    data=data,
    #    ssc=fixest.ssc(True, "none", True, "min", "min", False),
    # )
    # fit4_r = fixest.feols(
    #    ro.Formula("Y ~ X1 + i(f1, X2, ref = 2.0) | f2"),
    #    data=data,
    #    ssc=fixest.ssc(True, "none", True, "min", "min", False),
    # )
    # fit5_r = fixest.feols(
    #    ro.Formula("Y ~ X1 + i(f1, X2, ref = c(2.0, 3.0)) | f2"),
    #    data=data,
    #    ssc=fixest.ssc(True, "none", True, "min", "min", False),
    # )

    # create tuples: (pyfixest, fixest)
    fits = [
        (fit1, fit1_r),
        (fit2, fit2_r),
        # (fit3, fit3_r),
        # (fit4, fit4_r),
        # (fit5, fit5_r),
    ]

    for fit in fits:
        # test that coefficients match
        coef_py = fit[0].coef().values
        coef_r = stats.coef(fit[1])
        np.testing.assert_allclose(
            coef_py,
            coef_r,
            rtol=1e-04,
            atol=1e-04,
            err_msg="Coefficients do not match.",
        )

        # test that standard errors match
        se_py = fit[0].se().values
        se_r = fixest.se(fit[1])
        np.testing.assert_allclose(
            se_py, se_r, rtol=1e-04, atol=1e-04, err_msg="Standard errors do not match."
        )


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
    wald_pval_r = wald_r[1]

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
    se_py = fit_py.se().values
    se_r = fixest.se(fit_r)
    # np.testing.assert_allclose(
    #    se_py, se_r, rtol=1e-04, atol=1e-04, err_msg="Standard errors do not match."
    # )
