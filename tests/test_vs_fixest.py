import pytest
import re
import numpy as np
import pandas as pd
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data
from pyfixest.exceptions import NotImplementedError

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")

rtol = 1e-05
atol = 1e-05

iwls_maxiter = 1000
iwls_tol = 1e-10

small_test = False

@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("seed", [8711])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("model", ['Feols','Fepois'])
@pytest.mark.parametrize("inference", ['iid','hetero', {'CRV1':'group_id'}])



@pytest.mark.parametrize(
    "fml",
    [
        ("Y~X1"),
        ("Y~X1+X2"),
        ("Y~X1|f2"),
        ("Y~X1|f2+f3"),
        ("Y~X2|f2+f3"),
        ("log(Y) ~ X1"),
        ("Y ~ exp(X1)"),
        ("Y ~ C(f1)"),
        ("Y ~ X1 + C(f1)"),
        ("Y ~ X1 + C(f2)"),
        ("Y ~ X1 + C(f1) + C(f2)"),
        ("Y ~ X1 + C(f1) | f2"),
        ("Y ~ X1 + C(f1) | f2 + f3"),
        #("Y ~ X1 + C(f1):C(fe2)"),            # currently does not work as C():C() translation not implemented
        #("Y ~ X1 + C(f1):C(fe2) | f3"),       # currently does not work as C():C() translation not implemented
        #("Y~X1|f2^f3"),
        #("Y~X1|f1 + f2^f3"),
        #("Y~X1|f2^f3^f1"),
        ("Y ~ X1:X2"),
        ("Y ~ X1:X2 | f3"),
        ("Y ~ X1:X2 | f3 + f1"),
        #("log(Y) ~ X1:X2 | f3 + f1"),               # currently, causes big problems for Fepois (takes a long time)
        #("log(Y) ~ log(X1):X2 | f3 + f1"),          # currently, causes big problems for Fepois (takes a long time)
        #("Y ~  X2 + exp(X1) | f3 + f1"),            # currently, causes big problems for Fepois (takes a long time)
        ("Y ~ i(f1,X2)"),
        ("Y ~ i(f2,X2)"),
        ("Y ~ i(f1,X2) | f2"),
        ("Y ~ i(f1,X2) | f2 + f3"),
        #("Y ~ i(f1,X2, ref='1.0')"),               # currently does not work
        #("Y ~ i(f2,X2, ref='2.0')"),               # currently does not work
        #("Y ~ i(f1,X2, ref='3.0') | f2"),          # currently does not work
        #("Y ~ i(f1,X2, ref='4.0') | f2 + f3"),     # currently does not work
        ("Y ~ C(f1)"),
        ("Y ~ C(f1) + C(f2)"),
        #("Y ~ C(f1):X2"),                          # currently does not work as C():X translation not implemented
        #("Y ~ C(f1):C(f2)"),                       # currently does not work
        ("Y ~ C(f1) | f2"),

        ("Y ~ I(X1 ** 2)"),
        ("Y ~ I(X1 ** 2) + I(X2**4)"),
        ("Y ~ X1*X2"),
        ("Y ~ X1*X2 | f1+f2"),
        #("Y ~ X1/X2"),                             # currently does not work as X1/X2 translation not implemented
        #("Y ~ X1/X2 | f1+f2"),                     # currently does not work as X1/X2 translation not implemented

        #("Y ~ X1 + poly(X2, 2) | f1"),             # bug in formulaic in case of NAs in X1, X2

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
def test_single_fit(N, seed, beta_type, error_type, dropna, model, inference, fml):
    """
    test pyfixest against fixest via rpy2 (OLS, IV, Poisson)

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    """

    if model == "Feols":
        data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type, model = "Feols")
    else:
        data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type, model = "Fepois")

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
        pyfixest = Fixest(data=data).feols(fml, vcov=inference)
    except ValueError as e:
        if "is not of type 'O' or 'category'" in str(e):
            data["f1"] = pd.Categorical(data.f1.astype(str))
            data["f2"] = pd.Categorical(data.f2.astype(str))
            data["f3"] = pd.Categorical(data.f3.astype(str))
        else:
            raise ValueError("Code fails with an uninformative error message.")

    if model == "Feols":

        pyfixest = Fixest(data=data).feols(fml, vcov=inference)
        r_fixest = fixest.feols(
            ro.Formula(r_fml),
            vcov=r_inference,
            data=data_r,
            ssc=fixest.ssc(True, "none", True, "min", "min", False)
        )

        run_test = True

    else:

        # check if IV - only run poisson if not IV
        iv_check = Fixest(data=data).feols(fml, vcov="iid")

        if iv_check._is_iv:
            is_iv = True
            run_test = False
        else:
            is_iv = False
            run_test = True

            # check for separation in C() in first part of the formula
            if "C(" in fml.split("|")[0]:
                #find all variables contained in "C()"
                variables = re.findall(r"C\((.*?)\)", fml.split("|")[0])
                variables = [x.strip() for x in variables]
                #find the dependent variable
                dependent_variable = fml.split("~")[0].strip()
                Y = data[[dependent_variable]].values
                fe = data[variables].values
                ##check if there is separation
                na_separation = _check_separation(Y, fe)
                if na_separation:
                    data = data.drop(na_separation, axis = 0)
                    data_r = data_r.drop(na_separation, axis = 0)

            pyfixest = Fixest(data=data, iwls_tol = iwls_tol, iwls_maxiter = iwls_maxiter)

            try:
                pyfixest.fepois(fml, vcov=inference)
            except ValueError as exception:
                if "dependent variable must be a weakly positive" in str(exception):
                    return pytest.skip("Poisson model requires strictly positive dependent variable.")
                raise
            except NotImplementedError as exception:
                if "iid inference is not supported for non-linear models" in str(exception):
                    return pytest.skip("iid inference is not supported for non-linear models.")
                raise
            except RuntimeError as exception:
                if "Failed to converge after 1000000 iterations." in str(exception):
                    return pytest.skip("Maximum number of PyHDFE iterations reached. Nothing I can do here.")
                raise

            r_fixest = fixest.fepois(
                ro.Formula(r_fml),
                vcov=r_inference,
                data=data_r,
                ssc=fixest.ssc(True, "none", True, "min", "min", False),
                glm_iter = iwls_maxiter,
                glm_tol = iwls_tol
            )

            py_nobs = pyfixest.fetch_model(0).N
            r_nobs = stats.nobs(r_fixest)

            #if py_nobs != r_nobs:
            #    return pytest.skip("R and Py models run on different numbers observations.")

    if run_test:

        # get coefficients, standard errors, p-values, t-statistics, confidence intervals
        py_coef = pyfixest.coef().values
        py_se = pyfixest.se().values
        py_pval = pyfixest.pvalue().values
        py_tstat = pyfixest.tstat().values
        py_confint = pyfixest.confint().values.flatten()
        py_nobs = pyfixest.fetch_model(0).N


        # write list comprehension that sorts py_coef py_ses etc with np.sort
        py_coef, py_se, py_pval, py_tstat, py_confint = [np.sort(x) for x in [py_coef, py_se, py_pval, py_tstat, py_confint]]

        r_coef = stats.coef(r_fixest)
        r_se = fixest.se(r_fixest)
        r_pval = fixest.pvalue(r_fixest)
        r_tstat = fixest.tstat(r_fixest)
        r_confint = np.array(stats.confint(r_fixest)).flatten()
        r_nobs = stats.nobs(r_fixest)


        # write list comprehension that sorts py_coef py_ses etc with np.sort
        r_coef, r_se, r_pval, r_tstat, r_confint = [np.sort(x) for x in [r_coef, r_se, r_pval, r_tstat, r_confint]]



        #np.testing.assert_allclose(
        #    py_coef,
        #    r_coef,
        #    rtol = rtol,
        #    atol = atol,
        #    err_msg = "py_coef != r_coef"
        #)

        #np.testing.assert_allclose(
        #    py_se,
        #    r_se,
        #    rtol = rtol,
        #    atol = atol,
        #    err_msg = "py_se != r_se for iid errors"
        #)

        #np.testing.assert_allclose(
        #    py_pval,
        #    r_pval,
        #    rtol = rtol,
        #    atol = atol,
        #    err_msg = "py_pval != r_pval for iid errors"
        #)

        #np.testing.assert_allclose(
        #    py_tstat,
        #    r_tstat,
        #    rtol = rtol,
        #    atol = atol,
        #    err_msg = "py_tstat != r_tstat for iid errors"
        #)

        #np.testing.assert_allclose(
        #    py_confint,
        #    r_confint,
        #    rtol = rtol,
        #    atol = atol,
        #    err_msg = "py_confint != r_confint for iid errors"
        #)

        np.testing.assert_allclose(
            py_nobs,
            r_nobs,
            rtol = rtol,
            atol = atol,
            err_msg = "py_nobs != r_nobs"
        )



@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("seed", [17021])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize("dropna", [False, True])

@pytest.mark.parametrize(
    "fml_multi",
    [
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

        #("Y ~ i(f1,X2) | csw0(f2)"),
        #("Y ~ i(f1,X2) | sw0(f2)"),
        #("Y ~ i(f1,X2) | csw(f2, f3)"),
        #("Y ~ i(f1,X2) | sw(f2, f3)"),

        #("Y ~ i(f1,X2, ref = -5) | sw(f2, f3)"),
        #("Y ~ i(f1,X2, ref = -8) | csw(f2, f3)"),
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
        pyfixest = Fixest(data=data).feols(fml_multi)
    except ValueError as e:
        if "is not of type 'O' or 'category'" in str(e):
            data["f1"] = pd.Categorical(data.f1.astype(str))
            data["f2"] = pd.Categorical(data.f2.astype(str))
            data["f3"] = pd.Categorical(data.f3.astype(str))
            data[data == "nan"] = np.nan
            pyfixest = Fixest(data=data).feols(fml_multi)
        else:
            raise ValueError("Code fails with an uninformative error message.")

    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False)
    )

    for x, _ in range(0):

        mod = pyfixest.fetch_model(x)
        py_coef = mod.coef().values
        py_se = mod.se().values

        # sort py_coef, py_se
        py_coef, py_se = [np.sort(x) for x in [py_coef, py_se]]

        fixest_object = r_fixest.rx2(x+1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        #fixest_coef = stats.coef(r_fixest)
        #fixest_se = fixest.se(r_fixest)

        # sort fixest_coef, fixest_se
        fixest_coef, fixest_se = [np.sort(x) for x in [fixest_coef, fixest_se]]

        np.testing.assert_allclose(
            py_coef,
            fixest_coef,
            rtol = rtol,
            atol = atol,
            err_msg = "Coefs are not equal."
        )
        np.testing.assert_allclose(
            py_se,
            fixest_se,
            rtol = rtol,
            atol = atol,
            err_msg = "SEs are not equal."
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
    depvars = "c(" + ",".join(depvars.split("+")) + ")"

    if len(fml2) == 1:
        return depvars + "~" + fml_split[1]
    elif len(fml2) == 2:
        return depvars + "~" + fml_split[1] + "|" + fml2[1]
    else:
        return depvars + "~" + fml_split[1] + "|" + "|".join(fml2[1:])


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


def _check_separation(Y, fe):

    Y_help = pd.Series(np.where(Y.flatten() > 0, 1, 0))
    fe = pd.DataFrame(fe)

    separation_na = set()
    # loop over all elements of fe
    for x in fe.columns:
        ctab = pd.crosstab(Y_help, fe[x])
        null_column = ctab.xs(0)
        # fixed effect "nested" in Y == 0. cond 1: fixef combi only in nested in specific value of Y. cond 2: fixef combi only in nested in Y == 0
        sep_candidate = (np.sum(ctab > 0, axis=0).values == 1) & (
            null_column > 0
        ).values.flatten()
        droplist = ctab.xs(0)[sep_candidate].index.tolist()

        if len(droplist) > 0:
            dropset = set(np.where(fe[x].isin(droplist))[0])
            separation_na = separation_na.union(dropset)

    separation_na = list(separation_na)

    return separation_na