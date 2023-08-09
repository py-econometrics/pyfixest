import pytest
import re
import numpy as np
from numpy import log, exp
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
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
        # ("Y ~ X1 + I(X2, 2)"),
        ("Y ~ C(f1)"),
        ("Y ~ X1 + C(f1)"),
        ("Y ~ X1 + C(f2)"),
        ("Y ~ X1 + C(f1) + C(f2)"),
        ("Y ~ X1 + C(f1) | f2"),
        ("Y ~ X1 + C(f1) | f2 + f3"),
        # ("Y ~ X1 + C(f1):C(fe2)"),
        # ("Y ~ X1 + C(f1):C(fe2) | f3"),
        ("Y~X1|f2^f3"),
        ("Y~X1|f1 + f2^f3"),  # this one fails
        ("Y~X1|f2^f3^f1"),  # this one fails
        ("Y ~ X1:X2"),
        ("Y ~ X1:X2 | f3"),
        ("Y ~ X1:X2 | f3 + f1"),
        ("log(Y) ~ X1:X2 | f3 + f1"),
        ("log(Y) ~ log(X1):X2 | f3 + f1"),
        ("Y ~  X2 + exp(X1) | f3 + f1"),
        # ("Y ~ X1:C(X2) | X3"),
        # ("Y ~ C(X2):C(X3) | X4"),
    ],
)
def test_py_vs_r(N, seed, beta_type, error_type, fml):
    """
    test pyfixest against fixest via rpy2

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    """

    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

    vars = fml.split("~")[1].split("|")[0].split("+")

    # small intermezzo, as rpy2 does not drop NAs from factors automatically
    # note that fixes does this correctly
    # this currently does not yet work for C() interactions
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

    # suppress correction for fixed effects
    # fixest.setFixest_ssc(fixest.ssc(True, "nested", True, "min", "min", False))

    r_fml = _c_to_as_factor(fml)

    # iid errors
    pyfixest = Fixest(data=data).feols(fml, vcov="iid")

    py_coef = np.sort(pyfixest.coef())
    py_se = np.sort(pyfixest.se())
    py_pval = np.sort(pyfixest.pvalue())
    py_tstat = np.sort(pyfixest.tstat())
    py_confint = np.sort(pyfixest.confint())

    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se="iid",
        data=data_r,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_coef)), np.sort(stats.coef(r_fixest))):
        raise ValueError("py_coef != r_coef")
    if not np.allclose((np.array(py_se)), np.sort(fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for iid errors")
    if not np.allclose((np.array(py_pval)), np.sort(fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for iid errors")
    if not np.allclose(np.array(py_tstat), np.sort(fixest.tstat(r_fixest))):
        raise ValueError("py_tstat != r_tstat for iid errors")
    if not np.allclose(
        np.sort(np.array(py_confint).flatten()),
        np.sort(np.array(stats.confint(r_fixest)).flatten()),
    ):
        raise ValueError("py_confint != r_confint for iid errors")
    if not np.allclose(
        np.sort(np.array(py_confint).flatten()),
        np.sort(np.array(stats.confint(r_fixest)).flatten()),
    ):
        raise ValueError("py_confint != r_confint for iid errors")

    # heteroskedastic errors
    pyfixest.vcov("HC1")
    py_se = pyfixest.se().values
    py_pval = pyfixest.pvalue().values
    py_tstat = pyfixest.tstat().values
    py_confint = pyfixest.confint().values
    py_confint = pyfixest.confint().values

    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se="hetero",
        data=data_r,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for HC1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for HC1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for HC1 errors")
    if not np.allclose(
        np.sort(np.array(py_confint).flatten()),
        np.sort(np.array(stats.confint(r_fixest)).flatten()),
    ):
        raise ValueError("py_confint != r_confint for HC1 errors")
    if not np.allclose(
        np.sort(np.array(py_confint).flatten()),
        np.sort(np.array(stats.confint(r_fixest)).flatten()),
    ):
        raise ValueError("py_confint != r_confint for HC1 errors")

    # cluster robust errors
    pyfixest.vcov({"CRV1": "group_id"})
    py_se = pyfixest.se()
    py_pval = pyfixest.pvalue()
    py_tstat = pyfixest.tstat()
    py_confint = pyfixest.confint().values
    py_confint = pyfixest.confint().values

    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        cluster=ro.Formula("~group_id"),
        data=data_r,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for CRV1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for CRV1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for CRV1 errors")
    if not np.allclose(
        np.sort(np.array(py_confint).flatten()),
        np.sort(np.array(stats.confint(r_fixest)).flatten()),
    ):
        raise ValueError("py_confint != r_confint for CRV1 errors")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize(
    "fml_multi",
    [
        ("Y + Y2 ~X1"),
        ("Y + Y2 ~X1+X2"),
        ("Y + Y2 ~X1|f1"),
        ("Y + Y2 ~X1|f1+f2"),
        ("Y + Y2 ~X2|f2+f3"),
        ("Y + Y2 ~ sw(X1, X2)"),
        ("Y + Y2 ~ sw(X1, X2) |f1 "),
        ("Y + Y2 ~ csw(X1, X2)"),
        ("Y + Y2 ~ csw(X1, X2) | f2"),
        ("Y + Y2 ~ X1 + csw(X1, X2)"),
        ("Y + Y2 ~ X1 + csw(X1, X2) | f1"),
        ("Y + Y2 ~ X1 + csw0(X1, X2)"),
        ("Y + Y2 ~ X1 + csw0(X1, X2) | f1"),
        ("Y + Y2 ~ X1 | csw0(f1,f2)"),
        ("Y + Y2 ~ sw(X1, X2) | csw0(f1,f2,f3)"),
    ],
)
def test_py_vs_r2(N, seed, beta_type, error_type, fml_multi):
    """
    test pyfixest against fixest_multi objects
    """

    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_multi)

    pyfixest = Fixest(data=data).feols(fml_multi)
    py_coef = pyfixest.coef()
    py_se = pyfixest.se()
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    for x, _ in enumerate(r_fixest):
        fml = pyfixest.tidy().reset_index().fml.unique()[x]
        ix = pyfixest.tidy().reset_index().set_index("fml").xs(fml)
        py_coef = ix["Estimate"]
        py_se = ix["Std. Error"]

        fixest_object = r_fixest.rx2(x + 1)
        fixest_coef = fixest_object.rx2("coefficients")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        if not np.allclose((np.array(py_se)), (fixest.se(fixest_object))):
            raise ValueError("py_se != r_se for iid errors")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize(
    "fml_i",
    [
        # ("Y ~ i(X1,X2)"),
        # ("Y ~ i(X1,X2) | f2"),
        # ("Y ~ i(X1,X2) | f2 + f3"),
        # ("Y ~ i(X1,X2) | sw(f2, f3)"),
        # ("Y ~ i(X1,X2) | csw(f2, f3)"),
    ],
)
@pytest.mark.skip("interactions via i() produce pytest to get stuck")
def test_py_vs_r_i(N, seed, beta_type, error_type, fml_i):
    """
    test pyfixest against fixest_multi objects
    """

    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_i)

    pyfixest = Fixest(data=data).feols(fml_i, vcov="iid")
    py_coef = pyfixest.coef()
    py_se = pyfixest.se()
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se="iid",
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    for x, _ in enumerate(r_fixest):
        fml = pyfixest.tidy().reset_index().fml.unique()[x]
        ix = pyfixest.tidy().xs(fml)
        py_coef = ix["Estimate"]
        py_se = ix["Std. Error"]

        fixest_object = r_fixest.rx2(x + 1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        # if not np.allclose((np.array(py_se)), (fixest_se)):
        #    raise ValueError("py_se != r_se ")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize(
    "fml_C",
    [
        ("Y ~ C(f1)", "Y ~ as.factor(f1)"),
        # ("Y ~ C(X1) + X2", "Y ~ as.factor(X1) + X2"),
        # ("Y ~ C(X1):X2", "Y ~ as.factor(X1):X2"),
        # ("Y ~ C(X1):C(X2)", "Y ~ as.factor(X1):as.factor(X2)"),
        # ("Y ~ C(X1) | X2", "Y ~ as.factor(X1) | X2"),
    ],
)
def test_py_vs_r_C(N, seed, beta_type, error_type, fml_C):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)
    vars = fml.split("~")[1].split("|")[0].split("+")

    # small intermezzo, as rpy2 does not drop NAs from factors automatically
    # note that fixes does this correctly
    # this currently does not yet work for C() interactions

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

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    py_fml, r_fml = fml_C
    pyfixest = Fixest(data=data).feols(py_fml, vcov="iid")
    py_coef = pyfixest.coef()
    py_se = pyfixest.se()
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se="iid",
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_coef)), (stats.coef(r_fixest))):
        raise ValueError("py_coef != r_coef")

    # if not np.allcloseual((np.array(py_se)), (fixest.se(r_fixest))):
    #    raise ValueError("py_se != r_se ")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize(
    "fml_split",
    [
        ("Y ~ X1"),
        ("Y ~ X1 | X2 + X3"),
    ],
)
@pytest.mark.skip("split method not yet fully implemented")
def test_py_vs_r_split(N, seed, beta_type, error_type, fml_split):
    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    fml = "Y ~ X1 | X2 + X3"
    pyfixest = Fixest(data=data).feols(fml_split, vcov="iid", split="group_id")
    py_coef = pyfixest.coef()
    py_se = pyfixest.se()
    r_fixest = fixest.feols(
        ro.Formula(fml_split),
        se="iid",
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
        split=ro.Formula("~group_id"),
    )

    for x, _ in enumerate(r_fixest):
        fml = pyfixest.tidy().reset_index().fml.unique()[x]
        ix = pyfixest.tidy().xs(fml)
        py_coef = ix["Estimate"]
        py_se = ix["Std. Error"]

        fixest_object = r_fixest.rx2(x + 1)
        fixest_coef = fixest_object.rx2("coefficients")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        if not np.allclose((np.array(py_se)), (fixest.se(fixest_object))):
            raise ValueError("py_se != r_se ")


@pytest.mark.parametrize("N", [100, 1000, 10000])
@pytest.mark.parametrize("seed", [1234, 5678, 9012])
@pytest.mark.parametrize("beta_type", ["1", "2", "3"])
@pytest.mark.parametrize("error_type", ["1", "2", "3"])
@pytest.mark.parametrize(
    "fml_iv",
    [
        "Y ~ 1 | X1 ~ Z1",
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
        # "Y ~ X1 + X2 | X1 + X2 ~ Z1 + Z2",
        # tests of overidentified models
        "Y ~ 1 | X1 ~ Z1 + Z2",
        "Y ~ X2 | X1 ~ Z1 + Z2",
        "Y ~ X2 + C(f1) | X1 ~ Z1 + Z2",
        "Y ~ 1 | f1 | X1 ~ Z1 + Z2",
        "Y2 ~ 1 | f1 + f2 | X1 ~ Z1 + Z2",
        "Y2 ~  X2| f2 | X1 ~ Z1 + Z2",
    ],
)
def test_py_vs_r_iv(N, seed, beta_type, error_type, fml_iv):
    """
    tests for instrumental variables regressions
    """

    data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

    # iid errors
    pyfixest = Fixest(data=data).feols(fml_iv, vcov="iid")

    py_coef = np.sort(pyfixest.coef())
    py_se = np.sort(pyfixest.se())
    py_pval = np.sort(pyfixest.pvalue())
    py_tstat = np.sort(pyfixest.tstat())

    r_fixest = fixest.feols(
        ro.Formula(fml_iv),
        se="iid",
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_coef)), np.sort(stats.coef(r_fixest))):
        raise ValueError("py_coef != r_coef")
    if not np.allclose((np.array(py_se)), np.sort(fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for iid errors")
    if not np.allclose((np.array(py_pval)), np.sort(fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for iid errors")
    if not np.allclose(np.array(py_tstat), np.sort(fixest.tstat(r_fixest))):
        raise ValueError("py_tstat != r_tstat for iid errors")

    # heteroskedastic errors
    pyfixest.vcov("HC1")
    py_se = pyfixest.se()
    py_pval = pyfixest.pvalue()
    py_tstat = pyfixest.tstat()

    r_fixest = fixest.feols(
        ro.Formula(fml_iv),
        se="hetero",
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for HC1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for HC1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for HC1 errors")

    # cluster robust errors
    pyfixest.vcov({"CRV1": "group_id"})
    py_se = pyfixest.se()
    py_pval = pyfixest.pvalue()
    py_tstat = pyfixest.tstat()

    r_fixest = fixest.feols(
        ro.Formula(fml_iv),
        cluster=ro.Formula("~group_id"),
        data=data,
        ssc=fixest.ssc(True, "none", True, "min", "min", False),
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for CRV1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for CRV1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for CRV1 errors")


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
