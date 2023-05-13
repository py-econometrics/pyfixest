import pytest
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
stats = importr('stats')


@pytest.fixture
def data():
    return get_data(seed = 6574)


@pytest.mark.parametrize("fml", [


    ("Y~X1"),
    ("Y~X1+X2"),
    ("Y~X1|X2"),
    ("Y~X1|X2+X3"),
    ("Y~X2|X3+X4"),

    ("Y~X1|X2^X3"),
    ("Y~X1|X2^X3 + X4"),
    ("Y~X1|X2^X3^X4"),

    ("Y ~ X1:X2"),
    ("Y ~ X1:X2 | X3"),
    ("Y ~ X1:X2 | X3 + X4"),

    ("log(Y) ~ X1:X2 | X3 + X4"),
    ("log(Y) ~ log(X1):X2 | X3 + X4"),
    ("Y ~  X2 + exp(X1) | X3 + X4"),



    #("Y ~ C(X2)"),
    #("Y ~ X1 + C(X2)"),

    #("Y ~ X1:C(X2) | X3"),
    #("Y ~ C(X2):C(X3) | X4"),




])


def test_py_vs_r(data, fml):

    '''
    test pyfixest against fixest via rpy2

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    '''

    # suppress correction for fixed effects
    #fixest.setFixest_ssc(fixest.ssc(True, "nested", True, "min", "min", False))

    # iid errors
    pyfixest = Fixest(data = data).feols(fml, vcov = 'iid')

    py_coef = np.sort(pyfixest.coef()['Estimate'])
    py_se = np.sort(pyfixest.se()['Std. Error'])
    py_pval = np.sort(pyfixest.pvalue()['Pr(>|t|)'])
    py_tstat = np.sort(pyfixest.tstat()['t value'])

    r_fixest = fixest.feols(
        ro.Formula(fml),
        se = 'iid',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
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
    py_se = pyfixest.se()['Std. Error']
    py_pval = pyfixest.pvalue()['Pr(>|t|)']
    py_tstat = pyfixest.tstat()['t value']

    r_fixest = fixest.feols(
        ro.Formula(fml),
        se = 'hetero',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for HC1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for HC1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for HC1 errors")

    # cluster robust errors
    pyfixest.vcov({'CRV1':'group_id'})
    py_se = pyfixest.se()['Std. Error']
    py_pval = pyfixest.pvalue()['Pr(>|t|)']
    py_tstat = pyfixest.tstat()['t value']

    r_fixest = fixest.feols(
        ro.Formula(fml),
        cluster = ro.Formula('~group_id'),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for CRV1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for CRV1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for CRV1 errors")


@pytest.mark.parametrize("fml_multi", [

    ("Y + Y2 ~X1"),
    ("Y + Y2 ~X1+X2"),
    ("Y + Y2 ~X1|X2"),
    ("Y + Y2 ~X1|X2+X3"),
    ("Y + Y2 ~X2|X3+X4"),

    ("Y + Y2 ~ sw(X3, X4)"),
    ("Y + Y2 ~ sw(X3, X4) | X2"),

    ("Y + Y2 ~ csw(X3, X4)"),
    ("Y + Y2 ~ csw(X3, X4) | X2"),

    ("Y + Y2 ~ X1 + csw(X3, X4)"),
    ("Y + Y2 ~ X1 + csw(X3, X4) | X2"),

    ("Y + Y2 ~ X1 + csw0(X3, X4)"),
    ("Y + Y2 ~ X1 + csw0(X3, X4) | X2"),

    ("Y + Y2 ~ X1 | csw0(X3, X4)"),
    ("Y + Y2 ~ sw(X1, X2) | csw0(X3, X4)"),





])


def test_py_vs_r2(data, fml_multi):

    '''
    test pyfixest against fixest_multi objects
    '''

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_multi, False)

    pyfixest = Fixest(data = data).feols(fml_multi)
    py_coef = pyfixest.coef()['Estimate']
    py_se = pyfixest.se()['Std. Error']
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    for x, val in enumerate(r_fixest):

        i = pyfixest.tidy().index.unique()[x]
        ix = pyfixest.tidy().xs(i)
        py_coef = ix['Estimate']
        py_se = ix['Std. Error']

        fixest_object = r_fixest.rx2(x+1)
        fixest_coef = fixest_object.rx2("coefficients")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        if not np.allclose((np.array(py_se)), (fixest.se(fixest_object))):
            raise ValueError("py_se != r_se for iid errors")


@pytest.mark.parametrize("fml_i", [
    #("Y ~ i(X1,X2)"),
    #("Y ~ i(X1,X2) | X3"),
    #("Y ~ i(X1,X2) | X3 + X4"),
    #("Y ~ i(X1,X2) | sw(X3, X4)"),
    #("Y ~ i(X1,X2) | csw(X3, X4)"),
])

@pytest.mark.skip("interactions via i() produce pytest to get stuck")
def test_py_vs_r_i(data, fml_i):

    '''
    test pyfixest against fixest_multi objects
    '''

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_i, False)

    pyfixest = Fixest(data = data).feols(fml_i, vcov = 'iid')
    py_coef = pyfixest.coef()['Estimate']
    py_se = pyfixest.se()['Std. Error']
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se = 'iid',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    for x, val in enumerate(r_fixest):

        i = pyfixest.tidy().index.unique()[x]
        ix = pyfixest.tidy().xs(i)
        py_coef = ix['Estimate']
        py_se = ix['Std. Error']

        fixest_object = r_fixest.rx2(x+1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        #if not np.allclose((np.array(py_se)), (fixest_se)):
        #    raise ValueError("py_se != r_se ")



@pytest.mark.parametrize("fml_C", [
        ("Y ~ C(X2)", "Y ~ as.factor(X2)"),
        #("Y ~ C(X1) + X2", "Y ~ as.factor(X1) + X2"),
        #("Y ~ C(X1):X2", "Y ~ as.factor(X1):X2"),
        #("Y ~ C(X1):C(X2)", "Y ~ as.factor(X1):as.factor(X2)"),
        #("Y ~ C(X1) | X2", "Y ~ as.factor(X1) | X2"),
])

def test_py_vs_r_C(data, fml_C):


    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    py_fml, r_fml = fml_C
    pyfixest = Fixest(data = data).feols(py_fml, vcov = 'iid')
    py_coef = pyfixest.coef()['Estimate']
    py_se = pyfixest.se()['Std. Error']
    r_fixest = fixest.feols(
        ro.Formula(r_fml),
        se = 'iid',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    if not np.allclose((np.array(py_coef)), (stats.coef(r_fixest))):
        raise ValueError("py_coef != r_coef")

    #if not np.allcloseual((np.array(py_se)), (fixest.se(r_fixest))):
    #    raise ValueError("py_se != r_se ")


@pytest.mark.parametrize("fml_split", [
    ("Y ~ X1"),
    ("Y ~ X1 | X2 + X3"),
])

@pytest.mark.skip("split method not yet fully implemented")
def test_py_vs_r_split(data, fml_split):

    # suppress correction for fixed effects
    fixest.setFixest_ssc(fixest.ssc(True, "none", True, "min", "min", False))

    fml = "Y ~ X1 | X2 + X3"
    pyfixest = Fixest(data = data).feols(fml_split, vcov = 'iid', split = "group_id")
    py_coef = pyfixest.coef()['Estimate']
    py_se = pyfixest.se()['Std. Error']
    r_fixest = fixest.feols(
        ro.Formula(fml_split),
        se = 'iid',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False),
        split = ro.Formula("~group_id")
    )

    for x, _ in enumerate(r_fixest):

        i = pyfixest.tidy().index.unique()[x]
        ix = pyfixest.tidy().xs(i)
        py_coef = ix['Estimate']
        py_se = ix['Std. Error']

        fixest_object = r_fixest.rx2(x+1)
        fixest_coef = fixest_object.rx2("coefficients")

        if not np.allclose((np.array(py_coef)), (fixest_coef)):
            raise ValueError("py_coef != r_coef")
        if not np.allclose((np.array(py_se)), (fixest.se(fixest_object))):
            raise ValueError("py_se != r_se ")



@pytest.mark.parametrize("fml_iv", [

    "Y ~ X1 | X1 ~ Z1",
    #"Y ~ X1 + X2 | X1 ~ Z1",

    "Y ~ X1 | X2 | X1 ~ Z1",
    "Y ~ X1 | X2 + X3 | X1 ~ Z1",
    #"Y ~ X1 + X2| X3 | X1 ~ Z1",

])


def test_py_vs_r_iv(data, fml_iv):

    '''
    tests for instrumental variables regressions
    '''

    np.random.seed(1235)

    data["Z1"] = data["X1"] * np.random.normal(data.shape[0])

    # iid errors
    pyfixest = Fixest(data = data).feols(fml_iv, vcov = 'iid')

    py_coef = np.sort(pyfixest.coef()['Estimate'])
    py_se = np.sort(pyfixest.se()['Std. Error'])
    py_pval = np.sort(pyfixest.pvalue()['Pr(>|t|)'])
    py_tstat = np.sort(pyfixest.tstat()['t value'])

    fml_r = _py_fml_to_r_fml(fml_iv, True)

    r_fixest = fixest.feols(
        ro.Formula(fml_r),
        se = 'iid',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
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
    py_se = pyfixest.se()['Std. Error']
    py_pval = pyfixest.pvalue()['Pr(>|t|)']
    py_tstat = pyfixest.tstat()['t value']

    r_fixest = fixest.feols(
        ro.Formula(fml_r),
        se = 'hetero',
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for HC1 errors")
    #if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
    #    raise ValueError("py_pval != r_pval for HC1 errors")
    #if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
    #    raise ValueError("py_tstat != r_tstat for HC1 errors")

    # cluster robust errors
    pyfixest.vcov({'CRV1':'group_id'})
    py_se = pyfixest.se()['Std. Error']
    py_pval = pyfixest.pvalue()['Pr(>|t|)']
    py_tstat = pyfixest.tstat()['t value']

    r_fixest = fixest.feols(
        ro.Formula(fml_r),
        cluster = ro.Formula('~group_id'),
        data=data,
        ssc = fixest.ssc(True, "none", True, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest))):
        raise ValueError("py_se != r_se for CRV1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest))):
        raise ValueError("py_pval != r_pval for CRV1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest)):
        raise ValueError("py_tstat != r_tstat for CRV1 errors")




def _py_fml_to_r_fml(py_fml, is_iv = False):

    '''
    pyfixest multiple estimation fml syntax to fixest multiple depvar
    syntax converter,
    i.e. 'Y1 + X2 ~ X' -> 'c(Y1, Y2) ~ X'
    '''

    if is_iv == False:

        fml_split = py_fml.split("~")
        depvars = fml_split[0]
        covars = fml_split[1]
        depvars = "c(" +  ",".join(depvars.split("+")) + ")"

        return depvars + "~" + covars

    else:

        fml2 = py_fml.split("|")

        if len(fml2) == 2:

            covars = fml2[0].split("~")[1]
            depvar =  fml2[0].split("~")[0]
            endogvars = fml2[1].split("~")[0]
            exogvars = list(set(covars) - set(endogvars))
            if exogvars == []:
                exogvars = "1"

            return depvar + "~" + exogvars + "|" + fml2[1]

        elif len(fml2) == 3:

            covars = fml2[0].split("~")[1]
            depvar =  fml2[0].split("~")[0]
            endogvars = fml2[2].split("~")[0]
            exogvars = list(set(covars) - set(endogvars))
            if exogvars == []:
                exogvars = "1"

            return depvar + "~" + exogvars + "|" + fml2[1] + "|" +  fml2[2]







