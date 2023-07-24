import pytest
import numpy as np
from numpy import log, exp
from pyfixest.fixest import Fixest
from pyfixest.utils import get_data, get_poisson_data
from pyfixest.ssc_utils import ssc

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

@pytest.fixture
def data_poisson():
    return get_poisson_data(seed = 6574)

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

    #("log(Y) ~ X1:X2 | X3 + X4"),
    #("log(Y) ~ log(X1):X2 | X3 + X4"),
    ("Y ~  X2 + exp(X1) | X3 + X4"),



    #("Y ~ C(X2)"),
    #("Y ~ X1 + C(X2)"),

    #("Y ~ X1:C(X2) | X3"),
    #("Y ~ C(X2):C(X3) | X4"),

])


def test_py_vs_r_poisson(data_poisson, fml):

    '''
    test pyfixest against fixest via rpy2

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    '''

    # iid errors
    pyfixest = Fixest(data = data_poisson).fepois(fml, vcov = 'HC1', ssc = ssc(adj = False, cluster_adj = False))

    py_coef = np.sort(pyfixest.coef())
    py_se = np.sort(pyfixest.se())
    py_pval = np.sort(pyfixest.pvalue())
    py_tstat = np.sort(pyfixest.tstat())

    r_fixest = fixest.fepois(
        ro.Formula(fml),
        se = 'iid',
        data=data_poisson,
        ssc = fixest.ssc(False, "none", False, "min", "min", False)
    )

    if not np.allclose((np.array(py_coef)), np.sort(stats.coef(r_fixest))):
        raise ValueError("py_coef != r_coef")
    #if not np.allclose((np.array(py_se)), np.sort(fixest.se(r_fixest))):
    #    raise ValueError("py_se != r_se for iid errors")
    #if not np.allclose((np.array(py_pval)), np.sort(fixest.pvalue(r_fixest))):
    #    raise ValueError("py_pval != r_pval for iid errors")
    #if not np.allclose(np.array(py_tstat), np.sort(fixest.tstat(r_fixest))):
    #    raise ValueError("py_tstat != r_tstat for iid errors")

    # heteroskedastic errors
    pyfixest.vcov("HC1")
    py_se = pyfixest.se().values
    py_pval = pyfixest.pvalue().values
    py_tstat = pyfixest.tstat().values

    r_fixest = fixest.fepois(
        ro.Formula(fml),
        se = 'hetero',
        data=data_poisson,
        ssc = fixest.ssc(False, "none", False, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest)), rtol = 1e-3, atol = 1e-3):
        raise ValueError("py_se != r_se for HC1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest)), rtol = 1e-3, atol = 1e-3):
        raise ValueError("py_pval != r_pval for HC1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest), rtol = 1e-3, atol = 1e-3):
        raise ValueError("py_tstat != r_tstat for HC1 errors")

    # cluster robust errors
    pyfixest.vcov({'CRV1':'X4'})
    py_se = pyfixest.se()
    py_pval = pyfixest.pvalue()
    py_tstat = pyfixest.tstat()
    r_fixest = fixest.fepois(
        ro.Formula(fml),
        cluster = ro.Formula('~X4'),
        data=data_poisson,
        ssc = fixest.ssc(False, "none", False, "min", "min", False)
    )

    if not np.allclose((np.array(py_se)), (fixest.se(r_fixest)), rtol = 1e-3, atol = 1e-2):
        raise ValueError("py_se != r_se for CRV1 errors")
    if not np.allclose((np.array(py_pval)), (fixest.pvalue(r_fixest)), rtol = 1e-3, atol = 1e-2):
        raise ValueError("py_pval != r_pval for CRV1 errors")
    if not np.allclose(np.array(py_tstat), fixest.tstat(r_fixest), rtol = 1e-3, atol = 1e-2):
        raise ValueError("py_tstat != r_tstat for CRV1 errors")
