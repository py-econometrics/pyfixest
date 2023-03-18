import pytest
import numpy as np
import pandas as pd
from pyfixest.fixest import Fixest

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector, FloatVector
pandas2ri.activate()

fixest = importr("fixest")
stats = importr('stats')


@pytest.fixture
def data():

    # create data
    np.random.seed(1123487)
    N = 10000
    k = 5
    G = 25
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X[1] = np.random.choice(list(range(0, 5)), N, True)
    X[2] = np.random.choice(list(range(0, 10)), N, True)
    X[3] = np.random.choice(list(range(0, 10)), N, True)
    X[4] = np.random.normal(0, 1, N)

    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    Y = pd.DataFrame(Y)
    Y.rename(columns = {0:'Y'}, inplace = True)
    X = pd.DataFrame(X)

    data = pd.concat([Y, X], axis = 1)
    data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4', 4:'X5'}, inplace = True)

    data['group_id'] = cluster
    data['Y2'] = data.Y + np.random.normal(0, 1, N)

    data['Y'][0] = np.nan
    data['X1'][1] = np.nan
    #data['X2'][2] = np.nan




    return data


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

    #("Y ~ C(X2)"),
    #("Y ~ X1 + C(X2)"),

    #("Y ~ X1:C(X2) | X3"),
    #("Y ~ C(X2):C(X3) | X4"),




    #("Y ~ i(X1,X2)"),
    #("Y ~ i(X1,X2) | X3"),
    #("Y ~ i(X1,X2) | X3 + X4"),


])


def test_py_vs_r(data, fml):

    '''
    test pyfixest against fixest via rpy2

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    '''

    fixest.setFixest_ssc(fixest.ssc(True, "None", True, "min", "min", False))

    # iid errors
    pyfixest = Fixest(data = data).feols(fml, vcov = 'iid')
    py_coef = pyfixest.tidy()['Estimate']
    py_se = pyfixest.tidy()['Std. Error']
    r_fixest = fixest.feols(ro.Formula(fml), se = 'iid', data=data)

    np.array_equal((np.array(py_coef)), (stats.coef(r_fixest)))
    np.array_equal((np.array(py_se)), (fixest.se(r_fixest)))

    # heteroskedastic errors
    py_se = pyfixest.vcov("HC1").tidy()['Std. Error']
    r_fixest = fixest.feols(ro.Formula(fml), se = 'hetero',data=data)

    np.array_equal((np.array(py_se)), (fixest.se(r_fixest)))

    # cluster robust errors
    py_se = pyfixest.vcov({'CRV1':'group_id'}).tidy()['Std. Error']
    r_fixest = fixest.feols(ro.Formula(fml), cluster = ro.Formula('~group_id'), data=data)

    np.array_equal((np.array(py_se)), (fixest.se(r_fixest)))


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

    fixest.setFixest_ssc(fixest.ssc(True, "None", True, "min", "min", False))

    r_fml = _py_fml_to_r_fml(fml_multi)

    pyfixest = Fixest(data = data).feols(fml_multi, vcov = 'iid')
    py_coef = pyfixest.tidy()['Estimate']
    py_se = pyfixest.tidy()['Std. Error']
    r_fixest = fixest.feols(ro.Formula(r_fml), se = 'iid', data=data)

    for x, val in enumerate(r_fixest):

        i = pyfixest.tidy().index.unique()[x]
        ix = pyfixest.tidy().xs(i)
        py_coef = ix['Estimate']
        py_se = ix['Std. Error']

        fixest_object = r_fixest.rx2(x+1)
        fixest_coef = fixest_object.rx2("coefficients")
        fixest_se = fixest_object.rx2("se")

        np.array_equal((np.array(py_coef)), fixest_coef)
        np.array_equal((np.array(py_se)), (fixest.se(r_fixest)))


def _py_fml_to_r_fml(py_fml):

    '''
    pyfixest multiple estimation fml syntax to fixest multiple depvar
    syntax converter,
    i.e. 'Y1 + X2 ~ X' -> 'c(Y1, Y2) ~ X'
    '''

    fml_split = py_fml.split("~")
    depvars = fml_split[0]
    covars = fml_split[1]
    depvars = "c(" +  ",".join(depvars.split("+")) + ")"
    return depvars + "~" + covars







