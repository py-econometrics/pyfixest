import pytest
import numpy as np
import pandas as pd
from pyfixest.api import feols

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
    np.random.seed(1234)
    N = 1000
    k = 4
    G = 25
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X[1] = np.random.choice(list(range(0, 5)), N, True)
    X[2] = np.random.choice(list(range(0, 10)), N, True)
    X[3] = np.random.choice(list(range(0, 10)), N, True)

    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    Y = pd.DataFrame(Y)
    Y.rename(columns = {0:'Y'}, inplace = True)
    X = pd.DataFrame(X)

    data = pd.concat([Y, X], axis = 1)
    data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4'}, inplace = True)
    data['X4'] = data['X4'].astype('category').astype(str)
    data['X3'] = data['X3'].astype('category').astype(str)
    data['X2'] = data['X2'].astype('category').astype(str)
    data['group_id'] = cluster.astype(str)
    data['Y2'] = data.Y + np.random.normal(0, 1, N)

    return data


def test_py_vs_r(data):

    '''
    test pyfixest against fixest via rpy2

        - for multiple models
        - and multiple inference types
        - ... compare regression coefficients and standard errors
        - tba: t-statistics, covariance matrices, other metrics
    '''


    fmls = ["Y~X1", "Y ~X1 + X2", "Y~ X1 | X2", "Y~ X1|X2+X3"]

    for fml in fmls:

        # iid errors
        py_fixest = feols(fml, 'iid', data)
        r_fixest = fixest.feols(ro.Formula(fml), se = 'iid',data=data)
        # only test coef here
        np.allclose(np.array(py_fixest[0].coef), stats.coef(r_fixest), rtol = 1e-10)
        np.allclose(np.array(py_fixest[0].se), fixest.se(r_fixest), rtol = 1e-10)

        # heteroskedastic errors
        py_fixest = feols(fml, 'hetero', data)
        r_fixest = fixest.feols(ro.Formula(fml), se = 'hetero',data=data)
        np.allclose(np.array(py_fixest[0].se), fixest.se(r_fixest), rtol = 1e-10)

        # cluster robust errors
        py_fixest = feols(fml, {'CRV1':'group_id'}, data)
        r_fixest = fixest.feols(ro.Formula(fml), cluster = ro.Formula('~group_id'), data=data)
        np.allclose(np.array(py_fixest[0].se), fixest.se(r_fixest), rtol = 1e-10)






