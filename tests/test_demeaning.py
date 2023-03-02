import pytest
import pyhdfe
import numpy as np
from pyfixest.demean import demean

def test_demeaning_algo():

    np.random.seed(768512)
    N = 10000

    fixef_vars1 = np.random.choice([0, 1, 2, 3, 4, 5, 6], N).reshape(N, 1)
    fixef_vars2 = np.random.choice(list(range(0, 100)), N).reshape(N, 1)

    x = np.random.normal(0, 1, N).reshape(N, 1)
    y = np.random.normal(0, 1, N).reshape(N, 1)
    YX = np.concatenate([y,x], axis=1)


    # test 1
    # demeaning via pyfixest
    X_pyfixest = demean(x, fixef_vars1)

    # demeaning via pyhdfe
    algorithm = pyhdfe.create(ids=fixef_vars1, residualize_method='map')
    residualized = algorithm.residualize(YX)
    X_pyhdfe = residualized[:, 1:]

    np.allclose(X_pyfixest, X_pyhdfe, rtol = 1e-12)

    # test 2
    # demeaning via pyfixest
    X_pyfixest = demean(x, fixef_vars2)

    # demeaning via pyhdfe
    algorithm = pyhdfe.create(ids=fixef_vars2, residualize_method='map')
    residualized = algorithm.residualize(YX)
    X_pyhdfe = residualized[:, 1:]

    np.allclose(X_pyfixest, X_pyhdfe, rtol = 1e-12)

    # test 3
    # demeaning via pyfixest
    YX_pyfixest = demean(YX, fixef_vars2)

    # demeaning via pyhdfe
    algorithm = pyhdfe.create(ids=fixef_vars2, residualize_method='map')
    residualized = algorithm.residualize(YX)
    YX_pyhdfe = residualized

    np.allclose(YX_pyfixest, YX_pyhdfe, rtol = 1e-12)



    # test 3: 2wfe
    fixef_vars = np.concatenate([fixef_vars1, fixef_vars2], axis = 1)
    # demeaning via pyfixest
    X_pyfixest = demean(x, fixef_vars)

    # demeaning via pyhdfe
    algorithm = pyhdfe.create(ids=fixef_vars, residualize_method='map')
    YX = np.concatenate([y, x], axis=1)
    residualized = algorithm.residualize(YX)
    X_pyhdfe = residualized[:, 1:]

    np.allclose(X_pyfixest, X_pyhdfe, rtol = 1e-06)
