import pytest
import pyhdfe
import numpy as np
from pyfixest.demean import demean
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

fixest = importr("fixest")



def test_demean():

    np.random.seed(12308)

    N = 1000
    x = np.random.normal(0, 1, 2*N).reshape((N,2))
    flist = np.random.choice(list(range(100)), N*3).reshape((N,3))
    ones = np.repeat(1, N)
    weights = np.random.uniform(0, 1, N)

    # test against pyfixest
    YX = pyhdfe.create(flist).residualize(x)
    res_pyfixest_no_weights = demean(x, flist, ones)
    res_pyfixest_weights = demean(x, flist, weights)

    # test against fixest - no weights
    fixest_no_weights = np.array([
        -1.2488219, -0.1544995,
        0.5237495, -1.5003717,
        -0.2752967, -1.2759202,
        -1.4087871,  0.5274697,
        -0.4568021, -0.3470608,
        0.4429120, -0.6501140
    ])
    fixest_no_weights = fixest_no_weights.reshape((6,2))

    fixest_weights = np.array([
        -1.5682946, -0.2490128,
        0.4673151, -1.1191956,
        -0.4905845, -1.1513425,
        -1.6173110,  0.5508082,
        -0.1991782,  0.1129856,
        0.4853932, -0.9554187
    ])

    fixest_weights = fixest_weights.reshape((6,2))

    if not np.allclose(res_pyfixest_no_weights, YX):
        raise ValueError("demean() does not match pyhdfe.create().residualize()")

    if not np.allclose(res_pyfixest_no_weights[0:6], fixest_no_weights):
        raise ValueError("demean() does not match fixest (no weights)")

    if not np.allclose(res_pyfixest_weights[0:6], fixest_weights):
        raise ValueError("demean() does not match fixest (weights)")


