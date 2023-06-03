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
    weights = np.repeat(1, N)

    # test against pyfixest
    YX = pyhdfe.create(flist).residualize(x)
    res_pyfixest = demean(x, flist, weights)

    if not np.allclose(res_pyfixest, YX):
        raise ValueError("demean() does not match pyhdfe.create().residualize()")


    # test against fixest
    #x = ro.r.data(x, nrow = N, ncol = 2)
    #flist = ro.r.matrix(flist, nrow = N, ncol = 3)
    #weights = ro.r.matrix(weights, nrow = N, ncol = 1)
    #res_fixest = fixest.demean(x, flist, weights)

    #if not np.allclose(res_fixest, res_pyfixest):
    #    raise ValueError("demean() does not match fixest.demean()")


    # now with weights
    weights = np.random.uniform(0, 1, N)
    res_pyfixest = demean(x, flist, weights)

    #weights = ro.r.matrix(weights, nrow = N, ncol = 1)
    #res_fixest = fixest.demean(x, flist, weights)
    #if not np.allclose(res_fixest, res_pyfixest):
    #    raise ValueError("demean() does not match fixest.demean() with weights")

