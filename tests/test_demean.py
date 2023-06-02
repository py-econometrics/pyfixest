import pytest
import pyhdfe
import numpy as np
from pyfixest.demean import demean

def test_demean():

    np.random.seed(12308)
    N = 1000
    x = np.random.normal(0, 1, 2*N).reshape((N,2))
    flist = np.random.choice(list(range(100)), N*3).reshape((N,3))
    weights = np.repeat(1, N)

    YX = pyhdfe.create(flist).residualize(x)
    res_pyfixest = demean(x, flist, weights)

    if not np.allclose(res_pyfixest, YX):
        raise ValueError("demean() does not match pyhdfe.create().residualize()")

