import numpy as np
import pyhdfe

from pyfixest.estimation.demean_ import demean


def test_demean():
    rng = np.random.default_rng(929291)

    N = 10_000
    x = rng.normal(0, 1, 100 * N).reshape((N, 100))
    f1 = rng.choice(list(range(100)), N).reshape((N, 1))
    f2 = rng.choice(list(range(100)), N).reshape((N, 1))

    flist = np.concatenate((f1, f2), axis=1)

    # without weights
    weights = np.ones(N)
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, success = demean(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe, res_pyfixest)

    # with weights
    weights = rng.uniform(0, 1, N).reshape((N, 1))
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weights)
    res_pyfixest, success = demean(x, flist, weights.flatten(), tol=1e-10)
    assert np.allclose(res_pyhdfe, res_pyfixest)
