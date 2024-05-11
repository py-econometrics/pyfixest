import numpy as np
import pytest

import pyfixest as pf


@pytest.fixture
def data():
    return pf.get_data(N=1000)


@pytest.mark.parametrize(
    "fml", ["Y ~ X1 + f3", "Y ~ X1 + f3 | f1", "Y ~ X1 + f3 | f1 + f2"]
)
@pytest.mark.parametrize("resampvar", ["X1", "f3"])
@pytest.mark.parametrize("reps", [111, 212])
@pytest.mark.parametrize("algo_iterations", [None, 10])
def test_algos_internally(data, fml, resampvar, reps, algo_iterations):
    fit = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    res1 = fit.ritest(
        resampvar=resampvar,
        reps=reps,
        rng=rng1,
        type="randomization-c",
        choose_algorithm="slow",
        algo_iterations=algo_iterations,
        store_ritest_statistics=True,
    )
    ritest_stats1 = fit.ritest_statistics.copy()

    res2 = fit.ritest(
        resampvar=resampvar,
        reps=reps,
        rng=rng2,
        type="randomization-c",
        choose_algorithm="fast",
        algo_iterations=algo_iterations,
        store_ritest_statistics=True,
    )
    ritest_stats2 = fit.ritest_statistics.copy()

    assert np.allclose(res1.Estimate, res2.Estimate, atol=1e-3, rtol=1e-3)
    assert np.allclose(res1["Pr(>|t|)"], res2["Pr(>|t|)"], atol=1e-3, rtol=1e-3)
    assert np.allclose(ritest_stats1, ritest_stats2, atol=1e-3, rtol=1e-3)
