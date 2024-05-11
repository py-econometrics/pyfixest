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
@pytest.mark.parametrize("cluster", [None, "group_id"])
def test_algos_internally(data, fml, resampvar, reps, algo_iterations, cluster):
    fit = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    kwargs = {
        "resampvar": resampvar,
        "reps": reps,
        "type": "randomization-c",
        "algo_iterations": algo_iterations,
        "store_ritest_statistics": True,
        "cluster": cluster,
    }

    kwargs1 = kwargs.copy()
    kwargs2 = kwargs.copy()

    kwargs1["choose_algorithm"] = "slow"
    kwargs1["rng"] = rng1
    kwargs2["choose_algorithm"] = "fast"
    kwargs2["rng"] = rng2

    res1 = fit.ritest(**kwargs1)
    ritest_stats1 = fit.ritest_statistics.copy()

    res2 = fit.ritest(**kwargs2)
    ritest_stats2 = fit.ritest_statistics.copy()

    assert np.allclose(res1.Estimate, res2.Estimate, atol=1e-3, rtol=1e-3)
    assert np.allclose(res1["Pr(>|t|)"], res2["Pr(>|t|)"], atol=1e-3, rtol=1e-3)
    assert np.allclose(ritest_stats1, ritest_stats2, atol=1e-3, rtol=1e-3)
