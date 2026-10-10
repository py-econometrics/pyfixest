from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.post_estimation.ritest import _resample

matplotlib.use("Agg")  # Use a non-interactive backend


def test_fast_ritest_requires_numba(monkeypatch):
    """Fast ritest must surface a clear ImportError when numba is missing."""
    import pyfixest.estimation.post_estimation.ritest as ritest

    monkeypatch.setattr(ritest, "nb", None)

    fit = pf.feols("Y ~ X1", data=pf.get_data(N=50))

    with pytest.raises(ImportError, match=r"pyfixest\[numba\]"):
        fit.ritest("X1", reps=10, choose_algorithm="fast")


@pytest.mark.extended
@pytest.mark.parametrize("fml", ["Y~X1+f3", "Y~X1+f3|f1", "Y~X1+f3|f1+f2"])
@pytest.mark.parametrize("resampvar", ["X1", "f3"])
@pytest.mark.parametrize("reps", [111, 212])
@pytest.mark.parametrize("cluster", [None, "group_id"])
def test_algos_internally(data, fml, resampvar, reps, cluster):
    fit = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    kwargs = {
        "resampvar": resampvar,
        "reps": reps,
        "type": "randomization-c",
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
    ritest_stats1 = fit.ritest_statistics.statistics.copy()

    res2 = fit.ritest(**kwargs2)
    ritest_stats2 = fit.ritest_statistics.statistics.copy()

    assert np.allclose(res1.Estimate, res2.Estimate, atol=1e-8, rtol=1e-8)
    assert np.allclose(res1["Pr(>|t|)"], res2["Pr(>|t|)"], atol=1e-8, rtol=1e-8)
    assert np.allclose(ritest_stats1, ritest_stats2, atol=1e-8, rtol=1e-8)


@pytest.mark.extended
@pytest.mark.parametrize("fml", ["Y~X1+f3", "Y~X1+f3|f1"])
@pytest.mark.parametrize("resampvar", ["X1"])
@pytest.mark.parametrize("cluster", [None, "group_id"])
def test_randomization_t_vs_c(fml, resampvar, cluster):
    data = pf.get_data(N=300)

    fit1 = pf.feols(fml, data=data)
    fit2 = pf.feols(fml, data=data)

    rng1 = np.random.default_rng(12354)
    rng2 = np.random.default_rng(12354)

    fit1.ritest(
        resampvar="X1",
        type="randomization-c",
        rng=rng1,
        cluster=cluster,
        store_ritest_statistics=True,
        reps=100,
    )
    fit2.ritest(
        resampvar="X1",
        type="randomization-t",
        rng=rng2,
        cluster=cluster,
        store_ritest_statistics=True,
        reps=100,
    )

    # just weak test that both are somewhat close
    assert (
        np.abs(fit1.ritest_statistics.pvalue - fit2.ritest_statistics.pvalue) < 0.03
        if cluster is None
        else 0.06
    ), (
        f"P-values are too different for randomization-c and randomization-t tests for {fml} and {resampvar} and {cluster}."
    )


@pytest.fixture
def ritest_results():
    # Load the CSV file into a pandas DataFrame
    file_path = "tests/data/ritest_results.csv"
    results_df = pd.read_csv(file_path)
    results_df.set_index(["formula", "resampvar", "cluster"], inplace=True)
    return results_df


@pytest.fixture
def data():
    return pf.get_data(N=1000, seed=2999)


@pytest.mark.extended
@pytest.mark.parametrize("fml", ["Y~X1+f3", "Y~X1+f3|f1", "Y~X1+f3|f1+f2"])
@pytest.mark.parametrize("resampvar", ["X1", "f3", "X1=-0.75", "f3>0.05"])
@pytest.mark.parametrize("cluster", [None, "group_id"])
def test_vs_r(data, fml, resampvar, cluster, ritest_results):
    fit = pf.feols(fml, data=data)
    reps = 4000

    rng1 = np.random.default_rng(1234)

    kwargs = {
        "resampvar": resampvar,
        "reps": reps,
        "type": "randomization-c",
        "cluster": cluster,
    }

    kwargs1 = kwargs.copy()

    kwargs1["choose_algorithm"] = "fast"
    kwargs1["rng"] = rng1

    res1 = fit.ritest(**kwargs1)

    if cluster is not None:
        pval = ritest_results.xs(
            (fml, resampvar, cluster), level=("formula", "resampvar", "cluster")
        )["pval"].to_numpy()
        se = ritest_results.xs(
            (fml, resampvar, cluster), level=("formula", "resampvar", "cluster")
        )["se"].to_numpy()
        ci_lower = ritest_results.xs(
            (fml, resampvar, cluster), level=("formula", "resampvar", "cluster")
        )["ci_lower"].to_numpy()
    else:
        pval = ritest_results.xs(
            (fml, resampvar, "none"), level=("formula", "resampvar", "cluster")
        )["pval"].to_numpy()
        se = ritest_results.xs(
            (fml, resampvar, "none"), level=("formula", "resampvar", "cluster")
        )["se"].to_numpy()
        ci_lower = ritest_results.xs(
            (fml, resampvar, "none"), level=("formula", "resampvar", "cluster")
        )["ci_lower"].to_numpy()

    assert np.allclose(res1["Pr(>|t|)"], pval, rtol=0.005, atol=0.005)
    assert np.allclose(res1["Std. Error (Pr(>|t|))"], se, rtol=0.005, atol=0.005)
    assert np.allclose(res1["2.5% (Pr(>|t|))"], ci_lower, rtol=0.005, atol=0.005)


@pytest.mark.extended
def test_fepois_ritest():
    data = pf.get_data(model="Fepois")
    fit = pf.fepois("Y ~ X1*f3", data=data)
    fit.ritest(resampvar="f3", reps=2000, store_ritest_statistics=True)

    assert fit.ritest_statistics.statistics is not None
    assert np.allclose(
        fit.pvalue().xs("f3"), fit.ritest_statistics.pvalue, rtol=0.01, atol=0.01
    )


@pytest.fixture
def data_r_vs_t():
    return pf.get_data(N=5000, seed=2999)


@pytest.mark.parametrize("statistic", ["randomization-c", "randomization-t"])
@pytest.mark.parametrize("estimator", [pf.feols, pf.fepois])
def test_ritest_preserves_treatment_names(estimator, statistic, monkeypatch):
    """No-intercept RI retains controls whose names contain the treatment name."""
    data = pf.get_data(N=300, model="Fepois").dropna()
    data["D"] = (data.X1 > 0).astype(float)
    data["D_control"] = data.X2
    # Neither a pre-existing resampled column nor a substring in a control
    # should be overwritten or substituted in the formula.
    data["D_resampled"] = data.X1
    fml = "Y ~ D + D_control + {D * D_control} - 1"
    fit = estimator(fml, data=data)
    original = fit._data.copy(deep=True)
    reps = 5
    rng = np.random.default_rng(31)
    expected = []
    for _ in range(reps):
        resampled = original.copy(deep=False)
        resampled["D"] = _resample(
            resampvar_arr=original.D.to_numpy(), rng=rng
        ).flatten()
        reference = estimator(
            fml,
            data=resampled,
            vcov="iid" if statistic == "randomization-c" else "hetero",
        )
        values = (
            reference.coef() if statistic == "randomization-c" else reference.tstat()
        )
        expected.append(values["D"])

    monkeypatch.setattr(
        Formula, "formula", property(lambda self: "descriptive text, not a formula")
    )
    assert fit.model.formula == fit._fml == "descriptive text, not a formula"
    monkeypatch.setattr(Formula, "parse", _fail_on_reparse)
    fit.ritest(
        "D",
        reps=reps,
        type=statistic,
        choose_algorithm="slow",
        rng=np.random.default_rng(31),
        store_ritest_statistics=True,
    )
    np.testing.assert_allclose(fit.ritest_statistics.statistics, expected, rtol=1e-10)
    pd.testing.assert_frame_equal(fit._data, original)


def _fail_on_reparse(*args, **kwargs):
    raise AssertionError("Randomization inference must reuse the parsed formula.")
