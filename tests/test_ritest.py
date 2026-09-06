import matplotlib
import numpy as np
import pandas as pd
import pytest

import pyfixest as pf

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
    ritest_stats1 = fit._ritest_statistics.copy()

    res2 = fit.ritest(**kwargs2)
    ritest_stats2 = fit._ritest_statistics.copy()

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
        np.abs(fit1._ritest_pvalue - fit2._ritest_pvalue) < 0.03
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

    assert fit._ritest_statistics is not None
    assert np.allclose(fit.pvalue().xs("f3"), fit._ritest_pvalue, rtol=0.01, atol=0.01)


def test_fepois_slow_ritest_replays_estimation_contract(monkeypatch):
    """Poisson RI refits must retain offsets, formula context, and fit options."""
    import pyfixest.estimation as estimation

    data = pf.get_data(model="Fepois").dropna().iloc[:96].copy()
    data["treatment"] = (data["X2"] > data["X2"].median()).astype(int)
    data["exposure"] = np.random.default_rng(20260901).uniform(0.5, 3.0, len(data))

    def shift(values):
        return values + 0.125

    demeaner = pf.MapDemeaner(fixef_tol=1e-8, fixef_maxiter=20_000)
    ssc_config = pf.ssc(k_adj=False, G_adj=False)
    real_fepois = estimation.fepois
    fit = real_fepois(
        "Y ~ treatment + shift(X1) | f1",
        data=data,
        offset="log(exposure)",
        ssc=ssc_config,
        fixef_rm="none",
        iwls_tol=1e-10,
        iwls_maxiter=100,
        collin_tol=1e-8,
        separation_check=[],
        solver="np.linalg.lstsq",
        demeaner=demeaner,
        drop_intercept=True,
        context={"shift": shift},
    )

    refit_calls = []

    def recording_fepois(fml, *, data, vcov, **kwargs):
        refit_calls.append((fml, vcov, kwargs))
        return real_fepois(fml=fml, data=data, vcov=vcov, **kwargs)

    monkeypatch.setattr(estimation, "fepois", recording_fepois)
    fit.ritest(
        resampvar="treatment",
        reps=3,
        choose_algorithm="slow",
        rng=np.random.default_rng(1234),
    )

    assert len(refit_calls) == 3
    for fml, vcov, kwargs in refit_calls:
        assert fml == "Y ~ treatment_resampled + shift(X1) | f1"
        assert vcov == "iid"
        assert kwargs["weights"] is None
        assert kwargs["weights_type"] == "aweights"
        assert kwargs["ssc"] == ssc_config
        assert kwargs["fixef_rm"] == "none"
        assert kwargs["iwls_tol"] == 1e-10
        assert kwargs["iwls_maxiter"] == 100
        assert kwargs["collin_tol"] == 1e-8
        assert kwargs["separation_check"] == []
        assert kwargs["solver"] == "np.linalg.lstsq"
        assert kwargs["demeaner"] is demeaner
        assert kwargs["drop_intercept"] is True
        assert kwargs["offset"] == "log(exposure)"
        assert kwargs["context"]["shift"] is shift


def test_feglm_ritest_matches_ols_for_gaussian_family():
    """A Gaussian GLM is OLS, so both slow RI paths must resample identically."""
    data = pf.get_data().dropna()

    ols = pf.feols("Y ~ X1 + X2", data=data)
    ols.ritest(
        resampvar="X1",
        reps=25,
        rng=np.random.default_rng(9),
        choose_algorithm="slow",
        store_ritest_statistics=True,
    )

    glm = pf.feglm("Y ~ X1 + X2", data=data, family="gaussian")
    glm.ritest(
        resampvar="X1",
        reps=25,
        rng=np.random.default_rng(9),
        store_ritest_statistics=True,
    )

    np.testing.assert_allclose(
        ols._ritest_statistics, glm._ritest_statistics, rtol=1e-08, atol=1e-08
    )
    assert ols._ritest_pvalue == glm._ritest_pvalue


def test_feglm_ritest_is_centered_under_the_null():
    """Resampling a treatment with no effect must center the statistics at zero."""
    rng = np.random.default_rng(4242)
    N = 400
    data = pd.DataFrame(
        {
            "x": rng.normal(size=N),
            "d": rng.binomial(1, 0.5, size=N),
        }
    )
    data["y"] = rng.binomial(1, 1 / (1 + np.exp(-0.4 * data["x"])))

    fit = pf.feglm("y ~ d + x", data=data, family="logit")
    fit.ritest(
        resampvar="d",
        reps=100,
        rng=np.random.default_rng(11),
        store_ritest_statistics=True,
    )

    ri_stats = fit._ritest_statistics[1:]
    standard_error = np.std(ri_stats, ddof=1) / np.sqrt(len(ri_stats))
    assert abs(np.mean(ri_stats)) < 3 * standard_error
    assert 0.0 <= fit._ritest_pvalue <= 1.0


def test_feglm_slow_ritest_replays_estimation_contract(monkeypatch):
    """GLM refits must keep the family and the IRLS configuration of the fit."""
    import pyfixest.estimation as estimation

    rng = np.random.default_rng(20260906)
    N = 120
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "treatment": rng.binomial(1, 0.5, size=N),
            "f1": np.arange(N) % 4,
        }
    )
    data["y"] = rng.binomial(1, 0.5, size=N)

    ssc_config = pf.ssc(k_adj=False, G_adj=False)
    real_feglm = estimation.feglm
    fit = real_feglm(
        "y ~ treatment + x1 | f1",
        data=data,
        family="logit",
        ssc=ssc_config,
        iwls_tol=1e-10,
        iwls_maxiter=100,
        separation_check=[],
        accelerate=False,
    )
    refit_calls = []

    def recording_feglm(fml, *, data, vcov, **kwargs):
        refit_calls.append((fml, vcov, kwargs))
        return real_feglm(fml=fml, data=data, vcov=vcov, **kwargs)

    monkeypatch.setattr(estimation, "feglm", recording_feglm)
    fit.ritest(
        resampvar="treatment",
        reps=3,
        choose_algorithm="slow",
        rng=np.random.default_rng(1234),
    )

    assert len(refit_calls) == 3
    for fml, vcov, kwargs in refit_calls:
        assert fml == "y ~ treatment_resampled + x1 | f1"
        assert vcov == "iid"
        assert kwargs["family"] == "logit"
        assert kwargs["ssc"] == ssc_config
        assert kwargs["iwls_tol"] == 1e-10
        assert kwargs["iwls_maxiter"] == 100
        assert kwargs["separation_check"] == []
        assert kwargs["accelerate"] is False


@pytest.fixture
def data_r_vs_t():
    return pf.get_data(N=5000, seed=2999)
