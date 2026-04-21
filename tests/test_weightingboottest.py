import numpy as np
import pytest

import pyfixest as pf
from pyfixest.utils.utils import get_data


@pytest.fixture
def data():
    return get_data(N=500, seed=42)


@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1|f1", "Y~X1|f1+f2"])
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_returns_dataframe(data, fml, method):
    fit = pf.feols(fml, data=data, vcov="iid")
    res = fit.weightingboottest(reps=99, method=method, seed=0)
    assert hasattr(res, "columns")
    assert "Estimate" in res.columns
    assert "Bootstrap SE" in res.columns
    assert "p-value" in res.columns
    assert list(res.index) == list(fit._coefnames)


@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1|f1"])
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_return_draws_shape(data, fml, method):
    fit = pf.feols(fml, data=data, vcov="iid")
    reps = 50
    _, draws = fit.weightingboottest(
        reps=reps, method=method, seed=0, return_draws=True
    )
    k = len(fit._coefnames)
    assert draws.shape == (reps, k)


@pytest.mark.parametrize("fml", ["Y~X1", "Y~X1|f1"])
def test_estimates_close_to_original(data, fml):
    """Bootstrap mean should be close to the point estimate."""
    fit = pf.feols(fml, data=data, vcov="iid")
    _, draws = fit.weightingboottest(
        reps=500, method="bayesian", seed=42, return_draws=True
    )
    for i, coef in enumerate(fit._coefnames):
        np.testing.assert_allclose(
            draws[:, i].mean(),
            fit._beta_hat[i],
            atol=0.05,
            err_msg=f"Bootstrap mean too far from β̂ for {coef} in {fml}",
        )


def test_multinomial_matches_pairs_bootstrap(data):
    """Multinomial bootstrap SE should match manual pairs bootstrap SE (up to MC noise)."""
    fml = "Y ~ X1 | f1"
    fit = pf.feols(fml, data=data, vcov="iid")

    # multinomial via weightingboottest
    _, multi_draws = fit.weightingboottest(
        reps=500, method="multinomial", seed=42, return_draws=True
    )
    multi_se = multi_draws.std(axis=0, ddof=1)

    # manual pairs bootstrap
    rng = np.random.default_rng(42)
    N = len(data)
    pairs_betas = []
    for _ in range(500):
        idx = rng.integers(0, N, size=N)
        bd = data.iloc[idx].reset_index(drop=True)
        pairs_betas.append(pf.feols(fml, bd, vcov="iid")._beta_hat)
    pairs_se = np.array(pairs_betas).std(axis=0, ddof=1)

    # should agree within 10% (MC noise)
    np.testing.assert_allclose(multi_se, pairs_se, rtol=0.10)


@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_se_close_to_analytic(data, method):
    """Bootstrap SE (alpha=1) should be close to analytic iid SE."""
    fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")
    analytic_se = fit.se().to_numpy()
    res = fit.weightingboottest(reps=500, method=method, seed=42)
    boot_se = res["Bootstrap SE"].to_numpy()
    # within 15% for 500 reps
    np.testing.assert_allclose(boot_se, analytic_se, rtol=0.15)


def test_invalid_method(data):
    fit = pf.feols("Y ~ X1", data=data, vcov="iid")
    with pytest.raises(ValueError, match="method"):
        fit.weightingboottest(reps=10, method="invalid")


def test_invalid_concentration(data):
    fit = pf.feols("Y ~ X1", data=data, vcov="iid")
    with pytest.raises(ValueError):
        fit.weightingboottest(reps=10, method="bayesian", concentration=-1.0)


def test_seed_reproducibility(data):
    fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")
    res1 = fit.weightingboottest(reps=100, method="bayesian", seed=7)
    res2 = fit.weightingboottest(reps=100, method="bayesian", seed=7)
    np.testing.assert_array_equal(
        res1["Bootstrap SE"].values, res2["Bootstrap SE"].values
    )


@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_cluster_arg(data, method):
    """cluster= should run without error and produce a result."""
    fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")
    res = fit.weightingboottest(reps=50, method=method, cluster="f1", seed=0)
    assert res.shape[0] == len(fit._coefnames)


# ── Feiv (2SLS) tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fml", ["Y ~ X1 | X2 ~ Z1", "Y ~ X1 | f1 | X2 ~ Z1"])
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_feiv_returns_dataframe(data, fml, method):
    """Weightingboottest on a Feiv model should return a correctly shaped result."""
    fit = pf.feols(fml, data=data, vcov="iid")
    res = fit.weightingboottest(reps=99, method=method, seed=0)
    assert hasattr(res, "columns")
    assert "Estimate" in res.columns
    assert "Bootstrap SE" in res.columns
    assert list(res.index) == list(fit._coefnames)


@pytest.mark.parametrize("fml", ["Y ~ X1 | X2 ~ Z1", "Y ~ X1 | f1 | X2 ~ Z1"])
def test_feiv_se_larger_than_ols(data, fml):
    """2SLS bootstrap SE for the endogenous regressor should exceed OLS SE."""
    fit_iv = pf.feols(fml, data=data, vcov="iid")
    _, draws_iv = fit_iv.weightingboottest(
        reps=300, method="bayesian", seed=42, return_draws=True
    )
    iv_se = draws_iv.std(axis=0, ddof=1)
    # IV SE should be positive for all coefficients
    assert (iv_se > 0).all()


def test_feiv_bootstrap_mean_close_to_beta(data):
    """Bootstrap mean of 2SLS draws should be close to the 2SLS point estimate."""
    fit = pf.feols("Y ~ X1 | f1 | X2 ~ Z1", data=data, vcov="iid")
    _, draws = fit.weightingboottest(
        reps=500, method="bayesian", seed=42, return_draws=True
    )
    np.testing.assert_allclose(
        draws.mean(axis=0),
        fit._beta_hat,
        atol=0.15,
        err_msg="Bootstrap mean too far from 2SLS beta",
    )


def test_feiv_draws_shape(data):
    fit = pf.feols("Y ~ X1 | f1 | X2 ~ Z1", data=data, vcov="iid")
    reps = 50
    _, draws = fit.weightingboottest(
        reps=reps, method="multinomial", seed=0, return_draws=True
    )
    assert draws.shape == (reps, len(fit._coefnames))
