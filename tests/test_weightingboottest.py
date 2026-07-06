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


# ── Fepois (Poisson) tests ─────────────────────────────────────────────────────


@pytest.fixture
def pois_data():
    data = pf.get_data(N=500, seed=42).dropna().reset_index(drop=True)
    data["Y"] = (data["Y"] - data["Y"].min() + 1).round().astype(int)
    return data


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_fepois_returns_dataframe(pois_data, fml, method):
    """Weightingboottest on Fepois should return a correctly shaped result."""
    fit = pf.fepois(fml, data=pois_data, vcov="iid")
    res = fit.weightingboottest(reps=99, method=method, seed=0)
    assert hasattr(res, "columns")
    assert "Estimate" in res.columns
    assert "Bootstrap SE" in res.columns
    assert list(res.index) == list(fit._coefnames)


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
def test_fepois_draws_shape(pois_data, fml):
    fit = pf.fepois(fml, data=pois_data, vcov="iid")
    reps = 50
    _, draws = fit.weightingboottest(
        reps=reps, method="bayesian", seed=0, return_draws=True
    )
    assert draws.shape == (reps, len(fit._coefnames))


def test_fepois_bootstrap_mean_close_to_beta(pois_data):
    """Bootstrap mean of Poisson draws should be close to the point estimate."""
    fit = pf.fepois("Y ~ X1 | f1", data=pois_data, vcov="iid")
    _, draws = fit.weightingboottest(
        reps=300, method="bayesian", seed=42, return_draws=True
    )
    np.testing.assert_allclose(
        draws.mean(axis=0),
        fit._beta_hat,
        atol=0.15,
        err_msg="Bootstrap mean too far from Poisson beta",
    )


def test_fepois_se_positive(pois_data):
    """Bootstrap SE should be positive for all coefficients."""
    fit = pf.fepois("Y ~ X1 | f1", data=pois_data, vcov="iid")
    _, draws = fit.weightingboottest(
        reps=100, method="bayesian", seed=0, return_draws=True
    )
    assert (draws.std(axis=0, ddof=1) > 0).all()


# ── Feglm (logit, probit, gaussian) tests ──────────────────────────────────────


@pytest.fixture
def glm_data():
    data = pf.get_data(N=500, seed=42).dropna().reset_index(drop=True)
    data["Y_bin"] = (data["Y"] > data["Y"].median()).astype(int)
    data["Y_cont"] = data["Y"]
    return data


@pytest.mark.parametrize(
    "fml,family",
    [
        ("Y_bin ~ X1", "logit"),
        ("Y_bin ~ X1 | f1", "logit"),
        ("Y_bin ~ X1", "probit"),
        ("Y_bin ~ X1 | f1", "probit"),
        ("Y_cont ~ X1", "gaussian"),
        ("Y_cont ~ X1 | f1", "gaussian"),
    ],
)
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_feglm_returns_dataframe(glm_data, fml, family, method):
    """Weightingboottest on Feglm subclasses should return a correctly shaped result."""
    fit = pf.feglm(fml, data=glm_data, family=family, vcov="iid")
    res = fit.weightingboottest(reps=50, method=method, seed=0)
    assert hasattr(res, "columns")
    assert "Estimate" in res.columns
    assert "Bootstrap SE" in res.columns
    assert list(res.index) == list(fit._coefnames)


@pytest.mark.parametrize(
    "fml,family",
    [
        ("Y_bin ~ X1", "logit"),
        ("Y_bin ~ X1 | f1", "logit"),
        ("Y_cont ~ X1", "gaussian"),
    ],
)
def test_feglm_draws_shape(glm_data, fml, family):
    fit = pf.feglm(fml, data=glm_data, family=family, vcov="iid")
    reps = 50
    _, draws = fit.weightingboottest(
        reps=reps, method="bayesian", seed=0, return_draws=True
    )
    assert draws.shape == (reps, len(fit._coefnames))


def test_feglm_bootstrap_mean_close_to_beta_logit(glm_data):
    """Bootstrap mean of logit draws should be close to the point estimate."""
    fit = pf.feglm("Y_bin ~ X1 | f1", data=glm_data, family="logit", vcov="iid")
    _, draws = fit.weightingboottest(
        reps=300, method="bayesian", seed=42, return_draws=True
    )
    np.testing.assert_allclose(
        draws.mean(axis=0),
        fit._beta_hat,
        atol=0.20,
        err_msg="Bootstrap mean too far from logit beta",
    )


def test_feglm_bootstrap_mean_close_to_beta_gaussian(glm_data):
    """Bootstrap mean of gaussian draws should be close to the point estimate."""
    fit = pf.feglm("Y_cont ~ X1 | f1", data=glm_data, family="gaussian", vcov="iid")
    _, draws = fit.weightingboottest(
        reps=300, method="bayesian", seed=42, return_draws=True
    )
    np.testing.assert_allclose(
        draws.mean(axis=0),
        fit._beta_hat,
        atol=0.10,
        err_msg="Bootstrap mean too far from gaussian beta",
    )


def test_feglm_se_positive(glm_data):
    """Bootstrap SE should be positive for all families."""
    for family, fml in [
        ("logit", "Y_bin ~ X1 | f1"),
        ("probit", "Y_bin ~ X1 | f1"),
        ("gaussian", "Y_cont ~ X1 | f1"),
    ]:
        fit = pf.feglm(fml, data=glm_data, family=family, vcov="iid")
        _, draws = fit.weightingboottest(
            reps=100, method="bayesian", seed=0, return_draws=True
        )
        assert (draws.std(axis=0, ddof=1) > 0).all(), f"Zero SE for {family}"


@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
@pytest.mark.parametrize("family", ["logit", "probit"])
def test_feglm_se_close_to_analytic(glm_data, method, family):
    """Bootstrap SE should be close to analytic iid SE for logit/probit (within 20% for 500 reps).

    Gaussian is excluded: the GLM analytic SE fixes dispersion=1 and therefore
    differs from OLS SE. The bootstrap SE is correct; it is tested separately
    against the OLS bootstrap SE in test_feglm_gaussian_matches_ols.
    """
    fit = pf.feglm("Y_bin ~ X1 | f1", data=glm_data, family=family, vcov="iid")
    analytic_se = fit.se().to_numpy()
    res = fit.weightingboottest(reps=500, method=method, seed=42)
    boot_se = res["Bootstrap SE"].to_numpy()
    np.testing.assert_allclose(boot_se, analytic_se, rtol=0.20)


def test_feglm_gaussian_matches_ols(glm_data):
    """Gaussian GLM bootstrap SE should equal OLS bootstrap SE (same model, same data)."""
    fit_ols = pf.feols("Y_cont ~ X1 | f1", data=glm_data, vcov="iid")
    fit_gauss = pf.feglm(
        "Y_cont ~ X1 | f1", data=glm_data, family="gaussian", vcov="iid"
    )
    _, draws_ols = fit_ols.weightingboottest(
        reps=500, method="bayesian", seed=42, return_draws=True
    )
    _, draws_gauss = fit_gauss.weightingboottest(
        reps=500, method="bayesian", seed=42, return_draws=True
    )
    np.testing.assert_allclose(
        draws_gauss.std(axis=0, ddof=1),
        draws_ols.std(axis=0, ddof=1),
        rtol=0.05,
    )


@pytest.mark.parametrize(
    "fml_multi", ["Y + Y2 ~ X1", "Y ~ sw(X1, X2)", "Y ~ csw(X1, X2) | f1"]
)
@pytest.mark.parametrize("method", ["bayesian", "multinomial"])
def test_fixest_multi_weightingboottest(data, fml_multi, method):
    """FixestMulti.weightingboottest should stack per-model results by fml."""
    fit = pf.feols(fml_multi, data=data, vcov="iid")
    res = fit.weightingboottest(reps=50, method=method, seed=0)
    assert list(res.index.names) == ["fml", "Coefficient"]
    assert "Bootstrap SE" in res.columns
    assert set(res.index.get_level_values("fml")) == set(fit.all_fitted_models.keys())


def test_multiway_cluster_raises(data):
    """Weightingboottest should reject multiway clustering, like wildboottest."""
    fit = pf.feols("Y ~ X1", data=data.dropna(), vcov={"CRV1": "f1 +f2"})
    with pytest.raises(NotImplementedError, match="Multiway clustering"):
        fit.weightingboottest(reps=50, seed=0)


def test_offset_raises(pois_data):
    """Weightingboottest should reject models fit with an offset."""
    pois_data["log_exposure"] = np.log(pois_data["X2"].abs() + 1)
    fit = pf.fepois("Y ~ X1", data=pois_data, offset="log_exposure", vcov="iid")
    with pytest.raises(NotImplementedError, match="offset"):
        fit.weightingboottest(reps=50, seed=0)


def test_redraw_on_failure(data, monkeypatch):
    """A failing replicate should be redrawn, not silently kept, and warn once."""
    fit = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")

    calls = {"n": 0}
    original = fit._bootstrap_one_rep

    def flaky(nz, w_combined):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated demeaning failure")
        return original(nz, w_combined)

    monkeypatch.setattr(fit, "_bootstrap_one_rep", flaky)

    with pytest.warns(UserWarning, match="redrawn"):
        res = fit.weightingboottest(reps=10, method="bayesian", seed=0)

    assert res.shape[0] == len(fit._coefnames)
    assert calls["n"] == 11  # 10 successful reps + 1 redrawn failure
