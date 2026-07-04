"""Invariant tests for the weights wiring.

These tests encode the array-domain conventions documented on the `Feols`
class:

- `_weights` always holds the user weights (ones if unweighted) and is never
  reassigned; GLMs keep their IRLS working weights in `_irls_weights`.
- `_Y` / `_X` are raw-domain (demeaned, unweighted); `_Y_wls` / `_X_wls` are
  the sqrt(weights)-scaled counterparts.
- Frequency weights are equivalent to physically duplicating rows, for both
  point estimates and standard errors.
"""

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf


def _make_data(n=400, seed=871, poisson=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "f1": rng.integers(0, 8, n),
            "cluster": rng.integers(0, 15, n),
            "aw": rng.uniform(0.5, 2.0, n),
            "fw": rng.integers(1, 4, n),
        }
    )
    eta = 0.5 + 0.8 * df.x1 - 0.4 * df.x2 + 0.1 * df.f1
    if poisson:
        df["y"] = rng.poisson(np.exp(0.3 * (eta - eta.mean())))
    else:
        df["y"] = eta + rng.normal(size=n)
    return df


def _expand(df):
    "Duplicate each row `fw` times: the physical counterpart of fweights."
    return df.loc[df.index.repeat(df["fw"])].reset_index(drop=True)


# ------------------------------------------------------------------
# fweights == expanded rows
# ------------------------------------------------------------------


@pytest.mark.parametrize("fml", ["y ~ x1 + x2", "y ~ x1 + x2 | f1"])
@pytest.mark.parametrize(
    "vcov", ["iid", "hetero", {"CRV1": "cluster"}], ids=["iid", "hetero", "crv1"]
)
def test_feols_fweights_match_expanded(fml, vcov):
    df = _make_data()
    fit_fw = pf.feols(
        fml,
        data=df,
        weights="fw",
        weights_type="fweights",
        vcov=vcov,
        fixef_rm="none",
    )
    fit_ex = pf.feols(fml, data=_expand(df), vcov=vcov, fixef_rm="none")
    np.testing.assert_allclose(fit_fw.coef(), fit_ex.coef(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fit_fw.se(), fit_ex.se(), rtol=1e-8, atol=1e-10)
    assert fit_fw._N == fit_ex._N


@pytest.mark.parametrize("vcov", ["HC2", "HC3"])
def test_feols_fweights_match_expanded_hc2_hc3(vcov):
    # HC2 / HC3 are only supported without fixed effects
    df = _make_data()
    fit_fw = pf.feols(
        "y ~ x1 + x2", data=df, weights="fw", weights_type="fweights", vcov=vcov
    )
    fit_ex = pf.feols("y ~ x1 + x2", data=_expand(df), vcov=vcov)
    np.testing.assert_allclose(fit_fw.coef(), fit_ex.coef(), rtol=1e-8)
    np.testing.assert_allclose(fit_fw.se(), fit_ex.se(), rtol=1e-8)


@pytest.mark.parametrize("fml", ["y ~ x1 + x2", "y ~ x1 + x2 | f1"])
@pytest.mark.parametrize(
    "vcov", ["iid", "hetero", {"CRV1": "cluster"}], ids=["iid", "hetero", "crv1"]
)
def test_fepois_fweights_match_expanded(fml, vcov):
    df = _make_data(poisson=True)
    fit_fw = pf.fepois(
        fml,
        data=df,
        weights="fw",
        weights_type="fweights",
        vcov=vcov,
        fixef_rm="none",
        iwls_tol=1e-10,
    )
    fit_ex = pf.fepois(
        fml, data=_expand(df), vcov=vcov, fixef_rm="none", iwls_tol=1e-10
    )
    np.testing.assert_allclose(fit_fw.coef(), fit_ex.coef(), rtol=1e-6)
    np.testing.assert_allclose(fit_fw.se(), fit_ex.se(), rtol=1e-6)
    assert fit_fw._N == fit_ex._N


@pytest.mark.parametrize("vcov", ["iid", "hetero"])
def test_iv_fweights_match_expanded(vcov):
    df = _make_data()
    rng = np.random.default_rng(3)
    df["z1"] = df["x1"] + rng.normal(size=len(df))
    fit_fw = pf.feols(
        "y ~ x2 | x1 ~ z1",
        data=df,
        weights="fw",
        weights_type="fweights",
        vcov=vcov,
    )
    fit_ex = pf.feols("y ~ x2 | x1 ~ z1", data=_expand(df), vcov=vcov)
    np.testing.assert_allclose(fit_fw.coef(), fit_ex.coef(), rtol=1e-8)
    np.testing.assert_allclose(fit_fw.se(), fit_ex.se(), rtol=1e-8)


def test_aweights_fweights_same_point_estimates():
    df = _make_data()
    fit_a = pf.feols(
        "y ~ x1 + x2 | f1",
        data=df,
        weights="fw",
        weights_type="aweights",
        fixef_rm="none",
    )
    fit_f = pf.feols(
        "y ~ x1 + x2 | f1",
        data=df,
        weights="fw",
        weights_type="fweights",
        fixef_rm="none",
    )
    np.testing.assert_allclose(fit_a.coef(), fit_f.coef(), rtol=1e-10)


# ------------------------------------------------------------------
# WLS against the closed form
# ------------------------------------------------------------------


def test_wls_closed_form():
    df = _make_data()
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    w = df.aw.to_numpy()
    y = df.y.to_numpy()
    n, k = X.shape

    bread = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = bread @ (X.T @ (w * y))
    u = y - X @ beta

    fit_iid = pf.feols("y ~ x1 + x2", data=df, weights="aw", vcov="iid")
    np.testing.assert_allclose(fit_iid.coef().to_numpy(), beta, rtol=1e-8)
    vcov_iid = bread * np.sum(w * u**2) / (n - k)
    np.testing.assert_allclose(
        fit_iid.se().to_numpy(), np.sqrt(np.diag(vcov_iid)), rtol=1e-8
    )

    fit_hc1 = pf.feols("y ~ x1 + x2", data=df, weights="aw", vcov="HC1")
    scores = X * (w * u)[:, None]
    vcov_hc1 = n / (n - k) * bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(
        fit_hc1.se().to_numpy(), np.sqrt(np.diag(vcov_hc1)), rtol=1e-8
    )


# ------------------------------------------------------------------
# domain invariants on the fitted object
# ------------------------------------------------------------------


def test_weights_invariant_and_domains_feols():
    df = _make_data()
    fit = pf.feols("y ~ x1 + x2 | f1", data=df, weights="aw", fixef_rm="none")
    # _weights is exactly the user weights column
    np.testing.assert_allclose(fit._weights.flatten(), df["aw"].to_numpy())
    # _X / _Y raw, _X_wls / _Y_wls sqrt(weights)-scaled
    w_sqrt = np.sqrt(fit._weights)
    np.testing.assert_allclose(fit._X_wls, w_sqrt * fit._X)
    np.testing.assert_allclose(fit._Y_wls, w_sqrt * fit._Y)
    # raw-domain accessors: y = yhat + resid on the original scale
    np.testing.assert_allclose(
        fit.predict() + fit.resid(), df["y"].to_numpy(), rtol=1e-8
    )


def test_weights_invariant_unweighted_aliases():
    df = _make_data()
    fit = pf.feols("y ~ x1 + x2 | f1", data=df, fixef_rm="none")
    assert np.all(fit._weights == 1.0)
    assert fit._X_wls is fit._X
    assert fit._Y_wls is fit._Y


def test_weights_invariant_after_glm_fit():
    df = _make_data(poisson=True)
    fit = pf.fepois("y ~ x1 + x2 | f1", data=df, weights="aw", fixef_rm="none")
    # user weights untouched by the IRLS loop
    np.testing.assert_allclose(fit._weights.flatten(), df["aw"].to_numpy())
    # IRLS working weights live in their own attribute and differ
    assert fit._irls_weights.shape == fit._weights.shape
    assert not np.allclose(fit._irls_weights, fit._weights)

    fit0 = pf.fepois("y ~ x1 + x2 | f1", data=df, fixef_rm="none")
    assert np.all(fit0._weights == 1.0)


# ------------------------------------------------------------------
# post-estimation under weights
# ------------------------------------------------------------------


def test_predict_newdata_parity_weighted_fe():
    df = _make_data()
    fit = pf.feols("y ~ x1 + x2 | f1", data=df, weights="aw", fixef_rm="none")
    np.testing.assert_allclose(
        fit.predict(newdata=df), fit.predict(), rtol=0, atol=1e-4
    )


@pytest.mark.parametrize("use_weights", [False, True], ids=["unweighted", "aweights"])
def test_predict_newdata_parity_fepois_fe(use_weights):
    df = _make_data(poisson=True)
    weights = "aw" if use_weights else None
    fit = pf.fepois("y ~ x1 + x2 | f1", data=df, weights=weights, fixef_rm="none")
    np.testing.assert_allclose(
        fit.predict(newdata=df, type="link"),
        fit.predict(type="link"),
        rtol=0,
        atol=1e-4,
    )


def test_update_raises_for_weighted_models():
    df = _make_data()
    fit = pf.feols("y ~ x1", data=df, weights="aw")
    with pytest.raises(NotImplementedError, match="weights"):
        fit.update(df[["x1"]].to_numpy()[:5], df["y"].to_numpy()[:5])


def test_iv_weighted_diagnostics_smoke():
    df = _make_data()
    rng = np.random.default_rng(3)
    df["z1"] = df["x1"] + rng.normal(size=len(df))
    fit = pf.feols("y ~ x2 | f1 | x1 ~ z1", data=df, weights="aw", fixef_rm="none")
    assert np.isfinite(fit.coef()).all()
    fit.IV_Diag()
    assert np.isfinite(fit._eff_F)


def test_lean_clears_wls_arrays():
    df = _make_data()
    fit = pf.feols("y ~ x1 | f1", data=df, weights="aw", lean=True)
    for attr in ["_X", "_Y", "_X_wls", "_Y_wls", "_Z_wls", "_weights"]:
        assert not hasattr(fit, attr), f"lean model still holds {attr}"
