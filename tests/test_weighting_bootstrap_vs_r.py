import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

import pyfixest as pf
from pyfixest.estimation.post_estimation import weighting_bootstrap as wb

# Direct linear solves should agree to near machine precision.
LINEAR_RTOL = 1e-10
LINEAR_ATOL = 1e-10
# Iterative GLM solvers can stop at slightly different numerical tolerances.
GLM_RTOL = 1e-8
GLM_ATOL = 1e-8

pytestmark = pytest.mark.against_r_core


@pytest.fixture(scope="module")
def bootstrap_data():
    rng = np.random.default_rng(651)
    n_obs = 24
    g = np.repeat(np.arange(6), 4)
    x = rng.normal(size=n_obs)
    z1 = rng.normal(size=n_obs)
    z2 = rng.normal(size=n_obs)
    d = 0.4 * x + 0.9 * z1 - 0.6 * z2 + rng.normal(scale=0.4, size=n_obs)
    y = 0.7 * x + 1.2 * d + 0.3 * g + rng.normal(scale=0.5, size=n_obs)
    y_count = rng.poisson(np.exp(0.2 + 0.3 * x))
    return pd.DataFrame(
        {"y": y, "y_count": y_count, "x": x, "d": d, "z1": z1, "z2": z2, "g": g}
    )


@pytest.fixture(scope="module")
def prescribed_weights():
    return np.tile([2.0, 0.0, 1.0, 1.0], 6)


def _prescribe_weights(monkeypatch, weights):
    def draw_weights(**kwargs):
        return weights.copy()

    monkeypatch.setattr(wb, "_draw_bootstrap_weights", draw_weights)


def _r_data_with_weights(data, weights):
    active = weights > 0
    ro.globalenv["bootstrap_data"] = data.loc[active].reset_index(drop=True)
    ro.globalenv["bootstrap_weights"] = ro.FloatVector(weights[active])


def test_pairs_ols_fe_draw_matches_r_lm(
    bootstrap_data, prescribed_weights, monkeypatch
):
    _prescribe_weights(monkeypatch, prescribed_weights)
    fit = pf.feols("y ~ x | g", bootstrap_data, vcov="iid")
    _, draws = fit.pairs_bootstrap(2, return_draws=True)

    _r_data_with_weights(bootstrap_data, prescribed_weights)
    expected = np.asarray(
        ro.r(
            'unname(coef(lm(y ~ x + factor(g), data=bootstrap_data, weights=bootstrap_weights))["x"])'
        )
    )
    np.testing.assert_allclose(
        draws[:, 0], expected[0], rtol=LINEAR_RTOL, atol=LINEAR_ATOL
    )


def test_pairs_overidentified_iv_draw_matches_r_2sls(
    bootstrap_data, prescribed_weights, monkeypatch
):
    _prescribe_weights(monkeypatch, prescribed_weights)
    fit = pf.feols("y ~ x | d ~ z1 + z2", bootstrap_data, vcov="iid")
    _, draws = fit.pairs_bootstrap(2, return_draws=True)

    _r_data_with_weights(bootstrap_data, prescribed_weights)
    expected = np.asarray(
        ro.r(
            """
            X <- model.matrix(~ x + d, data=bootstrap_data)
            Z <- model.matrix(~ x + z1 + z2, data=bootstrap_data)
            sqrt_w <- sqrt(bootstrap_weights)
            Xw <- X * sqrt_w
            Zw <- Z * sqrt_w
            yw <- bootstrap_data$y * sqrt_w
            tXZ <- crossprod(Xw, Zw)
            tZZinv <- solve(crossprod(Zw))
            unname(solve(tXZ %*% tZZinv %*% t(tXZ),
                         tXZ %*% tZZinv %*% crossprod(Zw, yw)))
            """
        )
    )
    expected_draws = np.tile(expected.reshape(-1), (len(draws), 1))
    np.testing.assert_allclose(
        draws, expected_draws, rtol=LINEAR_RTOL, atol=LINEAR_ATOL
    )


def test_pairs_poisson_draw_matches_r_glm(
    bootstrap_data, prescribed_weights, monkeypatch
):
    _prescribe_weights(monkeypatch, prescribed_weights)
    fit = pf.feglm(
        "y_count ~ x",
        bootstrap_data,
        family="poisson",
        separation_check=[],
        accelerate=False,
        vcov="iid",
    )
    _, draws = fit.pairs_bootstrap(2, return_draws=True)

    _r_data_with_weights(bootstrap_data, prescribed_weights)
    expected = np.asarray(
        ro.r(
            "unname(coef(glm(y_count ~ x, data=bootstrap_data, "
            "weights=bootstrap_weights, family=poisson())))"
        )
    )
    expected_draws = np.tile(expected.reshape(-1), (len(draws), 1))
    np.testing.assert_allclose(draws, expected_draws, rtol=GLM_RTOL, atol=GLM_ATOL)
