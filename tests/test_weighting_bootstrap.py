import warnings
from inspect import signature

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.post_estimation import weighting_bootstrap as wb

# Dense dummy-variable fits and pyfixest refits solve the same objectives through
# different paths; these tolerances only accommodate floating-point solve order.
RTOL = 1e-7
ATOL = 1e-9


@pytest.fixture(scope="module")
def linear_data():
    rng = np.random.default_rng(421)
    n_groups = 8
    observations_per_group = 8
    g = np.repeat(np.arange(n_groups), observations_per_group)
    n_obs = len(g)
    x = rng.normal(size=n_obs)
    z1 = rng.normal(size=n_obs)
    z2 = rng.normal(size=n_obs)
    d = 0.8 * z1 + 0.5 * z2 + 0.3 * x + rng.normal(size=n_obs)
    group_effect = rng.normal(size=n_groups)[g]
    y = 0.7 * x + 1.1 * d + group_effect + rng.normal(size=n_obs)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "d": d,
            "z1": z1,
            "z2": z2,
            "g": g,
            "h": np.tile(np.arange(4), n_obs // 4),
            "aw": 0.5 + rng.random(n_obs),
        }
    )


@pytest.fixture(scope="module")
def glm_data():
    rng = np.random.default_rng(811)
    n_groups = 8
    observations_per_group = 15
    g = np.repeat(np.arange(n_groups), observations_per_group)
    n_obs = len(g)
    x = rng.normal(size=n_obs)
    group_effect = rng.normal(scale=0.25, size=n_groups)[g]
    probability = 1 / (1 + np.exp(-(0.4 * x + group_effect)))
    y_binary = rng.binomial(1, probability)
    y_count = rng.poisson(np.exp(0.2 + 0.25 * x + group_effect))
    y_gaussian = 0.6 * x + group_effect + rng.normal(size=n_obs)

    # Prevent full-sample FE separation while retaining stochastic outcomes.
    for group in range(n_groups):
        rows = np.flatnonzero(g == group)
        y_binary[rows[:2]] = [0, 1]
        y_count[rows[0]] = 1

    return pd.DataFrame(
        {
            "y_binary": y_binary,
            "y_count": y_count,
            "y_gaussian": y_gaussian,
            "x": x,
            "g": g,
            "aw": 0.4 + rng.random(n_obs),
        }
    )


def _draw_weights(
    *,
    n_obs: int,
    reps: int,
    seed: int,
    weight_distribution: str,
    alpha: float,
    cluster: np.ndarray | None = None,
):
    rng = np.random.default_rng(seed)
    if cluster is None:
        codes = np.arange(n_obs)
        n_units = n_obs
    else:
        codes = wb._factorize_bootstrap_cluster(cluster, n_obs)
        n_units = int(codes.max()) + 1
    return [
        wb._draw_bootstrap_weights(
            rng=rng,
            weight_distribution=weight_distribution,
            n_units=n_units,
            alpha=alpha,
        )[codes]
        for _ in range(reps)
    ]


def _bootstrap_with_draws(
    fit,
    *,
    reps,
    weight_distribution,
    alpha,
    cluster=None,
    seed=None,
):
    if weight_distribution == "dirichlet":
        return fit.bayesian_bootstrap(
            reps,
            alpha=alpha,
            cluster=cluster,
            seed=seed,
            return_draws=True,
        )
    return fit.pairs_bootstrap(
        reps,
        cluster=cluster,
        seed=seed,
        return_draws=True,
    )


def _dense_ols(data, weights, has_fixef):
    if has_fixef:
        fixed_effects = pd.get_dummies(data["g"], dtype=float).to_numpy()
        design = np.column_stack([data["x"], fixed_effects])
        n_reported = 1
    else:
        design = np.column_stack([np.ones(len(data)), data["x"]])
        n_reported = 2
    sqrt_weights = np.sqrt(weights)
    beta = np.linalg.lstsq(
        design * sqrt_weights[:, None],
        data["y"].to_numpy() * sqrt_weights,
        rcond=None,
    )[0]
    return beta[:n_reported]


def _dense_iv(data, weights, has_fixef):
    if has_fixef:
        fixed_effects = pd.get_dummies(data["g"], dtype=float).to_numpy()
        X = np.column_stack([data[["x", "d"]], fixed_effects])
        Z = np.column_stack([data[["x", "z1", "z2"]], fixed_effects])
        n_reported = 2
    else:
        X = np.column_stack([np.ones(len(data)), data[["x", "d"]]])
        Z = np.column_stack([np.ones(len(data)), data[["x", "z1", "z2"]]])
        n_reported = 3
    sqrt_weights = np.sqrt(weights)
    Xw = X * sqrt_weights[:, None]
    Zw = Z * sqrt_weights[:, None]
    Yw = data["y"].to_numpy() * sqrt_weights
    tZZinv = np.linalg.inv(Zw.T @ Zw)
    beta = np.linalg.solve(
        Xw.T @ Zw @ tZZinv @ Zw.T @ Xw,
        Xw.T @ Zw @ tZZinv @ Zw.T @ Yw,
    )
    return beta[:n_reported]


@pytest.mark.parametrize(
    "has_fixef,use_aweights,cluster,weight_distribution",
    [
        (False, False, None, "dirichlet"),
        (True, True, None, "dirichlet"),
        (False, False, None, "multinomial"),
        (True, True, None, "multinomial"),
        (False, True, "g", "multinomial"),
        (True, False, "g", "multinomial"),
    ],
)
def test_ols_draws_match_dense_weighted_fit(
    linear_data, has_fixef, use_aweights, cluster, weight_distribution
):
    formula = "y ~ x | g" if has_fixef else "y ~ x"
    weights_name = "aw" if use_aweights else None
    fit = pf.feols(formula, linear_data, weights=weights_name, vcov="iid")
    reps = 4
    seed = 19
    alpha = 1.4
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        alpha=alpha,
        cluster=cluster,
        seed=seed,
    )
    cluster_values = linear_data[cluster].to_numpy() if cluster else None
    bootstrap_weights = _draw_weights(
        n_obs=len(linear_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        alpha=alpha,
        cluster=cluster_values,
    )
    original_weights = linear_data["aw"].to_numpy() if use_aweights else 1
    expected = np.vstack(
        [
            _dense_ols(linear_data, original_weights * draw, has_fixef)
            for draw in bootstrap_weights
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("has_fixef,use_aweights", [(False, False), (True, True)])
@pytest.mark.parametrize("weight_distribution", ["dirichlet", "multinomial"])
def test_overidentified_iv_draws_match_dense_2sls(
    linear_data, has_fixef, use_aweights, weight_distribution
):
    formula = "y ~ x | d ~ z1 + z2"
    if has_fixef:
        formula = "y ~ x | g | d ~ z1 + z2"
    weights_name = "aw" if use_aweights else None
    fit = pf.feols(formula, linear_data, weights=weights_name, vcov="iid")
    reps = 3
    seed = 27
    alpha = 0.8
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        alpha=alpha,
        seed=seed,
    )
    bootstrap_weights = _draw_weights(
        n_obs=len(linear_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        alpha=alpha,
    )
    original_weights = linear_data["aw"].to_numpy() if use_aweights else 1
    expected = np.vstack(
        [
            _dense_iv(linear_data, original_weights * draw, has_fixef)
            for draw in bootstrap_weights
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "outcome,family,has_fixef,use_aweights",
    [
        ("y_count", "poisson", False, False),
        ("y_count", "poisson", True, True),
        ("y_binary", "logit", False, True),
        ("y_binary", "logit", True, False),
        ("y_binary", "probit", True, True),
    ],
)
@pytest.mark.parametrize("weight_distribution", ["dirichlet", "multinomial"])
def test_glm_draws_match_public_weighted_refits(
    glm_data, outcome, family, has_fixef, use_aweights, weight_distribution
):
    formula = f"{outcome} ~ x" + (" | g" if has_fixef else "")
    weights_name = "aw" if use_aweights else None
    fit = pf.feglm(
        formula,
        glm_data,
        family=family,
        weights=weights_name,
        separation_check=[],
        accelerate=False,
        vcov="iid",
    )
    reps = 2
    seed = 44
    alpha = 1.3
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        alpha=alpha,
        seed=seed,
    )
    bootstrap_weights = _draw_weights(
        n_obs=len(glm_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        alpha=alpha,
    )
    original_weights = glm_data["aw"].to_numpy() if use_aweights else 1
    expected = []
    for draw in bootstrap_weights:
        combined_weights = original_weights * draw
        active = combined_weights > 0
        ref_data = glm_data.loc[active].assign(_ref_weight=combined_weights[active])
        ref = pf.feglm(
            formula,
            ref_data,
            family=family,
            weights="_ref_weight",
            separation_check=[],
            accelerate=False,
            vcov="iid",
        )
        expected.append(ref._beta_hat)
    np.testing.assert_allclose(draws, np.vstack(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("weight_distribution", ["dirichlet", "multinomial"])
def test_gaussian_glm_draws_match_weighted_feols_refits(glm_data, weight_distribution):
    formula = "y_gaussian ~ x | g"
    fit = pf.feglm(
        formula,
        glm_data,
        family="gaussian",
        weights="aw",
        separation_check=[],
        accelerate=False,
        vcov="iid",
    )
    reps = 2
    seed = 61
    alpha = 1.2
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        alpha=alpha,
        seed=seed,
    )
    bootstrap_weights = _draw_weights(
        n_obs=len(glm_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        alpha=alpha,
    )
    expected = []
    for draw in bootstrap_weights:
        combined_weights = glm_data["aw"].to_numpy() * draw
        active = combined_weights > 0
        ref_data = glm_data.loc[active].assign(_ref_weight=combined_weights[active])
        expected.append(
            pf.feols(formula, ref_data, weights="_ref_weight", vcov="iid")._beta_hat
        )
    np.testing.assert_allclose(draws, np.vstack(expected), rtol=RTOL, atol=ATOL)


def test_initial_poisson_separation_aligns_bootstrap_inputs():
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 1, 2, 1, 1, 3, 2],
            "x": [-1.0, 0.0, 1.0] * 3,
            "g": np.repeat([0, 1, 2], 3),
            "aw": np.linspace(0.5, 1.5, 9),
        }
    )
    with pytest.warns(UserWarning, match="removed because of separation"):
        fit = pf.feglm(
            "y ~ x | g",
            data,
            family="poisson",
            weights="aw",
            separation_check=["fe"],
            accelerate=False,
        )
    _, draws = fit.bayesian_bootstrap(2, seed=3, return_draws=True)
    assert fit.n_separation_na == 3
    assert len(fit._user_weights) == len(fit._Y_untransformed) == 6
    assert draws.shape == (2, 1)


def test_bayesian_weighted_mean_has_beta_posterior_moments():
    rng = np.random.default_rng(918)
    reps = 100_000
    alpha = 2.0
    draws = np.array(
        [
            wb._draw_bootstrap_weights(
                rng=rng,
                weight_distribution="dirichlet",
                n_units=2,
                alpha=alpha,
            )[1]
            / 2
            for _ in range(reps)
        ]
    )
    expected_mean = 0.5
    expected_variance = 1 / (4 * (2 * alpha + 1))
    mean_mc_se = np.sqrt(expected_variance / reps)
    np.testing.assert_allclose(draws.mean(), expected_mean, atol=5 * mean_mc_se)
    np.testing.assert_allclose(draws.var(), expected_variance, rtol=0.02)


def test_pairs_inference_labels_and_pvalue():
    draws = np.array([[-3.0], [-0.5], [0.25], [2.0], [7.0]])
    beta_hat = np.array([1.5])
    coefnames = ("x",)
    table = wb._summarize_weighting_bootstrap(
        draws=draws,
        beta_hat=beta_hat,
        coefnames=coefnames,
        weight_distribution="multinomial",
        ci_level=0.8,
    )
    assert list(table.columns) == [
        "Estimate",
        "CI lower",
        "CI upper",
        "Bootstrap SE",
        "P-value",
        "interval",
    ]
    ci_lower, ci_upper = np.quantile(draws, [0.1, 0.9], axis=0)
    np.testing.assert_allclose(table.loc["x", "CI lower"], ci_lower[0])
    np.testing.assert_allclose(table.loc["x", "CI upper"], ci_upper[0])
    np.testing.assert_allclose(
        table.loc["x", "Bootstrap SE"], draws.std(axis=0, ddof=1)[0]
    )
    exceedances = np.sum(np.abs(draws - beta_hat) >= np.abs(beta_hat), axis=0)
    assert table.loc["x", "P-value"] == (1 + exceedances[0]) / (len(draws) + 1)
    assert table.loc["x", "interval"] == "percentile confidence"


def test_bayesian_inference_labels_and_tail_probability():
    draws = np.array([[-4.0], [-1.0], [0.5], [2.0], [3.0]])
    table = wb._summarize_weighting_bootstrap(
        draws=draws,
        beta_hat=np.array([1.0]),
        coefnames=("x",),
        weight_distribution="dirichlet",
        ci_level=0.8,
    )
    assert list(table.columns) == [
        "Original estimate",
        "Posterior mean",
        "Posterior SD",
        "CI lower",
        "CI upper",
        "Posterior tail probability",
        "interval",
    ]
    assert table.loc["x", "Original estimate"] == 1
    ci_lower, ci_upper = np.quantile(draws, [0.1, 0.9], axis=0)
    assert table.loc["x", "Posterior mean"] == draws.mean()
    np.testing.assert_allclose(
        table.loc["x", "Posterior SD"], draws.std(axis=0, ddof=1)[0]
    )
    np.testing.assert_allclose(table.loc["x", "CI lower"], ci_lower[0])
    np.testing.assert_allclose(table.loc["x", "CI upper"], ci_upper[0])
    assert table.loc["x", "Posterior tail probability"] == 0.8
    assert table.loc["x", "interval"] == "equal-tail credible"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"reps": 0}, "reps"),
        ({"reps": 1}, "reps"),
        ({"reps": 2.5}, "reps"),
        ({"reps": True}, "reps"),
        ({"reps": 2, "ci_level": 0}, "ci_level"),
        ({"reps": 2, "ci_level": 1}, "ci_level"),
        ({"reps": 2, "ci_level": np.nan}, "ci_level"),
    ],
)
@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_shared_argument_validation(linear_data, kwargs, match, api_name):
    fit = pf.feols("y ~ x", linear_data)
    with pytest.raises(ValueError, match=match):
        getattr(fit, api_name)(**kwargs)


@pytest.mark.parametrize("alpha", [0, np.inf, np.nan])
def test_bayesian_alpha_validation(linear_data, alpha):
    fit = pf.feols("y ~ x", linear_data)
    with pytest.raises(ValueError, match="alpha"):
        fit.bayesian_bootstrap(2, alpha=alpha)


def test_public_bootstrap_signatures_only_expose_relevant_arguments(linear_data):
    fit = pf.feols("y ~ x", linear_data)
    assert list(signature(fit.bayesian_bootstrap).parameters) == [
        "reps",
        "alpha",
        "cluster",
        "ci_level",
        "seed",
        "return_draws",
    ]
    assert list(signature(fit.pairs_bootstrap).parameters) == [
        "reps",
        "cluster",
        "ci_level",
        "seed",
        "return_draws",
    ]


def test_fixest_multi_bootstrap_signatures(linear_data):
    fit = pf.feols("y ~ sw(x, z1)", linear_data)
    assert list(signature(fit.bayesian_bootstrap).parameters) == [
        "reps",
        "alpha",
        "cluster",
        "ci_level",
        "seed",
    ]
    assert list(signature(fit.pairs_bootstrap).parameters) == [
        "reps",
        "cluster",
        "ci_level",
        "seed",
    ]


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_fweights_rejected_but_expanded_sample_supported(linear_data, api_name):
    collapsed = linear_data.iloc[:16].copy()
    collapsed["fw"] = np.tile([1, 2], 8)
    fit = pf.feols("y ~ x", collapsed, weights="fw", weights_type="fweights")
    with pytest.raises(NotImplementedError, match="expanded-sample multinomial"):
        getattr(fit, api_name)(2)

    expanded = collapsed.loc[collapsed.index.repeat(collapsed["fw"])].reset_index(
        drop=True
    )
    result = getattr(pf.feols("y ~ x", expanded), api_name)(2, seed=1)
    expected_columns = 7 if api_name == "bayesian_bootstrap" else 6
    assert result.shape == (2, expected_columns)


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_quantreg_rejected(linear_data, api_name):
    with pytest.warns(FutureWarning):
        fit = pf.quantreg("y ~ x", linear_data)
    with pytest.raises(NotImplementedError, match="quantile regression"):
        getattr(fit, api_name)(2)


@pytest.mark.parametrize(
    "fit_kwargs,match",
    [
        ({"lean": True}, "lean=False"),
        ({"store_data": False}, "store_data=True"),
    ],
)
@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_removed_bootstrap_state_has_informative_error(
    linear_data, fit_kwargs, match, api_name
):
    fit = pf.feols("y ~ x", linear_data, **fit_kwargs)
    with pytest.raises(ValueError, match=match):
        getattr(fit, api_name)(2)


@pytest.mark.parametrize("fit_kwargs", [{"lean": True}, {"store_data": False}])
def test_removed_bootstrap_state_is_not_retained(linear_data, fit_kwargs):
    fit = pf.feols("y ~ x | g | d ~ z1 + z2", linear_data, **fit_kwargs)
    for attribute in (
        "_X_untransformed_df",
        "_Z_untransformed_df",
        "_user_weights",
        "_fe_df",
    ):
        assert not hasattr(fit, attribute)


def test_missing_bootstrap_array_has_informative_error(linear_data):
    fit = pf.feols("y ~ x", linear_data)
    del fit._X_untransformed_df
    with pytest.raises(ValueError, match="missing _X_untransformed_df"):
        fit.bayesian_bootstrap(2)


@pytest.mark.parametrize("problem", ["missing", "one", "nan"])
def test_cluster_validation(linear_data, problem):
    data = linear_data.copy()
    cluster = "g"
    if problem == "missing":
        cluster = "not_a_column"
        match = "not found"
    elif problem == "one":
        data["bad_cluster"] = 1
        cluster = "bad_cluster"
        match = "at least two"
    else:
        data["bad_cluster"] = data["g"].astype(float)
        data.loc[0, "bad_cluster"] = np.nan
        cluster = "bad_cluster"
        match = "must not contain missing"
    fit = pf.feols("y ~ x", data, vcov="iid")
    with pytest.raises(ValueError, match=match):
        fit.pairs_bootstrap(2, cluster=cluster)


def test_multiway_cluster_fallback_rejected(linear_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = pf.feols("y ~ x", linear_data, vcov={"CRV1": "g + h"})
    with pytest.raises(NotImplementedError, match="Multiway clustering"):
        fit.pairs_bootstrap(2)


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_cluster_must_be_a_column_name(linear_data, api_name):
    fit = pf.feols("y ~ x", linear_data)
    with pytest.raises(TypeError, match="column name"):
        getattr(fit, api_name)(2, cluster=["g"])


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_offset_rejected(glm_data, api_name):
    data = glm_data.assign(offset=np.linspace(-0.1, 0.1, len(glm_data)))
    fit = pf.feglm("y_count ~ x", data, family="poisson", offset="offset", vcov="iid")
    with pytest.raises(NotImplementedError, match="offset"):
        getattr(fit, api_name)(2)


@pytest.mark.parametrize("model_type", ["quantreg", "fweights", "offset"])
def test_unsupported_models_do_not_retain_bootstrap_state(
    linear_data, glm_data, model_type
):
    if model_type == "quantreg":
        with pytest.warns(FutureWarning):
            fit = pf.quantreg("y ~ x", linear_data)
    elif model_type == "fweights":
        data = linear_data.assign(fw=np.tile([1, 2], len(linear_data) // 2))
        fit = pf.feols(
            "y ~ x | g | d ~ z1 + z2",
            data,
            weights="fw",
            weights_type="fweights",
        )
    else:
        data = glm_data.assign(offset=np.linspace(-0.1, 0.1, len(glm_data)))
        fit = pf.feglm(
            "y_count ~ x | g",
            data,
            family="poisson",
            offset="offset",
            separation_check=[],
            vcov="iid",
        )

    for attribute in (
        "_X_untransformed_df",
        "_Z_untransformed_df",
        "_user_weights",
        "_fe_df",
    ):
        assert not hasattr(fit, attribute)


def test_rank_deficient_draw_is_discarded(linear_data, monkeypatch):
    data = linear_data.iloc[:4].copy()
    data["x"] = [0.0, 0.0, 1.0, 2.0]
    fit = pf.feols("y ~ x", data)
    calls = 0

    def prescribed_weights(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.array([2.0, 2.0, 0.0, 0.0])
        return np.ones(4)

    monkeypatch.setattr(wb, "_draw_bootstrap_weights", prescribed_weights)
    with pytest.warns(UserWarning, match="conditional on the 2 successful draws"):
        _, draws = fit.pairs_bootstrap(2, return_draws=True)
    assert calls == 3
    assert np.isfinite(draws).all()


def test_replicate_induced_binary_separation_is_discarded(monkeypatch):
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 1, 1, 1],
            "x": [-2.0, 1.0, 2.0, -1.0, 0.0, 3.0],
        }
    )
    fit = pf.feglm("y ~ x", data, family="logit", separation_check=[], accelerate=False)
    calls = 0

    def prescribed_weights(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return np.ones(6)

    monkeypatch.setattr(wb, "_draw_bootstrap_weights", prescribed_weights)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, draws = fit.pairs_bootstrap(2, return_draws=True)
    assert calls == 3
    assert np.isfinite(draws).all()
    assert any(
        issubclass(warning.category, UserWarning)
        and "conditional" in str(warning.message)
        for warning in caught
    )
    assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)


def test_failure_attempts_are_bounded(linear_data, monkeypatch):
    fit = pf.feols("y ~ x", linear_data)
    calls = 0

    def always_fail(**kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("prescribed failure")

    monkeypatch.setattr(wb, "_fit_bootstrap_draw", always_fail)
    with pytest.raises(NonConvergenceError, match="after 4 attempts"):
        fit.bayesian_bootstrap(2)
    assert calls == 4


def test_cluster_draws_are_row_order_invariant(linear_data):
    shuffled = linear_data.sample(frac=1, random_state=91).reset_index(drop=True)
    fit = pf.feols("y ~ x", linear_data)
    shuffled_fit = pf.feols("y ~ x", shuffled)
    _, draws = fit.pairs_bootstrap(
        4,
        cluster="g",
        seed=7,
        return_draws=True,
    )
    _, shuffled_draws = shuffled_fit.pairs_bootstrap(
        4,
        cluster="g",
        seed=7,
        return_draws=True,
    )
    np.testing.assert_allclose(draws, shuffled_draws, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_multiple_estimation_matches_individual_models(linear_data, api_name):
    fit = pf.feols("y ~ sw(x, z1)", linear_data, vcov="iid")
    result = getattr(fit, api_name)(3, seed=12)
    assert list(result.index.names) == ["fml", "Coefficient"]

    for formula, model in fit.all_fitted_models.items():
        expected = getattr(model, api_name)(3, seed=12)
        pd.testing.assert_frame_equal(result.xs(formula, level="fml"), expected)


@pytest.mark.parametrize("api_name", ["bayesian_bootstrap", "pairs_bootstrap"])
def test_fitted_model_ssc_does_not_change_draws(linear_data, api_name):
    adjusted = pf.feols("y ~ x", linear_data, ssc=pf.ssc(k_adj=True))
    unadjusted = pf.feols("y ~ x", linear_data, ssc=pf.ssc(k_adj=False))
    _, adjusted_draws = getattr(adjusted, api_name)(3, seed=5, return_draws=True)
    _, unadjusted_draws = getattr(unadjusted, api_name)(3, seed=5, return_draws=True)
    np.testing.assert_array_equal(adjusted_draws, unadjusted_draws)
