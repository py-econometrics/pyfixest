import warnings

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals.vcov_utils import factorize_cluster_data

# Dense fits and public pyfixest refits solve the same objectives through
# different paths; these tolerances accommodate floating-point solve order.
RTOL = 1e-7
ATOL = 1e-9


@pytest.fixture(scope="module")
def linear_data():
    rng = np.random.default_rng(421)
    n_groups = 8
    observations_per_group = 8
    group = np.repeat(np.arange(n_groups), observations_per_group)
    n_obs = len(group)
    x = rng.normal(size=n_obs)
    z1 = rng.normal(size=n_obs)
    z2 = rng.normal(size=n_obs)
    d = 0.8 * z1 + 0.5 * z2 + 0.3 * x + rng.normal(size=n_obs)
    group_effect = rng.normal(size=n_groups)[group]
    y = 0.7 * x + 1.1 * d + group_effect + rng.normal(size=n_obs)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "d": d,
            "z1": z1,
            "z2": z2,
            "g": group,
            "h": np.tile(np.arange(4), n_obs // 4),
            "aw": 0.5 + rng.random(n_obs),
        }
    )


@pytest.fixture(scope="module")
def glm_data():
    rng = np.random.default_rng(811)
    n_groups = 8
    observations_per_group = 15
    group = np.repeat(np.arange(n_groups), observations_per_group)
    n_obs = len(group)
    x = rng.normal(size=n_obs)
    group_effect = rng.normal(scale=0.25, size=n_groups)[group]
    probability = 1 / (1 + np.exp(-(0.4 * x + group_effect)))
    y_binary = rng.binomial(1, probability)
    y_count = rng.poisson(np.exp(0.2 + 0.25 * x + group_effect))
    y_gaussian = 0.6 * x + group_effect + rng.normal(size=n_obs)

    for group_id in range(n_groups):
        rows = np.flatnonzero(group == group_id)
        y_binary[rows[:2]] = [0, 1]
        y_count[rows[0]] = 1

    return pd.DataFrame(
        {
            "y_binary": y_binary,
            "y_count": y_count,
            "y_gaussian": y_gaussian,
            "x": x,
            "g": group,
            "aw": 0.4 + rng.random(n_obs),
        }
    )


def _bootstrap_with_draws(
    fit,
    *,
    reps,
    weight_distribution,
    dirichlet_alpha,
    cluster=None,
    seed=None,
):
    if weight_distribution == "dirichlet":
        return fit.bootstrap_bayesian(
            reps,
            dirichlet_alpha=dirichlet_alpha,
            cluster=cluster,
            seed=seed,
            return_draws=True,
        )
    return fit.bootstrap_pairs(
        reps,
        cluster=cluster,
        seed=seed,
        return_draws=True,
    )


def _bootstrap_weights(
    *,
    n_obs,
    reps,
    seed,
    weight_distribution,
    dirichlet_alpha,
    cluster=None,
):
    rng = np.random.default_rng(seed)
    if cluster is None:
        row_to_unit = np.arange(n_obs)
        n_units = n_obs
    else:
        row_to_unit = factorize_cluster_data(cluster)[:, 0]
        n_units = int(row_to_unit.max()) + 1

    weights = []
    for _ in range(reps):
        if weight_distribution == "dirichlet":
            unit_weights = rng.dirichlet(np.full(n_units, dirichlet_alpha)) * n_units
        else:
            unit_weights = rng.multinomial(
                n_units, np.full(n_units, 1 / n_units)
            ).astype(float)
        weights.append(unit_weights[row_to_unit])
    return weights


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


@pytest.mark.parametrize(
    "has_fixef,use_aweights,vcov,cluster_arg,draw_cluster,weight_distribution",
    [
        (False, False, "iid", None, None, "dirichlet"),
        (True, True, "iid", "g", "g", "dirichlet"),
        (False, True, {"CRV1": "g"}, None, "g", "multinomial"),
        (True, False, "iid", "g", "g", "multinomial"),
    ],
)
def test_ols_draws_match_dense_weighted_fits(
    linear_data,
    has_fixef,
    use_aweights,
    vcov,
    cluster_arg,
    draw_cluster,
    weight_distribution,
):
    formula = "y ~ x | g" if has_fixef else "y ~ x"
    fit = pf.feols(
        formula,
        linear_data,
        weights="aw" if use_aweights else None,
        vcov=vcov,
    )
    reps = 4
    seed = 19
    dirichlet_alpha = 1.4
    inference, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
        cluster=cluster_arg,
        seed=seed,
    )
    sampling_units = (
        linear_data[draw_cluster].to_numpy() if draw_cluster is not None else None
    )
    weights = _bootstrap_weights(
        n_obs=len(linear_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
        cluster=sampling_units,
    )
    original_weights = linear_data["aw"].to_numpy() if use_aweights else 1
    expected = np.vstack(
        [
            _dense_ols(linear_data, original_weights * draw, has_fixef)
            for draw in weights
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=RTOL, atol=ATOL)
    expected_columns = (
        [
            "Original estimate",
            "Posterior mean",
            "Posterior SD",
            "CI lower",
            "CI upper",
            "interval",
        ]
        if weight_distribution == "dirichlet"
        else [
            "Estimate",
            "CI lower",
            "CI upper",
            "Bootstrap SE",
            "P-value",
            "interval",
        ]
    )
    assert list(inference.columns) == expected_columns


@pytest.mark.parametrize("weight_distribution", ["dirichlet", "multinomial"])
def test_overidentified_iv_draws_match_public_weighted_refits(
    linear_data, weight_distribution
):
    formula = "y ~ x + [d ~ z1 + z2] | g"
    fit = pf.feols(formula, linear_data, weights="aw", vcov="iid")
    reps = 3
    seed = 27
    dirichlet_alpha = 0.8
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    weights = _bootstrap_weights(
        n_obs=len(linear_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
    )
    expected = []
    for draw in weights:
        combined_weights = linear_data["aw"].to_numpy() * draw
        active = combined_weights > 0
        ref_data = linear_data.loc[active].assign(_ref_weight=combined_weights[active])
        expected.append(
            pf.feols(
                formula,
                ref_data,
                weights="_ref_weight",
                vcov="iid",
            )
            .coef()
            .to_numpy()
        )
    np.testing.assert_allclose(draws, np.vstack(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "outcome,family,has_fixef,use_aweights,weight_distribution",
    [
        ("y_count", "poisson", True, True, "multinomial"),
        ("y_binary", "logit", False, True, "dirichlet"),
        ("y_binary", "probit", True, False, "dirichlet"),
        ("y_gaussian", "gaussian", True, True, "multinomial"),
    ],
)
def test_glm_draws_match_public_weighted_refits(
    glm_data,
    outcome,
    family,
    has_fixef,
    use_aweights,
    weight_distribution,
):
    formula = f"{outcome} ~ x" + (" | g" if has_fixef else "")
    fit = pf.feglm(
        formula,
        glm_data,
        family=family,
        weights="aw" if use_aweights else None,
        separation_check=[],
        accelerate=False,
        vcov="iid",
    )
    reps = 2
    seed = 44
    dirichlet_alpha = 1.3
    _, draws = _bootstrap_with_draws(
        fit,
        reps=reps,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    weights = _bootstrap_weights(
        n_obs=len(glm_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        dirichlet_alpha=dirichlet_alpha,
    )
    original_weights = glm_data["aw"].to_numpy() if use_aweights else 1
    expected = []
    for draw in weights:
        combined_weights = original_weights * draw
        active = combined_weights > 0
        ref_data = glm_data.loc[active].assign(
            _ref_weight=np.asarray(combined_weights)[active]
        )
        expected.append(
            pf.feglm(
                formula,
                ref_data,
                family=family,
                weights="_ref_weight",
                separation_check=[],
                accelerate=False,
                vcov="iid",
            )
            .coef()
            .to_numpy()
        )
    np.testing.assert_allclose(draws, np.vstack(expected), rtol=RTOL, atol=ATOL)


def test_initial_poisson_separation_stays_row_aligned():
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
    _, draws = fit.bootstrap_bayesian(2, seed=3, return_draws=True)
    assert fit.n_separation_na == 3
    assert draws.shape == (2, 1)
    assert np.isfinite(draws).all()


@pytest.mark.parametrize("api_name", ["bootstrap_bayesian", "bootstrap_pairs"])
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"reps": 1}, "reps"),
        ({"reps": 2.5}, "reps"),
        ({"reps": True}, "reps"),
        ({"reps": 2, "level": 0}, "level"),
        ({"reps": 2, "level": np.nan}, "level"),
    ],
)
def test_public_argument_validation(linear_data, api_name, kwargs, match):
    fit = pf.feols("y ~ x", linear_data)
    with pytest.raises(ValueError, match=match):
        getattr(fit, api_name)(**kwargs)


@pytest.mark.parametrize("dirichlet_alpha", [0, np.inf, np.nan])
def test_dirichlet_alpha_validation(linear_data, dirichlet_alpha):
    fit = pf.feols("y ~ x", linear_data)
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        fit.bootstrap_bayesian(2, dirichlet_alpha=dirichlet_alpha)


@pytest.mark.parametrize("api_name", ["bootstrap_bayesian", "bootstrap_pairs"])
@pytest.mark.parametrize("problem", ["missing", "one", "nan", "type", "multiway"])
def test_cluster_validation(linear_data, api_name, problem):
    data = linear_data.copy()
    cluster = "g"
    vcov = "iid"
    error = ValueError
    if problem == "missing":
        cluster = "not_a_column"
        match = "not found"
    elif problem == "one":
        data["bad_cluster"] = 1
        cluster = "bad_cluster"
        match = "at least two"
    elif problem == "nan":
        data["bad_cluster"] = data["g"].astype(float)
        data.loc[0, "bad_cluster"] = np.nan
        cluster = "bad_cluster"
        match = "must not contain missing"
    elif problem == "type":
        cluster = ["g"]
        match = "column name"
        error = TypeError
    else:
        vcov = {"CRV1": "g + h"}
        cluster = None
        match = "Multiway clustering"
        error = NotImplementedError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = pf.feols("y ~ x", data, vcov=vcov)
    with pytest.raises(error, match=match):
        getattr(fit, api_name)(2, cluster=cluster)


@pytest.mark.parametrize("api_name", ["bootstrap_bayesian", "bootstrap_pairs"])
@pytest.mark.parametrize(
    "unsupported,match,error",
    [
        ("fweights", "expanded-sample multinomial", NotImplementedError),
        ("quantreg", "quantile regression", NotImplementedError),
        ("did", "not implemented for DiD results", NotImplementedError),
        ("lean", "lean=False", ValueError),
        ("store_data", "store_data=True", ValueError),
        ("offset", "offset", NotImplementedError),
    ],
)
def test_unsupported_result_paths(
    linear_data, glm_data, api_name, unsupported, match, error
):
    if unsupported == "fweights":
        data = linear_data.copy()
        data["fw"] = 1
        fit = pf.feols("y ~ x", data, weights="fw", weights_type="fweights")
    elif unsupported == "quantreg":
        with pytest.warns(FutureWarning):
            fit = pf.quantreg("y ~ x", linear_data)
    elif unsupported == "did":
        fit = pf.feols("y ~ x", linear_data)
        fit._method = "did2s"
    elif unsupported == "lean":
        fit = pf.feols("y ~ x", linear_data, lean=True)
    elif unsupported == "store_data":
        fit = pf.feols("y ~ x", linear_data, store_data=False)
    else:
        data = glm_data.assign(offset=np.linspace(-0.1, 0.1, len(glm_data)))
        fit = pf.feglm(
            "y_count ~ x",
            data,
            family="poisson",
            offset="offset",
            vcov="iid",
        )

    with pytest.raises(error, match=match):
        getattr(fit, api_name)(2)


def test_fixest_multi_does_not_expose_bootstrap_methods(linear_data):
    fit = pf.feols("y ~ sw(x, z1)", linear_data)
    assert not hasattr(fit, "bootstrap_bayesian")
    assert not hasattr(fit, "bootstrap_pairs")


def test_failed_pairs_replicates_remain_missing():
    data = pd.DataFrame({"y": [0.0, 1.5, 1.0], "x": [0.0, 1.0, 2.0]})
    fit = pf.feols("y ~ x - 1", data, vcov="iid")

    with pytest.warns(UserWarning, match="failed rows are `NaN`"):
        inference, draws = fit.bootstrap_pairs(5, seed=4, return_draws=True)

    assert draws.shape == (5, 1)
    assert np.isnan(draws).sum() == 1
    assert np.isfinite(inference.select_dtypes(include="number")).all().all()

    with pytest.raises(NonConvergenceError, match="at least 2 are required"):
        fit.bootstrap_pairs(2, seed=4)
