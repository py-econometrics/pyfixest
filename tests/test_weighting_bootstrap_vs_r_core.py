import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

import pyfixest as pf
from pyfixest.estimation.internals.vcov_utils import factorize_cluster_data

# Direct linear solves should agree to near machine precision. Iterative GLM
# solvers can stop at slightly different numerical tolerances.
LINEAR_RTOL = 1e-10
LINEAR_ATOL = 1e-10
GLM_RTOL = 1e-8
GLM_ATOL = 1e-8

pytestmark = pytest.mark.against_r_core


@pytest.fixture(scope="module")
def bootstrap_data():
    rng = np.random.default_rng(651)
    n_obs = 24
    group = np.repeat(np.arange(6), 4)
    x = rng.normal(size=n_obs)
    z1 = rng.normal(size=n_obs)
    z2 = rng.normal(size=n_obs)
    d = 0.4 * x + 0.9 * z1 - 0.6 * z2 + rng.normal(scale=0.4, size=n_obs)
    y = 0.7 * x + 1.2 * d + 0.3 * group + rng.normal(scale=0.5, size=n_obs)
    y_count = rng.poisson(np.exp(0.2 + 0.3 * x))
    return pd.DataFrame(
        {
            "y": y,
            "y_count": y_count,
            "x": x,
            "d": d,
            "z1": z1,
            "z2": z2,
            "g": group,
            "aw": 0.5 + rng.random(n_obs),
        }
    )


def _bootstrap_weights(
    *, n_obs, reps, seed, weight_distribution, cluster=None, dirichlet_alpha=1.0
):
    rng = np.random.default_rng(seed)
    if cluster is None:
        row_to_unit = np.arange(n_obs)
        n_units = n_obs
    else:
        row_to_unit = factorize_cluster_data(cluster)[:, 0]
        n_units = int(row_to_unit.max()) + 1

    draws = []
    for _ in range(reps):
        if weight_distribution == "dirichlet":
            unit_weights = rng.dirichlet(np.full(n_units, dirichlet_alpha)) * n_units
        else:
            unit_weights = rng.multinomial(
                n_units, np.full(n_units, 1 / n_units)
            ).astype(float)
        draws.append(unit_weights[row_to_unit])
    return draws


def _run_public_bootstrap(
    fit, *, reps, seed, weight_distribution, cluster=None, level=0.9
):
    if weight_distribution == "dirichlet":
        return fit.bootstrap_bayesian(
            reps,
            cluster=cluster,
            level=level,
            seed=seed,
            return_draws=True,
        )
    return fit.bootstrap_pairs(
        reps,
        cluster=cluster,
        level=level,
        seed=seed,
        return_draws=True,
    )


def _set_r_weights(data, weights):
    active = weights > 0
    ro.globalenv["bootstrap_data"] = data.loc[active].reset_index(drop=True)
    ro.globalenv["bootstrap_weights"] = ro.FloatVector(weights[active])


@pytest.mark.parametrize("weight_distribution", ["dirichlet", "multinomial"])
@pytest.mark.parametrize("model", ["ols_fe_cluster", "iv", "poisson"])
def test_public_draws_match_r_weighted_refits(
    bootstrap_data, weight_distribution, model
):
    reps = 3
    seed = 417
    cluster = "g" if model == "ols_fe_cluster" else None
    sampling_units = bootstrap_data["g"].to_numpy() if cluster else None
    weights = _bootstrap_weights(
        n_obs=len(bootstrap_data),
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        cluster=sampling_units,
    )

    if model == "ols_fe_cluster":
        fit = pf.feols("y ~ x | g", bootstrap_data, weights="aw", vcov={"CRV1": "g"})
    elif model == "iv":
        fit = pf.feols(
            "y ~ x + [d ~ z1 + z2]", bootstrap_data, weights="aw", vcov="iid"
        )
    else:
        fit = pf.feglm(
            "y_count ~ x",
            bootstrap_data,
            family="poisson",
            weights="aw",
            separation_check=[],
            accelerate=False,
            vcov="iid",
        )

    _, draws = _run_public_bootstrap(
        fit,
        reps=reps,
        seed=seed,
        weight_distribution=weight_distribution,
        cluster=cluster,
    )

    expected = []
    for draw in weights:
        combined_weights = bootstrap_data["aw"].to_numpy() * draw
        _set_r_weights(bootstrap_data, combined_weights)
        if model == "ols_fe_cluster":
            estimate = ro.r(
                "unname(coef(lm(y ~ x + factor(g), data=bootstrap_data, "
                'weights=bootstrap_weights))["x"])'
            )
        elif model == "iv":
            estimate = ro.r(
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
        else:
            estimate = ro.r(
                "unname(coef(glm(y_count ~ x, data=bootstrap_data, "
                "weights=bootstrap_weights, family=poisson())))"
            )
        expected.append(np.asarray(estimate).reshape(-1))

    tolerance = (
        {"rtol": GLM_RTOL, "atol": GLM_ATOL}
        if model == "poisson"
        else {"rtol": LINEAR_RTOL, "atol": LINEAR_ATOL}
    )
    np.testing.assert_allclose(draws, np.vstack(expected), **tolerance)


def test_pairs_summaries_match_r_boot(bootstrap_data):
    """Compare independently seeded public summaries with R boot 1.3-32."""
    reps = 3999
    level = 0.9
    seed = 924
    fit = pf.feols("y ~ x", bootstrap_data, vcov="iid")
    inference = fit.bootstrap_pairs(reps, level=level, seed=seed)

    ro.globalenv["bootstrap_data"] = bootstrap_data
    r_result = ro.r(
        f"""
        set.seed({seed})
        statistic <- function(data, frequencies) {{
            estimates <- coef(lm(y ~ x, data=data, weights=frequencies))
            names(estimates)[names(estimates) == "(Intercept)"] <- "Intercept"
            estimates
        }}
        result <- boot::boot(
            bootstrap_data,
            statistic=statistic,
            R={reps},
            stype="f"
        )
        list(
            estimate=unname(result$t0),
            se=apply(result$t, 2, sd),
            interval=apply(result$t, 2, quantile, probs=c(0.05, 0.95)),
            coefficient_names=names(result$t0)
        )
        """
    )
    coefficient_names = tuple(r_result.rx2("coefficient_names"))
    assert tuple(inference.index) == coefficient_names
    np.testing.assert_allclose(
        inference["Estimate"],
        np.asarray(r_result.rx2("estimate")),
        rtol=LINEAR_RTOL,
        atol=LINEAR_ATOL,
    )
    # The implementations use different RNG streams. These Monte Carlo
    # tolerances compare the promised distributional summaries, not draw order.
    np.testing.assert_allclose(
        inference["Bootstrap SE"],
        np.asarray(r_result.rx2("se")),
        rtol=0.08,
        atol=0.02,
    )
    np.testing.assert_allclose(
        inference[["CI lower", "CI upper"]].to_numpy().T,
        np.asarray(r_result.rx2("interval")),
        rtol=0.08,
        atol=0.05,
    )
