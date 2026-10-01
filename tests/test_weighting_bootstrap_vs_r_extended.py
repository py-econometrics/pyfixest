import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

import pyfixest as pf
from pyfixest.utils.check_r_install import check_r_install

_HAS_BAYESBOOT = check_r_install("bayesboot", strict=False)

pytestmark = [
    pytest.mark.against_r_extended,
    pytest.mark.skipif(
        not _HAS_BAYESBOOT,
        reason="R package bayesboot is not installed.",
    ),
]

LINEAR_RTOL = 1e-10
LINEAR_ATOL = 1e-10


def test_bayesian_summaries_match_r_bayesboot():
    """Compare Rubin's alpha=1 bootstrap with R bayesboot 0.2.3."""
    data = pd.DataFrame(
        {
            "y": [-1.0, 0.25, 0.5, 1.5, 2.0, 4.0],
            "x": [-2.0, -1.0, -0.25, 0.5, 1.25, 2.0],
        }
    )
    reps = 3999
    level = 0.9
    seed = 761
    fit = pf.feols("y ~ x", data, vcov="iid")
    inference = fit.bootstrap_bayesian(
        reps,
        dirichlet_alpha=1.0,
        level=level,
        seed=seed,
    )

    ro.globalenv["bootstrap_data"] = data
    r_result = ro.r(
        f"""
        statistic <- function(data, weights) {{
            estimates <- coef(lm(y ~ x, data=data, weights=weights))
            names(estimates)[names(estimates) == "(Intercept)"] <- "Intercept"
            estimates
        }}
        set.seed({seed})
        result <- bayesboot::bayesboot(
            bootstrap_data,
            statistic=statistic,
            R={reps},
            use.weights=TRUE
        )
        list(
            original=unname(statistic(
                bootstrap_data,
                rep(1 / nrow(bootstrap_data), nrow(bootstrap_data))
            )),
            posterior_mean=colMeans(result),
            posterior_sd=apply(result, 2, sd),
            interval=apply(result, 2, quantile, probs=c(0.05, 0.95)),
            coefficient_names=colnames(result)
        )
        """
    )
    coefficient_names = tuple(r_result.rx2("coefficient_names"))
    assert tuple(inference.index) == coefficient_names
    np.testing.assert_allclose(
        inference["Original estimate"],
        np.asarray(r_result.rx2("original")),
        rtol=LINEAR_RTOL,
        atol=LINEAR_ATOL,
    )
    # R and NumPy use different Dirichlet RNG streams. These Monte Carlo
    # tolerances compare posterior summaries rather than individual draws.
    np.testing.assert_allclose(
        inference["Posterior mean"],
        np.asarray(r_result.rx2("posterior_mean")),
        rtol=0.08,
        atol=0.03,
    )
    np.testing.assert_allclose(
        inference["Posterior SD"],
        np.asarray(r_result.rx2("posterior_sd")),
        rtol=0.08,
        atol=0.02,
    )
    np.testing.assert_allclose(
        inference[["CI lower", "CI upper"]].to_numpy().T,
        np.asarray(r_result.rx2("interval")),
        rtol=0.08,
        atol=0.05,
    )
