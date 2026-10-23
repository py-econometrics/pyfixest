"""RI-t must studentize observed and permuted coefficients identically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import pyfixest as pf
from pyfixest.estimation.post_estimation.ritest import _resample


@pytest.mark.parametrize(
    "vcov,cluster,h0",
    [
        ("iid", None, 0),
        ("hetero", None, 0.25),
        ("HC2", None, 0),
        ("HC3", None, 0),
        ({"CRV1": "s"}, None, 0),
        ("iid", "s", 0.25),
    ],
)
def test_randomization_t_matches_statsmodels(vcov, cluster, h0):
    rng = np.random.default_rng(8)
    n = 120
    data = pd.DataFrame(
        {
            "D": rng.integers(0, 2, n).astype(float),
            "x": rng.normal(size=n),
            "s": np.repeat(np.arange(12), 10),
        }
    )
    data["y"] = (
        0.3 * data.D
        + data.x
        + rng.normal(size=12)[data.s]
        + rng.normal(size=n) * (1 + data.D)
    )
    # Default SSC matches statsmodels OLS, HC1 and CRV1 finite-sample factors;
    # disable its extra n/(n-k) factor for the unadjusted HC2/HC3 reference.
    has_fe = isinstance(vcov, dict)
    ssc = pf.ssc(k_adj=not has_fe and vcov not in ["HC2", "HC3"], G_adj=not has_fe)
    fit = pf.feols(
        "y ~ D + x" + (" | s" if has_fe else ""), data=data, vcov=vcov, ssc=ssc
    )
    covariance_before = fit.variance_covariance
    reps = 19
    result = fit.ritest(
        f"D={h0}",
        type="randomization-t",
        cluster=cluster,
        reps=reps,
        rng=np.random.default_rng(1),
        store_ritest_statistics=True,
    )
    chosen = {"CRV1": cluster} if cluster is not None else vcov

    def statistic(treatment):
        design = sm.add_constant(np.column_stack([treatment, data.x]))
        if has_fe:
            design = np.column_stack(
                [design, pd.get_dummies(data.s, drop_first=True, dtype=float)]
            )
        if isinstance(chosen, dict):
            return sm.OLS(data.y, design).fit(
                cov_type="cluster",
                cov_kwds={"groups": data.s, "use_correction": not has_fe},
            )
        cov_type = {"hetero": "HC1"}.get(chosen, chosen)
        return (
            sm.OLS(data.y, design).fit()
            if cov_type == "iid"
            else sm.OLS(data.y, design).fit(cov_type=cov_type)
        )

    observed = statistic(data.D)
    expected_sample = (observed.params.iloc[1] - h0) / observed.bse.iloc[1]
    draw_rng = np.random.default_rng(1)
    draws = []
    for _ in range(reps):
        treatment = _resample(
            resampvar_arr=data.D.to_numpy(),
            clustervar_arr=None
            if cluster is None
            else data.s.to_numpy().reshape(-1, 1),
            rng=draw_rng,
            iterations=1,
        ).ravel()
        draws.append(statistic(treatment).tvalues.iloc[1])
    # Independent QR/SVD implementations agree to rounding on this full-rank design.
    np.testing.assert_allclose(
        fit.ritest_statistics.statistics,
        draws,
        rtol=1e-9,
        atol=1e-10,
        err_msg="Permutation t-statistics use a different covariance",
    )
    np.testing.assert_allclose(
        fit.ritest_statistics.sample_stat,
        expected_sample,
        rtol=1e-9,
        atol=1e-10,
        err_msg="Observed t-statistic uses a different covariance or null scale",
    )
    expected_p = np.mean(np.abs(draws[1:]) >= abs(expected_sample))
    assert result["Pr(>|t|)"] == expected_p
    assert fit.variance_covariance is covariance_before
