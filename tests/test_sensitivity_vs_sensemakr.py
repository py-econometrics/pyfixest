import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

import pyfixest as pf
from pyfixest.utils.check_r_install import check_r_install

_HAS_SENSEMAKR = check_r_install("sensemakr", strict=False)
_HAS_FIXEST = check_r_install("fixest", strict=False)

if _HAS_SENSEMAKR and _HAS_FIXEST:
    _R_SENSITIVITY_RESULTS = ro.r(
        """
        function(
            formula,
            data,
            treatment,
            benchmark,
            kd,
            ky,
            q,
            alpha,
            reduce,
            h0,
            use_fixest
        ) {
            if (use_fixest) {
                model <- fixest::feols(formula, data = data)
            } else {
                model <- stats::lm(formula, data = data)
            }
            stats <- sensemakr::sensitivity_stats(
                model,
                treatment = treatment,
                q = q,
                alpha = alpha
            )
            bounds <- sensemakr::ovb_bounds(
                model,
                treatment = treatment,
                benchmark_covariates = benchmark,
                kd = kd,
                ky = ky,
                alpha = alpha,
                reduce = reduce,
                h0 = h0
            )
            list(
                stats = c(
                    estimate = stats$estimate,
                    standard_error = stats$se,
                    degrees_of_freedom = stats$dof,
                    partial_r2 = stats$r2yd.x,
                    partial_f2 = stats$f2yd.x,
                    robustness_value = stats$rv_q,
                    robustness_value_alpha = stats$rv_qa
                ),
                bounds = unname(as.matrix(bounds[, c(
                    "r2dz.x",
                    "r2yz.dx",
                    "adjusted_estimate",
                    "adjusted_se",
                    "adjusted_t",
                    "adjusted_lower_CI",
                    "adjusted_upper_CI"
                )]))
            )
        }
        """
    )


pytestmark = [
    pytest.mark.against_r_extended,
    pytest.mark.skipif(
        not (_HAS_SENSEMAKR and _HAS_FIXEST),
        reason="R packages sensemakr and fixest are required.",
    ),
]

# Plain and fixed-effect OLS agree to near floating-point precision on this
# deterministic fixture; 1e-10 still leaves margin for cross-platform BLAS.
_PLAIN_RTOL = 1e-10
_FIXED_EFFECT_RTOL = 1e-10
_ATOL = 1e-10


@pytest.fixture(scope="module")
def sensitivity_reference_data() -> pd.DataFrame:
    rng = np.random.default_rng(918273)
    n_obs = 300
    group = np.repeat(np.arange(30), n_obs // 30)
    benchmark = rng.normal(size=n_obs)
    control = rng.normal(size=n_obs)
    treatment = 0.35 * benchmark + 0.2 * control + rng.normal(size=n_obs)
    group_effect = rng.normal(scale=0.5, size=30)[group]
    outcome = (
        0.7 * treatment
        + 0.45 * benchmark
        - 0.25 * control
        + group_effect
        + rng.normal(size=n_obs)
    )
    return pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
            "benchmark": benchmark,
            "control": control,
            "group": group,
        }
    )


@pytest.mark.parametrize(
    "formula, vcov, use_fixest, tolerance",
    [
        pytest.param(
            "outcome ~ treatment + benchmark + control",
            "iid",
            False,
            _PLAIN_RTOL,
            id="plain-iid",
        ),
        pytest.param(
            "outcome ~ treatment + benchmark + control",
            "hetero",
            False,
            _PLAIN_RTOL,
            id="plain-fitted-hetero-analysis-iid",
        ),
        pytest.param(
            "outcome ~ treatment + benchmark + control | group",
            "iid",
            True,
            _FIXED_EFFECT_RTOL,
            id="fixed-effects-iid",
        ),
    ],
)
@pytest.mark.parametrize("reduce", [True, False])
def test_sensitivity_statistics_and_bounds_match_sensemakr(
    sensitivity_reference_data,
    formula,
    vcov,
    use_fixest,
    tolerance,
    reduce,
):
    q = 0.8
    alpha = 0.05
    h0 = 0.2
    kd = [0.5, 1.0]
    ky = [1.0, 2.0]
    fit = pf.feols(formula, sensitivity_reference_data, vcov=vcov)

    if vcov == "iid":
        analysis = fit.sensitivity_analysis("treatment")
    else:
        with pytest.warns(UserWarning, match="uses IID standard errors"):
            analysis = fit.sensitivity_analysis("treatment")
    stats = analysis.sensitivity_stats(q=q, alpha=alpha)
    bounds = analysis.ovb_bounds(
        "benchmark",
        kd=kd,
        ky=ky,
        alpha=alpha,
        reduce=reduce,
        h0=h0,
    )

    r_results = _R_SENSITIVITY_RESULTS(
        ro.Formula(formula),
        sensitivity_reference_data,
        "treatment",
        "benchmark",
        np.asarray(kd),
        np.asarray(ky),
        q,
        alpha,
        reduce,
        h0,
        use_fixest,
    )

    py_stats = np.asarray(list(stats.to_dict().values()), dtype=float)
    r_stats = np.asarray(r_results.rx2("stats"), dtype=float)
    np.testing.assert_allclose(py_stats, r_stats, rtol=tolerance, atol=_ATOL)

    bound_columns = [
        "r2dz_x",
        "r2yz_dx",
        "adjusted_estimate",
        "adjusted_se",
        "adjusted_t",
        "adjusted_lower_ci",
        "adjusted_upper_ci",
    ]
    np.testing.assert_allclose(
        bounds[bound_columns].to_numpy(),
        np.asarray(r_results.rx2("bounds"), dtype=float),
        rtol=tolerance,
        atol=_ATOL,
    )
