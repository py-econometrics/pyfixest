import numpy as np
import pandas as pd
import pytest

import pyfixest as pf

# Frequency weights are tested in this module only. Every estimator is fit on
# aggregate rows carrying counts and on the literal expansion of those rows, and
# the two fits must agree on coefficients, every supported inference type,
# residuals, fitted values, sample sizes, and the estimator's fit statistics.

IWLS_KWARGS = {"iwls_tol": 1e-12, "iwls_maxiter": 200}
OLS_TOL = {"rtol": 1e-9, "atol": 1e-10}
# Collapsing repeated rows changes floating-point accumulation and therefore
# the IWLS stopping point. The covariance uses the working weights of the last
# iteration, so it tracks that stopping point more closely than the
# coefficients do; a tighter ``iwls_tol`` reaches the deviance noise floor and
# makes step-halving fail for the probit family.
GLM_TOL = {"rtol": 5e-6, "atol": 1e-9}


def _frequency_weight_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate rows with counts and their literal expansion.

    The frame carries a continuous, a count, and a binary response, an
    endogenous regressor with an instrument, a fixed effect with several rows
    per level, and a cluster variable that cuts across the fixed effect.
    """
    rng = np.random.default_rng(20260908)
    n_levels, rows_per_level = 6, 12
    n_rows = n_levels * rows_per_level

    fe = np.repeat([f"l{i}" for i in range(n_levels)], rows_per_level)
    fe_effect = np.repeat(rng.normal(scale=0.4, size=n_levels), rows_per_level)
    cluster = rng.integers(0, 5, size=n_rows)
    x = rng.normal(size=n_rows)
    z = rng.normal(size=n_rows)
    d = 0.8 * z + 0.3 * x + 0.2 * fe_effect + rng.normal(scale=0.5, size=n_rows)
    linear_predictor = -0.2 + 0.7 * x + fe_effect

    aggregate = pd.DataFrame(
        {
            "y": 1.0
            + 1.2 * d
            - 0.5 * x
            + fe_effect
            + rng.normal(scale=0.6, size=n_rows),
            "y_count": rng.poisson(np.exp(0.3 + 0.4 * x + fe_effect)),
            "y_binary": rng.binomial(1, 1 / (1 + np.exp(-linear_predictor))),
            "x": x,
            "d": d,
            "z": z,
            "fe": fe,
            "cluster": cluster,
            "count": rng.integers(1, 5, size=n_rows),
        }
    )
    expanded = (
        aggregate.loc[aggregate.index.repeat(aggregate["count"])]
        .drop(columns="count")
        .reset_index(drop=True)
    )
    return aggregate, expanded


def _fit(estimator: str, family: str | None, fml: str, data: pd.DataFrame, **kwargs):
    if estimator == "feols":
        return pf.feols(fml, data=data, **kwargs)
    if estimator == "fepois":
        return pf.fepois(fml, data=data, **IWLS_KWARGS, **kwargs)
    return pf.feglm(fml, data=data, family=family, **IWLS_KWARGS, **kwargs)


def _vcov_types(has_fe: bool, is_iv: bool, supports_crv3: bool) -> list:
    vcov_types: list = ["iid", "hetero", {"CRV1": "cluster"}]
    if not has_fe and not is_iv:
        vcov_types += ["HC2", "HC3"]
    if supports_crv3:
        vcov_types.append({"CRV3": "cluster"})
    return vcov_types


def _fit_statistics(estimator: str, family: str | None, has_fe: bool, is_iv: bool):
    if is_iv:
        return ["_pi_hat", "_f_stat_1st_stage", "_p_value_1st_stage"]
    if estimator == "fepois":
        return ["deviance", "_loglik", "_pearson_chi2"]
    if estimator == "feglm" and family != "gaussian":
        return ["deviance"]
    performance = ["_rmse", "_r2", "_adj_r2"]
    if has_fe:
        performance += ["_r2_within", "_adj_r2_within"]
    return performance


def _assert_matches_expansion(fit_weighted, fit_expanded, counts, vcov_types, tol):
    """Assert that a frequency-weighted fit equals the fit on repeated rows."""
    n_expanded = int(counts.sum())
    assert fit_weighted._N == fit_expanded._N == n_expanded
    assert fit_weighted._N_rows == len(counts)
    assert fit_expanded._N_rows == n_expanded

    np.testing.assert_allclose(
        fit_weighted.coef().to_numpy(),
        fit_expanded.coef().to_numpy(),
        err_msg="Coefficients differ",
        **tol,
    )
    for vcov_type in vcov_types:
        fit_weighted.vcov(vcov_type)
        fit_expanded.vcov(vcov_type)
        np.testing.assert_allclose(
            fit_weighted._vcov,
            fit_expanded._vcov,
            err_msg=f"Vcov differs for {vcov_type}",
            **tol,
        )
        np.testing.assert_allclose(
            fit_weighted.se().to_numpy(),
            fit_expanded.se().to_numpy(),
            err_msg=f"SEs differ for {vcov_type}",
            **tol,
        )

    np.testing.assert_allclose(
        np.repeat(fit_weighted.resid(), counts),
        fit_expanded.resid(),
        err_msg="Residuals differ",
        **tol,
    )
    if not fit_weighted._is_iv:  # predict() is unsupported for IV models
        np.testing.assert_allclose(
            np.repeat(fit_weighted.predict(), counts),
            fit_expanded.predict(),
            err_msg="Fitted values differ",
            **tol,
        )


FWEIGHT_CASES = [
    ("feols", None, "y ~ x"),
    ("feols", None, "y ~ x | fe"),
    ("feols", None, "y ~ x + [d ~ z]"),
    ("feols", None, "y ~ x + [d ~ z] | fe"),
    ("fepois", None, "y_count ~ x"),
    ("fepois", None, "y_count ~ x | fe"),
    ("feglm", "gaussian", "y ~ x"),
    ("feglm", "gaussian", "y ~ x | fe"),
    ("feglm", "logit", "y_binary ~ x"),
    ("feglm", "logit", "y_binary ~ x | fe"),
    ("feglm", "probit", "y_binary ~ x"),
    ("feglm", "probit", "y_binary ~ x | fe"),
]


@pytest.mark.parametrize(
    ("estimator", "family", "fml"),
    FWEIGHT_CASES,
    ids=[f"{e}-{f}-{fml}" if f else f"{e}-{fml}" for e, f, fml in FWEIGHT_CASES],
)
def test_fweights_match_literal_expansion(estimator, family, fml):
    """Frequency-weighted fits equal fits on the literally repeated rows."""
    aggregate, expanded = _frequency_weight_data()
    counts = aggregate["count"].to_numpy(dtype=np.int64)
    has_fe = fml.endswith("| fe")
    is_iv = "[" in fml
    tol = OLS_TOL if estimator == "feols" else GLM_TOL

    fit_weighted = _fit(
        estimator,
        family,
        fml,
        aggregate,
        weights="count",
        weights_type="fweights",
        vcov="iid",
    )
    fit_expanded = _fit(estimator, family, fml, expanded, vcov="iid")

    _assert_matches_expansion(
        fit_weighted,
        fit_expanded,
        counts,
        vcov_types=_vcov_types(
            has_fe, is_iv, supports_crv3=fit_weighted._support_crv3_inference
        ),
        tol=tol,
    )
    for statistic in _fit_statistics(estimator, family, has_fe, is_iv):
        np.testing.assert_allclose(
            getattr(fit_weighted, statistic),
            getattr(fit_expanded, statistic),
            err_msg=f"{statistic} differs",
            **tol,
        )
    if is_iv:
        fit_weighted.IV_Diag()
        fit_expanded.IV_Diag()
        np.testing.assert_allclose(
            fit_weighted._eff_F, fit_expanded._eff_F, err_msg="_eff_F differs", **tol
        )


def test_fweights_glm_sample_sizes_after_separation():
    """Effective and physical sample sizes stay distinct after separation."""
    aggregated = pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            "x": [-1.2, -0.4, 0.7, -1.1, -0.2, 0.9, -0.8, 0.1, 1.2, -0.9, 0.4, 1.4],
            "fe": np.repeat(list("abcd"), 3),
            "count": [1, 3, 2, 2, 1, 4, 1, 2, 3, 3, 2, 1],
        }
    )
    expanded = aggregated.loc[aggregated.index.repeat(aggregated["count"])].copy()

    with pytest.warns(UserWarning, match="observations removed because of separation"):
        fit_weighted = pf.feglm(
            "y ~ x | fe",
            data=aggregated,
            family="logit",
            weights="count",
            weights_type="fweights",
            vcov="HC1",
            separation_check=["fe"],
            iwls_tol=1e-11,
        )
    with pytest.warns(UserWarning, match="observations removed because of separation"):
        fit_expanded = pf.feglm(
            "y ~ x | fe",
            data=expanded,
            family="logit",
            vcov="HC1",
            separation_check=["fe"],
            iwls_tol=1e-11,
        )

    assert fit_weighted._N == fit_expanded._N == 19
    assert fit_weighted._N_rows == 9
    assert fit_expanded._N_rows == 19
    assert fit_weighted._observation_weights.n_effective == 19
    np.testing.assert_allclose(fit_weighted.coef(), fit_expanded.coef(), atol=1e-10)
    np.testing.assert_allclose(fit_weighted._vcov, fit_expanded._vcov, atol=1e-10)


def test_aweights():
    data = pf.get_data()
    data["weights"] = np.ones(data.shape[0])

    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1", data=data, weights_type="aweights")
    fit3 = pf.feols("Y ~ X1", data=data, weights="weights", weights_type="aweights")

    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)
    np.testing.assert_allclose(fit1.tidy().values, fit3.tidy().values)
