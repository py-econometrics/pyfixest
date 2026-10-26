"""Compare varying-slope fixed effects with R fixest.

R fixest is the reference for coefficients, standard errors, observation
counts, degrees of freedom, and fit statistics of models with `f[z]` and
`f[[z]]` fixed effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf

fixest = importr("fixest")
stats = importr("stats")

# Both sides solve the fixed effects to 1e-10 or tighter, so coefficients agree
# to about 1e-8 and standard errors, built on the residuals, to about 1e-7.
COEF_ATOL = 1e-8
SE_ATOL = 1e-7
DEMEANER = pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12, fixef_maxiter=10_000)
R_FIXEF_TOL = 1e-10

# (pyfixest formula, fixest formula)
OLS_FORMULAS = [
    ("Y ~ X1 | f1[X2]", "Y ~ X1 | f1[X2]"),
    ("Y ~ X1 | f1[[X2]]", "Y ~ X1 | f1[[X2]]"),
    ("Y ~ 0 + X1 | f1[[X2]]", "Y ~ 0 + X1 | f1[[X2]]"),
    ("Y ~ X1 | f1[X2, Z1] + f2", "Y ~ X1 | f1[X2, Z1] + f2"),
    ("Y ~ X1 | f1[[X2, Z1]] + f2", "Y ~ X1 | f1[[X2, Z1]] + f2"),
    ("Y ~ X1 | f1 + f1[[X2]] + f2[[Z1]]", "Y ~ X1 | f1 + f1[[X2]] + f2[[Z1]]"),
    ("Y ~ X1 | f1:f2[X2]", "Y ~ X1 | f1^f2[X2]"),
    ("Y ~ X1 | f1[X2**2]", "Y ~ X1 | f1[I(X2^2)]"),
]
IV_FORMULAS = [
    ("Y ~ X2 + [X1 ~ Z1] | f1[Z2]", "Y ~ X2 | f1[Z2] | X1 ~ Z1"),
    ("Y ~ X2 + [X1 ~ Z1] | f1[[Z2]] + f2", "Y ~ X2 | f1[[Z2]] + f2 | X1 ~ Z1"),
]
VCOVS = ["iid", "hetero", {"CRV1": "f1"}, {"CRV1": "f2"}, {"CRV1": "f1+f2"}]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    data = pf.get_data(N=1000, seed=76540251, beta_type="2", error_type="2")
    # Missing slope values drop rows like any other variable.
    data.loc[[3, 17], "X2"] = np.nan
    return data


def _r_vcov(vcov: str | dict[str, str]):
    return ro.Formula("~" + vcov["CRV1"]) if isinstance(vcov, dict) else vcov


def _r_coefnames(fit_r) -> list[str]:
    names = ro.r("function(fit) names(coef(fit))")(fit_r)
    return [
        name.replace("(Intercept)", "Intercept").removeprefix("fit_") for name in names
    ]


def _assert_matches_fixest(fit, fit_r) -> None:
    # fixest lists instrumented regressors first.
    names = _r_coefnames(fit_r)
    assert sorted(fit.coef().index) == sorted(names), "coefficient names != fixest"
    assert fit.sample_info.n_obs == int(stats.nobs(fit_r)[0]), "n_obs != fixest"
    assert fit.variance_covariance.df_k == int(fixest.degrees_freedom(fit_r, "k")[0]), (
        "df_k != fixest"
    )
    np.testing.assert_allclose(
        fit.coef()[names], stats.coef(fit_r), rtol=0, atol=COEF_ATOL, err_msg="coef"
    )
    np.testing.assert_allclose(
        fit.se()[names], fixest.se(fit_r), rtol=0, atol=SE_ATOL, err_msg="se"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("vcov", VCOVS, ids=str)
@pytest.mark.parametrize(("fml", "fml_r"), OLS_FORMULAS + IV_FORMULAS)
def test_varying_slopes_against_fixest(data, fml, fml_r, vcov, weights):
    fit = pf.feols(fml, data=data, vcov=vcov, weights=weights, demeaner=DEMEANER)
    r_kwargs = {} if weights is None else {"weights": ro.Formula(f"~{weights}")}
    fit_r = fixest.feols(
        ro.Formula(fml_r),
        data=data,
        vcov=_r_vcov(vcov),
        fixef_tol=R_FIXEF_TOL,
        **r_kwargs,
    )

    _assert_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize(("fml", "fml_r"), OLS_FORMULAS)
def test_varying_slopes_fit_statistics_against_fixest(data, fml, fml_r):
    fit = pf.feols(fml, data=data, demeaner=DEMEANER)
    fit_r = fixest.feols(ro.Formula(fml_r), data=data, fixef_tol=R_FIXEF_TOL)
    r2 = fixest.r2(fit_r, ro.StrVector(["r2", "ar2", "wr2", "war2"]))

    np.testing.assert_allclose(
        [
            fit.fitstat.r2,
            fit.fitstat.adj_r2,
            fit.fitstat.r2_within,
            fit.fitstat.adj_r2_within,
        ],
        np.asarray(r2),
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("fixed_effects", "n_dropped"), [("g", 1), ("g[z]", 1), ("g[[z]]", 0)]
)
@pytest.mark.parametrize("zero_loading", [True, False])
def test_varying_slopes_singletons_against_fixest(
    fixed_effects, n_dropped, zero_loading
):
    """Only effects with intercepts perfectly fit a singleton (#1740).

    With a zero slope covariate the singleton row is informative, and dropping
    it would change the estimate substantially.
    """
    rng = np.random.default_rng(7)
    n_obs = 121
    g = np.r_[np.repeat(np.arange(20), 6), 20]
    z = rng.normal(size=n_obs)
    z[-1] = 0.0 if zero_loading else 1.5
    x = rng.normal(size=n_obs)
    x[-1] = 10.0
    y = 1 + 1.7 * x + 0.4 * g * z + rng.normal(size=n_obs)
    y[-1] = 50.0
    data = pd.DataFrame({"y": y, "x": x, "g": g, "z": z})
    fml = f"y ~ x | {fixed_effects}"

    if n_dropped:
        with pytest.warns(UserWarning, match="1 singleton"):
            fit = pf.feols(fml, data=data, vcov="iid", demeaner=DEMEANER)
    else:
        fit = pf.feols(fml, data=data, vcov="iid", demeaner=DEMEANER)
    fit_r = fixest.feols(
        ro.Formula(fml),
        data=data,
        vcov="iid",
        fixef_rm="singletons",
        fixef_tol=R_FIXEF_TOL,
    )

    assert fit.sample_info.dropped_by_stage.singleton == n_dropped
    _assert_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "f1"}], ids=str)
@pytest.mark.parametrize(
    ("fml", "fml_r"), [OLS_FORMULAS[0], OLS_FORMULAS[1], IV_FORMULAS[0]]
)
def test_varying_slopes_frequency_weights_against_fixest(data, fml, fml_r, vcov):
    """Frequency weights match fixest on the literally expanded rows."""
    rng = np.random.default_rng(20261018)
    data = data.dropna().reset_index(drop=True)
    data["fweights"] = rng.integers(1, 4, size=len(data))
    expanded = data.loc[data.index.repeat(data["fweights"])].reset_index(drop=True)

    fit = pf.feols(
        fml,
        data=data,
        vcov=vcov,
        weights="fweights",
        weights_type="fweights",
        demeaner=DEMEANER,
    )
    fit_r = fixest.feols(
        ro.Formula(fml_r), data=expanded, vcov=_r_vcov(vcov), fixef_tol=R_FIXEF_TOL
    )

    _assert_matches_fixest(fit, fit_r)


# GLM IRLS converges to 1e-10 in deviance on both sides; coefficients agree to
# about 1e-6 and standard errors, through the final working weights, to 1e-5.
GLM_COEF_ATOL = 1e-6
GLM_SE_ATOL = 1e-5
GLM_FORMULAS = [
    ("Y ~ X1 | f1[X2]", "Y ~ X1 | f1[X2]"),
    ("Y ~ X1 | f1[[X2]]", "Y ~ X1 | f1[[X2]]"),
    ("Y ~ X1 | f1[[X2, Z1]] + f2", "Y ~ X1 | f1[[X2, Z1]] + f2"),
]


@pytest.fixture(scope="module")
def data_glm() -> pd.DataFrame:
    data = pf.get_data(N=1000, seed=7651, beta_type="2", error_type="2", model="Fepois")
    data["Y_binary"] = (data["Y"] > data["Y"].median()).astype(float)
    data.loc[[3, 17], "X2"] = np.nan
    return data


def _assert_glm_matches_fixest(fit, fit_r) -> None:
    names = _r_coefnames(fit_r)
    assert sorted(fit.coef().index) == sorted(names), "coefficient names != fixest"
    assert fit.sample_info.n_obs == int(stats.nobs(fit_r)[0]), "n_obs != fixest"
    assert fit.variance_covariance.df_k == int(fixest.degrees_freedom(fit_r, "k")[0]), (
        "df_k != fixest"
    )
    np.testing.assert_allclose(
        fit.coef()[names],
        stats.coef(fit_r),
        rtol=0,
        atol=GLM_COEF_ATOL,
        err_msg="coef",
    )
    np.testing.assert_allclose(
        fit.se()[names], fixest.se(fit_r), rtol=0, atol=GLM_SE_ATOL, err_msg="se"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "f1"}], ids=str)
@pytest.mark.parametrize("family", ["poisson", "logit", "probit", "gaussian"], ids=str)
@pytest.mark.parametrize(("fml", "fml_r"), GLM_FORMULAS)
def test_varying_slopes_glm_against_fixest(data_glm, fml, fml_r, family, vcov, weights):
    if family == "poisson":
        fit = pf.fepois(
            fml,
            data=data_glm,
            vcov=vcov,
            weights=weights,
            demeaner=DEMEANER,
            iwls_tol=1e-10,
        )
    else:
        fml = fml.replace("Y ~", "Y_binary ~") if family != "gaussian" else fml
        fml_r = fml_r.replace("Y ~", "Y_binary ~") if family != "gaussian" else fml_r
        fit = pf.feglm(
            fml,
            data=data_glm,
            family=family,
            vcov=vcov,
            weights=weights,
            demeaner=DEMEANER,
            iwls_tol=1e-10,
        )
    r_kwargs = {} if weights is None else {"weights": ro.Formula(f"~{weights}")}
    if family == "gaussian":
        # Gaussian GLM inference follows OLS; see the fixest compatibility ledger.
        fit_r = fixest.feols(
            ro.Formula(fml_r),
            data=data_glm,
            vcov=_r_vcov(vcov),
            fixef_tol=R_FIXEF_TOL,
            **r_kwargs,
        )
    else:
        r_family = {
            "poisson": stats.poisson(),
            "logit": stats.binomial(link="logit"),
            "probit": stats.binomial(link="probit"),
        }[family]
        fit_r = fixest.feglm(
            ro.Formula(fml_r),
            data=data_glm,
            family=r_family,
            vcov=_r_vcov(vcov),
            fixef_tol=R_FIXEF_TOL,
            glm_tol=1e-10,
            **r_kwargs,
        )

    _assert_glm_matches_fixest(fit, fit_r)


def _separation_data(*, mixed_signs: bool) -> pd.DataFrame:
    """Group 0 has only zero outcomes; its slope covariate has mixed or one sign."""
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(6), 20)
    z = np.tile([-1.0, 1.0], 60) if mixed_signs else np.tile([0.5, 1.5], 60)
    x = rng.normal(size=120)
    y = rng.poisson(np.exp(0.2 + 0.3 * x + 0.2 * z)).astype(float)
    y[g == 0] = 0.0
    return pd.DataFrame({"y": y, "x": x, "g": g, "z": z})


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("fixed_effects", "n_separated"), [("g", 20), ("g[z]", 20), ("g[[z]]", 0)]
)
@pytest.mark.parametrize("separation_check", [["fe"], ["fe", "ir"]], ids=str)
def test_varying_slopes_separation_against_fixest(
    fixed_effects, n_separated, separation_check
):
    """Slope-only effects have finite estimates in all-zero groups (#1741).

    Their slope covariate takes both signs, so no coefficient can send every
    fitted mean of the group to zero, and fixest keeps the group.
    """
    data = _separation_data(mixed_signs=True)
    fml = f"y ~ x | {fixed_effects}"

    if n_separated:
        with pytest.warns(UserWarning, match=f"{n_separated} observations removed"):
            fit = pf.fepois(
                fml,
                data=data,
                vcov="iid",
                separation_check=separation_check,
                demeaner=DEMEANER,
                iwls_tol=1e-10,
            )
    else:
        fit = pf.fepois(
            fml,
            data=data,
            vcov="iid",
            separation_check=separation_check,
            demeaner=DEMEANER,
            iwls_tol=1e-10,
        )
    fit_r = fixest.fepois(
        ro.Formula(fml), data=data, vcov="iid", fixef_tol=R_FIXEF_TOL, glm_tol=1e-10
    )

    assert fit.sample_info.dropped_by_stage.separation == n_separated
    _assert_glm_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
def test_varying_slopes_one_signed_separation_differs_from_fixest():
    """A slope-only group with one-signed covariate and all-zero outcomes is separated.

    Sending its slope to minus infinity drives every fitted mean of the group to
    zero. The `ir` check drops the group, while fixest keeps it. The coefficients
    agree, but the observation counts and therefore the standard errors differ.
    """
    data = _separation_data(mixed_signs=False)
    fml = "y ~ x | g[[z]]"

    with pytest.warns(UserWarning, match="20 observations removed"):
        fit = pf.fepois(
            fml,
            data=data,
            vcov="iid",
            separation_check=["fe", "ir"],
            demeaner=DEMEANER,
            iwls_tol=1e-10,
        )
    fit_r = fixest.fepois(
        ro.Formula(fml), data=data, vcov="iid", fixef_tol=R_FIXEF_TOL, glm_tol=1e-10
    )

    assert fit.sample_info.dropped_by_stage.separation == 20
    assert fit.sample_info.n_obs == 100
    assert int(stats.nobs(fit_r)[0]) == 120
    names = _r_coefnames(fit_r)
    np.testing.assert_allclose(
        fit.coef()[names], stats.coef(fit_r), rtol=0, atol=GLM_COEF_ATOL
    )
    assert not np.allclose(fit.se()[names], fixest.se(fit_r), rtol=0, atol=1e-5)
