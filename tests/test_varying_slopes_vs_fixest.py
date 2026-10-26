"""Compare varying-slope fixed effects with R fixest.

R fixest is the reference for coefficients, standard errors, observation
counts, degrees of freedom, residuals, and fit statistics of models with `f[z]`
and `f[[z]]` fixed effects. The formulas live in `tests/_varying_slope_cases.py`,
which `tests/test_formula_parse.py` also checks at the parsing layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.errors import NonConvergenceError
from tests._varying_slope_cases import (
    GLM_SLOPE_CASES,
    IV_SLOPE_CASES,
    OLS_SLOPE_CASES,
    SlopeCase,
)

fixest = importr("fixest")
stats = importr("stats")

# Both sides solve the fixed effects to 1e-10 or tighter, so coefficients agree
# to about 1e-8 and standard errors, built on the residuals, to about 1e-7.
COEF_ATOL = 1e-8
SE_ATOL = 1e-7
# Residuals carry the fixed-effect solver error of both sides.
RESID_ATOL = 1e-7
DEMEANER = pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12, fixef_maxiter=10_000)
R_FIXEF_TOL = 1e-10
VCOVS = ["iid", "hetero", {"CRV1": "f1"}, {"CRV1": "f2"}, {"CRV1": "f1+f2"}]


def _inference_cases(cases: tuple[SlopeCase, ...]) -> list:
    """Cross representative formulas with every vcov and weights option."""
    params = []
    for case in cases:
        options = (
            [(vcov, weights) for vcov in VCOVS for weights in (None, "weights")]
            if case.full_matrix
            else [("iid", None)]
        )
        params.extend(
            pytest.param(case, vcov, weights, id=f"{case.fml}-{vcov}-{weights}")
            for vcov, weights in options
        )
    return params


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


def _assert_matches_fixest(
    fit, fit_r, *, coef_atol: float = COEF_ATOL, se_atol: float = SE_ATOL
) -> None:
    # fixest lists instrumented regressors first.
    names = _r_coefnames(fit_r)
    assert sorted(fit.coef().index) == sorted(names), "coefficient names != fixest"
    assert fit.sample_info.n_obs == int(stats.nobs(fit_r)[0]), "n_obs != fixest"
    assert fit.variance_covariance.df_k == int(fixest.degrees_freedom(fit_r, "k")[0]), (
        "df_k != fixest"
    )
    np.testing.assert_allclose(
        fit.coef()[names], stats.coef(fit_r), rtol=0, atol=coef_atol, err_msg="coef"
    )
    np.testing.assert_allclose(
        fit.se()[names], fixest.se(fit_r), rtol=0, atol=se_atol, err_msg="se"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("case", "vcov", "weights"), _inference_cases(OLS_SLOPE_CASES + IV_SLOPE_CASES)
)
def test_varying_slopes_against_fixest(data, case, vcov, weights):
    fit = pf.feols(case.fml, data=data, vcov=vcov, weights=weights, demeaner=DEMEANER)
    r_kwargs = {} if weights is None else {"weights": ro.Formula(f"~{weights}")}
    fit_r = fixest.feols(
        ro.Formula(case.fml_r),
        data=data,
        vcov=_r_vcov(vcov),
        fixef_tol=R_FIXEF_TOL,
        **r_kwargs,
    )

    _assert_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize("case", OLS_SLOPE_CASES, ids=lambda case: case.fml)
def test_varying_slopes_fit_statistics_against_fixest(data, case):
    fit = pf.feols(case.fml, data=data, demeaner=DEMEANER)
    fit_r = fixest.feols(ro.Formula(case.fml_r), data=data, fixef_tol=R_FIXEF_TOL)
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
        err_msg="r2",
    )
    np.testing.assert_allclose(
        fit.resid()[:5],
        np.asarray(stats.residuals(fit_r))[:5],
        rtol=0,
        atol=RESID_ATOL,
        err_msg="resid",
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
@pytest.mark.parametrize("vcov", ["iid", {"CRV1": "g"}], ids=str)
@pytest.mark.parametrize("fixed_effects", ["g[z]", "g[[z]]"])
def test_varying_slopes_zero_slope_group_against_fixest(fixed_effects, vcov):
    """A group whose slope covariate is all zero has an unidentified slope."""
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(15), 20)
    z = rng.normal(size=g.size)
    z[g == 0] = 0.0
    x = rng.normal(size=g.size)
    y = 1 + x + 0.05 * g * z + rng.normal(size=g.size)
    data = pd.DataFrame({"y": y, "x": x, "g": g, "z": z})
    fml = f"y ~ x | {fixed_effects}"

    fit = pf.feols(fml, data=data, vcov=vcov, demeaner=DEMEANER)
    fit_r = fixest.feols(
        ro.Formula(fml), data=data, vcov=_r_vcov(vcov), fixef_tol=R_FIXEF_TOL
    )

    _assert_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize("slope_type", ["float", "int"])
@pytest.mark.parametrize("level_type", ["float", "str", "categorical"])
def test_varying_slopes_column_types_against_fixest(data, level_type, slope_type):
    """Levels and slopes accept the same column types as other model terms."""
    data = data.dropna().reset_index(drop=True)
    levels = data["f1"].astype(int)
    data["f1"] = {
        "float": data["f1"],
        "str": levels.astype(str),
        "categorical": pd.Categorical(levels),
    }[level_type]
    data["s"] = {
        "float": data["X2"],
        "int": np.round(3 * data["X2"]).astype(int),
    }[slope_type]
    fml = "Y ~ X1 | f1[s] + f2[[s]]"

    fit = pf.feols(fml, data=data, demeaner=DEMEANER)
    fit_r = fixest.feols(ro.Formula(fml), data=data, fixef_tol=R_FIXEF_TOL)

    _assert_matches_fixest(fit, fit_r)


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    ("fml", "single_fmls"),
    [
        (
            "Y ~ X1 | sw(f1, f1[X2], f1[[X2]], f1[X2, Z1])",
            [
                "Y ~ X1 | f1",
                "Y ~ X1 | f1[X2]",
                "Y ~ X1 | f1[[X2]]",
                "Y ~ X1 | f1[X2, Z1]",
            ],
        ),
        (
            "Y ~ X1 | csw0(f1[[X2]], f2)",
            ["Y ~ X1", "Y ~ X1 | f1[[X2]]", "Y ~ X1 | f1[[X2]] + f2"],
        ),
    ],
)
@pytest.mark.parametrize("estimator", ["feols", "fepois"])
def test_varying_slopes_multiple_estimation_against_fixest(
    data, data_glm, fml, single_fmls, estimator
):
    """Each model of a multiple estimation matches its single fixest fit.

    Single fits are the reference because fixest 0.14.0 multiple estimation
    drops the intercept of slope-only models (see the compatibility ledger).
    Models with different slopes on the same levels must not share a demeaning
    cache entry.
    """
    if estimator == "feols":
        multi = pf.feols(fml, data=data, demeaner=DEMEANER)
        fits_r = [
            fixest.feols(ro.Formula(single), data=data, fixef_tol=R_FIXEF_TOL)
            for single in single_fmls
        ]
        tolerances = {}
    else:
        multi = pf.fepois(fml, data=data_glm, demeaner=DEMEANER, iwls_tol=1e-10)
        fits_r = [
            fixest.fepois(
                ro.Formula(single), data=data_glm, fixef_tol=R_FIXEF_TOL, glm_tol=1e-10
            )
            for single in single_fmls
        ]
        tolerances = {"coef_atol": GLM_COEF_ATOL, "se_atol": GLM_SE_ATOL}

    for fit, fit_r in zip(multi.to_list(), fits_r, strict=True):
        _assert_matches_fixest(fit, fit_r, **tolerances)


@pytest.mark.against_r_core
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "f1"}], ids=str)
@pytest.mark.parametrize(
    ("fml", "fml_r"),
    [
        (IV_SLOPE_CASES[0].fml, IV_SLOPE_CASES[0].fml_r),
        (OLS_SLOPE_CASES[0].fml, OLS_SLOPE_CASES[0].fml_r),
        (OLS_SLOPE_CASES[1].fml, OLS_SLOPE_CASES[1].fml_r),
    ],
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
# The relative IRLS stopping rule of 1e-10 on deviances of order 1e3.
GLM_DEVIANCE_ATOL = 1e-7


def _glm_cases() -> list:
    params = []
    for case in GLM_SLOPE_CASES:
        for family in ("poisson", "logit", "probit", "gaussian"):
            marks = (
                pytest.mark.xfail(
                    # Weighted probit fails in step-halving instead.
                    raises=(NonConvergenceError, RuntimeError),
                    strict=True,
                    reason="Binomial IRLS does not converge with interacted slopes",
                )
                if family in ("logit", "probit") and ":" in case.fml
                else ()
            )
            params.append(
                pytest.param(case, family, marks=marks, id=f"{case.fml}-{family}")
            )
    return params


@pytest.fixture(scope="module")
def data_glm() -> pd.DataFrame:
    data = pf.get_data(N=1000, seed=7651, beta_type="2", error_type="2", model="Fepois")
    data["Y_binary"] = (data["Y"] > data["Y"].median()).astype(float)
    data.loc[[3, 17], "X2"] = np.nan
    return data


@pytest.mark.against_r_core
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "f1"}], ids=str)
@pytest.mark.parametrize(("case", "family"), _glm_cases())
def test_varying_slopes_glm_against_fixest(data_glm, case, family, vcov, weights):
    fml, fml_r = case.fml, case.fml_r
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

    _assert_matches_fixest(fit, fit_r, coef_atol=GLM_COEF_ATOL, se_atol=GLM_SE_ATOL)
    if family != "gaussian":
        np.testing.assert_allclose(
            fit.fitstat.deviance,
            float(np.asarray(fit_r.rx2("deviance"))[0]),
            rtol=0,
            atol=GLM_DEVIANCE_ATOL,
            err_msg="deviance",
        )
    if family == "poisson":
        # pyfixest defines the log-likelihood for Poisson fits only.
        np.testing.assert_allclose(
            fit.fitstat.loglik,
            float(np.asarray(fit_r.rx2("loglik"))[0]),
            rtol=0,
            atol=GLM_DEVIANCE_ATOL,
            err_msg="loglik",
        )


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
    _assert_matches_fixest(fit, fit_r, coef_atol=GLM_COEF_ATOL, se_atol=GLM_SE_ATOL)


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
