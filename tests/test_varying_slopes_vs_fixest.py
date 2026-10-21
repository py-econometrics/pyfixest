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
