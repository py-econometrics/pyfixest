"""Compare public estimator results with the pinned pyfixest release.

This suite mirrors the structure of `tests/test_vs_fixest.py`: the same data
fixtures, the same parametrization over the shared formula tuples in
`tests/_feols_test_cases.py`, and one test per estimator. Only the reference
side differs. Instead of calling R through rpy2, each `baseline.check(...)`
compares against a value recorded by running *this file* under the pinned
release, which makes the suite fast enough for an edit loop.

It is a regression alarm, not an external correctness oracle: a bug already
present in the pinned release is recorded, not caught. `test_vs_fixest.py`
remains the authoritative evidence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.utils.utils import ssc
from tests._feols_test_cases import (
    FEOLS_FORMULA_F3_CASES,
    FEOLS_FORMULAS,
    build_feols_data_variants,
    glm_fmls,
    iv_fmls,
    ols_fmls,
    ssc_formula_vcov_dropna_cases,
)
from tests._feols_test_cases import (
    convert_f3 as _convert_f3,
)
from tests._release_baseline import (
    RELEASE_VERSION,
    Baseline,
    data_digest,
    newest_release_tag,
    release_number,
)

INFERENCE = ["iid", "hetero", {"CRV1": "group_id"}]

# Behaviour that changed after the pinned release. Each entry names its source
# so a widened tolerance or a skipped quantity stays reviewable; everything not
# listed here is compared at the shared default in _release_baseline.py.
GLM_GAUSSIAN_MATCHES_FEOLS = (
    "feglm(family='gaussian') now matches feols exactly, which changed both its "
    "small-sample correction and its confidence-interval reference distribution "
    "(changelog: GLM API and Behavior)"
)
GLM_BINOMIAL_START_VALUES = (
    "logit and probit IRLS initialise at new values (changelog: GLM API and Behavior)"
)
FEPOIS_IRLS_ACCELERATION = (
    "fepois gained step-halving, inner-tolerance tightening and warm starts, "
    "which move the intermediate IRLS weights (changelog: GLM API and Behavior)"
)
QUANTREG_T_REFERENCE = (
    "quantreg confidence intervals now use the t reference distribution "
    "instead of the normal"
)
WEIGHTED_RMSE_MATCHES_FIXEST = (
    "the RMSE of weighted fits divides the weighted sum of squared residuals by "
    "the sum of the weights instead of the number of observations, as fixest"
)

# The IRLS rewrite moves fepois' converged point slightly; the measured drift
# across this matrix stays below 2e-6.
FEPOIS_TOLERANCE = {"rtol": 1e-5, "reason": FEPOIS_IRLS_ACCELERATION}
# New logit/probit start values, plus an absolute floor: a confidence-interval
# bound can land near zero while the coefficient it belongs to does not, and a
# purely relative bound is meaningless there.
GLM_BINOMIAL_TOLERANCE = {
    "rtol": 1e-4,
    "atol": 1e-6,
    "reason": GLM_BINOMIAL_START_VALUES,
}

# f3 dtype handling is an input-parsing concern, so it is crossed once rather
# than against every inference and weighting combination.
FEOLS_F3_DTYPE_CASES = [
    (fml, f3_type) for fml, f3_type in FEOLS_FORMULA_F3_CASES if f3_type != "str"
]

# The small-sample options only rescale the vcov and df_t, so the full option
# grid is crossed with representative formulas instead of all of them: no fixed
# effects, one, two with missing values and two-way clustering, and interacted.
SSC_CASES = [
    ("Y ~ X1 + X2 + f1", False, "hetero"),
    ("Y ~ X1 + X2 | f1", False, "f1"),
    ("Y ~ X1 + X2 | f1 + f2", True, "f1+f2"),
    ("Y ~ X1 + X2 | f1:f2", False, "iid"),
]
assert set(SSC_CASES) <= set(ssc_formula_vcov_dropna_cases), (
    "SSC_CASES must remain a subset of the cases test_vs_fixest.py checks against R"
)


@pytest.fixture(scope="module")
def data_feols():
    data = pf.get_data(N=1000, seed=76540251, beta_type="2", error_type="2")
    rng = np.random.default_rng(20260511)
    data["fweights"] = rng.integers(1, 5, data.shape[0]).astype(float)
    return data


@pytest.fixture(scope="module")
def data_feols_variants(data_feols):
    return build_feols_data_variants(data_feols)


@pytest.fixture(scope="module")
def data_fepois():
    data = pf.get_data(N=1000, seed=7651, beta_type="2", error_type="2", model="Fepois")
    data.where(data != "nan", np.nan, inplace=True)
    data = _convert_f3(data, "str")
    rng = np.random.default_rng(20260511)
    data["offset_var"] = np.log(rng.uniform(0.5, 3.0, data.shape[0]))
    data.iloc[10, data.columns.get_loc("offset_var")] = np.nan
    # Binary outcome for logit/probit, as in test_vs_fixest.py.
    data["Y_bin"] = (data["Y"] > 0).astype(int)
    return data


@pytest.fixture(scope="module")
def ssc_data():
    data = {}
    for model, data_model in [("feols", "Feols"), ("fepois", "Fepois")]:
        base = pf.get_data(model=data_model)
        data[(model, False)] = base
        data[(model, True)] = base.dropna()
    return data


@pytest.fixture(scope="module")
def data_quantreg():
    # As in test_quantreg.py, except that X2 always enters the outcome: the
    # data set is fixed here rather than rebuilt per formula.
    data = pf.get_data(N=5_000, seed=3131)
    rng = np.random.default_rng(3993)
    data["Y"] = 1 + 2 * data["X1"] + 3 * data["X2"] + rng.normal(size=len(data))
    return data


def _covariance_field(mod, name: str):
    """Read a `VarianceCovariance` field, or its private legacy attribute.

    The pinned release predates `fit.variance_covariance` and stores `_vcov`, `_df_k`,
    and `_df_t` on the model.
    """
    if hasattr(mod, "variance_covariance"):
        return getattr(mod.variance_covariance, name)
    return getattr(mod, f"_{name}")


def _check_structure(baseline: Baseline, mod) -> None:
    baseline.check_exact("coefnames", list(mod._coefnames))
    # The pinned release predates `EstimationSample`; `_N` was its `n_obs`.
    nobs = mod.sample_info.n_obs if hasattr(mod, "sample_info") else mod._N
    baseline.check_exact("nobs", int(nobs))
    baseline.check_exact("df_k", int(_covariance_field(mod, "df_k")))
    baseline.check_exact("df_t", int(_covariance_field(mod, "df_t")))


def _fit_statistic(mod, name: str, legacy: str) -> float:
    """Read a fit statistic; the pinned release predates `FitStatistics`."""
    if hasattr(mod, "fitstat"):
        return getattr(mod.fitstat, name)
    return getattr(mod, legacy)


def _check_fit(baseline: Baseline, mod, *, confint: bool = True) -> None:
    """Check the full estimate and inference vectors."""
    _check_structure(baseline, mod)
    baseline.check("coef", mod.coef())
    baseline.check("se", mod.se())
    baseline.check("tstat", mod.tstat())
    baseline.check("pvalue", mod.pvalue())
    if confint:
        baseline.check("confint", mod.confint())
    # se covers the vcov diagonal; one norm keeps the off-diagonal block in
    # scope without recording an O(k^2) matrix per case.
    vcov = np.asarray(_covariance_field(mod, "vcov"))
    baseline.check("vcov_offdiag", np.linalg.norm(vcov - np.diag(np.diag(vcov))))


def _check_fit_at_x1(baseline: Baseline, mod, **tolerance) -> None:
    """Check the IRLS estimators at `X1`, as test_vs_fixest.py does.

    Some `ols_fmls` entries are near-singular under a Poisson likelihood and
    produce non-finite standard errors for the interacted columns, so both
    suites compare the well-conditioned `X1` row rather than the full vector.
    The structural check still covers every coefficient name.
    """
    _check_structure(baseline, mod)
    baseline.check("coef", mod.coef().xs("X1"), **tolerance)
    baseline.check("se", mod.se().xs("X1"), **tolerance)
    baseline.check("tstat", mod.tstat().xs("X1"), **tolerance)
    baseline.check("pvalue", mod.pvalue().xs("X1"), **tolerance)
    baseline.check("confint", mod.confint().xs("X1"), **tolerance)


def test_release_pin_is_current():
    """Nudge when a newer release exists than the one the baseline is pinned to.

    Rolling the pin is what brings back the comparisons that the documented
    differences above currently skip, so this warns rather than failing.
    """
    newest = newest_release_tag()
    pinned = release_number(RELEASE_VERSION)
    if newest is None or pinned is None:
        pytest.skip("no comparable release tag available in this checkout")
    if release_number(newest) > pinned:
        warnings.warn(
            f"The release contract is pinned to pyfixest {RELEASE_VERSION}, but "
            f"{newest} has been released. Roll it with "
            "`pixi run roll-release-baseline`, then re-audit the documented "
            "differences at the top of this file.",
            UserWarning,
            stacklevel=1,
        )


def test_release_data_is_unchanged(data_feols, data_fepois, ssc_data, baseline):
    """Guard against a change to the shared DGP silently rebasing the suite."""
    baseline.check_exact("data_feols", data_digest(data_feols))
    baseline.check_exact("data_fepois", data_digest(data_fepois))
    baseline.check_exact("ssc_feols", data_digest(ssc_data[("feols", False)]))
    baseline.check_exact("ssc_fepois", data_digest(ssc_data[("fepois", False)]))


@pytest.mark.parametrize(
    "case",
    [
        "iid",
        "implicit_intercept",
        "fixed_effects",
        "two_clusters",
        "two_clusters_conventional",
        "split_full",
        "split_group",
        "iv",
        "poisson",
    ],
)
def test_legacy_table_metadata(case, data_feols, data_fepois, baseline):
    """Compare the legacy attributes table integrations read from fitted models."""
    if case == "poisson":
        mod = pf.fepois("Y ~ 1 + X1", data_fepois)
    elif case == "iv":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = pf.feols("Y ~ X2 | X1 ~ Z1", data_feols)
    elif case in ("split_full", "split_group"):
        fits = pf.feols("Y ~ 1 + X1", data_feols, fsplit="f2").to_list()
        mod = fits[0 if case == "split_full" else 1]
    else:
        formula = (
            "Y ~ X1"
            if case == "implicit_intercept"
            else "Y ~ 1 + X1"
            if case == "iid"
            else "Y ~ X1 | f1"
        )
        vcov = "iid"
        if case == "fixed_effects":
            vcov = {"CRV1": "f1"}
        elif case.startswith("two_clusters"):
            vcov = {"CRV1": "f1+f2"}
        options = (
            {"ssc": ssc(G_df="conventional")} if case.endswith("conventional") else {}
        )
        mod = pf.feols(formula, data_feols, vcov=vcov, **options)

    if case in ("iv", "implicit_intercept"):
        baseline.skip(
            "_fml",
            reason="formula rendering now makes intercepts and IV syntax explicit",
        )
    else:
        baseline.check_exact("_fml", mod._fml)
    for name in (
        "_method",
        "_fixef",
        "_has_fixef",
        "_depvar",
        "_sample_split_var",
        "_N",
        "_vcov_type",
        "_clustervar",
        "_G",
    ):
        baseline.check_exact(name, getattr(mod, name))
    split_value = mod._sample_split_value
    baseline.check_exact("split_value_type", type(split_value).__name__)
    baseline.check_exact("split_value_repr", repr(split_value))
    for name in (
        "_vcov",
        "_pvalue",
        "_rmse",
        "_r2",
        "_r2_within",
        "_adj_r2",
        "_adj_r2_within",
    ):
        value = getattr(mod, name)
        baseline.check_exact(f"{name}_shape", list(np.shape(value)))
        baseline.check(name, value, **(FEPOIS_TOLERANCE if case == "poisson" else {}))
    if case == "poisson":
        baseline.check("deviance", mod.deviance, **FEPOIS_TOLERANCE)
    else:
        baseline.check_exact("deviance", mod.deviance)
    if case == "iv":
        baseline.check("_f_stat_1st_stage", mod._f_stat_1st_stage)


@pytest.mark.parametrize("dropna", [False, True])
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("fml", FEOLS_FORMULAS)
def test_single_fit_feols(
    data_feols_variants, dropna, inference, weights, fml, baseline
):
    data = data_feols_variants[(dropna, "str")]

    mod = pf.feols(
        fml=baseline.fml(fml),
        data=data,
        vcov=inference,
        weights=weights,
        ssc=ssc(k_adj=True, G_adj=True),
    )

    _check_fit(baseline, mod)
    baseline.check("resid", mod.resid()[0:5])
    baseline.check("predict", mod.predict()[0:5])
    if weights is None:
        baseline.check("rmse", _fit_statistic(mod, "rmse", "_rmse"))
    else:
        baseline.skip("rmse", reason=WEIGHTED_RMSE_MATCHES_FIXEST)
    for name in ("r2", "adj_r2", "r2_within", "adj_r2_within"):
        baseline.check(name, _fit_statistic(mod, name, f"_{name}"))


@pytest.mark.parametrize("fml,f3_type", FEOLS_F3_DTYPE_CASES)
def test_single_fit_feols_f3_dtypes(data_feols_variants, fml, f3_type, baseline):
    """Cross the fixed-effect dtype axis once, at a single inference setting."""
    mod = pf.feols(
        fml=baseline.fml(fml),
        data=data_feols_variants[(False, f3_type)],
        vcov="iid",
        ssc=ssc(k_adj=True, G_adj=True),
    )

    _check_fit(baseline, mod)


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("fml", ols_fmls)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("offset", [False, True])
def test_single_fit_fepois(data_fepois, inference, fml, weights, offset, baseline):
    mod = pf.fepois(
        fml=baseline.fml(fml),
        data=data_fepois,
        vcov=inference,
        ssc=ssc(k_adj=True, G_adj=True),
        iwls_tol=1e-10,
        iwls_maxiter=100,
        weights=weights,
        offset="offset_var" if offset else None,
    )

    _check_fit_at_x1(baseline, mod, **FEPOIS_TOLERANCE)
    baseline.check("deviance", _fit_statistic(mod, "deviance", "deviance"))
    baseline.check("loglik", _fit_statistic(mod, "loglik", "_loglik"))
    baseline.check("pearson_chi2", _fit_statistic(mod, "pearson_chi2", "_pearson_chi2"))
    baseline.check("resid", mod.resid()[0:5], **FEPOIS_TOLERANCE)
    baseline.check(
        "irls_weights",
        (
            mod.working_state.working_weights
            if hasattr(mod, "working_state")
            else mod._irls_weights
        ).flatten()[0:5],
        **FEPOIS_TOLERANCE,
    )


# fweights are only exercised here; test_vs_fixest.py crosses IV with
# [None, "weights"] because fixest has no frequency-weight counterpart.
IV_WEIGHTS = [(None, "aweights"), ("weights", "aweights"), ("fweights", "fweights")]


@pytest.mark.parametrize("weights,weights_type", IV_WEIGHTS)
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("fml", iv_fmls)
def test_single_fit_iv(
    data_feols_variants, weights, weights_type, inference, fml, baseline
):
    mod = pf.feols(
        fml=baseline.fml(fml),
        data=data_feols_variants[(False, "str")],
        vcov=inference,
        weights=weights,
        weights_type=weights_type,
        ssc=ssc(k_adj=True, G_adj=True),
    )

    _check_fit(baseline, mod)
    baseline.check("resid", mod.resid()[0:5])


# The pinned release predates weighted GLM and the poisson family on feglm, so
# those fits have no counterpart to record. test_vs_fixest.py covers both
# against R, and test_single_fit_fepois covers the Poisson numerics here.
@pytest.mark.parametrize("family", ["logit", "probit", "gaussian"])
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("fml", glm_fmls)
def test_single_fit_feglm(data_fepois, family, inference, fml, baseline):
    fml = fml.replace("Y", "Y_bin", 1) if family in ("logit", "probit") else fml

    mod = pf.feglm(
        fml=baseline.fml(fml),
        data=data_fepois,
        family=family,
        vcov=inference,
        ssc=ssc(k_adj=True, G_adj=True),
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )

    baseline.check("deviance", _fit_statistic(mod, "deviance", "deviance"))

    # `resid()` returned the IRLS working residual in the pinned release and
    # returns the response residual now, so both are compared through the
    # attributes that mean the same thing in either version.
    tolerance = {} if family == "gaussian" else GLM_BINOMIAL_TOLERANCE
    baseline.check(
        "resid_response",
        (
            mod.working_state.response_residuals
            if hasattr(mod, "working_state")
            else mod._u_hat_response
        )[0:5],
        **tolerance,
    )
    baseline.check(
        "resid_working",
        (
            mod.working_state.working_residuals
            if hasattr(mod, "working_state")
            else mod._u_hat_working
        )[0:5],
        **tolerance,
    )

    if family == "gaussian":
        # Coefficients and residuals still agree to machine precision; only the
        # variance-derived quantities moved.
        _check_structure(baseline, mod)
        baseline.check("coef", mod.coef().xs("X1"))
        for name in ("se", "tstat", "pvalue", "confint"):
            baseline.skip(name, reason=GLM_GAUSSIAN_MATCHES_FEOLS)
    else:
        _check_fit_at_x1(baseline, mod, **GLM_BINOMIAL_TOLERANCE)


@pytest.mark.parametrize("fml,dropna,vcov", SSC_CASES)
@pytest.mark.parametrize("weights", [None, "weights"])
@pytest.mark.parametrize("k_adj", [True, False])
@pytest.mark.parametrize("G_adj", [True, False])
@pytest.mark.parametrize("k_fixef", ["full", "none", "nonnested"])
@pytest.mark.parametrize("model", ["feols", "fepois"])
def test_ssc(
    ssc_data, fml, dropna, weights, vcov, k_adj, G_adj, k_fixef, model, baseline
):
    kwargs = {"iwls_tol": 1e-10, "iwls_maxiter": 100} if model == "fepois" else {}

    mod = getattr(pf, model)(
        fml=baseline.fml(fml),
        data=ssc_data[(model, dropna)],
        vcov=vcov if vcov in ("iid", "hetero") else {"CRV1": vcov},
        weights=weights,
        ssc=ssc(k_adj=k_adj, G_adj=G_adj, G_df="min", k_fixef=k_fixef),
        **kwargs,
    )

    _check_fit(baseline, mod)


@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 + X2"])
@pytest.mark.parametrize("quantile", [0.02, 0.35, 0.5, 0.9])
@pytest.mark.parametrize("method", ["fn", "pfn"])
@pytest.mark.parametrize("vcov", ["nid", {"CRV1": "group_id"}])
def test_quantreg(data_quantreg, fml, quantile, method, vcov, baseline):
    mod = pf.quantreg(
        fml,
        data=data_quantreg,
        vcov=vcov,
        quantile=quantile,
        method=method,
        tol=1e-6,
        ssc=ssc(k_adj=False, G_adj=False),
        seed=83838,
    )

    _check_fit(baseline, mod, confint=False)
    baseline.skip("confint", reason=QUANTREG_T_REFERENCE)
    baseline.check("resid", mod.resid()[0:5])


def _check_legacy_metadata(baseline, mod):
    """Check an explicit release inventory, independent of the alias registry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name in (
            "_N_rows",
            "n_separation_na",
            "_weights_name",
            "_weights_type",
            "_has_weights",
            "_sample_split_var",
            "_n_fe",
        ):
            baseline.check_exact(name, getattr(mod, name))
        # 0.60's set omits GLM separation, which the current sample includes.
        if mod.n_separation_na == 0:
            baseline.check_exact("_na_index", sorted(mod._na_index))
            baseline.check_exact("_na_index_type", type(mod._na_index).__name__)
        baseline.skip(
            "_k_fe",
            reason="The legacy encoded Series is removed; use "
            "fit.fixef_counts.n_levels_by_fe for final-sample level counts.",
        )


@pytest.mark.parametrize(
    "vcov,G_df,weights,weights_type,fixef,split,rhs",
    [
        ({"CRV1": "group_id"}, "min", "weights", "aweights", " | f1", None, "X1 + X2"),
        (
            {"CRV1": "group_id+f2"},
            "conventional",
            "weights",
            "aweights",
            " | f2 + f1",
            None,
            "X1 + X2",
        ),
        ("iid", "min", None, "aweights", "", None, "X1 + X2"),
        ("hetero", "min", "fweights", "fweights", " | f1", "f2", "X1 + X2"),
        ("iid", "min", "weights", "aweights", " | f1", None, "0"),
    ],
)
def test_deprecated_attributes_feols(
    data_feols_variants, vcov, G_df, weights, weights_type, fixef, split, rhs, baseline
):
    mod = pf.feols(
        fml=f"Y ~ {rhs}{fixef}",
        data=data_feols_variants[(True, "str")],
        vcov=vcov,
        weights=weights,
        weights_type=weights_type,
        split=split,
        ssc=ssc(G_df=G_df),
    )
    if split:
        mod = mod.to_list()[0]
    if fixef:
        mod.fixef()

    _check_legacy_metadata(baseline, mod)
    if rhs != "0":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            baseline.check_exact("G", [int(g) for g in mod._G])


@pytest.mark.parametrize("weights,weights_type", IV_WEIGHTS)
def test_deprecated_attributes_iv(data_feols_variants, weights, weights_type, baseline):
    mod = pf.feols(
        fml="Y ~ X2 | f1 | X1 ~ Z1",
        data=data_feols_variants[(True, "str")],
        vcov={"CRV1": "group_id"},
        weights=weights,
        weights_type=weights_type,
    )

    _check_legacy_metadata(baseline, mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        baseline.check_exact("G", [int(g) for g in mod._G])


@pytest.mark.parametrize(
    "weights,separated", [(None, False), ("weights", False), (None, True)]
)
def test_deprecated_attributes_fepois(data_fepois, weights, separated, baseline):
    data = data_fepois.copy()
    if separated:
        group = data["f1"].dropna().iloc[0]
        data.loc[data["f1"] == group, "Y"] = 0
    mod = pf.fepois(
        fml="Y ~ X1 + X2 | f1",
        data=data,
        vcov={"CRV1": "group_id"},
        weights=weights,
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )
    mod.fixef()
    _check_legacy_metadata(baseline, mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        baseline.check_exact("separation_check", mod.separation_check)
        baseline.check("_pearson_chi2", mod._pearson_chi2, **FEPOIS_TOLERANCE)
        for name in ("_loglik_null", "_pseudo_r2"):
            value = getattr(mod, name)
            if weights is None:
                assert np.isfinite(value), name
                baseline.check(name, value, **FEPOIS_TOLERANCE)
            else:
                assert value is None, name
                baseline.check_exact(name, value)


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_deprecated_attributes_feglm(data_fepois, family, baseline):
    mod = pf.feglm(
        fml="Y_bin ~ X1 + X2 | f1",
        data=data_fepois,
        family=family,
        vcov="hetero",
        iwls_tol=1e-10,
        iwls_maxiter=100,
    )
    mod.fixef()
    _check_legacy_metadata(baseline, mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        baseline.check_exact("separation_check", mod.separation_check)


# Explicit inventory: this module must also run with the 0.60 wheel, which does
# not have the current alias registry. The registry test checks for omissions.
ALIAS_EXACT = {
    "Feols": (
        "_fml",
        "_method",
        "_is_iv",
        "_model_name",
        "_fixef",
        "_has_fixef",
        "_icovars",
        "_depvar",
        "_sample_split_var",
        "_drop_singletons",
        "_drop_intercept",
        "_store_data",
        "_copy_data",
        "_lean",
        "_context",
        "_offset_name",
        "_ssc_dict",
        "_solver",
        "_has_weights",
        "_weights_name",
        "_weights_type",
        "_N",
        "_N_rows",
        "n_separation_na",
        "_n_fe",
        "_vcov_type",
        "_vcov_type_detail",
        "_is_clustered",
        "_clustervar",
        "_G",
        "_df_k",
        "_df_t",
        "_dfn",
        "_dfd",
        "_collin_vars",
        "_collin_index",
        "_support_hac_inference",
        "_support_decomposition",
        "_supports_wildboottest",
        "_supports_cluster_causal_variance",
        "_support_crv3_inference",
        "deviance",
    ),
    "Feiv": (
        "_non_exo_instruments",
        "_p_iv",
        "_collin_vars_z",
        "_collin_index_z",
        "_support_iid_inference",
    ),
    "Feglm": (
        "_accelerate",
        "maxiter",
        "tol",
        "separation_check",
        "_support_iid_inference",
    ),
    "Quantreg": ("_quantile", "_quantile_tol", "_quantile_maxiter", "_seed"),
}
ALIAS_NUMERIC = {
    "Feols": (
        "_collin_tol",
        "_vcov",
        "_se",
        "_tstat",
        "_pvalue",
        "_conf_int",
        "_rmse",
        "_r2",
        "_r2_within",
        "_adj_r2",
        "_adj_r2_within",
        "_wald_statistic",
        "_f_statistic",
        "_p_value",
    ),
    "Feiv": ("_f_stat_1st_stage", "_p_value_1st_stage", "_eff_F"),
    "Feglm": ("deviance",),
    "Fepois": ("_loglik", "_loglik_null", "_pseudo_r2", "_pearson_chi2"),
}
ALIAS_SPECIAL = {
    "_na_index": "Compare the sorted set and its type in the existing release cases.",
    "_sample_split_value": "Compare the split value separately; the unsplit sentinel changed.",
    "_model_1st_stage": "Compare the first-stage model's coefficients, not its identity.",
    "get_performance": "Compare the callable's result and resulting fit statistics.",
    "_demeaner": "Compare the backend and options on a fit with an explicit MAP backend.",
    "_fixef_tol": "Compare on a fit with an explicit MAP backend.",
    "_fixef_maxiter": "Compare on a fit with an explicit MAP backend.",
    "_ritest_sample_stat": "Compare after running randomization inference.",
    "_res_cohort_eventtime_dict": "Compare saturated event-study design separately.",
    "_yname": "Compare saturated event-study design separately.",
    "_gname": "Compare saturated event-study design separately.",
    "_tname": "Compare saturated event-study design separately.",
    "_idname": "Compare saturated event-study design separately.",
    "_att": "Compare saturated event-study design separately.",
}
ALIAS_EXCEPTIONS = {
    "_model_spec": "The formulaic specification and fixed-effect encoding changed.",
    "_r2_adj": "This compatibility name was added for table integrations after 0.60.",
    "_ritest_statistics": "The resampling implementation produces different draws for the same seed.",
    "_ritest_pvalue": "The p-value depends on resampling draws that changed since 0.60.",
}


def test_legacy_alias_release_inventory(data_feols, baseline):
    """Compare alias values on representative fits with the released wheel."""
    data = data_feols.dropna().reset_index(drop=True)
    data["Y_bin"] = (data.Y > 0).astype(int)
    data["Y_count"] = np.random.default_rng(2026).poisson(np.exp(0.3 * data.X1))
    fits = {
        "Feols": pf.feols("Y ~ X1 + X2 | f1", data, vcov={"CRV1": "f1"}),
        "Feiv": pf.feols("Y ~ X2 | f1 | X1 ~ Z1", data, vcov={"CRV1": "f1"}),
        "Feglm": pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit"),
        "Fepois": pf.fepois("Y_count ~ X1 + X2 | f1", data),
        "Quantreg": pf.quantreg("Y ~ X1 + X2", data, quantile=0.4, vcov="nid"),
    }
    fits["Feiv"].IV_Diag()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for kind, names in ALIAS_EXACT.items():
            for name in names:
                baseline.check_exact(f"{kind}.{name}", getattr(fits[kind], name))
        for kind, names in ALIAS_NUMERIC.items():
            for name in names:
                tolerance = FEPOIS_TOLERANCE if kind == "Fepois" else {}
                if kind == "Feglm":
                    tolerance = GLM_BINOMIAL_TOLERANCE
                baseline.check(f"{kind}.{name}", getattr(fits[kind], name), **tolerance)
        baseline.check(
            "Feiv._model_1st_stage.coef", fits["Feiv"]._model_1st_stage.coef()
        )
        baseline.check_exact("get_performance.result", fits["Feols"].get_performance())
        map_fit = pf.feols("Y ~ X1 | f1", data, demeaner=pf.MapDemeaner())
        for name in ("_fixef_tol", "_fixef_maxiter"):
            baseline.check_exact(name, getattr(map_fit, name))
        baseline.check_exact("_demeaner.type", type(map_fit._demeaner).__name__)
        baseline.check_exact("_demeaner.backend", map_fit._demeaner.backend)
        split = pf.feols("Y ~ X1", data, split="f2").to_list()[0]
        baseline.check_exact("_sample_split_value", split._sample_split_value)
        ri = pf.feols("Y ~ X1 + X2", data)
        ri.ritest(
            "X1",
            reps=9,
            rng=np.random.default_rng(2026),
            store_ritest_statistics=True,
            choose_algorithm="slow",
        )
        baseline.check("_ritest_sample_stat", ri._ritest_sample_stat)
        event_data = pd.read_csv("pyfixest/did/data/df_het.csv")
        event = pf.event_study(
            event_data,
            yname="dep_var",
            idname="unit",
            tname="year",
            gname="g",
            estimator="saturated",
        )
        for name in ("_yname", "_gname", "_tname", "_idname", "_att"):
            baseline.check_exact(name, getattr(event, name))
        cohorts = event._res_cohort_eventtime_dict
        baseline.check_exact("event.cohorts", sorted(cohorts))
        for cohort, values in cohorts.items():
            baseline.check(f"event.{cohort}.time", values["time"])
            estimates = values["est"]
            baseline.check_exact(
                f"event.{cohort}.coefficients", estimates["Coefficient"].tolist()
            )
            baseline.check(f"event.{cohort}.estimates", estimates["Estimate"])
        for name, reason in ALIAS_EXCEPTIONS.items():
            baseline.skip(name, reason=reason)
