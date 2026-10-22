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
    # One unweighted fit covers the quantile-specific array representations.
    if (fml, quantile, method, vcov) == ("Y ~ X1 + X2", 0.5, "fn", "nid"):
        arrays = _legacy_attributes(mod, ["_Y", "_X", "_Z", "_u_hat"])
        _check_legacy_rows(baseline, arrays)
        for name, values in arrays.items():
            baseline.check_exact(f"{name}_shape", list(values.shape))


# The 0.60 attribute names now resolve through deprecation shims that rebuild
# the release's values from the typed state, including the square-root-weight
# scaling of the stored arrays. Weighted fits are where that scaling shows.
def _legacy_attributes(mod, names):
    """Read 0.60 attribute names; current pyfixest warns that they are deprecated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return {name: np.asarray(getattr(mod, name)) for name in names}


def _check_legacy_rows(baseline, arrays, **tolerance):
    """Check the first rows of each array, as the residual checks do."""
    for name, values in arrays.items():
        baseline.check(name, values[0:5], **tolerance)


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

    names = ["_Y", "_X", "_u_hat", "_weights"]
    if rhs != "0":
        names.extend(["_scores", "_Z"])
    if fixef:
        names.append("_sumFE")
    arrays = _legacy_attributes(mod, names)
    _check_legacy_rows(baseline, arrays)
    for name, value in arrays.items():
        baseline.check_exact(f"{name}_shape", list(value.shape))
    _check_legacy_metadata(baseline, mod)
    if rhs == "0":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            baseline.check_exact("_Z", mod._Z)
        return  # FE-only fits have no coefficient inference or sandwich.
    for name, values in _legacy_attributes(
        mod, ["_hessian", "_tZX", "_bread", "_ssc"]
    ).items():
        baseline.check(name, values)
        baseline.check_exact(f"{name}_shape", list(values.shape))
    baseline.check_exact("G", [int(g) for g in _legacy_attributes(mod, ["_G"])["_G"]])


@pytest.mark.parametrize("weights,weights_type", IV_WEIGHTS)
def test_deprecated_attributes_iv(data_feols_variants, weights, weights_type, baseline):
    mod = pf.feols(
        fml="Y ~ X2 | f1 | X1 ~ Z1",
        data=data_feols_variants[(True, "str")],
        vcov={"CRV1": "group_id"},
        weights=weights,
        weights_type=weights_type,
    )

    arrays = _legacy_attributes(mod, ["_Y", "_X", "_Z", "_u_hat", "_X_hat", "_v_hat"])
    _check_legacy_rows(baseline, arrays)
    for name, value in arrays.items():
        baseline.check_exact(f"{name}_shape", list(value.shape))
    _check_legacy_metadata(baseline, mod)
    for name, values in _legacy_attributes(mod, ["_bread", "_pi_hat"]).items():
        baseline.check(name, values)
    baseline.check_exact("G", [int(g) for g in _legacy_attributes(mod, ["_G"])["_G"]])


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

    arrays = _legacy_attributes(
        mod, ["_Y", "_X", "_Z", "_u_hat", "_weights", "_scores", "_sumFE"]
    )
    _check_legacy_rows(baseline, arrays, **FEPOIS_TOLERANCE)
    for name, values in _legacy_attributes(mod, ["_hessian", "_bread"]).items():
        baseline.check(name, values, **FEPOIS_TOLERANCE)
        baseline.check_exact(f"{name}_shape", list(values.shape))
    baseline.check_exact("Y_shape", list(arrays["_Y"].shape))


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

    arrays = _legacy_attributes(
        mod,
        [
            "_Y",
            "_X",
            "_Z",
            "_u_hat",
            "_weights",
            "_scores",
            "_sumFE",
            "_Y_hat_link",
            "_Y_hat_response",
        ],
    )
    _check_legacy_rows(baseline, arrays, **GLM_BINOMIAL_TOLERANCE)
    for name, values in arrays.items():
        baseline.check_exact(f"{name}_shape", list(values.shape))
    assert not np.allclose(arrays["_Y_hat_link"], arrays["_Y_hat_response"])
