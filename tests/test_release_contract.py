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
from tests._release_baseline import Baseline, data_digest

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


def _check_structure(baseline: Baseline, mod) -> None:
    baseline.check_exact("coefnames", list(mod._coefnames))
    baseline.check_exact("nobs", int(mod._N))
    baseline.check_exact("df_k", int(mod._df_k))
    baseline.check_exact("df_t", int(mod._df_t))


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
    vcov = np.asarray(mod._vcov)
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
    baseline.check("deviance", mod.deviance)
    baseline.check("resid", mod.resid()[0:5], **FEPOIS_TOLERANCE)
    baseline.check("irls_weights", mod._irls_weights.flatten()[0:5], **FEPOIS_TOLERANCE)


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

    baseline.check("deviance", mod.deviance)

    # `resid()` returned the IRLS working residual in the pinned release and
    # returns the response residual now, so both are compared through the
    # attributes that mean the same thing in either version.
    tolerance = {} if family == "gaussian" else GLM_BINOMIAL_TOLERANCE
    baseline.check("resid_response", mod._u_hat_response[0:5], **tolerance)
    baseline.check("resid_working", mod._u_hat_working[0:5], **tolerance)

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
