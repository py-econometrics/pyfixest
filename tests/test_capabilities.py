"""Tests for the declarative post-estimation capability contract."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.capabilities import (
    FEATURE_ERRORS,
    FEATURES,
    FIXED_EFFECTS,
    FREQUENCY_WEIGHTS,
    IV,
    NON_FREQUENCY_WEIGHTS,
    WEIGHTED,
    Capabilities,
    FitFeatures,
    require_support,
    support_matrix,
    supported,
    unless,
    unsupported,
    unsupported_estimator,
)

PLAIN = FitFeatures(estimator="feols")
WITH_FIXEF = FitFeatures(estimator="feols", has_fixef=True)
AWEIGHTED = FitFeatures(estimator="feols", has_weights=True, weights_kind="aweights")
FWEIGHTED = FitFeatures(estimator="feols", has_weights=True, weights_kind="fweights")
IV_FIT = FitFeatures(estimator="feols", is_iv=True)


def test_rule_helpers_report_the_expected_reasons():
    assert supported()(PLAIN) is None
    assert unsupported("IV models")(PLAIN) == "IV models"
    assert (
        unsupported_estimator(FitFeatures(estimator="quantreg"))
        == "models of type 'quantreg'"
    )


@pytest.mark.parametrize(
    ("condition", "holds_for", "not_for"),
    [
        (IV, IV_FIT, PLAIN),
        (WEIGHTED, AWEIGHTED, PLAIN),
        (FIXED_EFFECTS, WITH_FIXEF, PLAIN),
        (FREQUENCY_WEIGHTS, FWEIGHTED, AWEIGHTED),
        (NON_FREQUENCY_WEIGHTS, AWEIGHTED, FWEIGHTED),
    ],
)
def test_named_conditions_separate_the_fits_they_describe(
    condition, holds_for, not_for
):
    predicate, reason = condition
    assert predicate(holds_for)
    assert not predicate(not_for)
    assert unless(condition)(holds_for) == reason
    assert unless(condition)(not_for) is None


def test_unless_reports_the_first_matching_condition():
    rule = unless(FIXED_EFFECTS, WEIGHTED)
    both = FitFeatures(
        estimator="feols", has_fixef=True, has_weights=True, weights_kind="aweights"
    )
    assert rule(both) == "models with fixed effects"
    assert rule(AWEIGHTED) == "models with weights"


def test_capabilities_default_every_feature_to_unsupported():
    """A feature a class does not declare is unsupported for that estimator."""
    reasons = Capabilities().evaluate(FitFeatures(estimator="fepois"))
    assert list(reasons) == list(FEATURES)
    assert set(reasons.values()) == {"models of type 'fepois'"}


def test_capabilities_evaluate_mixes_declared_and_default_rules():
    declaration = Capabilities(predict=supported(), hac=unless(FREQUENCY_WEIGHTS))
    reasons = declaration.evaluate(FWEIGHTED)
    assert reasons["predict"] is None
    assert reasons["hac"] == "models with frequency weights"
    assert reasons["ccv"] == "models of type 'feols'"


def test_require_support_passes_silently_when_supported():
    require_support(
        capabilities=Capabilities(predict=supported()),
        feature="predict",
        features=PLAIN,
        subject="predict()",
    )


def test_require_support_raises_the_registered_type_and_message():
    with pytest.raises(
        NotImplementedError,
        match=r"^Randomization inference is not supported for models with weights\.$",
    ):
        require_support(
            capabilities=Capabilities(ritest=unless(WEIGHTED)),
            feature="ritest",
            features=AWEIGHTED,
            subject="Randomization inference",
        )

    with pytest.raises(
        VcovTypeNotSupportedError,
        match=r"^CRV3 inference is not supported for IV models\.$",
    ):
        require_support(
            capabilities=Capabilities(crv3=unsupported("IV models")),
            feature="crv3",
            features=IV_FIT,
            subject="CRV3 inference",
        )


def test_feature_errors_cover_every_feature():
    assert set(FEATURE_ERRORS) == set(FEATURES)
    assert FEATURE_ERRORS["crv3"] is VcovTypeNotSupportedError
    assert {FEATURE_ERRORS[feature] for feature in FEATURES if feature != "crv3"} == {
        NotImplementedError
    }


def test_support_matrix_shape_and_selected_entries():
    matrix = support_matrix()

    assert list(matrix.index) == list(FEATURES)
    assert matrix.index.name == "feature"
    assert list(matrix.columns) == ["feols", "feiv", "fepois", "feglm", "quantreg"]
    assert matrix.to_numpy().dtype == np.dtype(bool)

    assert matrix.loc["crv3", "feols"]
    assert not matrix.loc["crv3", "feiv"]
    assert matrix.loc["crv3", "fepois"]
    assert not matrix.loc["crv3", "feglm"]
    assert matrix.loc["nid", "quantreg"]
    assert not matrix.loc["nid", "feols"]
    assert matrix.loc["predict", "quantreg"]
    assert not matrix.loc["prediction_errors", "quantreg"]


# ---------------------------------------------------------------------------
# The declared table, enforced through the public methods that consult it.
# ---------------------------------------------------------------------------

_N = 60


@pytest.fixture(scope="module")
def capability_data() -> pd.DataFrame:
    """Build a tiny deterministic sample every estimator in the matrix can fit."""
    rng = np.random.default_rng(2718)
    treatment = rng.binomial(1, 0.5, size=_N).astype(float)
    instrument = rng.normal(size=_N)
    endogenous = 0.8 * instrument + rng.normal(scale=0.5, size=_N)
    return pd.DataFrame(
        {
            "Y": 1.0 + 0.5 * treatment + 0.4 * endogenous + rng.normal(size=_N),
            "Y_count": rng.poisson(2.0, size=_N),
            "Y_bin": rng.binomial(1, 0.5, size=_N),
            "D": treatment,
            "X2": rng.normal(size=_N),
            "Z1": instrument,
            "endog": endogenous,
            "t": np.arange(_N, dtype=float),
            "f1": np.tile(np.arange(6), _N // 6),
            "cluster": np.tile(np.arange(5), _N // 5),
            "aw": np.linspace(0.5, 1.5, _N),
            "fw": np.tile([1.0, 2.0], _N // 2),
        }
    )


def _fit(kind: str, data: pd.DataFrame):
    if kind == "ols":
        return pf.feols("Y ~ D + X2", data=data)
    if kind == "ols_aweights":
        return pf.feols("Y ~ D + X2", data=data, weights="aw", weights_type="aweights")
    if kind == "ols_fweights":
        return pf.feols("Y ~ D + X2", data=data, weights="fw", weights_type="fweights")
    if kind == "ols_fixef":
        return pf.feols("Y ~ D + X2 | f1", data=data)
    if kind == "iv":
        return pf.feols("Y ~ X2 + [endog ~ Z1]", data=data)
    if kind == "iv_fixef":
        return pf.feols("Y ~ X2 + [endog ~ Z1] | f1", data=data)
    if kind == "poisson":
        return pf.fepois("Y_count ~ D + X2", data=data)
    if kind == "logit":
        return pf.feglm("Y_bin ~ D + X2", data=data, family="logit")
    if kind == "quantreg":
        with pytest.warns(FutureWarning, match="experimental"):
            return pf.quantreg("Y ~ D + X2", data=data, quantile=0.5)
    raise AssertionError(f"unknown fit kind {kind}")


def _call(fit, feature: str):
    """Invoke `feature` with the smallest arguments that reach its support gate."""
    if feature == "crv3":
        return fit.vcov({"CRV3": "cluster"})
    if feature == "hac":
        return fit.vcov("NW", vcov_kwargs={"time_id": "t", "lag": 1})
    if feature == "nid":
        return fit.vcov("nid")
    if feature == "wildboottest":
        return fit.wildboottest(param=fit._coefnames[1], reps=11, seed=45)
    if feature == "ccv":
        return fit.ccv(treatment="D", cluster="cluster", n_splits=2, seed=45)
    if feature == "decompose":
        return fit.decompose(decomp_var="D", only_coef=True)
    if feature == "ritest":
        return fit.ritest(resampvar="D", reps=2)
    if feature == "fixef":
        return fit.fixef()
    if feature == "predict":
        return fit.predict()
    if feature == "prediction_errors":
        return fit.predict(se_fit=True)
    if feature == "update":
        return fit.update(np.ones((1, fit._k)), np.zeros(1))
    if feature == "savi":
        return fit.evalue()
    raise AssertionError(f"unknown feature {feature}")


# Features whose method only the OLS leaf defines. A fit whose class does not
# implement one has no such attribute at all, so the capability reason arrives
# as an `AttributeError` from `BaseRegression.__getattr__` rather than as the
# exception the feature registers.
CLASS_LEVEL_METHODS = {
    "wildboottest": "wildboottest",
    "ccv": "ccv",
    "decompose": "decompose",
    "update": "update",
    "savi": "evalue",
}


# (fit, feature, declared reason or None when the feature must be available).
# `fixef` appears only for fits that absorb fixed effects, because the missing
# fixed-effect ValueError is raised before the support gate.
SUPPORT_CASES = [
    ("ols", "crv3", None),
    ("ols", "hac", None),
    ("ols", "nid", "models of type 'feols'"),
    ("ols", "wildboottest", None),
    ("ols", "ccv", None),
    ("ols", "decompose", None),
    ("ols", "ritest", None),
    ("ols", "predict", None),
    ("ols", "prediction_errors", None),
    ("ols", "update", None),
    ("ols", "savi", None),
    ("ols_aweights", "crv3", None),
    ("ols_aweights", "hac", None),
    ("ols_aweights", "wildboottest", "models with weights"),
    ("ols_aweights", "ccv", "models with weights"),
    ("ols_aweights", "decompose", "models with non-frequency weights"),
    ("ols_aweights", "ritest", "models with weights"),
    ("ols_aweights", "predict", None),
    ("ols_aweights", "prediction_errors", "models with weights"),
    ("ols_aweights", "update", "models with weights"),
    ("ols_aweights", "savi", "models with weights"),
    ("ols_fweights", "hac", "models with frequency weights"),
    ("ols_fweights", "decompose", None),
    ("ols_fweights", "savi", "models with weights"),
    ("ols_fixef", "crv3", None),
    ("ols_fixef", "hac", None),
    ("ols_fixef", "wildboottest", None),
    ("ols_fixef", "ccv", "models with fixed effects"),
    ("ols_fixef", "ritest", None),
    ("ols_fixef", "fixef", None),
    ("ols_fixef", "predict", None),
    ("ols_fixef", "prediction_errors", "models with fixed effects"),
    ("ols_fixef", "update", "models with fixed effects"),
    ("ols_fixef", "savi", "models with fixed effects"),
    ("iv", "crv3", "IV models"),
    ("iv", "hac", None),
    ("iv", "wildboottest", "IV models"),
    ("iv", "ccv", "IV models"),
    ("iv", "decompose", "IV models"),
    ("iv", "ritest", "IV models"),
    ("iv", "predict", "IV models"),
    ("iv", "prediction_errors", "IV models"),
    ("iv", "update", "IV models"),
    ("iv", "savi", "IV models"),
    ("iv_fixef", "fixef", "IV models"),
    ("poisson", "crv3", None),
    ("poisson", "hac", None),
    ("poisson", "nid", "models of type 'fepois'"),
    ("poisson", "wildboottest", "models of type 'fepois'"),
    ("poisson", "ccv", "models of type 'fepois'"),
    ("poisson", "decompose", "models of type 'fepois'"),
    ("poisson", "ritest", None),
    ("poisson", "predict", None),
    ("poisson", "prediction_errors", "models of type 'fepois'"),
    ("poisson", "update", "models of type 'fepois'"),
    ("poisson", "savi", "models of type 'fepois'"),
    ("logit", "crv3", "models of type 'feglm'"),
    ("logit", "hac", None),
    ("logit", "nid", "models of type 'feglm'"),
    ("logit", "wildboottest", "models of type 'feglm'"),
    ("logit", "ccv", "models of type 'feglm'"),
    ("logit", "decompose", "models of type 'feglm'"),
    ("logit", "ritest", "models of type 'feglm'"),
    ("logit", "predict", None),
    ("logit", "prediction_errors", "models of type 'feglm'"),
    ("logit", "update", "models of type 'feglm'"),
    ("logit", "savi", "models of type 'feglm'"),
    ("quantreg", "crv3", "models of type 'quantreg'"),
    ("quantreg", "hac", "models of type 'quantreg'"),
    ("quantreg", "nid", None),
    ("quantreg", "wildboottest", "models of type 'quantreg'"),
    ("quantreg", "ccv", "models of type 'quantreg'"),
    ("quantreg", "decompose", "models of type 'quantreg'"),
    ("quantreg", "ritest", "models of type 'quantreg'"),
    ("quantreg", "predict", None),
    ("quantreg", "prediction_errors", "models of type 'quantreg'"),
    ("quantreg", "update", "models of type 'quantreg'"),
    ("quantreg", "savi", "models of type 'quantreg'"),
]


@pytest.mark.parametrize(
    ("kind", "feature", "reason"),
    SUPPORT_CASES,
    ids=[f"{kind}-{feature}" for kind, feature, _ in SUPPORT_CASES],
)
def test_declared_support_matches_the_public_methods(
    capability_data: pd.DataFrame, kind: str, feature: str, reason: "str | None"
):
    """Every declared cell is what the corresponding public method enforces."""
    fit = _fit(kind, capability_data)
    assert fit._capabilities.evaluate(fit._fit_features)[feature] == reason

    if reason is None:
        # A supported cell must not raise the errors the contract owns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _call(fit, feature)
        return

    method = CLASS_LEVEL_METHODS.get(feature)
    if method is not None and not hasattr(type(fit), method):
        # The method is not defined for this class at all; the reason reaches
        # the caller through the attribute lookup instead.
        with pytest.raises(
            AttributeError,
            match=rf"has no attribute '{method}'\. .* is not supported for {reason}\.",
        ):
            _call(fit, feature)
        return

    with pytest.raises(
        FEATURE_ERRORS[feature], match=rf"is not supported for {reason}\.$"
    ):
        _call(fit, feature)


@pytest.mark.parametrize("kind", ["poisson", "logit", "quantreg"])
def test_ols_only_methods_are_absent_from_non_ols_results(
    capability_data: pd.DataFrame, kind: str
):
    """Method placement, not a runtime check, answers support for these."""
    fit = _fit(kind, capability_data)

    for method in ("ccv", "decompose", "update", "wildboottest", "pvalue_savi"):
        assert not hasattr(fit, method)

    with pytest.raises(
        AttributeError,
        match=r"has no attribute 'decompose'\. Decomposition is not supported "
        r"for models of type '\w+'\. Call capabilities\(\)",
    ):
        getattr(fit, "decompose")  # noqa: B009

    # Any other missing name keeps the plain message.
    with pytest.raises(
        AttributeError, match=r"object has no attribute 'not_a_pyfixest_method'$"
    ):
        getattr(fit, "not_a_pyfixest_method")  # noqa: B009


def test_capabilities_accessor_reports_supported_and_reason(
    capability_data: pd.DataFrame,
):
    """The accessor reports the support of the fit in hand, not of its class."""
    fit = _fit("ols_aweights", capability_data)
    table = fit.capabilities()

    assert list(table.index) == list(FEATURES)
    assert table.index.name == "feature"
    assert list(table.columns) == ["supported", "reason"]

    assert table.loc["predict", "supported"]
    assert table.loc["predict", "reason"] is None
    assert not table.loc["ccv", "supported"]
    assert table.loc["ccv", "reason"] == "models with weights"

    plain = _fit("ols", capability_data).capabilities()
    assert plain.loc["ccv", "supported"]
    assert plain.loc["ccv", "reason"] is None


@pytest.mark.parametrize("kind", ["quantreg", "logit", "poisson"])
def test_non_ols_fits_reject_prediction_errors(
    capability_data: pd.DataFrame, kind: str
):
    """Both prediction-error arguments hit the gate, not only `se_fit`.

    The OLS residual-variance formula behind `se_fit` and `interval` describes
    neither a conditional quantile nor a GLM mean, so both arguments must be
    rejected rather than silently returning point predictions.
    """
    fit = _fit(kind, capability_data)
    reason = fit._capabilities.evaluate(fit._fit_features)["prediction_errors"]

    for kwargs in ({"se_fit": True}, {"interval": "prediction"}):
        with pytest.raises(
            NotImplementedError,
            match=rf"Prediction with standard errors is not supported for {reason}\.",
        ):
            fit.predict(**kwargs)


def test_unweighted_fits_report_no_weights_kind():
    """The estimation API keeps a default `weights_type` on unweighted fits."""
    fit = pf.feols("Y ~ X1", data=pf.get_data(), weights_type="fweights")
    assert fit._fit_features.weights_kind is None
    assert not FREQUENCY_WEIGHTS[0](fit._fit_features)


def test_difference_in_differences_results_keep_their_refit_gates():
    """A relabelled `Feols` result must not be refit as if it were plain OLS."""
    data = pd.read_csv("pyfixest/did/data/df_het.csv").iloc[:2000]
    fit = pf.did2s(
        data,
        yname="dep_var",
        first_stage="~ 0 | state + year",
        second_stage="~ i(rel_year, ref=-1.0)",
        treatment="treat",
        cluster="state",
    )

    assert fit._fit_features.is_did
    reasons = fit._capabilities.evaluate(fit._fit_features)
    for feature in ("ritest", "update", "savi"):
        assert reasons[feature] == "difference-in-differences results"

    with pytest.raises(NotImplementedError, match="difference-in-differences results"):
        fit.update(np.ones((1, fit._k)), np.zeros(1))
    with pytest.raises(NotImplementedError, match="difference-in-differences results"):
        fit.evalue()
