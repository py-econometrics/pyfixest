"""Tests for the declarative post-estimation capability contract."""

import pytest

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
