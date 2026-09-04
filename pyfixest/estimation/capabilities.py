"""Declarative support contract for post-estimation and inference features.

One table per model class answers a single question: can this fitted result run
this feature? Each gated method asks the table instead of re-deriving support
from `_is_iv`, `_has_weights`, `_has_fixef`, or the mutable `_method` label, so
an unsupported combination fails with a specific error rather than
reinterpreting one estimator's arrays as another's.

A class declares a `Capabilities` table of `Rule` callables, one per `Feature`.
A rule receives the `FitFeatures` of one fitted result and returns either
`None`, meaning supported, or the reason the fit is unsupported.
Argument-level restrictions -- `update(inplace=True)`,
`decompose(only_coef=...)`, multiway-cluster limits, or the covariance types
SAVI accepts -- stay with their methods; only fit-level support lives here.

See [Which Methods Does Each Estimator Support?](/how-to/supported-methods.qmd)
for the rendered table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, get_args

import pandas as pd

from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.internals.literals import (
    EstimatorKind,
    FamilyOptions,
    WeightsTypeOptions,
)

Feature: TypeAlias = Literal[
    "crv3",
    "hac",
    "nid",
    "wildboottest",
    "ccv",
    "decompose",
    "ritest",
    "fixef",
    "predict",
    "prediction_errors",
    "update",
    "savi",
]

FEATURES: Final[tuple[Feature, ...]] = get_args(Feature)

# `_method` labels that the difference-in-differences entry points write onto an
# otherwise ordinary `Feols` result. Their coefficients and covariance come from
# a two-step or event-study estimator, so features that refit the retained OLS
# design would silently report inference for a different model.
DID_METHOD_LABELS: Final[frozenset[str]] = frozenset({"did2s", "twfe", "saturated"})


@dataclass(frozen=True, slots=True, kw_only=True)
class FitFeatures:
    """Fit-level properties that decide post-estimation support.

    Attributes
    ----------
    estimator : EstimatorKind
        Estimator that produced the fit, taken from the model class rather than
        from the mutable `_method` label.
    family : FamilyOptions or None
        GLM family of the fit, and None outside GLMs.
    is_iv : bool
        Whether the fit instruments an endogenous regressor.
    has_fixef : bool
        Whether fixed effects were absorbed.
    has_weights : bool
        Whether the user supplied observation weights.
    weights_kind : WeightsTypeOptions or None
        Interpretation of the observation weights, and None when the fit is
        unweighted. An unweighted fit reports None even though the estimation
        API still carries its default `weights_type`.
    is_did : bool
        Whether a difference-in-differences entry point relabelled this result.
        Such results reuse `Feols` as their result class but carry two-step or
        event-study estimates and covariance.
    """

    estimator: EstimatorKind
    family: FamilyOptions | None = None
    is_iv: bool = False
    has_fixef: bool = False
    has_weights: bool = False
    weights_kind: WeightsTypeOptions | None = None
    is_did: bool = False


# A rule returns the reason a fit is unsupported, or None when it is supported.
Rule: TypeAlias = Callable[[FitFeatures], str | None]

# A predicate on a fit paired with the reason it makes a feature unavailable.
Condition: TypeAlias = tuple[Callable[[FitFeatures], bool], str]


def unsupported_estimator(features: FitFeatures) -> str:
    """Reject every fit, naming the estimator that produced it.

    This is the default rule of every `Capabilities` field: a feature a class
    does not declare is unsupported for that estimator.

    Parameters
    ----------
    features : FitFeatures
        Properties of the fitted result under test.

    Returns
    -------
    str
        The reason, naming the estimator.
    """
    return f"models of type '{features.estimator}'"


def supported() -> Rule:
    """Build a rule that accepts every fit of the declaring estimator.

    Returns
    -------
    Rule
        A rule that always returns None.
    """

    def rule(features: FitFeatures) -> str | None:
        return None

    return rule


def unsupported(reason: str) -> Rule:
    """Build a rule that rejects every fit of the declaring estimator.

    Parameters
    ----------
    reason : str
        The reason, phrased as the object of "is not supported for", such as
        `"IV models"`.

    Returns
    -------
    Rule
        A rule that always returns `reason`.
    """

    def rule(features: FitFeatures) -> str | None:
        return reason

    return rule


def unless(*conditions: Condition) -> Rule:
    """Build a rule that accepts a fit unless one of `conditions` holds.

    Parameters
    ----------
    *conditions : Condition
        Predicate and reason pairs, tested in order. The first predicate that
        holds decides the message, so list the most specific condition first.

    Returns
    -------
    Rule
        A rule returning the reason of the first matching condition, else None.
    """

    def rule(features: FitFeatures) -> str | None:
        for holds, reason in conditions:
            if holds(features):
                return reason
        return None

    return rule


IV: Final[Condition] = (lambda features: features.is_iv, "IV models")
WEIGHTED: Final[Condition] = (
    lambda features: features.has_weights,
    "models with weights",
)
FIXED_EFFECTS: Final[Condition] = (
    lambda features: features.has_fixef,
    "models with fixed effects",
)
FREQUENCY_WEIGHTS: Final[Condition] = (
    lambda features: features.weights_kind == "fweights",
    "models with frequency weights",
)
NON_FREQUENCY_WEIGHTS: Final[Condition] = (
    lambda features: features.has_weights and features.weights_kind != "fweights",
    "models with non-frequency weights",
)
DIFFERENCE_IN_DIFFERENCES: Final[Condition] = (
    lambda features: features.is_did,
    "difference-in-differences results",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Capabilities:
    """The post-estimation features one model class supports.

    Every field is a `Rule` evaluated against the `FitFeatures` of a fitted
    result. Fields left out default to `unsupported_estimator`, so a class
    declares only what it supports and any feature added later starts out
    unsupported everywhere.

    Attributes
    ----------
    crv3 : Rule
        `vcov={"CRV3": ...}` cluster-jackknife covariance.
    hac : Rule
        Newey-West and Driscoll-Kraay covariance.
    nid : Rule
        `vcov="nid"`, the quantile-regression sandwich with an estimated
        conditional density.
    wildboottest : Rule
        Wild cluster bootstrap.
    ccv : Rule
        Causal cluster variance of Abadie et al. (2023).
    decompose : Rule
        Gelbach (2016) decomposition.
    ritest : Rule
        Randomization inference.
    fixef : Rule
        Recovery of the absorbed fixed-effect coefficients.
    predict : Rule
        Point predictions, in sample and on new data.
    prediction_errors : Rule
        Prediction standard errors and prediction intervals.
    update : Rule
        Sherman-Morrison coefficient updates.
    savi : Rule
        Safe anytime-valid e-values, sequential p-values, and confidence
        sequences.
    """

    crv3: Rule = unsupported_estimator
    hac: Rule = unsupported_estimator
    nid: Rule = unsupported_estimator
    wildboottest: Rule = unsupported_estimator
    ccv: Rule = unsupported_estimator
    decompose: Rule = unsupported_estimator
    ritest: Rule = unsupported_estimator
    fixef: Rule = unsupported_estimator
    predict: Rule = unsupported_estimator
    prediction_errors: Rule = unsupported_estimator
    update: Rule = unsupported_estimator
    savi: Rule = unsupported_estimator

    def evaluate(self, features: FitFeatures) -> dict[Feature, str | None]:
        """Evaluate every rule against one fitted result.

        Parameters
        ----------
        features : FitFeatures
            Properties of the fitted result under test.

        Returns
        -------
        dict
            Maps each feature to None when it is supported, and otherwise to
            the reason it is not.
        """
        return {feature: getattr(self, feature)(features) for feature in FEATURES}


FEATURE_ERRORS: Final[Mapping[Feature, type[Exception]]] = MappingProxyType(
    {
        **dict.fromkeys(FEATURES, NotImplementedError),
        # A rejected covariance type is a covariance error, not a missing
        # method, and callers have caught it as such since CRV3 was added.
        "crv3": VcovTypeNotSupportedError,
    }
)


def require_support(
    *,
    capabilities: Capabilities,
    feature: Feature,
    features: FitFeatures,
    subject: str,
) -> None:
    """Raise unless `capabilities` support `feature` for this fit.

    Parameters
    ----------
    capabilities : Capabilities
        The declaration of the model class that produced the fit.
    feature : Feature
        The feature the caller is about to run.
    features : FitFeatures
        Properties of the fitted result under test.
    subject : str
        How the message names the feature, such as `"CRV3 inference"`. It
        becomes the subject of "... is not supported for ...".

    Raises
    ------
    Exception
        The type registered for `feature` in `FEATURE_ERRORS`:
        `VcovTypeNotSupportedError` for `"crv3"` and `NotImplementedError`
        otherwise.
    """
    reason = getattr(capabilities, feature)(features)
    if reason is None:
        return
    raise FEATURE_ERRORS[feature](f"{subject} is not supported for {reason}.")


def support_matrix() -> pd.DataFrame:
    """Return the feature support of every estimator at a plain fit.

    The table evaluates each model class's declaration for an unweighted fit
    without fixed effects, so it shows what an estimator supports in principle.
    Weights, fixed effects, and instruments can withdraw a feature from an
    individual fit; call
    [`capabilities()`](/reference/estimation.models.feols_.Feols.capabilities.qmd)
    on a fitted result for the support of that fit.

    Returns
    -------
    pandas.DataFrame
        Boolean support, indexed by feature, with one column per estimator:
        `feols`, `feiv`, `fepois`, `feglm`, and `quantreg`.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    pf.estimation.support_matrix()
    ```
    """
    from pyfixest.estimation.models.feglm_ import Feglm
    from pyfixest.estimation.models.feiv_ import Feiv
    from pyfixest.estimation.models.feols_ import Feols
    from pyfixest.estimation.models.fepois_ import Fepois
    from pyfixest.estimation.quantreg.quantreg_ import Quantreg

    plain_fits: dict[str, tuple[type[Feols], FitFeatures]] = {
        "feols": (Feols, FitFeatures(estimator="feols")),
        "feiv": (Feiv, FitFeatures(estimator="feols", is_iv=True)),
        "fepois": (Fepois, FitFeatures(estimator="fepois", family="poisson")),
        "feglm": (Feglm, FitFeatures(estimator="feglm", family="logit")),
        "quantreg": (Quantreg, FitFeatures(estimator="quantreg")),
    }

    return pd.DataFrame(
        {
            name: [
                reason is None
                for reason in model._capabilities.evaluate(features).values()
            ]
            for name, (model, features) in plain_fits.items()
        },
        index=pd.Index(FEATURES, name="feature"),
    )
