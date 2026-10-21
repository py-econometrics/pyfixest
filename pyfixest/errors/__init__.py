from __future__ import annotations

from formulaic.errors import FactorEvaluationError


class FixedEffectEvaluationError(FactorEvaluationError):
    r"""
    A fixed-effect factor or encoding cannot be evaluated.

    Check that fixed-effect columns exist, their transformations evaluate, and
    their values support the requested encoding.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """


class FixedEffectInteractionError(Exception):
    r"""
    A fixed-effect interaction is invalid.

    Check the fixed-effects part of the formula and use supported interaction
    syntax, with every referenced column present in the data.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class CovariateInteractionError(Exception):
    r"""
    A covariate interaction cannot be constructed.

    Check the interacting variable names, types, and the supported formula syntax.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class DuplicateKeyError(Exception):
    r"""
    A formula component or option contains a duplicate key.

    Remove the duplicate entry identified in the exception message.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class EndogVarsAsCovarsError(Exception):
    r"""
    An endogenous variable is also listed as an ordinary covariate.

    Place the endogenous regressor only inside the instrumental-variable block
    `[endogenous ~ instruments]`, not on the ordinary right-hand side.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class InstrumentsAsCovarsError(Exception):
    r"""
    An excluded instrument is also listed as an ordinary covariate.

    Separate included exogenous covariates from excluded instruments in the
    instrumental-variable formula.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class UnderDeterminedIVError(Exception):
    r"""
    The instrumental-variable model has too few excluded instruments.

    Supply at least as many excluded instruments as endogenous regressors, and
    check identification after collinear terms are removed.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class UnsupportedMultipleEstimationSyntax(Exception):
    r"""
    The formula combines unsupported multiple-estimation syntax.

    Use the supported stepwise or multiple-outcome forms from the formula
    tutorial, or estimate the specifications in separate calls.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class VcovTypeNotSupportedError(Exception):
    r"""
    The requested covariance estimator is unavailable for this model.

    Choose an inference method supported by the fitted model and estimation path;
    check the model-support reference for restrictions.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class NanInClusterVarError(Exception):
    r"""
    A clustering variable contains missing values.

    Remove or resolve missing cluster identifiers in the estimation sample
    before requesting clustered standard errors.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class DepvarIsNotNumericError(Exception):
    r"""
    The dependent variable is not numeric.

    Convert the outcome to a supported numeric representation before fitting,
    checking that the conversion preserves its intended meaning.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class NonConvergenceError(Exception):
    r"""
    The estimation algorithm failed to converge.

    Inspect the exception message, model identification, scaling, and the
    estimator-specific tolerance and iteration controls before refitting.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class MatrixNotFullRankError(Exception):
    r"""
    A matrix required by the estimation path is not full rank.

    Inspect collinear regressors, fixed effects, and instrument relevance, then
    remove redundant terms or use an identified specification.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class EmptyDesignMatrixError(Exception):
    r"""
    No usable regressor columns remain in the design matrix.

    Check the formula, missing-value filtering, and regressors absorbed by the
    fixed effects; include an estimable regressor for paths that require one.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class FeatureDeprecationError(Exception):
    r"""
    The requested feature has been removed or superseded.

    Use the replacement named in the exception message.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class EmptyVcovError(Exception):
    r"""
    A coefficient covariance matrix is unavailable.

    Check that inference was computed for the model and that the fitted result
    retains the state required by the requested inference operation.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


class MissingModelDataError(RuntimeError):
    r"""
    Required fitted-model state was removed by storage options.

    Refit with the data or intermediate state required by the requested operation.
    See [model storage](/reference-guide/model-storage.qmd) for `lean` and
    `store_data` restrictions.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """


class FormulaSyntaxError(Exception):
    r"""
    The model formula cannot be parsed as supported fixest syntax.

    Check separators, fixed effects, instrumental-variable brackets, and
    multiple-estimation syntax in the formula tutorial.

    See Also
    --------
    [Errors and troubleshooting](/reference-guide/errors.qmd) : Failure types and remedies.
    [Formula syntax](/tutorials/formula-syntax.qmd) : Supported model specifications.
    """

    pass


__all__ = [
    "CovariateInteractionError",
    "DepvarIsNotNumericError",
    "DuplicateKeyError",
    "EmptyDesignMatrixError",
    "EmptyVcovError",
    "EndogVarsAsCovarsError",
    "FeatureDeprecationError",
    "FixedEffectInteractionError",
    "FormulaSyntaxError",
    "InstrumentsAsCovarsError",
    "MatrixNotFullRankError",
    "MissingModelDataError",
    "NanInClusterVarError",
    "NonConvergenceError",
    "UnderDeterminedIVError",
    "UnsupportedMultipleEstimationSyntax",
    "VcovTypeNotSupportedError",
]
