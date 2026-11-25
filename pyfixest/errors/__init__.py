from __future__ import annotations

from formulaic.errors import FactorEvaluationError


class FixedEffectEvaluationError(FactorEvaluationError):
    """A parsed fixed-effect factor or encoding could not be evaluated."""


class FixedEffectInteractionError(Exception):  # noqa: D101
    pass


class CovariateInteractionError(Exception):  # noqa: D101
    pass


class DuplicateKeyError(Exception):  # noqa: D101
    pass


class EndogVarsAsCovarsError(Exception):  # noqa: D101
    pass


class InstrumentsAsCovarsError(Exception):  # noqa: D101
    pass


class UnderDeterminedIVError(Exception):  # noqa: D101
    pass


class IVIdentificationError(Exception):  # noqa: D101
    pass


class UnsupportedMultipleEstimationSyntax(Exception):  # noqa: D101
    pass


class VcovTypeNotSupportedError(Exception):  # noqa: D101
    pass


class NanInClusterVarError(Exception):  # noqa: D101
    pass


class DepvarIsNotNumericError(Exception):  # noqa: D101
    pass


class NonConvergenceError(Exception):  # noqa: D101
    pass


class MatrixNotFullRankError(Exception):  # noqa: D101
    pass


class EmptyDesignMatrixError(Exception):  # noqa: D101
    pass


class FeatureDeprecationError(Exception):  # noqa: D101
    pass


class EmptyEstimationSampleError(ValueError):
    """No observations are available to fit a model."""


class EmptyVcovError(Exception):
    """The fit has no variance-covariance matrix because it estimated no coefficients."""

    def __init__(
        self,
        message: str = (
            "This model has no variance-covariance matrix because it estimated "
            "no coefficients: its formula only demeans the outcome by fixed "
            "effects, such as 'Y ~ 1 | f1'. Add regressors to the formula to get "
            "standard errors and inference."
        ),
    ) -> None:
        super().__init__(message)


class MissingModelDataError(RuntimeError):
    """Required model state was removed by fitted-model storage options."""


class FormulaSyntaxError(Exception):  # noqa: D101
    pass


class PyfixestDeprecationWarning(FutureWarning):
    """A pyfixest feature is deprecated and will be removed in a future release.

    A `FutureWarning` subclass, so Python shows it by default in scripts and
    notebooks, and users can filter pyfixest's deprecations on their own.
    """


__all__ = [
    "CovariateInteractionError",
    "DepvarIsNotNumericError",
    "DuplicateKeyError",
    "EmptyDesignMatrixError",
    "EmptyEstimationSampleError",
    "EmptyVcovError",
    "EndogVarsAsCovarsError",
    "FeatureDeprecationError",
    "FixedEffectInteractionError",
    "FormulaSyntaxError",
    "IVIdentificationError",
    "InstrumentsAsCovarsError",
    "MatrixNotFullRankError",
    "MissingModelDataError",
    "NanInClusterVarError",
    "NonConvergenceError",
    "PyfixestDeprecationWarning",
    "UnderDeterminedIVError",
    "UnsupportedMultipleEstimationSyntax",
    "VcovTypeNotSupportedError",
]
