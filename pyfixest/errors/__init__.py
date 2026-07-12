"""Public exception hierarchy for pyfixest."""


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


class EmptyVcovError(Exception):  # noqa: D101
    pass


class FormulaSyntaxError(Exception):  # noqa: D101
    pass


class MissingStoredDataError(AttributeError):
    """Raised when post-estimation needs data omitted from a fitted model."""


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
    "MissingStoredDataError",
    "NanInClusterVarError",
    "NonConvergenceError",
    "UnderDeterminedIVError",
    "UnsupportedMultipleEstimationSyntax",
    "VcovTypeNotSupportedError",
]
