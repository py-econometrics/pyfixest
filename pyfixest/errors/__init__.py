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


class MultiEstNotSupportedError(Exception):  # noqa: D101
    pass


class NanInClusterVarError(Exception):  # noqa: D101
    pass


class DepvarIsNotNumericError(Exception):  # noqa: D101
    pass


class NotImplementedError(Exception):  # noqa: D101
    pass


class NonConvergenceError(Exception):  # noqa: D101
    pass


class MatrixNotFullRankError(Exception):  # noqa: D101
    pass


class EmptyDesignMatrixError(Exception):  # noqa: D101
    pass


__all__ = [
    "FixedEffectInteractionError",
    "CovariateInteractionError",
    "DuplicateKeyError",
    "EndogVarsAsCovarsError",
    "InstrumentsAsCovarsError",
    "UnderDeterminedIVError",
    "UnsupportedMultipleEstimationSyntax",
    "VcovTypeNotSupportedError",
    "MultiEstNotSupportedError",
    "NanInClusterVarError",
    "DepvarIsNotNumericError",
    "NotImplementedError",
    "NonConvergenceError",
    "MatrixNotFullRankError",
    "EmptyDesignMatrixError",
]
