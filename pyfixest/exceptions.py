class FixedEffectInteractionError(Exception):
    pass


class CovariateInteractionError(Exception):
    pass


class DuplicateKeyError(Exception):
    pass


class EndogVarsAsCovarsError(Exception):
    pass


class InstrumentsAsCovarsError(Exception):
    pass


class UnderDeterminedIVError(Exception):
    pass


class UnsupportedMultipleEstimationSyntax(Exception):
    pass


class VcovTypeNotSupportedError(Exception):
    pass


class MultiEstNotSupportedError(Exception):
    pass


class MatrixNotFullRankError(Exception):
    pass


class NanInClusterVarError(Exception):
    pass


class DepvarIsNotNumericError(Exception):
    pass


class NotImplementedError(Exception):
    pass


class NonConvergenceError(Exception):
    pass
