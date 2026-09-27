"""Exception classes raised by pyfixest."""

from __future__ import annotations

import warnings


class PyfixestError(Exception):
    """
    Base class of every exception that pyfixest defines.

    Catch `PyfixestError` to handle any pyfixest-specific failure. Subclasses
    that also derive from a built-in exception, such as
    `DepvarIsNotNumericError` from `TypeError`, are caught by that built-in too.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pyfixest as pf
    from pyfixest.errors import PyfixestError

    data = pf.get_data()
    data["Y"] = pd.Categorical(data["Y"].astype(str))

    try:
        pf.feols("Y ~ X1", data=data)
    except PyfixestError as error:
        print(f"{type(error).__name__}: {error}")
    ```
    """


class FormulaSyntaxError(PyfixestError):
    """The formula is not valid fixest formula syntax."""


class EndogVarsAsCovarsError(PyfixestError):
    """An endogenous variable also appears as a covariate."""


class InstrumentsAsCovarsError(PyfixestError):
    """An instrument also appears as a covariate."""


class UnderDeterminedIVError(PyfixestError):
    """The IV model has fewer instruments than endogenous variables."""


class DepvarIsNotNumericError(PyfixestError, TypeError):
    """The dependent variable is not numeric."""


class VcovTypeNotSupportedError(PyfixestError):
    """The model does not support the requested variance-covariance estimator."""


class NanInClusterVarError(PyfixestError):
    """A cluster variable contains missing values."""


class NonConvergenceError(PyfixestError):
    """An iterative estimation algorithm did not converge."""


class EmptyVcovError(PyfixestError):
    """The model has no variance-covariance matrix, e.g. because it has no coefficients."""


class MissingModelDataError(PyfixestError, RuntimeError):
    """Required model state was removed by fitted-model storage options."""


# Deprecated classes stay importable through the module `__getattr__` below,
# which emits a `FutureWarning` on access.
class FixedEffectInteractionError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class CovariateInteractionError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class DuplicateKeyError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class UnsupportedMultipleEstimationSyntax(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class MatrixNotFullRankError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class EmptyDesignMatrixError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


class FeatureDeprecationError(PyfixestError):
    """Deprecated: pyfixest never raises this error."""


_DEPRECATED_ERRORS: dict[str, type[PyfixestError]] = {
    error.__name__: error
    for error in (
        FixedEffectInteractionError,
        CovariateInteractionError,
        DuplicateKeyError,
        UnsupportedMultipleEstimationSyntax,
        MatrixNotFullRankError,
        EmptyDesignMatrixError,
        FeatureDeprecationError,
    )
}

del (
    FixedEffectInteractionError,
    CovariateInteractionError,
    DuplicateKeyError,
    UnsupportedMultipleEstimationSyntax,
    MatrixNotFullRankError,
    EmptyDesignMatrixError,
    FeatureDeprecationError,
)


def __getattr__(name: str) -> type[PyfixestError]:
    if name in _DEPRECATED_ERRORS:
        warnings.warn(
            f"`pyfixest.errors.{name}` is deprecated and will be removed in a "
            "future release. pyfixest never raises it; catch "
            "`pyfixest.errors.PyfixestError` to handle any pyfixest error.",
            FutureWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ERRORS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepvarIsNotNumericError",
    "EmptyVcovError",
    "EndogVarsAsCovarsError",
    "FormulaSyntaxError",
    "InstrumentsAsCovarsError",
    "MissingModelDataError",
    "NanInClusterVarError",
    "NonConvergenceError",
    "PyfixestError",
    "UnderDeterminedIVError",
    "VcovTypeNotSupportedError",
]
