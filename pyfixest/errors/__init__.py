"""Exception classes raised by pyfixest."""

from __future__ import annotations


class PyfixestError(Exception):
    """
    Base class of the exceptions defined in `pyfixest.errors`.
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


class FormulaSyntaxError(PyfixestError, ValueError):
    """The formula is not valid fixest formula syntax."""


class EndogVarsAsCovarsError(PyfixestError, ValueError):
    """An endogenous variable also appears as a covariate."""


class InstrumentsAsCovarsError(PyfixestError, ValueError):
    """An instrument also appears as a covariate."""


class UnderDeterminedIVError(PyfixestError, ValueError):
    """The IV model has fewer instruments than endogenous variables."""


class DepvarIsNotNumericError(PyfixestError, TypeError):
    """The dependent variable is not numeric."""


class VcovTypeNotSupportedError(PyfixestError):
    """The model does not support the requested variance-covariance estimator."""


class NanInClusterVarError(PyfixestError, ValueError):
    """A cluster variable contains missing values."""


class NonConvergenceError(PyfixestError):
    """An iterative estimation algorithm did not converge."""


class EmptyVcovError(PyfixestError):
    """The model has no variance-covariance matrix, e.g. because it has no coefficients."""


class MissingModelDataError(PyfixestError, RuntimeError):
    """Required model state was removed by fitted-model storage options."""


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
