from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

DidEstimator = Literal["did2s", "twfe", "saturated"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DidDesign:
    """Difference-in-differences design a fitted model was estimated on.

    Published as `fit.did_design` by
    [event_study()](/reference/did.estimation.event_study.qmd) and
    [did2s()](/reference/did.estimation.did2s.qmd), and absent on fits that no
    DiD wrapper produced. The wrappers fit ordinary `feols()` models, so this
    value, rather than the estimation method of the fitted model, records which
    DiD estimator was used.

    Parameters
    ----------
    estimator : {"did2s", "twfe", "saturated"}
        The DiD estimator the wrapper ran.
    yname : str
        Name of the outcome variable.
    cluster : str or None
        Name of the variable the covariance is clustered on. `None` only when
        a DiD class was instantiated directly without one.
    idname : str or None
        Name of the unit identifier variable. `None` for
        [did2s()](/reference/did.estimation.did2s.qmd), which takes the two
        stage formulas instead of a panel design.
    tname : str or None
        Name of the calendar-period variable, or `None` as for `idname`.
    gname : str or None
        Name of the variable holding the unit-specific period of initial
        treatment, or `None` as for `idname`.
    xfml : str or None
        Formula of the additional covariates, `None` when there are none.
    att : bool or None
        Whether the average treatment effect on the treated was estimated
        instead of the canonical event study with all leads and lags, or
        `None` as for `idname`.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.event_study(
        pf.get_motherhood_event_study_data(),
        yname="log_earnings",
        idname="unit",
        tname="year",
        gname="g",
        estimator="twfe",
    )
    fit.did_design
    ```
    """

    estimator: DidEstimator
    yname: str
    cluster: str | None
    idname: str | None = None
    tname: str | None = None
    gname: str | None = None
    xfml: str | None = None
    att: bool | None = None


class DidFit(Protocol):
    """Structural contract of the fitted model a DiD wrapper returns.

    The wrappers return the generic `Feols` that `feols()` fitted, so they
    publish their design record on it rather than on a DiD-specific result
    class.
    """

    did_design: DidDesign


class DID(ABC):
    """
    A class used to represent the DID (Differences-in-Differences) model.

    Attributes
    ----------
    data : pandas.DataFrame
        The DataFrame containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the identifier variable.
    tname : str
        Variable name for calendar period. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted.
    gname : str
        unit-specific time of initial treatment. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted. Never treated units must
        have a value of 0.
    xfml : str
        The formula for the covariates.
    att : str
        Whether to estimate the average treatment effect on the treated (ATT) or
        the canonical event study design with all leads and lags. Default is True.
    cluster : str
        The name of the cluster variable.
    """

    @abstractmethod
    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        cluster: str | None = None,
        xfml: str | None = None,
        att: bool = True,
    ):
        # do some checks here

        self._data = data.copy()
        self._yname = yname
        self._idname = idname
        self._tname = tname
        self._gname = gname
        self._xfml = xfml
        self._att = att
        self._cluster = cluster

        # check if tname and gname are of type int (either int 64, 32, 8)

        for var in [self._tname, self._gname]:
            if self._data[var].dtype not in [
                "int64",
                "int32",
                "int8",
                "float64",
                "float32",
            ]:
                raise ValueError(
                    f"""The variable {var} must be of a numeric type, and more
                    specifically, in the format YYYYMMDDHHMMSS. I.e. either 2012, 2013,
                    etc. or 201201, 201202, 201203 etc."""
                )

        # create a treatment variable
        self._data["is_treated"] = (
            self._data[self._tname] >= self._data[self._gname]
        ) * (self._data[self._gname] > 0)
        self._data = self._data.merge(
            self._data.assign(
                first_treated_period=self._data[self._tname] * self._data["is_treated"]
            )
            .groupby(self._idname)["first_treated_period"]
            .apply(lambda x: x[x > 0].min()),
            on=self._idname,
        )
        self._data["rel_time"] = (
            self._data[self._tname] - self._data["first_treated_period"]
        )
        self._data["first_treated_period"] = (
            self._data["first_treated_period"].replace(np.nan, 0).astype(int)
        )
        self._data["rel_time"] = self._data["rel_time"].replace(np.nan, np.inf)

    @abstractmethod
    def estimate(self):  # noqa: D102
        pass

    @abstractmethod
    def vcov(self):  # noqa: D102
        pass

    @abstractmethod
    def iplot(self):  # noqa: D102
        pass

    @abstractmethod
    def tidy(self):  # noqa: D102
        pass

    @abstractmethod
    def summary(self):  # noqa: D102
        pass
