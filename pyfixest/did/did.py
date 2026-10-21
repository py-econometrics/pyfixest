from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DID(ABC):
    r"""
    Internal base class for difference-in-differences estimator wrappers.

    Use [event_study()](/reference/did.estimation.event_study.qmd) to fit a
    model through the supported public entry point. These wrappers retain
    constructor inputs in private state; they are not the fitted results that
    users receive. Their plotting and summary hooks are estimator plumbing,
    not the supported fitted-result interface.

    The `att` argument is retained but does not change the static TWFE/DID2S
    specification. For a dynamic DID2S formula, use
    [did2s()](/reference/did.estimation.did2s.qmd) directly.

    See Also
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) : Public estimation arguments and returned results.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Static and dynamic workflows.
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
