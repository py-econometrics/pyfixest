from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


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

        # Normalize integer storage before computing signed relative periods.
        for var in [self._tname, self._gname]:
            if pd.api.types.is_integer_dtype(self._data[var].dtype):
                if self._data[var].isna().any():
                    raise ValueError(
                        f"The variable {var} must not contain missing values."
                    )
                if (
                    pd.api.types.is_unsigned_integer_dtype(self._data[var].dtype)
                    and (self._data[var] > np.iinfo(np.int64).max).any()
                ):
                    raise ValueError(
                        f"The variable {var} must fit in a signed 64-bit integer."
                    )
                self._data[var] = self._data[var].astype(np.int64)
            elif self._data[var].dtype not in ["float64", "float32"]:
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
