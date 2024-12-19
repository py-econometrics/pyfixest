from abc import ABC, abstractmethod
from typing import Optional

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
        Variable name for calendar period. Must be a numeric (int or float) and
        of the same data type as the gname variable.
    gname : str
        unit-specific time of initial treatment. Must be an numeric (int or float)
        and of the same data type as the tname variable. More concretely, needs to be
        included in the values of tname. Values of 0 are considered as never treated.
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
        cluster: Optional[str] = None,
        xfml: Optional[str] = None,
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
        if self._data[self._tname].dtype not in [
            "int64",
            "int32",
            "int8",
            "float64",
            "float32",
        ]:
            raise ValueError(
                f"""The variable {self._tname} must be of a numeric type,
                but it is of type {self._data[self._tname].dtype}."""
            )
        if self._data[self._gname].dtype not in [
            "int64",
            "int32",
            "int8",
            "float64",
            "float32",
        ]:
            raise ValueError(
                f"""The variable {self._tname} must be of a numeric type, but it is
                of type {self._data[self._gname].dtype}.
                """
            )

        # check if gname is included in tname values
        self._tname_unique = self._data[self._tname].unique()
        self._gname_unique = self._data[self._gname].unique()

        if np.isin(0, self._tname_unique).any():
            raise ValueError(
                f"""The value 0 was found in the 'tname' variable {self._tname}.
                This value is reserved for never treated groups and cannot be used as a treatment period.
                """
            )

        allowed_g_values = [*self._tname_unique.tolist(), 0]
        for gval in self._gname_unique:
            if gval not in allowed_g_values:
                raise ValueError(
                    f"""The variable {gval} was found in the 'gname' variable {self._gname} but not in 'tname' {self._tname}.
                    All values of 'gname' must be included in 'tname' or be equal to 0 (for never treated groups).
                    """
                )

        # create a treatment variable
        self._data["ATT"] = (self._data[self._tname] >= self._data[self._gname]) * (
            self._data[self._gname] > 0
        )

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
