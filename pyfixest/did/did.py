from abc import ABC, abstractmethod


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
    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
        # do some checks here

        self._data = data.copy()
        self._yname = yname
        self._idname = idname
        self._tname = tname
        self._gname = gname
        self._xfml = xfml
        self._att = att
        self._cluster = cluster

        # if self._xfml is not None:
        #    raise NotImplementedError("Covariates are not yet supported.")
        # if self._att is False:
        #    raise NotImplementedError(
        #        "Event study design with leads and lags is not yet supported."
        #    )

        # check if idname, tname and gname are in data
        # if self._idname not in self._data.columns:
        #    raise ValueError(f"The variable {self._idname} is not in the data.")
        # if self._tname not in self._data.columns:
        #    raise ValueError(f"The variable {self._tname} is not in the data.")
        # if self._gname not in self._data.columns:
        #    raise ValueError(f"The variable {self._gname} is not in the data.")

        # check if tname and gname are of type int (either int 64, 32, 8)
        if self._data[self._tname].dtype not in [
            "int64",
            "int32",
            "int8",
            "float64",
            "float32",
        ]:
            raise ValueError(
                f"""The variable {self._tname} must be of a numeric type, and more
                specifically, in the format YYYYMMDDHHMMSS. I.e. either 2012, 2013,
                etc. or 201201, 201202, 201203 etc."""
            )
        if self._data[self._gname].dtype not in [
            "int64",
            "int32",
            "int8",
            "float64",
            "float32",
        ]:
            raise ValueError(
                f"""The variable {self._tname} must be of a numeric type, and more
                specifically, in the format YYYYMMDDHHMMSS. I.e. either 2012, 2013,
                etc. or 201201, 201202, 201203 etc."""
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
