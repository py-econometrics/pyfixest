from abc import ABC, abstractmethod


class DID(ABC):
    @abstractmethod
    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted.
            gname: unit-specific time of initial treatment. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted. Never treated units
                   must have a value of 0.
            xfml: The formula for the covariates.
            estimator: The estimator to use. Options are "did2s".
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
            cluster: The name of the cluster variable.
        Returns:
            None
        """

        # do some checks here

        self._data = data.copy()
        self._yname = yname
        self._idname = idname
        self._tname = tname
        self._gname = gname
        self._xfml = xfml
        self._att = att
        self._cluster = cluster

        #if self._xfml is not None:
        #    raise NotImplementedError("Covariates are not yet supported.")
        #if self._att is False:
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
        if self._data[self._tname].dtype not in ["int64", "int32", "int8", "float64", "float32"]:
            raise ValueError(
                f"The variable {self._tname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS."
            )
        if self._data[self._gname].dtype not in ["int64", "int32", "int8", "float64", "float32"]:
            raise ValueError(
                f"The variable {self._gname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS."
            )

        # check if there is a never treated unit
        #if 0 not in self._data[self._gname].unique():
        #    raise ValueError(f"There must be at least one unit that is never treated.")

        # create a treatment variable
        self._data["ATT"] = (self._data[self._tname] >= self._data[self._gname]) * (
            self._data[self._gname] > 0
        )

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def vcov(self):
        pass

    @abstractmethod
    def iplot(self):
        pass

    @abstractmethod
    def tidy(self):
        pass

    @abstractmethod
    def summary(self):
        pass
