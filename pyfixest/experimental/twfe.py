from pyfixest.estimation import feols
from pyfixest.exceptions import NotImplementedError
from pyfixest.experimental.did import DID


class TWFE(DID):

    """
    Estimate a Difference-in-Differences model using the two-way fixed effects estimator.
    """

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
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
            cluster (str): The name of the cluster variable.
        Returns:
            None
        """

        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)

        self._estimator = "twfe"

        if self._xfml is not None:
            self._fml = f"{yname} ~ ATT + {xfml} | {idname} + {tname}"
        else:
            self._fml = f"{yname} ~ ATT | {idname} + {tname}"

    def estimate(self):
        _fml = self._fml
        _data = self._data

        fit = feols(fml=_fml, data=_data)
        self._fit = fit

        return fit

    def vcov(self):
        """
        Method not needed. The vcov matrix is calculated via the `Feols` object.
        """

        pass

    def iplot(
        self,
        alpha=0.05,
        figsize=(500, 300),
        yintercept=None,
        xintercept=None,
        rotate_xticks=0,
        title="TWFE Event Study Estimate",
        coord_flip=False,
    ):
        self.iplot(
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

    def tidy(self):
        return self.tidy()

    def summary(self):
        return self.summary()