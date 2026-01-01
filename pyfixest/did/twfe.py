from typing import Optional, cast

import pandas as pd

from pyfixest.did.did import DID
from pyfixest.estimation import feols
from pyfixest.estimation.feols_ import Feols


class TWFE(DID):
    """
    Estimate a Two-way Fixed Effects model.

    Estimate a Difference-in-Differences model using the two-way fixed effects
    estimator.

    Attributes
    ----------
    data: pandas.DataFrame
        The DataFrame containing all variables.
    yname: str
        The name of the dependent variable.
    idname: str
        The name of the id variable.
    tname: str
        Variable name for calendar period. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted.
    gname: str
        Unit-specific time of initial treatment. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted. Never treated units
        must have a value of 0.
    xfml: str
        The formula for the covariates.
    att: bool
        Whether to estimate the average treatment effect on the treated (ATT) or the
        canonical event study design with all leads and lags. Default is True.
    cluster: Optional[str]
        The name of the cluster variable.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        xfml: Optional[str] = None,
        att: bool = True,
        cluster: Optional[str] = "idname",
    ) -> None:
        super().__init__(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )

        self._estimator = "twfe"

        if self._xfml is not None:
            self._fml = f"{yname} ~ is_treated + {xfml} | {idname} + {tname}"
        else:
            self._fml = f"{yname} ~ is_treated | {idname} + {tname}"

    def estimate(self):
        """Estimate the TWFE model."""
        _fml = self._fml
        _data = self._data

        fit = cast(Feols, feols(fml=_fml, data=_data))
        self._fit = fit

        return fit

    def vcov(self):
        """
        Variance-covariance matrix.

        The vcov matrix is calculated via the [Feols(/reference/Feols.qmd) object.

        Notes
        -----
        Method not needed.
        """
        pass

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (500, 300),
        yintercept: Optional[int] = None,
        xintercept: Optional[int] = None,
        rotate_xticks: int = 0,
        title: str = "TWFE Event Study Estimate",
        coord_flip: bool = False,
    ):
        """Plot TWFE estimates."""
        self.iplot(
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

    def tidy(self):  # noqa: D102
        return self.tidy()

    def summary(self):  # noqa: D102
        return self.summary()
