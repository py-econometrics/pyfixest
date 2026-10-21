from typing import cast

import pandas as pd

from pyfixest.did.did import DID
from pyfixest.estimation import feols
from pyfixest.estimation.models.feols_ import Feols


class TWFE(DID):
    r"""
    Internal wrapper for a static two-way fixed-effects treatment model.

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

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        xfml: str | None = None,
        att: bool = True,
        cluster: str | None = "idname",
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

        The vcov matrix is calculated via the [Feols](/reference/estimation.models.feols_.Feols.qmd) object.

        Notes
        -----
        Method not needed.
        """
        pass

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (500, 300),
        yintercept: int | None = None,
        xintercept: int | None = None,
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
