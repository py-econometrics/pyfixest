from typing import cast

import numpy as np
import pandas as pd

from pyfixest.did.did import DID
from pyfixest.estimation import feols
from pyfixest.estimation.internals.literals import VcovTypeOptions
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.report.visualize import _HAS_LETS_PLOT, _coefplot


class LPDID(DID):
    r"""
    Fitted local-projections difference-in-differences results.

    Returned by [lpdid()](/reference/did.estimation.lpdid.qmd); use that
    function to estimate a model. This result has its own small interface:
    `fit.tidy()` returns the coefficient table, `fit.summary()` returns the
    same table, and `fit.iplot()` plots it. It is not a `Feols` result.

    Notes
    -----
    The coefficient table contains Estimate, Std. Error, t value, Pr(>|t|),
    2.5%, 97.5%, and N (the observations used for each regression). For a
    dynamic model its index names event times as `time_to_treatment::<h>`;
    relative period -1 is the omitted reference. A pooled ATT has one row.

    This result does not expose fitted-data attributes, predictions, covariance
    updates, or the general `Feols` post-estimation interface. Its `vcov()`
    estimator hook is not an operation for retrieving or changing inference.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_motherhood_event_study_data(n_per_country=8, seed=2026)
    fit = pf.lpdid(
        data, yname="log_earnings", idname="unit", tname="year", gname="g",
        pre_window=-3, post_window=3, att=False,
    )
    fit.tidy()
    ```

    See Also
    --------
    [lpdid()](/reference/did.estimation.lpdid.qmd) : Arguments and pooled or dynamic examples.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Estimator selection and interpretation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        xfml: str | None,
        att: bool,
        cluster: str,
        vcov: VcovTypeOptions | dict[str, str] | None = None,
        pre_window: int | None = None,
        post_window: int | None = None,
        never_treated: int | None = 0,
    ):
        # if att:
        #    raise NotImplementedError("ATT is not yet supported.")

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
        if not (isinstance(xfml, str) or xfml is None):
            raise TypeError("xfml must be a string or None.")

        data = data.copy()
        data.sort_values([idname, tname], inplace=True)
        data[f"{yname}_lag"] = data.groupby(idname)[yname].shift(1)

        data["treat"] = np.where(data[gname] <= data[tname], 1, 0)
        data["rel_year"] = data[tname] - data[gname]

        # handle never treated units
        data["treat"] = np.where(data[gname] == never_treated, 0, data["treat"])
        data["rel_year"] = np.where(
            data[gname] == never_treated, np.inf, data["rel_year"]
        )

        data["treat_diff"] = data["treat"] - data.groupby(idname)["treat"].shift(1)
        data["treat_diff"] = np.where(data["treat_diff"] < 0, 0, data["treat_diff"])

        self._data = data

        rel_years = np.unique(data["rel_year"])
        rel_years = rel_years[np.isfinite(rel_years)]

        pre_window_int = int(np.min(rel_years)) if pre_window is None else pre_window
        post_window_int = int(np.max(rel_years)) if post_window is None else post_window

        # check that pre_window is in rel_years
        if pre_window_int not in rel_years:
            raise ValueError(f"pre_window must be in {rel_years}")
        # check that post_window is in rel_years
        if post_window_int not in rel_years:
            raise ValueError(f"post_window must be in {rel_years}")

        pre_window_int = np.abs(pre_window_int)

        if vcov is None:
            vcov = {"CRV1": idname}

        self._vcov = vcov
        self._pre_window = pre_window_int
        self._post_window = post_window_int
        self._never_treated = never_treated
        self._xfml = xfml
        self._estimator = "lpdid"

    def estimate(self):
        """Estimate the DID model."""
        self._coeftable = _lpdid_estimate(
            data=self._data,
            yname=self._yname,
            idname=self._idname,
            tname=self._tname,
            vcov=self._vcov,
            pre_window=self._pre_window,
            post_window=self._post_window,
            att=self._att,
            xfml=self._xfml,
        )

    def vcov(self):  # noqa: D102
        pass

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple[int, int] | None = None,
        yintercept: int | None = None,
        xintercept: int | None = None,
        rotate_xticks: int = 0,
        title: str = "LPDID Event Study Estimate",
        coord_flip: bool = False,
        plot_backend: str = "lets_plot" if _HAS_LETS_PLOT else "matplotlib",
    ):
        r"""
        Create coefficient plots.

        Parameters
        ----------
        alpha : float, optional
            Only 0.05 (the default) is currently supported. The stored coefficient
            table contains fixed 95% limits named `2.5%` and `97.5%`; this method
            does not recompute intervals. Other values request missing columns
            and can raise KeyError.
        figsize : tuple[int, int], optional
            Size of the plot (width, height): in pixels for lets-plot, in
            inches for matplotlib. Defaults to `None`, which uses (500, 300) for
            lets-plot and `(10, 6)` for matplotlib.
        yintercept : float, optional
            Value at which to draw a horizontal reference line on the y-axis.
            Defaults to `None`.
        xintercept : float, optional
            Value at which to draw a vertical reference line on the x-axis.
            Defaults to `None`.
        rotate_xticks : int, optional
            Rotation angle for x-axis tick labels. Defaults to 0.
        title : str, optional
            Title of the plot.
        coord_flip : bool, optional
            Whether to flip the coordinates of the plot. Defaults to `False`.
        plot_backend: str, optional
            `lets_plot` if installed, otherwise `matplotlib`.

        Returns
        -------
        matplotlib.figure.Figure or lets_plot.plot.core.PlotSpec
            Figure for the matplotlib backend or plot specification for lets-plot,
            showing coefficient estimates with confidence intervals. Save with
            `fig.savefig(...)` or `lets_plot.ggsave(plot, ...)`, respectively.

        See Also
        --------
        [LPDID](/reference/did.lpdid.LPDID.qmd) : Fitted coefficient table and example.
        [iplot()](/reference/report.iplot.qmd) : Plot `i()` terms from `Feols` results.
        """
        df = self._coeftable.copy().reset_index()
        df["fml"] = "lpdid"

        return _coefplot(
            plot_backend=plot_backend,
            df=df,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            flip_coord=coord_flip,
        )

    def tidy(self):
        r"""
        Return the coefficient table as a pandas `DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Estimates, standard errors, t statistics, p values, 95% confidence
            limits, and N; dynamic rows are indexed by relative period.
            The stored table itself is returned, so copy it before changing it.

        See Also
        --------
        [LPDID](/reference/did.lpdid.LPDID.qmd) : Result structure and executable example.
        """
        return self._coeftable

    def summary(self):
        r"""
        Return the coefficient table as a pandas `DataFrame`.

        Unlike `pf.summary()`, this method returns a table rather than printing.

        Returns
        -------
        pandas.DataFrame
            The same stored coefficient table returned by `fit.tidy()`.

        See Also
        --------
        [LPDID](/reference/did.lpdid.LPDID.qmd) : Result structure and executable example.
        """
        return self._coeftable


def _lpdid_estimate(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    pre_window: int,
    post_window: int,
    att: bool = True,
    vcov: VcovTypeOptions | dict[str, str] | None = None,
    xfml: str | None = None,
) -> pd.DataFrame:
    """
    Estimate a  Difference-in-Differences / Event Study Model via Local Projections.

    Parameters
    ----------
    data: pandas.DataFrame
        The `DataFrame` containing all variables.
    yname: str
        The name of the dependent variable.
    idname: str
        The name of the id variable.
    tname: str
        Variable name for calendar period.
    gname:  str
        unit-specific time of initial treatment.
    vcov: VcovTypeOptions, dict[str, str], None
        The name of the cluster variable. If `None`, then defaults to `{"CRV1": idname}`.
        Either `iid`, `hetero`, or a dictionary, e.g. `{"CRV1": idname}` or
        {`CRV3`: "idname"}. You can pass anything that is accepted by the vcov
        argument of feols.
    pre_window: int
        The number of periods before the treatment to include in the
        estimation. Default is `None`, which means that the `pre_window` is set to
        the minimum relative year in the data.
    post_window: int
        The number of periods after the treatment to include in the
        estimation. Default is `None`, which means that the `post_window` is set to
        the maximum relative year in the data.
    never_treated: int
        Value in gname that indicates that a unit was never treated.
        By default, never treated units are assumed to have value gname = 0.
    att: bool
        Whether to estimate the average treatment effect on the treated (ATT)
        or a canonical event study design with all leads and lags.
        Default is `True`.
    xfml: str
        Optional formula for the covariates, e.g. "X1 + X2 + X3". Added to
        every local projection regression.

    Returns
    -------
    pandas.DataFrame
        A `DataFrame` with the estimated coefficients.
    """
    # the implementation here is highly influenced by Alex Cardazzi's R
    # code for the lpdid package: https://github.com/alexCardazzi/lpdid

    fit_all = []
    reweight = False  # noqa: F841

    if xfml is None:
        fml = f"Dy ~ treat_diff | {tname}"
    else:
        fml = f"Dy ~ treat_diff + {xfml} | {tname}"

    if att:
        # post window

        data[f"{yname}_post"] = _pooled_adjustment(data, yname, post_window, idname)
        data["Dy"] = data[f"{yname}_post"] - data[f"{yname}_lag"]
        sample_idx_post = (data["treat_diff"] == 1) | (
            data.groupby(idname)["treat"].shift(-post_window) == 0
        )
        fit_post = cast(Feols, feols(fml=fml, data=data[sample_idx_post], vcov=vcov))
        fit_tidy_post = fit_post.tidy().xs("treat_diff")
        fit_tidy_post["N"] = int(fit_post.sample_info.n_obs)

        res = pd.DataFrame(fit_tidy_post).T

    else:
        for h in range(post_window + 1):
            data["Dy"] = data.groupby(idname)[yname].shift(-h) - data[f"{yname}_lag"]

            sample_idx = (data["treat_diff"] == 1) | (
                data.groupby(idname)["treat"].shift(-h) == 0
            )

            fit = cast(Feols, feols(fml=fml, data=data[sample_idx], vcov=vcov))

            fit_tidy = cast(pd.Series, fit.tidy().xs("treat_diff"))
            fit_tidy["N"] = int(fit.sample_info.n_obs)
            fit_tidy.name = h  # type: ignore[union-attr]
            fit_all.append(fit_tidy)

        for h in range(pre_window + 1):
            if h <= 1:
                # skip the reference period
                continue

            data["Dy"] = data.groupby(idname)[yname].shift(h) - data[f"{yname}_lag"]
            sample_idx = (data["treat_diff"] == 1) | (data["treat"] == 0)

            fit = cast(Feols, feols(fml=fml, data=data[sample_idx], vcov=vcov))

            fit_tidy = cast(pd.Series, fit.tidy().xs("treat_diff"))
            fit_tidy["N"] = int(fit.sample_info.n_obs)
            fit_tidy.name = -h  # type: ignore[union-attr]
            fit_all.append(fit_tidy)

        res = pd.DataFrame(fit_all).sort_index()
        res.index.name = "Coefficient"
        res.index = res.index.map(lambda x: f"time_to_treatment::{x}")

    return res


def _pooled_adjustment(
    df: pd.DataFrame, y: str, pool_lead: int, idname: str
) -> np.ndarray:
    """
    Calculate post-treatment means rather than just using a single y value from t+h.

    Parameters
    ----------
    - df: pandas.DataFrame
        The dataset used in the analysis.
    - y: str
        The column name that denotes the outcome variable.
    - pool_lead: int
        The number of post-periods that should be used when calculating the
        post-treatment mean.
    - idname: str
        The name of the id variable.

    Returns
    -------
    np.ndarray
        The average of all future values in the analysis.
    """
    # Initialize lead variable
    x = np.zeros(df.shape[0])

    # Calculate lead sum
    for k in range(0, pool_lead + 1, 1):
        x += df.groupby(idname)[y].shift(-k).to_numpy().astype(float)

    # Average the lead sum
    x /= pool_lead + 1

    return x
