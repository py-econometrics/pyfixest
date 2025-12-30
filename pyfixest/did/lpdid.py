from typing import Optional, Union, cast

import numpy as np
import pandas as pd

from pyfixest.did.did import DID
from pyfixest.estimation.estimation import feols
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.literals import VcovTypeOptions
from pyfixest.report.visualize import _HAS_LETS_PLOT, _coefplot


class LPDID(DID):
    """
    A class used to represent the Local Projections Diff-in-Diff Estimator.

    Attributes
    ----------
    data : pandas.DataFrame
        The DataFrame containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the identifier variable.
    tname : str
        The name of the time variable.
    gname : str
        The name of the group variable.
    xfml : str
        The transformation to apply to the data.
    att : str
        The attribute to consider in the model.
    cluster : str
        The cluster to consider in the model.
    vcov : str
        The type of variance-covariance matrix to use.
    pre_window : tuple
        The pre-treatment window to consider in the model.
    post_window : tuple
        The post-treatment window to consider in the model.
    never_treated : bool
        Whether to consider never-treated units in the model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        xfml: str,
        att: bool,
        cluster: str,
        vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
        pre_window: Optional[int] = None,
        post_window: Optional[int] = None,
        never_treated: Optional[int] = 0,
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
        assert isinstance(xfml, str) or xfml is None, "xfml must be a string or None"

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
        figsize: Optional[tuple[int, int]] = None,
        yintercept: Optional[int] = None,
        xintercept: Optional[int] = None,
        rotate_xticks: int = 0,
        title: str = "LPDID Event Study Estimate",
        coord_flip: bool = False,
        plot_backend: str = "lets_plot" if _HAS_LETS_PLOT else "matplotlib",
    ):
        """
        Create coefficient plots.

        Parameters
        ----------
        alpha : float, optional
            Significance level for visualization options. Defaults to 0.05.
        figsize : tuple[int, int], optional
            Size of the plot (width, height) in inches. Defaults to (500, 300).
        yintercept : float, optional
            Value to set as the y-axis intercept (vertical line).
            Defaults to None.
        xintercept : float, optional
            Value to set as the x-axis intercept (horizontal line).
            Defaults to None.
        rotate_xticks : int, optional
            Rotation angle for x-axis tick labels. Defaults to 0.
        title : str, optional
            Title of the plot.
        coord_flip : bool, optional
            Whether to flip the coordinates of the plot. Defaults to False.
        plot_backend: str, optional
            The plotting backend to use between "lets_plot" (default) and "matplotlib".

        Returns
        -------
        lets-plot figure
            A lets-plot or matplotlib figure with coefficient estimates and confidence intervals.
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

    def tidy(self):  # noqa: D102
        return self._coeftable

    def summary(self):  # noqa: D102
        return self._coeftable


def _lpdid_estimate(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    pre_window: int,
    post_window: int,
    att: bool = True,
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    xfml: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate a  Difference-in-Differences / Event Study Model via Linear Projections.

    Parameters
    ----------
    data: pandas.DataFrame
        The DataFrame containing all variables.
    yname: str
        The name of the dependent variable.
    idname: str
        The name of the id variable.
    tname: str
        Variable name for calendar period.
    gname:  str
        unit-specific time of initial treatment.
    vcov: VcovTypeOptions, dict[str, str], None
        The name of the cluster variable. If None, then defaults to {"CRV1": idname}.
        Either "iid", "hetero", or a dictionary, e.g. {"CRV1": idname} or
        {"CRV3": "idname"}. You can pass anything that is accepted by the vcov
        argument of feols.
    pre_window: int
        The number of periods before the treatment to include in the
        estimation. Default is None, which means that the pre_window is set to
        the minimum relative year in the data.
    post_window: int
        The number of periods after the treatment to include in the
        estimation. Default is None, which means that the post_window is set to
        the maximum relative year in the data.
    never_treated: int
        Value in gname that indicates that a unit was never treated.
        By default, never treated units are assumed to have value gname = 0.
    att: bool
        Whether to estimate the average treatment effect on the treated (ATT)
        or a canonical event study design with all leads and lags.
        Default is True.
    xfml: str
        Optional formula for the covariates. Not yet supported.
        E.g. "X1 + X2 + X3".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the estimated coefficients.
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
        fit_tidy_post["N"] = int(fit_post._N)

        res = pd.DataFrame(fit_tidy_post).T

    else:
        for h in range(post_window + 1):
            data["Dy"] = data.groupby(idname)[yname].shift(-h) - data[f"{yname}_lag"]

            sample_idx = (data["treat_diff"] == 1) | (
                data.groupby(idname)["treat"].shift(-h) == 0
            )

            fit = cast(Feols, feols(fml=fml, data=data[sample_idx], vcov=vcov))

            fit_tidy = cast(pd.Series, fit.tidy().xs("treat_diff"))
            fit_tidy["N"] = int(fit._N)
            fit_tidy.name = h
            fit_all.append(fit_tidy)

        for h in range(pre_window + 1):
            if h <= 1:
                # skip the reference period
                continue

            data["Dy"] = data.groupby(idname)[yname].shift(h) - data[f"{yname}_lag"]
            sample_idx = (data["treat_diff"] == 1) | (data["treat"] == 0)

            fit = cast(Feols, feols(fml=fml, data=data[sample_idx], vcov=vcov))

            fit_tidy = cast(pd.Series, fit.tidy().xs("treat_diff"))
            fit_tidy["N"] = int(fit._N)
            fit_tidy.name = -h
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
