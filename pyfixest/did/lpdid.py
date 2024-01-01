import pandas as pd
import numpy as np

from pyfixest.estimation import feols
from pyfixest.visualize import _coefplot
from pyfixest.did.did import DID

from typing import Optional, Union, Dict


class LPDID(DID):
    def __init__(
        self,
        data,
        yname,
        idname,
        tname,
        gname,
        xfml,
        att,
        cluster,
        vcov,
        pre_window,
        post_window,
        never_treated,
    ):
        # if att:
        #    raise NotImplementedError("ATT is not yet supported.")

        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)
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

        if pre_window is None:
            pre_window = int(np.min(rel_years))
        if post_window is None:
            post_window = int(np.max(rel_years))

        # check that pre_window is in rel_years
        if pre_window not in rel_years:
            raise ValueError(f"pre_window must be in {rel_years}")
        # check that post_window is in rel_years
        if post_window not in rel_years:
            raise ValueError(f"post_window must be in {rel_years}")

        pre_window = np.abs(pre_window)

        if vcov is None:
            vcov = {"CRV1": idname}

        self._vcov = vcov
        self._pre_window = pre_window
        self._post_window = post_window
        self._never_treated = never_treated
        self._xfml = xfml
        self._estimator = "lpdid"

    def estimate(self):
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

    def vcov(self):
        pass

    def iplot(
        self,
        alpha=0.05,
        figsize=(500, 300),
        yintercept=None,
        xintercept=None,
        rotate_xticks=0,
        title="LPDID Event Study Estimate",
        coord_flip=False,
    ):
        df = self._coeftable
        df["fml"] = "lpdid"

        return _coefplot(
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
        return self._coeftable

    def summary(self):
        return self._coeftable


def lpdid(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    pre_window: Optional[int] = None,
    post_window: Optional[int] = None,
    never_treated: int = 0,
    att: bool = True,
    xfml=None,
) -> pd.DataFrame:
    """
    Estimate a  Difference-in-Differences / Event Study Model via Local Projections.
    Args:
        data: The DataFrame containing all variables.
        yname: The name of the dependent variable.
        idname: The name of the id variable.
        tname: Variable name for calendar period.
        gname: unit-specific time of initial treatment.
        vcov: The name of the cluster variable. If None, then defaults to {"CRV1": idname}. Either "iid", "hetero", or a dictionary, e.g. {"CRV1": idname} or
              {"CRV3": "idname"}. You can pass anything that is accepted by the vcov argument of feols.
        pre_window: The number of periods before the treatment to include in the estimation. Default is None, which means that the pre_window is set to the minimum
                    relative year in the data.
        post_window: The number of periods after the treatment to include in the estimation. Default is None, which means that the post_window is set to the maximum
                     relative year in the data.
        never_treated: Value in gname that indicates that a unit was never treated. By default, never treated units are assumed to
                       have value gname = 0.
        att: Whether to estimate the pooled average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags / the ATT for each period. Default is False.

        xfml: Optional formula for the covariates. Not yet supported. E.g. "X1 + X2 + X3".
    Returns:
        A data frame with the estimated coefficients.
    """

    FIT = LPDID(
        data=data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        cluster={"CRV1": idname},  # just something to pass DID input checks
        vcov=vcov,
        pre_window=pre_window,
        post_window=post_window,
        never_treated=never_treated,
        att=att,
        xfml=xfml,
    )

    FIT.estimate()

    return FIT


def _lpdid_estimate(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    pre_window: Optional[int] = None,
    post_window: Optional[int] = None,
    att: bool = True,
    xfml=None,
) -> pd.DataFrame:
    """ "
    Estimate a  Difference-in-Differences / Event Study Model via Linear Projections.
    Args:
        data: The DataFrame containing all variables.
        yname: The name of the dependent variable.
        idname: The name of the id variable.
        tname: Variable name for calendar period.
        gname: unit-specific time of initial treatment.
        vcov: The name of the cluster variable. If None, then defaults to {"CRV1": idname}. Either "iid", "hetero", or a dictionary, e.g. {"CRV1": idname} or
              {"CRV3": "idname"}. You can pass anything that is accepted by the vcov argument of feols.
        pre_window: The number of periods before the treatment to include in the estimation. Default is None, which means that the pre_window is set to the minimum
                    relative year in the data.
        post_window: The number of periods after the treatment to include in the estimation. Default is None, which means that the post_window is set to the maximum
                     relative year in the data.
        never_treated: Value in gname that indicates that a unit was never treated. By default, never treated units are assumed to
                       have value gname = 0.
        att: Whether to estimate the average treatment effect on the treated (ATT) or a canonical event study design with all leads and lags. Default is True.
        xfml: Optional formula for the covariates. Not yet supported. E.g. "X1 + X2 + X3".
    Returns:
        A data frame with the estimated coefficients.
    """

    # the implementation here is highly influenced by Alex Cardazzi's R
    # code for the lpdid package: https://github.com/alexCardazzi/lpdid

    fit_all = []
    reweight = False

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
        fit_post = feols(fml=fml, data=data[sample_idx_post], vcov=vcov)
        fit_tidy_post = fit_post.tidy().xs("treat_diff")
        fit_tidy_post["N"] = int(fit_post._N)

        res = pd.DataFrame(fit_tidy_post).T

    else:
        for h in range(post_window + 1):
            data["Dy"] = data.groupby(idname)[yname].shift(-h) - data[f"{yname}_lag"]

            sample_idx = (data["treat_diff"] == 1) | (
                data.groupby(idname)["treat"].shift(-h) == 0
            )

            fit = feols(fml=fml, data=data[sample_idx], vcov=vcov)

            fit_tidy = fit.tidy().xs("treat_diff")
            fit_tidy["N"] = int(fit._N)
            fit_tidy.name = h
            fit_all.append(fit_tidy)

        for h in range(pre_window + 1):
            if h <= 1:
                # skip the reference period
                continue

            data["Dy"] = data.groupby(idname)[yname].shift(h) - data[f"{yname}_lag"]
            sample_idx = (data["treat_diff"] == 1) | (data["treat"] == 0)

            fit = feols(fml=fml, data=data[sample_idx], vcov=vcov)

            fit_tidy = fit.tidy().xs("treat_diff")
            fit_tidy["N"] = int(fit._N)
            fit_tidy.name = -h
            fit_all.append(fit_tidy)

        res = pd.DataFrame(fit_all).sort_index()
        res.index.name = "Coefficient"
        res.index = res.index.map(lambda x: f"time_to_treatment::{x}")

    return res


def _pooled_adjustment(df, y, pool_lead, idname):
    """
    Calculate post-treatment means rather than just using a single y value from t+h.

    Parameters:
    - df (pd.DataFrame): The dataset used in the analysis.
    - y (str): The column name that denotes the outcome variable.
    - pool_lead (int): The number of post-periods that should be used when calculating the post-treatment mean.
    - idname (str): The name of the id variable.

    Returns:
    - pd.Series: The average of all future values in the analysis.
    """
    # Initialize lead variable
    x = 0

    # Calculate lead sum
    for k in range(0, pool_lead + 1, 1):
        x += df.groupby(idname)[y].shift(-k)

    # Average the lead sum
    x /= pool_lead + 1

    return x
