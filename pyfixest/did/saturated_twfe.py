from typing import Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyfixest.estimation.estimation import feols
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.literals import (
    VcovTypeOptions,
)

from .did2s import DID


class SaturatedEventStudy(DID):
    """
    Saturated event study with cohort-specific effect curves.

    Attributes
    ----------
    data : pd.DataFrame
        Dataframe containing the data.
    yname : str
        Name of the outcome variable.
    idname : str
        Name of the unit identifier variable.
    tname : str
        Name of the time variable.
    gname : str
        Name of the treatment variable.
    cluster : str
        The name of the cluster variable.
    xfml : str
        The formula for the fixed effects.
    att : bool
        Whether to use the average treatment effect.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        att: bool = True,
        cluster: Optional[str] = None,
        xfml: Optional[str] = None,
    ):
        super().__init__(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            cluster=cluster,
            xfml=xfml,
            att=att,
        )
        self._estimator = "Saturated Event Study"

        # create a treatment variable
        self._data["ATT"] = (self._data[self._tname] >= self._data[self._gname]) * (
            self._data[self._gname] > 0
        )

    def estimate(self):
        """
        Estimate the model.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the estimates.
        """
        self.mod = _saturated_event_study(
            self._data,
            outcome=self._yname,
            treatment="ATT",
            time_id=self._tname,
            unit_id=self._idname,
        )

        return self.mod

    # !TODO - implement the rest of the methods
    def vcov(self):
        """
        Get the covariance matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the covariance matrix.
        """
        pass

    def iplot(self):
        """Plot DID estimates."""
        pass

    def tidy(self):
        """Tidy result dataframe."""
        return self.mod.tidy()

    def summary(self):
        """Get summary table."""
        return self.mod.summary()


######################################################################


def _saturated_event_study(
    df: pd.DataFrame,
    outcome: str = "outcome",
    treatment: str = "treated",
    time_id: str = "time",
    unit_id: str = "unit",
    ax: plt.Axes = None,
):
    ######################################################################
    # this chunk creates gname internally here - assume that data already contains it?
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["first_treated_period"] = (
        df["first_treated_period"].replace(np.nan, 0).astype("int")
    )
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    cohort_dummies = pd.get_dummies(
        df.first_treated_period, drop_first=True, prefix="cohort_dummy"
    )
    df_int = pd.concat([df, cohort_dummies], axis=1)
    ######################################################################
    # formula
    ff = f"""
                {outcome} ~
                {"+".join([f"i(rel_time, {x}, ref = -1.0)" for x in df_int.filter(like="cohort_dummy", axis=1).columns])}
                | {unit_id} + {time_id}
                """
    m = feols(ff, df_int, vcov={"CRV1": unit_id})
    # !TODO move this into a separate function / integrate logic into iplot method that overrides feols's plot
    if ax:
        # plot
        res = m.tidy()
        # create a dict with cohort specific effect curves
        res_dict: dict[str, dict[str, pd.DataFrame | np.ndarray]] = {}
        for c in cohort_dummies.columns:
            res_cohort = res.filter(like=c, axis=0)
            event_time = (
                res_cohort.index.str.extract(r"\[T\.(-?\d+\.\d+)\]")
                .astype(float)
                .values.flatten()
            )
            res_dict[c] = {"est": res_cohort, "time": event_time}

        cmp = plt.get_cmap("Set1")
        for i, (k, v) in enumerate(res_dict.items()):
            ax.plot(v["time"], v["est"]["Estimate"], marker=".", label=k, color=cmp(i))
            ax.fill_between(
                v["time"], v["est"]["2.5%"], v["est"]["97.5%"], alpha=0.2, color=cmp(i)
            )
        ax.axvline(-1, color="black", linestyle="--")
        ax.axhline(0, color="black", linestyle=":")
    return m


def test_treatment_heterogeneity(
    df: pd.DataFrame,
    outcome: str = "Y_it",
    treatment: str = "W_it",
    unit_id: str = "unit_id",
    time_id: str = "time_id",
    retmod: bool = False,
):
    """
    Test for treatment heterogeneity in the event study design.

    For details, see https://github.com/apoorvalal/TestingInEventStudies

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    outcome : str
        Name of the outcome variable.
    treatment : str
        Name of the treatment variable.
    unit_id : str
        Name of the unit identifier variable.
    time_id : str
        Name of the time variable.
    retmod : bool
        Whether to return the model object.

    Returns
    -------
    float
        The p-value of the test.

    Examples
    --------
    ```{python}
    import pyfixest as pf
    from pyfixest.utils.dgps import get_sharkfin

    df_one_cohort = get_sharkfin()
    df_one_cohort.head()

    pf.test_treatment_heterogeneity(
        df_one_cohort,
        outcome = "Y",
        treatment = "treat",
        unit_id = "unit",
        time_id = "year"
    )
    ```
    """
    # Get treatment timing info
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["first_treated_period"] = (
        df["first_treated_period"].replace(np.nan, 0).astype("int")
    )
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    # Create dummies but drop TWO cohorts - one serves as base for pooled effects
    cohort_dummies = pd.get_dummies(
        df.first_treated_period, drop_first=True, prefix="cohort_dummy"
    ).iloc[
        :, 1:
    ]  # drop an additional cohort - drops interactions for never treated and baseline

    df_int = pd.concat([df, cohort_dummies], axis=1)

    # Modified formula with base effects + cohort-specific deviations
    ff = f"""
    {outcome} ~
    i(rel_time, ref=-1.0) +
    {"+".join([f"i(rel_time, {x}, ref = -1.0)" for x in df_int.filter(like="cohort_dummy", axis=1).columns])}
    | {unit_id} + {time_id}
    """

    model: Feols = cast(Feols, feols(ff, df_int, vcov={"CRV1": unit_id}))
    P = model.coef().shape[0]

    if retmod:
        return model
    mmres = model.tidy().reset_index()
    mmres[["time", "cohort"]] = mmres.Coefficient.str.split(":", expand=True)
    mmres["time"] = mmres.time.str.extract(r"\[T\.(-?\d+\.\d+)\]").astype(float)
    mmres["cohort"] = mmres.cohort.str.extract(r"(\d+)")
    # indices of coefficients that are deviations from common event study coefs
    event_study_coefs = mmres.loc[~(mmres.cohort.isna()) & (mmres.time > 0)].index
    # Method 2 (K x P) - more efficient
    K = len(event_study_coefs)
    R2 = np.zeros((K, P))
    for i, idx in enumerate(event_study_coefs):
        R2[i, idx] = 1

    test_result = model.wald_test(R=R2, distribution="chi2")
    return test_result["pvalue"]


def _test_dynamics(
    df: pd.DataFrame,
    outcome: str = "Y",
    treatment: str = "W",
    time_id: str = "time",
    unit_id: str = "unit",
    vcv: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
):
    if vcv is None:
        vcv = {"CRV1": unit_id}

    # Fit models
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    restricted: Feols = cast(
        Feols, feols(f"{outcome} ~ i({treatment}) | {unit_id} + {time_id}", df)
    )
    unrestricted: Feols = cast(
        Feols,
        feols(f"{outcome} ~ i(rel_time, ref=0) | {unit_id} + {time_id}", df, vcov=vcv),
    )
    # Get the restricted estimate
    restricted_effect = restricted.coef().iloc[0]
    # Create R matrix - each row tests one event study coefficient
    # against restricted estimate
    n_evstudy_coefs = unrestricted.coef().shape[0]
    R = np.eye(n_evstudy_coefs)
    # q vector is the restricted estimate repeated
    q = np.repeat(restricted_effect, n_evstudy_coefs)
    # Conduct Wald test
    pv = unrestricted.wald_test(R=R, q=q, distribution="chi2")["pvalue"]
    return pv
