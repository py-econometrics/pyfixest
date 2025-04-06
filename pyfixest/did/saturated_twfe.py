from typing import Optional
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation.estimation import feols

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
        self.mod, self._res_cohort_eventtime_dict = _saturated_event_study(
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
        cmp = plt.get_cmap("Set1")

        fig, ax = plt.subplots(figsize=(10, 6))

        for cohort, values in self._res_cohort_eventtime_dict.items():
            time = np.array(values["time"], dtype=float)
            est = values["est"]["Estimate"].astype(float).values
            ci_lower = values["est"]["2.5%"].astype(float).values
            ci_upper = values["est"]["97.5%"].astype(float).values

            ax.plot(time, est, marker="o", label=cohort, color=cmp(len(ax.lines)))
            ax.fill_between(
                time, ci_lower, ci_upper, alpha=0.3, color=cmp(len(ax.lines))
            )

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Coefficient (with 95% CI)")
        ax.set_title("Event Study Estimates by Cohort")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def tidy(self):
        """Tidy result dataframe."""
        return self.mod.tidy()

    def summary(self):
        """Get summary table."""
        return self.mod.summary()

    def test_treatment_heterogeneity(self) -> pd.Series:
        """
        Test for treatment heterogeneity in the event study design.

        Parameters
        ----------
        by : str, optional

                The type of test to perform. Can be either "cohort" or "time".
                Default is "cohort". If "cohort", tests for treatment heterogeneity
                across cohorts as in Lal (2025). See https://arxiv.org/abs/2503.05125
                for details.
        """
        return _test_treatment_heterogeneity(
            model = self.mod if isinstance(self, SaturatedEventStudy) else self,
        )

    def aggregate(self, agg = "cohort", weighting: Optional[str] = None) -> pd.Series:
        """
        Aggregate the fully interacted event study estimates by relative time, cohort, and time.

        Parameters
        ----------
        agg : str, optional

                The type of aggregation to perform. Can be either "att" or "cohort" or "period".
                Default is "att". If "att", computes the average treatment effect on the treated.
                If "cohort", computes the average treatment effect by cohort. If "period",
                computes the average treatment effect by period.

        weighting : str, optional

                    The type of weighting to use. Can be either ...

        Returns
        -------
        pd.Series
            A Series containing the aggregated estimates.
        """

        model = self.mod if isinstance(self, SaturatedEventStudy) else self

        cohort_event_dict = model._res_cohort_eventtime_dict
        cohort_list = list(cohort_event_dict.keys())
        period_set = sorted(set(t for x in cohort_list for t in cohort_event_dict[x]["time"].tolist()))

        coefs = model._beta_hat
        coefnames = model._coefnames
        # just set to one for now, change later
        weights = np.ones(len(coefs))

        if agg == "cohort":

            df_agg = pd.DataFrame(index=period_set, columns=["coef", "se", "pval", "tstat","conf_lower", "conf_upper"])
            df_agg.index.name = "period"

        for period in period_set:
            R = np.zeros(len(coefs))
            for cohort in cohort_list:
                cohort_pattern = rf"\[(?:T\.?)?{re.escape(str(period))}\]:.*$"
                matched_indices = [i for i, name in enumerate(coefnames) if re.search(cohort_pattern, name)]
                R[matched_indices] = weights[matched_indices]

            coef = np.sum(R * coefs)
            se = np.sqrt(R @ model._vcov @ R.T)
            tstat = coef / se
            pval = 2 * (1 - norm.cdf(np.abs(tstat)))
            conf_lower = coef - 1.96 * se
            conf_upper = coef + 1.96 * se

            row = {
                "coef": coef,
                "se": se,
                "tstat": tstat,
                "pval": pval,
                "conf_lower": conf_lower,
                "conf_upper": conf_upper
            }

            df_agg.loc[period] = row

        return df_agg



######################################################################


def _saturated_event_study(
    df: pd.DataFrame,
    outcome: str = "outcome",
    treatment: str = "treated",
    time_id: str = "time",
    unit_id: str = "unit",
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
                {"+".join([f"i(rel_time, {x}, ref = -1.0)" for x in cohort_dummies.columns.tolist()])}
                | {unit_id} + {time_id}
                """
    m = feols(ff, df_int, vcov={"CRV1": unit_id})

    res = m.tidy()
    # create a dict with cohort specific effect curves
    res_cohort_eventtime_dict: dict[str, dict[str, pd.DataFrame | np.ndarray]] = {}
    for cohort in cohort_dummies.columns:
        res_cohort = res.filter(like=cohort, axis=0)
        event_time = (
            res_cohort.index.str.extract(r"\[(?:T\.)?(-?\d+(?:\.\d+)?)\]")
            .astype(float)
            .values.flatten()
        )
        res_cohort_eventtime_dict[cohort] = {"est": res_cohort, "time": event_time}

    return m, res_cohort_eventtime_dict


def _test_treatment_heterogeneity(
    model: SaturatedEventStudy,
) -> pd.Series:
    """
    Test for treatment heterogeneity in the event study design.

    For details, see https://github.com/apoorvalal/TestingInEventStudies

    Parameters
    ----------
    model : SaturatedEventStudy
        The fitted event study model

    Returns
    -------
    pd.Series

            A Series containing the p-value of the test and the test statistic.
    """
    mmres = model.tidy().reset_index()
    P = mmres.shape[0]
    mmres[["time", "cohort"]] = mmres.Coefficient.str.split(":", expand=True)
    mmres["time"] = mmres.time.str.extract(r"\[(?:T\.)?(-?\d+(?:\.\d+)?)\]").astype(float)
    mmres["cohort"] = mmres.cohort.str.extract(r"(\d+)")
    # indices of coefficients that are deviations from common event study coefs
    event_study_coefs = mmres.loc[~(mmres.cohort.isna()) & (mmres.time > 0)].index
    # Method 2 (K x P) - more efficient
    K = len(event_study_coefs)
    R2 = np.zeros((K, P))
    for i, idx in enumerate(event_study_coefs):
        R2[i, idx] = 1

    test_result = model.wald_test(R=R2, distribution="chi2")
    return test_result
