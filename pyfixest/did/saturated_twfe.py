import re
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation.estimation import feols
from pyfixest.estimation.feols_ import Feols

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
        Additional covariates to include in the model.
    att : bool
        Whether to use the average treatment effect.
    display_warning: bool
        Whether to display (some) warning messages.

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
        display_warning: bool = True,
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

        if display_warning:
            warnings.warn(
                "The SaturatedEventStudyClass is currently in beta. Please report any issues you may encounter."
            )

    def estimate(self) -> Feols:
        """
        Estimate the model.

        Returns
        -------
        Feols
            The fitted Feols model object.
        """
        self.mod, self._res_cohort_eventtime_dict = _saturated_event_study(
            self._data,
            outcome=self._yname,
            time_id=self._tname,
            unit_id=self._idname,
            cluster=self._cluster,
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

        _, ax = plt.subplots(figsize=(10, 6))

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
            model=self.mod if isinstance(self, SaturatedEventStudy) else self,
        )

    def aggregate(
        self, agg="period", weighting: Optional[str] = "shares"
    ) -> pd.DataFrame:
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

                    The type of weighting to use. Can be either 'shares' or 'variance'.

        Returns
        -------
        pd.Series
            A Series containing the aggregated estimates.
        """
        if agg not in ["period"]:
            raise ValueError("agg must be either 'period'")

        if weighting not in ["shares"]:
            raise ValueError("weighting must be 'shares'.")

        model = self.mod if isinstance(self, SaturatedEventStudy) else self

        cohort_event_dict = model._res_cohort_eventtime_dict
        cohort_list = list(cohort_event_dict.keys())
        period_set = sorted(
            set(t for x in cohort_list for t in cohort_event_dict[x]["time"].tolist())
        )

        coefs = model._beta_hat
        se = model._se
        coefnames = model._coefnames

        if weighting == "shares":
            weights_df = compute_period_weights(
                data=model._data,
                cohort=model._gname,
                period="rel_time",
                treatment="is_treated",
            ).set_index([self._gname, "rel_time"])

        treated_periods = list(period_set)

        df_agg = pd.DataFrame(
            index=pd.Index(treated_periods, name="period"),
            columns=["Estimate", "Std. Error", "t value", "Pr(>|t|)", "2.5%", "97.5%"],
        )

        for period in treated_periods:
            R = np.zeros(len(coefs))
            for cohort in cohort_list:
                cohort_pattern = rf"^(?:.+)::{period}:(?:.+)::{cohort}$"
                match_idx = [
                    i
                    for i, name in enumerate(coefnames)
                    if re.search(cohort_pattern, name)
                ]
                cohort_int = int(cohort.replace("cohort_dummy_", ""))
                R[match_idx] = (
                    weights_df.xs((cohort_int, period)).values[0]
                    if weighting == "shares"
                    else 1 / se[match_idx]
                )

            if weighting == "variance":
                R /= np.sum(R)

            res_dict = _compute_lincomb_stats(R=R, coefs=coefs, vcov=model._vcov)
            df_agg.loc[period] = pd.Series(res_dict)

        return df_agg

    def iplot_aggregate(self, agg="period", weighting: Optional[str] = "shares"):
        """
        Plot the aggregated estimates.

        Parameters
        ----------
        agg : str, optional
            The type of aggregation to perform. Can be either "att" or "cohort" or "period".
            Default is "att". If "att", computes the average treatment effect on the treated.
            If "cohort", computes the average treatment effect by cohort. If "period",
            computes the average treatment effect by period.

        weighting : str, optional
            The type of weighting to use. Can be either 'shares' or 'variance'.

        Returns
        -------
        None
        """
        df_agg = self.aggregate(agg=agg, weighting=weighting)

        time = np.array(df_agg.index, dtype=float).astype(float)
        est = df_agg["Estimate"].values.astype(float)
        ci_lower = df_agg["2.5%"].values.astype(float)
        ci_upper = df_agg["97.5%"].values.astype(float)

        cmp = plt.get_cmap("Set1")
        _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, est, marker="o", color=cmp(len(ax.lines)))
        ax.fill_between(time, ci_lower, ci_upper, alpha=0.3, color=cmp(len(ax.lines)))

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Coefficient (with 95% CI)")
        ax.set_title("Event Study Estimates")
        ax.legend()
        plt.tight_layout()
        plt.show()


def _compute_lincomb_stats(R: np.ndarray, coefs: np.ndarray, vcov: np.ndarray) -> dict:
    """
    Compute linear combination of coefficients and statistics of interest.

    Parameters
    ----------
    R : np.ndarray
        1D numpy array of weights (same length as coefs),
    coefs: np.ndarray
        1D numpy array of coefficient estimates,
    vcov: np.ndarray
        The covariance matrix of coefs (same dimension as len(coefs)),

    Returns
    -------
    dict
        A dictionary containing the coefficient, standard error, t-statistic,
        p-value, and confidence interval bounds.
    """
    coef_val = np.sum(R * coefs)
    se = np.sqrt(R @ vcov @ R)
    tstat = coef_val / se
    pval = 2 * (1 - norm.cdf(abs(tstat)))
    z = norm.ppf(0.975)

    ci_margin = z * se
    conf_lower = coef_val - ci_margin
    conf_upper = coef_val + ci_margin

    return {
        "Estimate": coef_val,
        "Std. Error": se,
        "t value": tstat,
        "Pr(>|t|)": pval,
        "2.5%": conf_lower,
        "97.5%": conf_upper,
    }


def _saturated_event_study(
    df: pd.DataFrame,
    outcome: str,
    time_id: str,
    unit_id: str,
    cluster: Optional[str] = None,
):
    ff = f"{outcome} ~ i(rel_time, first_treated_period, ref = -1.0, ref2=0.0) | {unit_id} + {time_id}"
    m = feols(fml=ff, data=df, vcov={"CRV1": cluster})  # type: ignore
    res = m.tidy().reset_index()
    res = res.join(
        res["Coefficient"].str.extract(
            r".+::(?P<time>.+):.+::(?P<cohort>.+)", expand=True
        )
    )
    res["time"] = res["time"].astype(float)
    # create a dict with cohort specific effect curves
    res_cohort_eventtime_dict: dict[str, dict[str, pd.DataFrame | np.ndarray]] = {}
    for cohort, res_cohort in res.groupby("cohort"):
        event_time = res_cohort["time"].to_numpy()
        res_cohort_eventtime_dict[cohort] = {"est": res_cohort, "time": event_time}

    return m, res_cohort_eventtime_dict


def _test_treatment_heterogeneity(
    model: Feols,
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
    mmres[["time", "cohort"]] = mmres["Coefficient"].str.extract(
        r".+::(?P<time>.+):.+::(?P<cohort>.+)", expand=True
    )
    mmres["time"] = mmres["time"].astype(float)
    # indices of coefficients that are deviations from common event study coefs
    event_study_coefs = mmres.loc[~(mmres.cohort.isna()) & (mmres.time > 0)].index
    # Method 2 (K x P) - more efficient
    K = len(event_study_coefs)
    R2 = np.zeros((K, P))
    for i, idx in enumerate(event_study_coefs):
        R2[i, idx] = 1

    test_result = model.wald_test(R=R2, distribution="chi2")
    return test_result


def compute_period_weights(
    data: pd.DataFrame,
    cohort: str = "g",
    period: str = "rel_time",
    treatment: str = "treatment",
    include_grid: bool = True,
) -> pd.DataFrame:
    """
    Compute Sun & Abraham interaction weights for all relative times.

    For l < 0, weight_{g,l} = n^0_{g,l} / Σ_{g'} n^0_{g',l}, where
      n^0_{g,l} = count of (cohort=g, period=l) still untreated but eventually treated.
    For l > 0, weight_{g,l} = n^1_{g,l} / Σ_{g'} n^1_{g',l}, where
      n^1_{g,l} = count of (cohort=g, period=l) already treated.
    For l = 0, weight is set to zero in the final grid.

    Parameters
    ----------
    data : pd.DataFrame
        Must include columns for `cohort`, `period`, and `treatment`.
    cohort : str
        Column name of cohort/adoption group.
    period : str
        Column name of relative time (lead/lag) indicator.
    treatment : str
        Column name of treatment indicator (0/1).
    include_grid : bool, default True
        If True, returns a full (cohort x period) grid with zero-filled weights.

    Returns
    -------
    pd.DataFrame
        Columns [cohort, period, weight]. If `include_grid`, every
        combination appears (with weight=0 where not defined).
    """
    df = data[[cohort, period, treatment]].copy()
    ever_treated = df.loc[df[treatment] == 1, cohort].unique()

    # post-treatment cells (l > 0)
    post = (
        df[df[treatment] == 1]
        .groupby([cohort, period])
        .size()
        .reset_index(name="n_grel")
    )
    post = post[post[period] >= 0]
    denom_post = post.groupby(period)["n_grel"].sum().reset_index(name="n_rel")
    post = post.merge(denom_post, on=period)
    post["weight"] = post["n_grel"] / post["n_rel"]

    # pre-treatment cells (l < 0)
    pre = (
        df[(df[treatment] == 0) & (df[cohort].isin(ever_treated))]
        .groupby([cohort, period])
        .size()
        .reset_index(name="n_grel")
    )
    pre = pre[pre[period] < 0]
    denom_pre = pre.groupby(period)["n_grel"].sum().reset_index(name="n_rel")
    pre = pre.merge(denom_pre, on=period)
    pre["weight"] = pre["n_grel"] / pre["n_rel"]

    # combine and (optionally) fill out the full grid
    out = pd.concat(
        [pre[[cohort, period, "weight"]], post[[cohort, period, "weight"]]],
        ignore_index=True,
    )

    if include_grid:
        all_periods = sorted(df[period].unique())
        grid = pd.MultiIndex.from_product(
            [list(ever_treated), all_periods], names=[cohort, period]
        ).to_frame(index=False)
        out = grid.merge(out, on=[cohort, period], how="left")
        out["weight"] = out["weight"].fillna(0.0)

    return out[[cohort, period, "weight"]]
