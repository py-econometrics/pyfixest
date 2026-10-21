import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.estimation import feols
from pyfixest.estimation.internals.model_state import WaldTest
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.utils.dev_utils import _find_stack_level

from .did2s import DID


@dataclass(frozen=True, slots=True, kw_only=True)
class EventStudyDesign:
    r"""
    Event-study design a fitted event study was estimated on.

    Published as `fit.event_study_design` by
    [event_study()](/reference/did.estimation.event_study.qmd) with
    `estimator="saturated"`. It is the only record of the design the fitted
    model retains; the generic fitted-model classes carry no DiD state.

    Parameters
    ----------
    yname : str
        Name of the outcome variable.
    idname : str
        Name of the unit identifier variable.
    tname : str
        Name of the calendar-period variable.
    gname : str
        Name of the variable holding the unit-specific period of initial
        treatment.
    att : bool
        The `att` argument passed to `event_study()`. It is recorded only:
        the saturated estimator always estimates the full event study with
        one coefficient per cohort and relative period.
    cohort_event_times : Mapping[str, Mapping[str, Any]]
        Cohort-specific event-study curves, keyed by treatment cohort. Each
        value holds the tidy coefficient table of that cohort (`"est"`) and
        its event times (`"time"`).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.event_study(
        pf.get_motherhood_event_study_data(n_per_country=8, seed=2026),
        yname="log_earnings",
        idname="unit",
        tname="year",
        gname="g",
        estimator="saturated",
    )
    design = fit.event_study_design
    design.gname, sorted(design.cohort_event_times)
    ```

    See Also
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) : Creates `fit.event_study_design` for saturated fits.
    [Saturated event-study results](/reference/did.saturated_twfe.SaturatedEventStudy.qmd) : Aggregation and plotting operations.
    """

    yname: str
    idname: str
    tname: str
    gname: str
    att: bool
    cohort_event_times: Mapping[str, Mapping[str, Any]]


class SaturatedEventStudy(DID):
    r"""
    Saturated event study with cohort-specific effect curves.

    Use [event_study()](/reference/did.estimation.event_study.qmd) with
    `estimator="saturated"` to obtain a fitted `Feols` result. That fit has
    `event_study_design` and the additional operations documented here:
    `aggregate()`, `iplot()`, `iplot_aggregate()`, and
    `test_treatment_heterogeneity()`.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data; the public `event_study()` entry point documents the columns.
    yname, idname, tname, gname : str
        Outcome, unit, numeric period, and first-treatment-period columns.
        Never-treated units use `gname=0`.
    att : bool, optional
        Recorded in the design but currently ignored; the full event study is fitted.
    cluster : str, optional
        Clustering variable. `event_study()` defaults this to `idname`.
    xfml : str, optional
        Currently ignored by the saturated estimator.
    display_warning : bool, optional
        Display the beta-status warning. Default is `True`.

    Notes
    -----
    Fits a fully saturated event study that interacts every treatment cohort
    with every relative period. Each cohort has its own set of event time
    effects, which avoids the comparisons between already-treated units that
    bias static two-way fixed effects under heterogeneous treatment effects
    (Sun and Abraham 2021,
    [Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2020.09.006)).

    Examples
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) with
    `estimator="saturated"` returns a
    [Feols](/reference/estimation.models.feols_.Feols.qmd) fit and attaches
    the `aggregate()`, `iplot()`, `iplot_aggregate()`, and
    `test_treatment_heterogeneity()` methods documented here to it.
    `aggregate()` aggregates the cohort-specific effects by event time.

    ```{python}
    import pyfixest as pf

    data = pf.get_motherhood_event_study_data(n_per_country=8, seed=2026)

    fit = pf.event_study(
        data,
        yname="log_earnings",
        idname="unit",
        tname="year",
        gname="g",
        estimator="saturated",
    )
    fit.aggregate().head()
    ```

    `iplot_aggregate()` plots the aggregated effects.

    ```{python}
    fit.iplot_aggregate()
    ```

    See Also
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) : Public estimator entry point.
    [EventStudyDesign](/reference/did.saturated_twfe.EventStudyDesign.qmd) : Design retained by the fitted result.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Cohort curves and aggregation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        att: bool = True,
        cluster: str | None = None,
        xfml: str | None = None,
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
                "The SaturatedEventStudyClass is currently in beta. Please report any issues you may encounter.",
                UserWarning,
                stacklevel=_find_stack_level(),
            )

    def estimate(self) -> Feols:
        """
        Estimate the model.

        Returns
        -------
        Feols
            The fitted `Feols` model object. It carries the event-study design
            as `event_study_design`.
        """
        self.mod, cohort_event_times = _saturated_event_study(
            self._data,
            outcome=self._yname,
            time_id=self._tname,
            unit_id=self._idname,
            cluster=self._cluster,
        )
        self.mod.event_study_design = EventStudyDesign(
            yname=self._yname,
            idname=self._idname,
            tname=self._tname,
            gname=self._gname,
            att=self._att,
            cohort_event_times=cohort_event_times,
        )

        return self.mod

    # !TODO - implement the rest of the methods
    def vcov(self):
        r"""
        Get the covariance matrix.

        Returns
        -------
        None
            This estimator hook is unimplemented. On the `Feols` result returned by
            `event_study()`, inspect `fit.variance_covariance.vcov` instead.
        """
        pass

    def iplot(self):
        r"""
        Display one event-time curve per treatment cohort with 95% confidence bands.

        Returns
        -------
        None
            Displays a matplotlib figure using `plt.show()`.

        See Also
        --------
        [Saturated event-study results](/reference/did.saturated_twfe.SaturatedEventStudy.qmd) : Obtain a fit and follow the cohort/aggregation example.
        [event_study()](/reference/did.estimation.event_study.qmd) : Use `estimator="saturated"` to access this operation.
        """
        import matplotlib.pyplot as plt

        model = self.mod if isinstance(self, SaturatedEventStudy) else self

        cmp = plt.get_cmap("Set1")

        _, ax = plt.subplots(figsize=(10, 6))

        for cohort, values in model.event_study_design.cohort_event_times.items():
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

    def test_treatment_heterogeneity(self) -> WaldTest:
        r"""
        Wald test on the post-treatment cohort-specific event study coefficients.

        Despite its name, the current implementation tests the joint null that
        all cohort-specific coefficients at event times greater than 0 are
        zero. It does not test whether the effects are equal across cohorts
        as in Lal (2025, [arXiv](https://arxiv.org/abs/2503.05125)), and it
        leaves out event time 0.

        Returns
        -------
        WaldTest
            The chi2 statistic of the joint test and its p-value.

        See Also
        --------
        [Saturated event-study results](/reference/did.saturated_twfe.SaturatedEventStudy.qmd) : Obtain a fit and follow the cohort/aggregation example.
        [event_study()](/reference/did.estimation.event_study.qmd) : Use `estimator="saturated"` to access this operation.
        """
        return _test_treatment_heterogeneity(
            model=self.mod if isinstance(self, SaturatedEventStudy) else self,
        )

    def aggregate(self, agg="period", weighting: str | None = "shares") -> pd.DataFrame:
        r"""
        Aggregate the cohort-specific event study estimates by event time.

        Parameters
        ----------
        agg : str, optional
            The aggregation level. Only `"period"`, the default, is supported:
            the cohort-specific effects are aggregated by event time.
        weighting : str, optional
            The weighting scheme. Only `"shares"`, the default, is supported:
            each cohort enters an event time with its share of the cohort
            observations at that event time (Sun and Abraham 2021).

        Returns
        -------
        pd.DataFrame
            The aggregated estimate per event time, with standard error,
            t value, p value, and the bounds of the 95% confidence interval.

        See Also
        --------
        [Saturated event-study results](/reference/did.saturated_twfe.SaturatedEventStudy.qmd) : Obtain a fit and follow the cohort/aggregation example.
        [event_study()](/reference/did.estimation.event_study.qmd) : Use `estimator="saturated"` to access this operation.
        """
        if agg not in ["period"]:
            raise ValueError("agg must be either 'period'")

        if weighting not in ["shares"]:
            raise ValueError("weighting must be 'shares'.")

        model = self.mod if isinstance(self, SaturatedEventStudy) else self
        design = model.event_study_design

        cohort_event_dict = design.cohort_event_times
        cohort_list = list(cohort_event_dict.keys())
        period_set = sorted(
            set(t for x in cohort_list for t in cohort_event_dict[x]["time"].tolist())
        )

        coefs = model._beta_hat
        se = model.coeftable.se
        coefnames = model._coefnames

        if weighting == "shares":
            weights_df = compute_period_weights(
                data=model._data,
                cohort=design.gname,
                period="rel_time",
                treatment="is_treated",
            ).set_index([design.gname, "rel_time"])

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

            res_dict = _compute_lincomb_stats(
                R=R, coefs=coefs, vcov=model.variance_covariance.vcov
            )
            df_agg.loc[period] = pd.Series(res_dict)

        return df_agg

    def iplot_aggregate(self, agg="period", weighting: str | None = "shares"):
        r"""
        Plot the aggregated estimates.

        Parameters
        ----------
        agg : str, optional
            The aggregation level. Only `"period"`, the default, is supported:
            the cohort-specific effects are aggregated by event time.
        weighting : str, optional
            The weighting scheme. Only `"shares"`, the default, is supported:
            each cohort enters an event time with its share of the cohort
            observations at that event time (Sun and Abraham 2021).

        Returns
        -------
        None
            Displays a matplotlib figure using `plt.show()`; does not return it.

        See Also
        --------
        [Saturated event-study results](/reference/did.saturated_twfe.SaturatedEventStudy.qmd) : Obtain a fit and follow the cohort/aggregation example.
        [event_study()](/reference/did.estimation.event_study.qmd) : Use `estimator="saturated"` to access this operation.
        """
        import matplotlib.pyplot as plt

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
    cluster: str | None = None,
):
    ff = f"{outcome} ~ i(rel_time, first_treated_period, ref = -1, ref2=0) | {unit_id} + {time_id}"
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
        res_cohort_eventtime_dict[str(cohort)] = {"est": res_cohort, "time": event_time}

    return m, res_cohort_eventtime_dict


def _test_treatment_heterogeneity(
    model: Feols,
) -> WaldTest:
    """
    Wald test on the post-treatment cohort-specific event study coefficients.

    Tests the joint null that all cohort-specific coefficients at event times
    greater than 0 are zero. This is not the test of equal effects across
    cohorts in Lal (2025, [arXiv](https://arxiv.org/abs/2503.05125)); see
    https://github.com/apoorvalal/TestingInEventStudies for that test.

    Parameters
    ----------
    model : Feols
        The fitted saturated event study model.

    Returns
    -------
    WaldTest
        The chi2 statistic of the joint test and its p-value.
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

    return model.wald_test(R=R2, distribution="chi2")


def compute_period_weights(
    data: pd.DataFrame,
    cohort: str = "g",
    period: str = "rel_time",
    treatment: str = "treatment",
    include_grid: bool = True,
) -> pd.DataFrame:
    r"""
    Compute Sun & Abraham interaction weights for all relative times.

    For l < 0, weight_{g,l} = n^0_{g,l} / Σ_{g'} n^0_{g',l}, where
      n^0_{g,l} = count of (cohort=g, period=l) still untreated but eventually treated.
    For l > 0, weight_{g,l} = n^1_{g,l} / Σ_{g'} n^1_{g',l}, where
      n^1_{g,l} = count of (cohort=g, period=l) already treated.
    Event time 0 uses treated cohort shares, like other post-treatment periods.
    Cells without observations have zero weight when `include_grid=True`.

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
        If `True`, returns a full (cohort x period) grid with zero-filled weights.

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
