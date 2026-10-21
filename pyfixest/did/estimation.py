from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pyfixest.did.did2s import DID2S, _did2s_estimate
from pyfixest.did.lpdid import LPDID
from pyfixest.did.saturated_twfe import SaturatedEventStudy
from pyfixest.did.twfe import TWFE
from pyfixest.estimation.internals.literals import VcovTypeOptions
from pyfixest.estimation.models.feols_ import Feols


def event_study(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    xfml: str | None = None,
    cluster: str | None = None,
    estimator: str | None = "twfe",
    att: bool | None = True,
):
    r"""
    Estimate Event Study Model.

    This function allows for the estimation of treatment effects using different
    estimators. It supports `twfe` for the two-way fixed effects estimator,
    `did2s` for the two-step DID2S estimator of Gardner (2022,
    [arXiv](https://arxiv.org/abs/2207.05943)), and `saturated` for a
    saturated event-study specification with cohort-specific effects as in
    Sun and Abraham (2021,
    [Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2020.09.006)).

    The `twfe` and `did2s` estimators fit a static model with a single
    `is_treated` coefficient. The `saturated` estimator fits the full event
    study with one coefficient per cohort and relative period. For a dynamic
    DID2S event study, call [did2s()](/reference/did.estimation.did2s.qmd)
    directly.

    Parameters
    ----------
    data : DataFrame
        The `DataFrame` containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the id variable.
    tname : str
        Variable name for calendar period.
    gname : str
        Unit-specific period of initial treatment, in the same numeric time
        scale as `tname`. Never-treated units must have a value of 0.
        Use consecutive period indices when differences represent event time.
    xfml : str, optional
        The formula for the covariates, e.g. "X1 + X2". Default is `None`.
        Used by the `twfe` and `did2s` estimators; the `saturated` estimator
        currently ignores it.
    cluster : str, optional
        The name of the cluster variable. If `None`, defaults to idname.
        With ``estimator="did2s"``, inference is fixed at estimation time, and
        post-estimation ``vcov()`` updates are rejected.
    estimator : str, optional
        The estimator to use. Options are `twfe` (default), `did2s`, and
        `saturated`.
    att : bool, optional
        Default is `True`. Currently has no effect: every estimator ignores it,
        and `att=False` does not produce an event study with leads and lags.

    Returns
    -------
    Feols
        A fitted model object of class [Feols](/reference/estimation.models.feols_.Feols.qmd).
        With `estimator = "saturated"`, the fit additionally carries its
        [EventStudyDesign](/reference/did.saturated_twfe.EventStudyDesign.qmd) as
        `event_study_design` and provides the `aggregate()`, `iplot_aggregate()`,
        `iplot()`, and `test_treatment_heterogeneity()` methods.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_motherhood_event_study_data(n_per_country=8, seed=2026)
    fit = pf.event_study(
        data, yname="log_earnings", idname="unit", tname="year", gname="g",
        estimator="twfe",
    )
    fit.tidy()
    ```

    For cohort-specific event-time curves, choose the saturated estimator:

    ```{python}
    fit_saturated = pf.event_study(
        data, yname="log_earnings", idname="unit", tname="year", gname="g",
        estimator="saturated",
    )
    fit_saturated.aggregate().head()
    ```

    See Also
    --------
    [did2s()](/reference/did.estimation.did2s.qmd) : Specify a dynamic or pooled two-stage model directly.
    [panelview()](/reference/did.visualize.panelview.qmd) : Inspect treatment timing before fitting.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Compare estimators and event-time plots.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    for name, value in (
        ("yname", yname),
        ("idname", idname),
        ("tname", tname),
        ("gname", gname),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
    if not (isinstance(xfml, str) or xfml is None):
        raise TypeError("xfml must be a string or None.")
    if not isinstance(estimator, str):
        raise TypeError(
            "estimator must be a string: one of 'twfe', 'did2s', or 'saturated'."
        )
    if not isinstance(att, bool):
        raise TypeError("att must be a boolean.")
    if not (isinstance(cluster, str) or cluster is None):
        raise TypeError("cluster must be a string or None.")

    cluster = idname if cluster is None else cluster

    if estimator == "did2s":
        did2s = DID2S(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )

        fit = did2s.estimate()
        _mark_as_did2s(fit)

    elif estimator == "twfe":
        twfe = TWFE(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )
        fit = twfe.estimate()

        fit.vcov(vcov={"CRV1": cluster})
        fit.model = replace(fit.model, method="twfe")
        # ritest(), update(), and SAVI have not supported event-study fits.
        fit.capabilities = replace(
            fit.capabilities,
            randomization_inference=False,
            sherman_morrison_update=False,
            anytime_valid_inference=False,
        )

    elif estimator == "saturated":
        saturated = SaturatedEventStudy(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )
        fit = saturated.estimate()
        fit.vcov(vcov={"CRV1": cluster})

        fit.model = replace(fit.model, method="saturated")
        # ritest(), update(), and SAVI have not supported event-study fits.
        fit.capabilities = replace(
            fit.capabilities,
            randomization_inference=False,
            sherman_morrison_update=False,
            anytime_valid_inference=False,
        )
        fit.iplot = saturated.iplot.__get__(fit, type(fit))  # type: ignore
        test_treatment_heterogeneity = saturated.test_treatment_heterogeneity.__get__(
            fit, type(fit)
        )
        fit.test_treatment_heterogeneity = test_treatment_heterogeneity  # type: ignore
        fit.aggregate = saturated.aggregate.__get__(fit, type(fit))  # type: ignore
        fit.iplot_aggregate = saturated.iplot_aggregate.__get__(fit, type(fit))  # type: ignore

    else:
        raise NotImplementedError("Estimator not supported")

    # update inference with vcov matrix
    fit.get_inference()

    return fit


def did2s(
    data: pd.DataFrame,
    yname: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    cluster: str,
    weights: str | None = None,
):
    r"""
    Estimate a Difference-in-Differences model using Gardner's two-step DID2S estimator.

    Implements the two-stage difference-in-differences estimator of Gardner
    (2022, [arXiv](https://arxiv.org/abs/2207.05943)): the first stage
    estimates the fixed effects on untreated observations, the second stage
    regresses the residualized outcome on the treatment variables.

    Parameters
    ----------
    data : pd.DataFrame
        The `DataFrame` containing all variables.
    yname : str
        The name of the dependent variable.
    first_stage : str
        The formula for the first stage, starting with `~`. It must contain
        fixed effects, e.g. "~ 0 | unit + year".
    second_stage : str
        The formula for the second stage, starting with `~`. It must not
        contain fixed effects.
    treatment : str
        The name of the treatment variable. It must be boolean or numeric with
        exactly two distinct values; it is converted to boolean. The first
        stage is fitted on the observations where `treatment` is `False`.
    cluster : str
        The name of the cluster variable.
        Inference is fixed at estimation time, and
        post-estimation ``vcov()`` updates are rejected.
    weights : str, optional
        Name of the column in `data` that contains the observation weights.
        Default is `None`, which weights all observations equally.

    Returns
    -------
    Feols
        A fitted model object of class [Feols](/reference/estimation.models.feols_.Feols.qmd).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_motherhood_event_study_data(n_per_country=8, seed=2026)
    fit = pf.did2s(
        data, yname="log_earnings",
        first_stage="~ 0 | unit + year",
        second_stage="~ i(rel_year, ref=-1)",
        treatment="treat", cluster="country",
    )
    fit.tidy().head()
    ```

    Plot the dynamic effects, or replace the second stage for a pooled effect:

    ```{python}
    pf.iplot(fit, coord_flip=False, plot_backend="matplotlib")
    pooled = pf.did2s(
        data, yname="log_earnings", first_stage="~ 0 | unit + year",
        second_stage="~ treat", treatment="treat", cluster="country",
    )
    pooled.tidy()
    ```

    See Also
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) : Choose a static or saturated event-study specification.
    [panelview()](/reference/did.visualize.panelview.qmd) : Inspect treatment timing before fitting.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Compare estimators and event-time plots.
    """
    fit = _did2s_estimate(
        data=data.copy(),
        yname=yname,
        _first_stage=first_stage,
        _second_stage=second_stage,
        treatment=treatment,
        cluster=cluster,
        weights=weights,
    )
    fit.get_inference()  # update inference with correct vcov matrix
    _mark_as_did2s(fit)

    return fit


def _mark_as_did2s(fit: Feols) -> None:
    """Record that a fit came from the DID2S estimator."""
    fit.model = replace(fit.model, method="did2s")
    fit.capabilities = replace(
        fit.capabilities,
        covariance_update=False,
        hac_inference=False,
        multiway_clustering=False,
        wildboottest=False,
        cluster_causal_variance=False,
        decomposition=False,
        prediction=False,
        fixed_effect_recovery=False,
        randomization_inference=False,
        sherman_morrison_update=False,
        crv3_inference=False,
        anytime_valid_inference=False,
    )


def lpdid(
    data: pd.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    vcov: VcovTypeOptions | dict[str, str] | None = None,
    pre_window: int | None = None,
    post_window: int | None = None,
    never_treated: int = 0,
    att: bool = True,
    xfml=None,
) -> LPDID:
    r"""
    Local projections approach to estimation.

    Estimate a Difference-in-Differences / Event Study Model via the Local
    Projections Approach of Dube, Girardi, Jordà and Taylor (2023,
    [NBER Working Paper 31184](https://www.nber.org/papers/w31184)).

    Parameters
    ----------
    data : DataFrame
        The `DataFrame` containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the id variable.
    tname : str
        Variable name for calendar period.
    gname : str
        Unit-specific period of initial treatment.
    vcov : str, dict, optional
        The type of inference to employ. Defaults to `{"CRV1": idname}`.
        Options include `iid`, `hetero`, or a dictionary like `{"CRV1": idname}`.
    pre_window : int, optional
        The earliest relative period to include in the estimation, given as a
        negative relative period that exists in the data (e.g. -20 for 20
        periods before treatment). Default is the minimum relative period in
        the data.
    post_window : int, optional
        The latest relative period to include in the estimation, e.g. 20 for 20
        periods after treatment. It must exist in the data. Default is the
        maximum relative period in the data.
    never_treated : int, optional
        Value in gname indicating units never treated. Default is 0.
    att : bool, optional
        If `True` (default), estimates a single pooled average treatment effect
        on the treated over relative periods 0 to `post_window`; `pre_window`
        is then not used. If `False`, estimates one coefficient per relative
        period from `pre_window` to `post_window`.
    xfml : str, optional
        Formula for the covariates, e.g. "X1 + X2", added to every local
        projection regression. Default is `None`.

    Returns
    -------
    LPDID
        A fitted [LPDID](/reference/did.lpdid.LPDID.qmd) object. Use its
        `tidy()` method for a `DataFrame` of the estimated coefficients and its
        `iplot()` method to plot them.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_motherhood_event_study_data(n_per_country=8, seed=2026)
    fit = pf.lpdid(
        data, yname="log_earnings", idname="unit", tname="year", gname="g",
        vcov={"CRV1": "country"}, pre_window=-3, post_window=3, att=False,
    )
    fit.tidy()
    fit.iplot(plot_backend="matplotlib", coord_flip=False)
    ```

    For a pooled effect over relative periods 0 through 3:

    ```{python}
    pooled = pf.lpdid(
        data, yname="log_earnings", idname="unit", tname="year", gname="g",
        vcov={"CRV1": "country"}, post_window=3, att=True,
    )
    pooled.tidy()
    ```

    See Also
    --------
    [event_study()](/reference/did.estimation.event_study.qmd) : Choose a static or saturated event-study specification.
    [panelview()](/reference/did.visualize.panelview.qmd) : Inspect treatment timing before fitting.
    [Difference-in-differences tutorial](/tutorials/difference-in-differences.qmd) : Compare estimators and event-time plots.
    """
    FIT = LPDID(
        data=data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        cluster="",  # just something to pass DID input checks
        vcov=vcov,
        pre_window=pre_window,
        post_window=post_window,
        never_treated=never_treated,
        att=att,
        xfml=xfml,
    )

    FIT.estimate()

    return FIT
