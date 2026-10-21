from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.api.feglm import feglm
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.internals.literals import (
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import Ssc, capture_context


def fepois(
    fml: str,
    data: DataFrameType,
    vcov: VcovTypeOptions | dict[str, str] | None = None,
    vcov_kwargs: dict[str, str | int] | None = None,
    weights: str | None = None,
    weights_type: WeightsTypeOptions = "aweights",
    offset: str | None = None,
    ssc: Ssc | Mapping[str, Any] | None = None,
    fixef_rm: FixedRmOptions = "singleton",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-09,
    separation_check: list[str] | None = None,
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner: AnyDemeaner | None = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: int | Mapping[str, Any] | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    accelerate: bool = True,
) -> Feols | Fepois | FixestMulti:
    """
    Estimate Poisson regression with fixed effects using the `ppmlhdfe` algorithm.

    Call as `pf.fepois(...)` after `import pyfixest as pf`. Use this estimator
    for count outcomes, or rates represented by counts with an exposure offset.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax, e.g.
        `"Y ~ X1 + X2 | FE1 + FE2"`. The optional part after `|` lists the
        fixed effects. Special syntax includes stepwise regressions (`sw()`,
        `sw0()`), cumulative stepwise regressions (`csw()`, `csw0()`), multiple
        dependent variables (`sw(Y1, Y2) ~ X`), interaction of variables
        (`i(X1, X2)`), and interacted fixed effects (`fe1:fe2`). See the
        [formula syntax tutorial](/tutorials/formula-syntax.qmd).

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : VcovTypeOptions | dict[str, str] | None, optional
        Type of variance-covariance matrix for inference. Options include `iid`,
        `hetero`, `HC1`, `HC2`, `HC3`, `NW` for Newey-West HAC standard errors,
        `DK` for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference.
        `HC2` and `HC3` are only available for models without fixed effects.
        If `None`, `iid` is used.
        Note that NW and DK require to pass additional keyword arguments via the `vcov_kwargs` argument.
        For time-series HAC, you need to pass the '`time_id`' column. For panel-HAC, you need to
        pass both '`time_id`' and '`panel_id`'. See `vcov_kwargs` for details.

    vcov_kwargs : dict[str, str | int] | None, optional
        Additional keyword arguments to pass to the vcov function. These keywords include
        "lag" for the number of lags to use in the Newey-West (NW) and Driscoll-Kraay (DK) HAC standard errors,
        "`time_id`" for the time ID used for NW and DK standard errors, and "`panel_id`" for the panel
        identifier used for NW and DK standard errors. Currently, the time difference between consecutive time
        periods is always treated as 1. More flexible time-step selection is work in progress.

    weights : str | None, optional
        Default is `None`. Weights for weighted Poisson regression. If `None`, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    weights_type : WeightsTypeOptions, optional
        Options include `aweights` or `fweights`. `aweights` implement analytic or
        precision weights, while `fweights` implement frequency weights. Frequency weights
        are useful for compressed count data where identical observations are aggregated.
        For details see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.

    offset : str | None, optional
        Default is `None`. Formulaic expression that evaluates to one numeric
        column in `data`. The offset is added to the linear predictor with its
        coefficient fixed at 1. This is useful for modeling rates when exposure
        differs across observations, e.g. `offset="log(population)"`.

    ssc : Ssc | Mapping[str, Any] | None, optional
        Small sample correction for inference, created with
        [ssc()](/reference/utils.utils.ssc.qmd). If `None`, uses the default
        settings of `pf.ssc()`.

    fixef_rm : FixedRmOptions, optional
        Specifies whether to drop singleton fixed effects.
        Can be equal to "singleton" (default) or `none`.
        "singleton" will drop singleton fixed effects. This will not impact point
        estimates but it will impact standard errors.

    iwls_tol : float, optional
        Tolerance for IWLS convergence, by default 1e-08.

    iwls_maxiter : int, optional
        Maximum number of iterations for IWLS convergence, by default 25.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-09.

    separation_check : list[str] | None, optional
        Methods to identify and drop separated observations: "fe" (drop
        observations in fixed-effect levels whose outcome is never positive)
        and/or "ir" (iterative rectifier). Executes `["fe"]` by default (when
        `None`). Pass an empty list to skip the check. The check only runs for
        models with fixed effects.

    solver : SolverOptions, optional
        The solver to use for the regression. Can be "`np.linalg.lstsq`",
        "`np.linalg.solve`", "`scipy.linalg.solve`" and "`scipy.sparse.linalg.lsqr`".
        Defaults to "`scipy.linalg.solve`".

    demeaner : AnyDemeaner | None, optional
        Typed demeaner configuration. Controls the fixed-effects demeaning
        backend, tolerance, and iteration limits. Accepts a `MapDemeaner`
        or `LsmrDemeaner` instance. Defaults to
        `LsmrDemeaner()` (within LSMR, auto preconditioning, `tol=1e-8`,
        `maxiter=1_000`).
        For other options - including the optional Numba backend and the
        torch-based LSMR backends - see the
        [Demeaner Backends vignette](/how-to/demeaner-backends.qmd).

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default `False`.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default `True`.
        If set to `False`, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of the
        estimation function. For example, the input data set is re-indexed
        within the function.

    store_data : bool, optional
        Keep the estimation data and model matrix on the fitted result. Defaults
        to `True`. If `False`, operations that require them raise
        `MissingModelDataError`; ordinary predictions on supplied `newdata` can
        still work for supported models. See the
        [storage guide](/reference-guide/model-storage.qmd) for operation-specific
        requirements.

    lean : bool, optional
        Remove large retained arrays as well as the estimation data. Defaults to
        `False`. Coefficients and their fitted inference remain available, but
        residuals, in-sample predictions and covariance updates require omitted
        state. Set `vcov` at estimation time. Ordinary predictions on supplied
        `newdata` remain possible for supported models without absorbed fixed
        effects; see the [storage guide](/reference-guide/model-storage.qmd).

    context : int | Mapping[str, Any] | None, optional
        Additional context variables to be used by formulaic during the
        creation of the model matrix. This can include custom factorization
        functions, transformations, or any other variables that need to be
        available in the formula environment. An int is a frame offset
        relative to the caller (`0` = the frame that calls `fepois()`); a
        mapping supplies the names directly.

    split : str | None, optional
        A column name, e.g. `split="var"`. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit : str | None, optional
        This argument is the same as split but also includes the full sample as the first estimation.

    accelerate : bool, optional
        Whether to use acceleration tricks developed in the ppmlhdfe paper (warm start and adaptive fixed effects
        tolerance) for models with fixed effects. Results agree with the unaccelerated fit up to the
        convergence tolerance and are obtained faster, so we recommend to always set it to `True`.

    Returns
    -------
    `Fepois` | `FixestMulti`
        A [Fepois](/reference/estimation.models.fepois_.Fepois.qmd) fitted
        result for a single specification. Multiple specifications or sample
        splitting with `split`/`fsplit` return
        [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd).
        Use `fits.to_list()` to access the individual fits.

    Notes
    -----
    Poisson fits support coefficient inference, prediction, fixed-effect
    recovery and unweighted randomization inference. They do not support
    wild bootstrap, causal cluster variance, decomposition, online updates
    or SAVI. Prediction defaults to `type="link"` (log expected counts);
    use `type="response"` for expected counts. Prediction standard errors
    and intervals are unsupported; the current GLM path ignores `interval`.

    See [fitted results](/reference-guide/fitted-results.qmd) for common operations,
    [model support](/reference-guide/model-support.qmd) for estimator restrictions,
    and [model storage](/reference-guide/model-storage.qmd) for retained-data requirements.

    Examples
    --------
    Fit a count model with cluster-robust inference:

    ```{python}
    import pyfixest as pf

    data = pf.get_data(model="Fepois")
    fit = pf.fepois(
        "Y ~ X1 + X2 | f1 + f2", data=data, vcov={"CRV1": "f1"}
    )
    fit.tidy()
    fit.predict(type="response")[:5]
    ```

    To model rates, keep the outcome as a count and put log exposure in
    an offset whose coefficient is fixed at one:

    ```{python}
    import numpy as np

    data["population"] = np.random.default_rng(123).integers(
        50_000, 500_000, size=len(data)
    )
    fit_rate = pf.fepois(
        "Y ~ X1 + X2 | f1 + f2", data=data, offset="log(population)"
    )
    fit_rate.coef()
    ```

    See Also
    --------
    [pf.feglm()](/reference/estimation.api.feglm.feglm.qmd) supports additional GLM families.
    [pf.feols()](/reference/estimation.api.feols.feols.qmd) documents shared formula and inference options.
    See the [demeaner guide](/how-to/demeaner-backends.qmd) for computation options.
    """
    # Thin wrapper: fepois is exactly feglm(family="poisson"). Capture the
    # context here, as an integer frame offset passed on to feglm would resolve
    # relative to this frame rather than the user's.
    context = {} if context is None else capture_context(context)
    return feglm(
        fml=fml,
        data=data,
        family="poisson",
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        weights_type=weights_type,
        offset=offset,
        ssc=ssc,
        fixef_rm=fixef_rm,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
        solver=solver,
        demeaner=demeaner,
        drop_intercept=drop_intercept,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        context=context,
        split=split,
        fsplit=fsplit,
        accelerate=accelerate,
    )
