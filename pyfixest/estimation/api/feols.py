from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.api.utils import (
    _estimation_input_checks,
    _resolve_ssc,
    _resolve_vcov,
)
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.demeaner_options import (
    _resolve_demeaner,
    _warn_if_experimental_torch_demeaner,
)
from pyfixest.estimation.internals.literals import (
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.estimation.internals.model_state import EstimationOptions
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.plan_ import plan_formulas
from pyfixest.estimation.runner import run_estimation
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import Ssc, capture_context


def feols(
    fml: str,
    data: DataFrameType,
    vcov: VcovTypeOptions | dict[str, str] | None = None,
    vcov_kwargs: dict[str, str | int] | None = None,
    weights: str | None = None,
    ssc: Ssc | Mapping[str, Any] | None = None,
    fixef_rm: FixedRmOptions = "singleton",
    collin_tol: float = 1e-09,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    weights_type: WeightsTypeOptions = "aweights",
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner: AnyDemeaner | None = None,
    use_compression: bool = False,
    reps: int = 100,
    context: int | Mapping[str, Any] | None = None,
    seed: int | None = None,
    split: str | None = None,
    fsplit: str | None = None,
) -> Feols | FixestMulti:
    """
    Estimate ordinary or instrumental-variables least squares with fixed effects.

    Call as `pf.feols(...)` after `import pyfixest as pf`. Put absorbed fixed
    effects after `|` and an IV/2SLS instrument block inside `[D ~ Z]`.

    Parameters
    ----------
    fml : str
        A formula string using fixest formula syntax. The part before the
        first `|` holds the dependent variable and the covariates; an optional
        second part after `|` lists the fixed effects, e.g.
        `"Y ~ X1 + X2 | FE1 + FE2"`. Instrumental variables are written in
        brackets inside the covariate part, e.g. `"Y ~ X2 + [X1 ~ Z1] | FE1 + FE2"`,
        where `X1` is the endogenous variable and `Z1` the instrument.
        Special syntax includes stepwise regressions (`sw()`, `sw0()`),
        cumulative stepwise regressions (`csw()`, `csw0()`), multiple
        dependent variables (`sw(Y1, Y2) ~ X1`), interaction of variables
        (`i(X1, X2)`), and interacted fixed effects (`fe1:fe2`). See the
        [formula syntax tutorial](/tutorials/formula-syntax.qmd).

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : VcovTypeOptions | dict[str, str] | None, optional
        Type of variance-covariance matrix for inference. Options include `iid`,
        `hetero`, `HC1`, `HC2`, `HC3`, `NW` for Newey-West HAC standard errors,
        `DK` for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference,
        e.g. `{"CRV1": "f1"}`. For multiway clustering, join the cluster variables
        with `+`, e.g. `{"CRV1": "f1 + f2"}` or `{"CRV1": "f1 + f2 + f3"}`.
        `HC2` and `HC3` are only available for models without fixed effects
        and without instruments. `nid` is only supported by `quantreg()`. If `None`, `iid` is used.
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
        Default is `None`. Name of the column in `data` that contains the
        observation weights. If `None`, all observations are weighted equally.

    ssc : Ssc | Mapping[str, Any] | None, optional
        Small sample correction for inference, created with
        [ssc()](/reference/utils.utils.ssc.qmd). If `None`, uses the default
        settings of `pf.ssc()`.

    fixef_rm : FixedRmOptions, optional
        Specifies whether to drop singleton fixed effects.
        Can be equal to "singleton" (default) or `none`.
        "singleton" will drop singleton fixed effects. This will not impact point
        estimates but it will impact standard errors.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-09.

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
        Discard large estimation arrays, residuals, fitted values, and solver
        state after fitting. Defaults to `False`. Coefficients and inference
        remain available; operations requiring discarded state raise
        `MissingModelDataError`. See the [storage guide](/reference-guide/model-storage.qmd).

    weights_type : WeightsTypeOptions, optional
        Options include `aweights` (the default) or `fweights`. `aweights` implement analytic or
        precision weights, while `fweights` implement frequency weights. For details
        see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.

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

    use_compression : bool, optional
        Deprecated and no longer supported. Passing `use_compression=True`
        raises a `NotImplementedError`. For out-of-memory regression on large
        datasets, consider using the
        [duckreg](https://github.com/py-econometrics/duckreg) package instead.

    reps : int, optional
        Deprecated legacy argument of the removed compressed regression. It is
        still validated (a positive integer) but has no effect.

    context : int | Mapping[str, Any] | None, optional
        Additional context variables to be used by formulaic during the
        creation of the model matrix. This can include custom factorization
        functions, transformations, or any other variables that need to be
        available in the formula environment. An int is a frame offset
        relative to the caller (`0` = the frame that calls `feols()`, so
        all names visible there are available); a mapping supplies the names
        directly.

    seed : int | None, optional
        Deprecated legacy argument of the removed compressed regression. It is
        still validated (an integer) but has no effect.

    split : str | None, optional
        A column name, e.g. `split="var"`. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit : str | None, optional
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    `Feols` | `Feiv` | `FixestMulti`
        A [Feols](/reference/estimation.models.feols_.Feols.qmd) fitted result
        for a single OLS specification, or
        [Feiv](/reference/estimation.models.feiv_.Feiv.qmd) for IV/2SLS.
        Multiple specifications or sample splitting with `split`/`fsplit`
        return [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd);
        use `fits.to_list()` or `fits.fetch_model(0)` to access individual fits.

    Notes
    -----
    An OLS fit supports coefficient inference, prediction, fixed-effect
    recovery and further post-estimation operations, subject to each
    method's model and storage restrictions. Wild bootstrap requires
    unweighted OLS. SAVI additionally requires no absorbed fixed effects.
    IV fits do not support prediction, fixed-effect recovery, wild bootstrap,
    randomization inference, decomposition, causal cluster variance,
    online updates, SAVI, or CRV3/HC2/HC3 covariance.

    See [fitted results](/reference-guide/fitted-results.qmd) for common operations,
    [model support](/reference-guide/model-support.qmd) for estimator restrictions,
    and [model storage](/reference-guide/model-storage.qmd) for retained-data requirements.

    Examples
    --------
    Fit a model and inspect its coefficient table:

    ```{python}
    import pyfixest as pf

    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data=data)
    fit.tidy()
    ```

    Cluster standard errors by a column in the estimation data:

    ```{python}
    fit_cluster = pf.feols(
        "Y ~ X1 + X2 | f1 + f2", data=data, vcov={"CRV1": "f1"}
    )
    fit_cluster.confint()
    ```

    Use an instrument block for IV/2SLS, or stepwise syntax for several fits:

    ```{python}
    fit_iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1 + f2", data=data)
    fit_iv.coef()
    fits = pf.feols("Y ~ sw(X1, X2) | f1", data=data)
    pf.etable(fits)
    ```

    See Also
    --------
    [pf.fepois()](/reference/estimation.api.fepois.fepois.qmd) fits count models;
    [pf.quantreg()](/reference/estimation.api.quantreg.quantreg.qmd) fits conditional quantiles.
    See [formula syntax](/tutorials/formula-syntax.qmd) for multiple models and interactions,
    and [estimation examples](/how-to/estimation-examples.qmd) for custom transforms,
    prediction and post-estimation workflows.
    """
    ssc = _resolve_ssc(ssc)
    vcov_spec = _resolve_vcov(vcov, vcov_kwargs)
    context = {} if context is None else capture_context(context)
    demeaner = _resolve_demeaner(demeaner)
    _warn_if_experimental_torch_demeaner(demeaner)

    if not isinstance(use_compression, bool):
        raise TypeError("The function argument `use_compression` must be of type bool.")

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov_spec,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        weights_type=weights_type,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
    )

    if use_compression:
        raise NotImplementedError(
            "The `use_compression` argument is deprecated and no longer supported. "
            "The compressed regression and Mundlak implementation has been removed. "
            "For out-of-memory regression on large datasets, consider using the "
            "`duckreg` package (https://github.com/py-econometrics/duckreg) instead."
        )

    options = EstimationOptions(
        ssc=ssc,
        drop_singletons=fixef_rm == "singleton",
        drop_intercept=drop_intercept,
        weights=weights,
        weights_type=weights_type,
        offset=None,
        collin_tol=collin_tol,
        solver=solver,
        demeaner=demeaner,
        store_data=store_data,
        copy_data=copy_data,
        lean=lean,
        context=context,
    )
    config = EstimationConfig(
        method="feols",
        data=data,
        formulas=tuple(Formula.parse(fml)),
        options=options,
        vcov=vcov_spec,
        split=split,
        fsplit=fsplit,
    )

    plan = plan_formulas(config)
    return run_estimation(config, plan, apply_retention=True)
