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
    EstimationMethod,
    FamilyOptions,
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.model_state import GlmEstimationOptions
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import plan_formulas
from pyfixest.estimation.runner import run_estimation
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import Ssc, capture_context

# Poisson goes through the Fepois model class (a Feglm subclass); the other
# families dispatch to their dedicated model classes.
_GLM_METHODS: dict[FamilyOptions, EstimationMethod] = {
    "poisson": "fepois",
    "logit": "feglm-logit",
    "probit": "feglm-probit",
    "gaussian": "feglm-gaussian",
}


def feglm(
    fml: str,
    data: DataFrameType,
    family: FamilyOptions,
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
    Estimate generalized linear models with fixed effects.

    Call as `pf.feglm(..., family=...)` after `import pyfixest as pf`.

    Supported families: `logit`, `probit`, and `gaussian` (see
    [Feglm](/reference/estimation.models.feglm_.Feglm.qmd)), and `poisson`
    (see [Fepois](/reference/estimation.models.fepois_.Fepois.qmd)).

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

    family : FamilyOptions
        The family of the GLM model. Options include "gaussian", "logit", "probit",
        and "poisson". Passing "poisson" produces the same result as calling
        `pf.fepois()` and returns a `Fepois` object.

    vcov : VcovTypeOptions | dict[str, str] | None, optional
        Type of variance-covariance matrix for inference. Options include `iid`,
        `hetero`, `HC1`, `HC2`, `HC3`, `NW` for Newey-West HAC standard errors,
        `DK` for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference.
        `HC2` and `HC3` are only available for models without fixed effects.
        CRV3 is only available for `family="poisson"`.
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
        Default is `None`. Name of the column in `data` to be used as observation
        weights. When supplied, IRLS minimizes the weighted deviance.

    weights_type : WeightsTypeOptions, optional
        Type of weights variable. Either `aweights` (analytic / precision
        weights) or `fweights` (frequency weights). Defaults to `aweights`.

    offset : str | None, optional
        Default is `None`. Formulaic expression that evaluates to one numeric
        offset column on the link scale. Only supported with
        `family='poisson'`. For exposure adjustments, use an expression such as
        `offset="log(population)"`.

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
        models with fixed effects. The "fe" rule is designed for Poisson models but currently
        runs for every family: for `family="gaussian"` it can drop valid
        observations (pass `separation_check=[]` to keep them, as `feols()`
        does), and for `family="logit"`/`"probit"` it does not drop
        fixed-effect levels whose outcome is always 1.

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
        relative to the caller (`0` = the frame that calls `feglm()`); a
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
    `Feglm` | `Fepois` | `FixestMulti`
        A family-specific [Feglm](/reference/estimation.models.feglm_.Feglm.qmd)
        fitted result for `"logit"`, `"probit"` or `"gaussian"`, or
        [Fepois](/reference/estimation.models.fepois_.Fepois.qmd) for `"poisson"`.
        Multiple specifications or sample splitting with `split`/`fsplit`
        return [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd).
        Use `fits.to_list()` to access the individual fits.

    Notes
    -----
    Logit, probit and Gaussian GLM fits support coefficient inference,
    prediction and fixed-effect recovery. They do not support wild bootstrap,
    randomization inference, causal cluster variance, decomposition, online
    updates, SAVI or CRV3 inference. For the Poisson family, see
    [pf.fepois()](/reference/estimation.api.fepois.fepois.qmd).

    Coefficients and predictions default to the link scale. Use
    `fit.predict(type="response")` for predictions on the outcome scale.
    Prediction standard errors and intervals are unsupported; the current
    GLM path ignores `interval`. Use the
    [marginal effects guide](/how-to/marginaleffects.qmd) for response-scale effects.

    See [fitted results](/reference-guide/fitted-results.qmd) for common operations,
    [model support](/reference-guide/model-support.qmd) for estimator restrictions,
    and [model storage](/reference-guide/model-storage.qmd) for retained-data requirements.

    Examples
    --------
    Fit a binary-outcome model and inspect response-scale predictions:

    ```{python}
    import numpy as np
    import pyfixest as pf

    data = pf.get_data()
    data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)
    fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data=data, family="logit")
    fit.tidy()
    fit.predict(type="response")[:5]
    ```

    Compare logit and probit specifications:

    ```{python}
    fit_probit = pf.feglm(
        "Y_bin ~ X1 + X2 | f1", data=data, family="probit"
    )
    pf.etable([fit, fit_probit])
    ```

    See Also
    --------
    [pf.fepois()](/reference/estimation.api.fepois.fepois.qmd) is the Poisson convenience function;
    [pf.feols()](/reference/estimation.api.feols.feols.qmd) fits least-squares models.
    See the [marginal effects guide](/how-to/marginaleffects.qmd) for applied workflows.

    References
    ----------
    - Bergé, L. (2018). Efficient estimation of maximum likelihood models with
      multiple fixed-effects: the R package FENmlm.
      [CREA Discussion Paper](https://ideas.repec.org/p/luc/wpaper/18-13.html).
    - Correia, S., Guimaraes, P., & Zylkin, T. (2019). ppmlhdfe: Fast Poisson
      Estimation with High-Dimensional Fixed Effects.
      [The Stata Journal](https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691).
    - Stammann, A. (2018). Fast and Feasible Estimation of Generalized Linear
      Models with High-Dimensional k-way Fixed Effects.
      [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815).
    """
    _validate_literal_argument(family, FamilyOptions)
    if offset is not None and family != "poisson":
        raise ValueError(
            "The `offset` argument is only supported with `family='poisson'`."
        )

    if separation_check is None:
        separation_check = ["fe"]
    ssc = _resolve_ssc(ssc)
    vcov_spec = _resolve_vcov(vcov, vcov_kwargs)

    context = {} if context is None else capture_context(context)
    demeaner = _resolve_demeaner(demeaner)
    _warn_if_experimental_torch_demeaner(demeaner)

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
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        separation_check=separation_check,
    )

    options = GlmEstimationOptions(
        ssc=ssc,
        drop_singletons=fixef_rm == "singleton",
        drop_intercept=drop_intercept,
        weights=weights,
        weights_type=weights_type,
        offset=offset,
        collin_tol=collin_tol,
        solver=solver,
        demeaner=demeaner,
        store_data=store_data,
        copy_data=copy_data,
        lean=lean,
        context=context,
        maxiter=iwls_maxiter,
        tol=iwls_tol,
        separation_check=separation_check,
        accelerate=accelerate,
    )
    config = EstimationConfig(
        method=_GLM_METHODS[family],
        data=data,
        formulas=tuple(Formula.parse(fml)),
        options=options,
        vcov=vcov_spec,
        split=split,
        fsplit=fsplit,
    )

    plan = plan_formulas(config)
    if plan.is_iv:
        raise NotImplementedError("IV estimation is not supported for GLMs.")

    return run_estimation(config, plan, apply_retention=True)
