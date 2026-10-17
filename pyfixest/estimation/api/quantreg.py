from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation.api.utils import (
    _estimation_input_checks,
    _resolve_ssc,
    _resolve_vcov,
)
from pyfixest.estimation.config import EstimationConfig, QuantileProcess
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.literals import (
    FixedRmOptions,
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.estimation.internals.model_state import QuantregEstimationOptions
from pyfixest.estimation.plan_ import plan_formulas
from pyfixest.estimation.runner import run_estimation
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import Ssc, capture_context


def _quantreg_input_checks(
    quantile: float | list[float], tol: float, maxiter: int | None
):
    "Run custom input checks for quantreg."
    if isinstance(quantile, list):
        if not all(isinstance(q, float) for q in quantile):
            raise ValueError("quantile must be a list of floats")

        if not all(0.0 < q < 1.0 for q in quantile):
            raise ValueError("quantile must be between 0 and 1")
    else:
        # single quantile provided
        if not isinstance(quantile, float):
            raise TypeError("quantile must be a float")
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")

    # tol must always be in (0, 1)
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must be in (0, 1)")

    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be greater than 0")


def quantreg(
    fml: str,
    data: DataFrameType,
    vcov: VcovTypeOptions | dict[str, str] | None = "nid",
    quantile: float | list[float] = 0.5,
    method: QuantregMethodOptions = "fn",
    multi_method: QuantregMultiOptions = "cfm1",
    tol: float = 1e-06,
    maxiter: int | None = None,
    ssc: Ssc | Mapping[str, Any] | None = None,
    collin_tol: float = 1e-09,
    separation_check: list[str] | None = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: int | Mapping[str, Any] | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    seed: int | None = None,
):
    """
    Estimate conditional quantiles using the interior-point algorithm of Portnoy and Koenker (1997).

    Call as `pf.quantreg(...)` after `import pyfixest as pf`. The default
    quantile is the conditional median (`quantile=0.5`).

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        In contrast to `feols()` and `feglm()`, no fixed effects formula syntax is supported.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : VcovTypeOptions | dict[str, str] | None, optional
        Type of variance-covariance matrix for inference, `nid` by default.
        Supported are `iid`, `hetero` (`HC1`, `HC2` and `HC3` are aliases for
        `hetero`), `nid`, and one-way cluster robust errors via a dictionary,
        e.g. `{"CRV1": "cluster_var"}`.
        The `iid`, `hetero` and cluster robust estimators are based on a
        kernel-based estimator as in Powell (1991). The `nid` method implements
        the robust sandwich estimator proposed in Hendricks and Koenker (1992).
        Cluster robust inference follows Parente and Santos Silva (2016).
        CRV3, multiway clustering and HAC (`NW`, `DK`) inference are not
        supported.

    quantile : float or list[float]
        The quantile to estimate, or a list of quantiles fitted jointly as a
        quantile regression process. Each must be between 0 and 1.

    method : QuantregMethodOptions, optional
        The method to use for the quantile regression, either "fn" (the
        default) or "pfn".
        "fn" implements the Frisch-Newton interior point algorithm
        described in Portnoy and Koenker (1997).
        The "pfn" method implements a variant of the
        algorithm proposed by Portnoy and Koenker (1997) including preprocessing steps, which
        a) can speed up the algorithm if N is very large but b) assumes independent observations.
        For details, you can either take a look at the Portnoy and Koenker paper, or "Fast Algorithms for the Quantile Regression Process"
        by Chernozhukov, Fernández-Val, and Melly (2019).

    multi_method : QuantregMultiOptions, optional
        Controls the algorithm for running the quantile regression process.
        Only relevant if more than one quantile regression is fit in one `quantreg` call.
        Options are 'cfm1' (the default), which implements Algorithm 2 of
        Chernozhukov, Fernández-Val and Melly (2019), and 'cfm2', which
        implements their Algorithm 3.

    tol : float, optional
        The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the
        algorithm stops when the duality gap falls below `tol`.

    maxiter : int, optional
        The maximum number of iterations. If `None`, maxiter = the number of observations in the model
        (as in R's quantreg package via nit(3) = n).

    ssc : Ssc | Mapping[str, Any] | None, optional
        Small sample correction for inference, created with
        [ssc()](/reference/utils.utils.ssc.qmd). If `None`, uses the default
        settings of `pf.ssc()`. Note that by default, R's quantreg and Stata's
        qreg2 do not use small sample corrections. To match their behavior, set
        `ssc = pf.ssc(k_adj=False, G_adj=False)`.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-09.

    separation_check : list[str] | None, optional
        Not used in quantile regression. The value is validated as in
        `feglm()` but has no effect.

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
        relative to the caller (`0` = the frame that calls `quantreg()`); a
        mapping supplies the names directly.

    split : str | None, optional
        A column name, e.g. `split="var"`. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit : str | None, optional
        This argument is the same as split but also includes the full sample as the first estimation.

    seed : int | None, optional
        A random seed for reproducibility. If `None`, no seed is set.
        Only relevant for the "pfn" method.
        The "fn" method is deterministic and does not require a seed.

    Returns
    -------
    Quantreg | `FixestMulti`
        A [Quantreg](/reference/estimation.quantreg.quantreg_.Quantreg.qmd)
        fitted result for a single specification and quantile. A list of
        quantiles, multiple specifications, or sample splitting with
        `split`/`fsplit` returns
        [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd).
        Use `fits.to_list()` to access individual fits.

    Notes
    -----
    Quantile fits support coefficient inference and point prediction, but
    not absorbed fixed effects, weights, IV, fixed-effect recovery, wild
    bootstrap, randomization inference, causal cluster variance,
    decomposition, online updates or SAVI. CRV3, multiway clustering and
    HAC covariance are unsupported. Predictions are fitted conditional
    quantiles, not conditional means.

    See [fitted results](/reference-guide/fitted-results.qmd) for common operations,
    [model support](/reference-guide/model-support.qmd) for estimator restrictions,
    and [model storage](/reference-guide/model-storage.qmd) for retained-data requirements.

    Examples
    --------
    Fit a conditional median with the default `"nid"` covariance estimator:

    ```{python}
    import pyfixest as pf

    data = pf.get_data()
    fit = pf.quantreg("Y ~ X1 + X2", data=data, quantile=0.5)
    fit.tidy()
    ```

    Fit several quantiles and plot their coefficients:

    ```{python}
    fits = pf.quantreg(
        "Y ~ X1 + X2", data=data, quantile=[0.1, 0.5, 0.9]
    )
    figure, axes = pf.qplot(fits)
    ```

    See Also
    --------
    [pf.qplot()](/reference/report.qplot.qmd) plots quantile regression paths;
    [pf.feols()](/reference/estimation.api.feols.feols.qmd) fits conditional means.
    See the [quantile regression tutorial](/tutorials/quantile-regression.qmd)
    for fast fitting, inference and visualization.
    """
    # WLS currently not supported for quantile regression
    weights = None
    weights_type: WeightsTypeOptions = "aweights"
    solver: SolverOptions = "np.linalg.solve"
    # fixed effects are rejected for quantile regression; use the feols default
    fixef_rm: FixedRmOptions = "singleton"

    ssc = _resolve_ssc(ssc)

    context = {} if context is None else capture_context(context)

    if isinstance(vcov, str) and vcov in ["HC1", "HC2", "HC3"]:
        vcov = "hetero"
    vcov_spec = _resolve_vcov(vcov, None)

    _quantreg_input_checks(quantile, tol, maxiter)

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

    options = QuantregEstimationOptions(
        ssc=ssc,
        drop_singletons=fixef_rm == "singleton",
        drop_intercept=drop_intercept,
        weights=weights,
        weights_type=weights_type,
        offset=None,
        collin_tol=collin_tol,
        solver=solver,
        demeaner=MapDemeaner(),
        store_data=store_data,
        copy_data=copy_data,
        lean=lean,
        context=context,
        # a quantile process carries its first quantile; each child fit
        # gets its own via `dataclasses.replace`
        quantile=quantile[0] if isinstance(quantile, list) else quantile,
        method=method,
        quantile_tol=tol,
        quantile_maxiter=maxiter,
        seed=seed,
    )
    config = EstimationConfig(
        method="quantreg",
        data=data,
        formulas=tuple(Formula.parse(fml)),
        options=options,
        vcov=vcov_spec,
        split=split,
        fsplit=fsplit,
        quantile_process=QuantileProcess(quantiles=quantile, multi_method=multi_method)
        if isinstance(quantile, list)
        else None,
    )

    plan = plan_formulas(config)
    if plan.is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Quantile Regression"
        )

    return run_estimation(config, plan, apply_retention=True)
