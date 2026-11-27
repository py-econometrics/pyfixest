from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation.api.utils import (
    _estimation_input_checks,
    _resolve_ssc,
    _resolve_vcov,
    _warn_argument_without_effect,
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
    quantiles = quantile if isinstance(quantile, list) else [quantile]
    if not quantiles:
        raise ValueError(
            "quantile must include at least one value. For example, pass quantile=0.5 "
            "or quantile=[0.25, 0.5, 0.75]."
        )
    for index, value in enumerate(quantiles):
        option = f"quantile[{index}]" if isinstance(quantile, list) else "quantile"
        if not isinstance(value, float):
            error = ValueError if isinstance(quantile, list) else TypeError
            raise error(f"{option} must be a float; got {value!r}.")
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{option} must be between 0 and 1, exclusive; got {value!r}."
            )

    if not isinstance(tol, Real):
        raise TypeError(f"tol must be a real number; got {tol!r}.")
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must be greater than 0 and less than 1; got {tol!r}.")

    if maxiter is not None:
        if isinstance(maxiter, bool) or not isinstance(maxiter, Integral):
            raise TypeError(
                f"maxiter must be a positive integer or None; got {maxiter!r}."
            )
        if maxiter <= 0:
            raise ValueError(f"maxiter must be greater than 0; got {maxiter!r}.")


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
    Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997).
    Note that the interior point algorithm assumes independent observations.

    Parameters
    ----------
    fml : str
        A two-sided formula using fixest formula syntax.
        In contrast to `feols()` and `feglm()`, no fixed effects formula syntax is supported.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    quantile : float or list[float]
        The quantile to estimate, or a list of quantiles fitted jointly as a
        quantile regression process. Each must be between 0 and 1.

    method : QuantregMethodOptions, optional
        The method to use for the quantile regression. Currently, only "fn" is supported.
        In the future, will be either "fn" or "pfn".
        "fn" implements the Frisch-Newton interior point algorithm
        described in Portnoy and Koenker (1997).
        The "pfn" method implements a variant of the
        algorithm proposed by Portnoy and Koenker (1997) including preprocessing steps, which
        a) can speed up the algorithm if N is very large but b) assumes independent observations.
        For details, you can either take a look at the Portnoy and Koenker paper, or "Fast Algorithms for the Quantile Regression Process"
        by Chernozhukov, Fernández-Val, and Melly (2019).

    multi_method: QuantregMultiMethodOpitons, optional
        Controls the algorithm for running the quantile regression process.
        Only relevant if more than one quantile regression is fit in one `quantreg` call.
        Options are 'cmf1', which is the default and implements algorithm 2 from Chernozhukov et al,
        'cmf2', which implements their algorithm 3, and 'none', which just loops over separate model
        calls.

    tol : float, optional
        The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the
        algorithm stops when the relative change in the duality gap is less than tol.

    maxiter : int, optional
        The maximum number of iterations. If None, maxiter = the number of observations in the model
        (as in R's quantreg package via nit(3) = n).

    vcov : VcovTypeOptions or dict[str, str], optional
        Variance estimator. Default is "nid", the robust sandwich estimator of
        Hendricks and Koenker (1993). Also supported are "iid", "hetero" (to which
        "HC1", "HC2", and "HC3" are equivalent), and one-way cluster-robust
        inference following Parente and Santos Silva (2016), requested with a
        dictionary such as {"CRV1": "cluster_var"}. "iid", "hetero", and the
        cluster-robust estimator are kernel-based, as in Powell (1991). None means
        "iid".

    ssc : Ssc or Mapping[str, Any], optional
        The small-sample correction for inference, created with `ssc()`. Default is
        None, which uses the defaults of `ssc()`.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-09.

    separation_check : list[str], optional
        Deprecated; has no effect. Passing a value other than None emits a
        `PyfixestDeprecationWarning`.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True. If False, the
        data is not copied, which can save memory but may lead to unintended changes
        in the input data.

    store_data : bool, optional
        Whether the fitted model retains the data, by default True. Set to
        False to save memory. Accessing `fit.data` or calling a post-estimation
        operation that needs the data, such as `predict()` or `vcov()`, then
        raises `MissingModelDataError`.

    lean: bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods, which raise `MissingModelDataError` when required
        attributes were removed. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    context : int or Mapping[str, Any], optional
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    split : str, optional
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit : str, optional
        This argument is the same as split but also includes the full sample as the first estimation.

    seed: int, optional
        A random seed for reproducibility. If None, no seed is set.
        Only relevant for the "pfn" method.
        The "fn" method is deterministic and does not require a seed.

    Returns
    -------
    object
        An instance of the [Quantreg](/reference/estimation.quantreg.quantreg_.Quantreg.qmd) class or [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd) class for multiple models specified via `fml`.

    Notes
    -----
    `Quantreg` inherits from
    [Feols](/reference/estimation.models.feols_.Feols.qmd). A quantile
    regression fit therefore supports the same post-estimation methods as an OLS
    fit, among them `vcov()`, `tidy()`, `coef()`, `confint()` and `predict()`.
    See the Post-Estimation Methods section of the function reference for the
    full list.

    Fixed effects are not yet supported. Multiway clustering is not supported
    for the cluster-robust variance estimator.

    R's quantreg and Stata's qreg2 do not apply small-sample corrections by
    default. To match their behavior, pass
    `ssc=pf.ssc(k_adj=False, G_adj=False)`.

    Examples
    --------
    The following example regresses `Y` on `X1` and `X2` at the median (0.5 quantile):

    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.summary()
    ```

    To fit multiple quantiles in one call:

    ```{python}
    fits = pf.quantreg("Y ~ X1 + X2", data, quantile=[0.1, 0.5, 0.9])
    pf.qplot(fits)
    ```

    Arguments such as `split`, `fsplit`, `context`, `lean`, and `copy_data`
    behave as in `feols()`, but quantile regression does not support fixed-effects
    formula syntax. For details around inference, fast fitting, and visualization
    of the full quantile regression process, see the
    [quantile regression tutorial](/tutorials/quantile-regression.html).
    """
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
    if separation_check is not None:
        _warn_argument_without_effect(function="quantreg", argument="separation_check")

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
            "quantreg() does not support an instrument block. Use feols() if the "
            "intended model is linear IV. See https://pyfixest.org/tutorials/quantile-regression.html"
        )

    return run_estimation(config, plan, apply_retention=True)
