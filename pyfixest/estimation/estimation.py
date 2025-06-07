import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Optional, Union

import pandas as pd

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    FixedRmOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.options import options
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import capture_context
from pyfixest.utils.api_input_checks import _check_type, _check_value, EstimationInputs


def autofill_with_options(func):
    """
    Decorator to autofill the arguments of the estimation functions with the global options.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if value is None and hasattr(options, name):
                bound.arguments[name] = getattr(options, name)
        return func(**bound.arguments)

    return wrapper


@autofill_with_options
def feols(
    fml: Optional[str] = None,
    data: Optional[DataFrameType] = None,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    weights: Union[None, str] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: Optional[FixedRmOptions] = None,
    fixef_tol: Optional[float] = None,
    collin_tol: Optional[float] = None,
    drop_intercept: Optional[bool] = None,
    copy_data: Optional[bool] = None,
    store_data: Optional[bool] = None,
    lean: Optional[bool] = None,
    weights_type: Optional[WeightsTypeOptions] = None,
    solver: Optional[SolverOptions] = None,
    demeaner_backend: Optional[DemeanerBackendOptions] = None,
    use_compression: Optional[bool] = None,
    reps: Optional[int] = None,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    seed: Optional[int] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
) -> Union[Feols, FixestMulti]:
    """
    Estimate a linear regression models with fixed effects using fixest formula syntax.

    Parameters
    ----------
    fml : str
        A three-sided formula string using fixest formula syntax.
        Syntax: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1". "|" separates dependent variable,
        fixed effects, and instruments. Special syntax includes stepwise regressions,
        cumulative stepwise regression, multiple dependent variables,
        interaction of variables (i(X1,X2)), and interacted fixed effects (fe1^fe2).

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid",
        "hetero", "HC1", "HC2", "HC3", or a dictionary for CRV1/CRV3 inference.

    weights : Union[None, str], optional.
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : FixedRmOptions
        Specifies whether to drop singleton fixed effects.
        Options: "none" (default), "singleton".

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

    fixef_tol: float, optional
        Tolerance for the fixed effects demeaning algorithm. Defaults to 1e-08.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `fepois`.
        For example, the input data set is re-index within the function.
        As far as I know, the only other relevant case is
        when using interacted fixed effects, in which case you'll find
        a column with interacted fixed effects in the data set.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object. This has
        impact on post-estimation capabilities that rely on the data, e.g. `predict()`
        or `vcov()`.

    lean: bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    weights_type: WeightsTypeOptions, optional
        Options include `aweights` or `fweights`. `aweights` implement analytic or
        precision weights, while `fweights` implement frequency weights. For details
        see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.

    solver : SolverOptions, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".
        Defaults to "scipy.linalg.solve".

    demeaner_backend: DemeanerBackendOptions, optional
        The backend to use for demeaning. Can be either "numba", "jax", or "rust".
        Defaults to "numba".

    use_compression: bool
        Whether to use sufficient statistics to losslessly fit the regression model
        on compressed data. False by default. If True, the model is estimated on
        compressed data, which can lead to a significant speed-up for large data sets.
        See the paper by Wong et al (2021) for more details https://arxiv.org/abs/2102.11297.
        Note that if `use_compression = True`, inference is lossless. If standard errors are
        clustered, a wild cluster bootstrap is employed. Parameters for the wild bootstrap
        can be specified via the `reps` and `seed` arguments. Additionally, note that for one-way
        fixed effects, the estimation method uses a Mundlak transform to "control" for the
        fixed effects. For two-way fixed effects, a two-way Mundlak transform is employed.
        For two-way fixed effects, the Mundlak transform is only identical to a two-way
        fixed effects model if the data set is a panel. We do not provide any checks for the
        panel status of the data set.

    reps: int
        Number of bootstrap repetitions. Only relevant for boostrap inference applied to
        compute cluster robust errors when `use_compression = True`.

    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    seed: Optional[int]
        Seed for the random number generator. Only relevant for boostrap inference applied to
        compute cluster robust errors when `use_compression = True`.

    split: Optional[str]
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit: Optional[str]
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    object
        An instance of the [Feols(/reference/Feols.qmd) class or `FixestMulti`
        class for multiple models specified via `fml`.

    Examples
    --------
    As in `fixest`, the [Feols(/reference/Feols.qmd) function can be used to
    estimate a simple linear regression model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for
    `f1` and `f2`: fixed effects are specified after the `|` symbol.

    ```{python}
    import pyfixest as pf
    import pandas as pd
    import numpy as np

    data = pf.get_data()

    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```

    Calling `feols()` returns an instance of the [Feols(/reference/Feols.qmd)
    class. The `summary()` method can be used to print the results.

    An alternative way to retrieve model results is via the `tidy()` method, which
    returns a pandas dataframe with the estimated coefficients, standard errors,
    t-statistics, and p-values.

    ```{python}
    fit.tidy()
    ```

    You can also access all elements in the tidy data frame by dedicated methods,
    e.g. `fit.coef()` for the coefficients, `fit.se()` for the standard errors,
    `fit.tstat()` for the t-statistics, and `fit.pval()` for the p-values, and
    `fit.confint()` for the confidence intervals.

    The employed type of inference can be specified via the `vcov` argument. If
    vcov is not provided, `PyFixest` employs the `fixest` default of iid inference,
    unless there are fixed effects in the model, in which case `feols()` clusters
    the standard error by the first fixed effect (CRV1 inference).

    ```{python}
    fit1 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov="iid")
    fit2 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov="hetero")
    fit3 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
    ```

    Supported inference types are "iid", "hetero", "HC1", "HC2", "HC3", and
    "CRV1"/"CRV3". Clustered standard errors are specified via a dictionary,
    e.g. `{"CRV1": "f1"}` for CRV1 inference with clustering by `f1` or
    `{"CRV3": "f1"}` for CRV3 inference with clustering by `f1`. For two-way
    clustering, you can provide a formula string, e.g. `{"CRV1": "f1 + f2"}` for
    CRV1 inference with clustering by `f1`.

    ```{python}
    fit4 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1 + f2"})
    ```

    Inference can be adjusted post estimation via the `vcov` method:

    ```{python}
    fit.summary()
    fit.vcov("iid").summary()
    ```

    The `ssc` argument specifies the small sample correction for inference. In
    general, `feols()` uses all of `fixest::feols()` defaults, but sets the
    `fixef.K` argument to `"none"` whereas the `fixest::feols()` default is `"nested"`.
    See here for more details: [link to github](https://github.com/py-econometrics/pyfixest/issues/260).

    `feols()` supports a range of multiple estimation syntax, i.e. you can estimate
    multiple models in one call. The following example estimates two models, one with
    fixed effects for `f1` and one with fixed effects for `f2` using the `sw()` syntax.

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | sw(f1, f2)", data)
    type(fit)
    ```

    The returned object is an instance of the `FixestMulti` class. You can access
    the results of the first model via `fit.fetch_model(0)` and the results of
    the second model via `fit.fetch_model(1)`. You can compare the model results
    via the `etable()` function:

    ```{python}
    pf.etable(fit)
    ```

    Other supported multiple estimation syntax include `sw0()`, `csw()` and `csw0()`.
    While `sw()` adds variables in a "stepwise" fashion, `csw()` does so cumulatively.

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | csw(f1, f2)", data)
    pf.etable(fit)
    ```

    The `sw0()` and `csw0()` syntax are similar to `sw()` and `csw()`, but start
    with a model that excludes the variables specified in `sw()` and `csw()`:

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | sw0(f1, f2)", data)
    pf.etable(fit)
    ```

    The `feols()` function also supports multiple dependent variables. The following
    example estimates two models, one with `Y1` as the dependent variable and one
    with `Y2` as the dependent variable.

    ```{python}
    fit = pf.feols("Y + Y2 ~ X1 | f1 + f2", data)
    pf.etable(fit)
    ```

    It is possible to combine different multiple estimation operators:

    ```{python}
    fit = pf.feols("Y + Y2 ~ X1 | sw(f1, f2)", data)
    pf.etable(fit)
    ```

    In general, using muliple estimation syntax can improve the estimation time
    as covariates that are demeaned in one model and are used in another model do
    not need to be demeaned again: `feols()` implements a caching mechanism that
    stores the demeaned covariates.

    Additionally, you can fit models on different samples via the split and fsplit
    arguments. The split argument splits the sample according to the variable
    specified in the argument, while the fsplit argument also includes the full
    sample in the estimation.

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data, split = "f1")
    pf.etable(fit)
    ```

    Besides OLS, `feols()` also supports IV estimation via three part formulas:

    ```{python}
    fit = pf.feols("Y ~  X2 | f1 + f2 | X1 ~ Z1", data)
    fit.tidy()
    ```
    Here, `X1` is the endogenous variable and `Z1` is the instrument. `f1` and `f2`
    are the fixed effects, as before. To estimate IV models without fixed effects,
    simply omit the fixed effects part of the formula:

    ```{python}
    fit = pf.feols("Y ~  X2 | X1 ~ Z1", data)
    fit.tidy()
    ```

    Last, `feols()` supports interaction of variables via the `i()` syntax.
    Documentation on this is tba.

    You can pass custom transforms via the `context` argument. If you set `context = 0`, all
    functions from the level of the call to `feols()` will be available:

    ```{python}
    def _lspline(series: pd.Series, knots: list[float]) -> np.array:
        'Generate a linear spline design matrix for the input series based on knots.'
        vector = series.values
        columns = []

        for i, knot in enumerate(knots):
            column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
            columns.append(column)
            vector = vector - column

        # Add the remainder as the last column
        columns.append(vector)

        # Combine columns into a design matrix
        return np.column_stack(columns)

    spline_split = _lspline(data["X2"], [0, 1])
    data["X2_0"] = spline_split[:, 0]
    data["0_X2_1"] = spline_split[:, 1]
    data["1_X2"] = spline_split[:, 2]

    explicit_fit = pf.feols("Y ~ X2_0 + 0_X2_1 + 1_X2 | f1 + f2", data=data)
    # set context = 0 to make _lspline available for feols' internal call to Formulaic.model_matrix
    context_captured_fit = pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=data, context = 0)
    # or provide it as a dict / mapping
    context_captured_fit_map = pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=data, context = {"_lspline":_lspline})

    pf.etable([explicit_fit, context_captured_fit, context_captured_fit_map])
    ```

    After fitting a model via `feols()`, you can use the `predict()` method to
    get the predicted values:

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.predict()[0:5]
    ```

    The `predict()` method also supports a `newdata` argument to predict on new data,
    which returns a numpy array of the predicted values:

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.predict(newdata=data)[0:5]
    ```

    Last, you can plot the results of a model via the `coefplot()` method:

    ```{python}
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.coefplot()
    ```

    We can conduct a regression decomposition via the `decompose()` method, which implements
    a regression decomposition following the method developed in Gelbach (2016):

    ```{python}
    import re
    import pyfixest as pf
    from pyfixest.utils.dgps import gelbach_data

    data_gelbach = gelbach_data(nobs = 1000)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data_gelbach)

    # simple decomposition
    res = fit.decompose(param = "x1")
    pf.make_table(res)

    # group covariates via "combine_covariates" argument
    res = fit.decompose(param = "x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]})
    pf.make_table(res)

    # group covariates via regex
    res = fit.decompose(param="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
    ```

    Objects of type `Feols` support a range of other methods to conduct inference.
    For example, you can run a wild (cluster) bootstrap via the `wildboottest()` method:

    ```{python}
    fit = pf.feols("Y ~ X1 + X2", data)
    fit.wildboottest(param = "X1", reps=1000)
    ```
    would run a wild bootstrap test for the coefficient of `X1` with 1000
    bootstrap repetitions.

    For a wild cluster bootstrap, you can specify the cluster variable
      via the `cluster` argument:

    ```{python}
    fit.wildboottest(param = "X1", reps=1000, cluster="group_id")
    ```

    The `ritest()` method can be used to conduct randomization inference:

    ```{python}
    fit.ritest(resampvar = "X1", reps=1000)
    ```

    Last, you can compute the cluster causal variance estimator by Athey et
    al by using the `ccv()` method:

    ```{python}
    import numpy as np
    rng = np.random.default_rng(1234)
    data["D"] = rng.choice([0, 1], size = data.shape[0])
    fit_D = pf.feols("Y ~ D", data = data)
    fit_D.ccv(treatment = "D", cluster = "group_id")
    ```
    """
    context = {} if context is None else capture_context(context)

    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)

    args = locals()
    filtered_args = {
        k: args[k] for k in EstimationInputs.__dataclass_fields__ if k in args
    }
    EstimationInputs(**filtered_args).validate()

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        weights_type=weights_type,
        use_compression=use_compression,
        reps=reps,
        seed=seed,
        split=split,
        fsplit=fsplit,
        context=context,
    )

    estimation = "feols" if not use_compression else "compression"

    fixest._prepare_estimation(
        estimation, fml, vcov, weights, ssc, fixef_rm, drop_intercept
    )

    # demean all models: based on fixed effects x split x missing value combinations
    fixest._estimate_all_models(
        vcov,
        collin_tol=collin_tol,
        solver=solver,
        demeaner_backend=demeaner_backend,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)


@autofill_with_options
def fepois(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: FixedRmOptions = "none",
    fixef_tol: float = 1e-08,
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-10,
    separation_check: Optional[list[str]] = None,
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner_backend: DemeanerBackendOptions = "numba",
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
) -> Union[Feols, Fepois, FixestMulti]:
    """
    Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        Syntax: "Y ~ X1 + X2 | FE1 + FE2". "|" separates left-hand side and fixed
        effects.
        Special syntax includes:
        - Stepwise regressions (sw, sw0)
        - Cumulative stepwise regression (csw, csw0)
        - Multiple dependent variables (Y1 + Y2 ~ X)
        - Interaction of variables (i(X1,X2))
        - Interacted fixed effects (fe1^fe2)
        Compatible with formula parsing via the formulaic module.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid",
        "hetero", "HC1", "HC2", "HC3", or a dictionary for CRV1/CRV3 inference.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : FixedRmOptions
        Specifies whether to drop singleton fixed effects.
        Options: "none" (default), "singleton".

    fixef_tol: float, optional
        Tolerance for the fixed effects demeaning algorithm. Defaults to 1e-08.

    iwls_tol : Optional[float], optional
        Tolerance for IWLS convergence, by default 1e-08.

    iwls_maxiter : Optional[float], optional
        Maximum number of iterations for IWLS convergence, by default 25.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

    separation_check: list[str], optional
        Methods to identify and drop separated observations.
        Either "fe" or "ir". Executes "fe" by default (when None).

    solver : SolverOptions, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".
        Defaults to "scipy.linalg.solve".

    demeaner_backend: DemeanerBackendOptions, optional
        The backend to use for demeaning. Can be either "numba", "jax", or "rust".
        Defaults to "numba".

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `fepois`.
        For example, the input data set is re-index within the function.
        As far as I know, the only other relevant case is
        when using interacted fixed effects, in which case you'll find
        a column with interacted fixed effects in the data set.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object. This has
        impact on post-estimation capabilities that rely on the data, e.g. `predict()`
        or `vcov()`.

    lean: bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    split: Optional[str]
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit: Optional[str]
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    object
        An instance of the `Fepois` class or an instance of class `FixestMulti`
        for multiple models specified via `fml`.

    Examples
    --------
    The `fepois()` function can be used to estimate a simple Poisson regression
    model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for
    `f1` and `f2`: fixed effects are specified after the `|` symbol.

    ```{python}
    import pyfixest as pf

    data = pf.get_data(model = "Fepois")
    fit = pf.fepois("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```

    For more examples on the use of other function arguments, please take a look at the documentation of the [feols()](https://py-econometrics.github.io/pyfixest/reference/estimation.estimation.feols.html#pyfixest.estimation.estimation.feols) function.
    """
    context = {} if context is None else capture_context(context)

    # WLS currently not supported for Poisson regression
    weights = None
    weights_type = "aweights"

    args = locals()
    filtered_args = {
        k: args[k] for k in EstimationInputs.__dataclass_fields__ if k in args
    }
    EstimationInputs(**filtered_args).validate()

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        context=context,
    )

    fixest._prepare_estimation(
        "fepois", fml, vcov, weights, ssc, fixef_rm, drop_intercept
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
        solver=solver,
        demeaner_backend=demeaner_backend,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)


@autofill_with_options
def feglm(
    fml: Optional[str] = None,
    data: Optional[DataFrameType] = None,  # type: ignore
    family: Optional[str] = None,
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: Optional[FixedRmOptions] = None,
    fixef_tol: Optional[float] = None,
    iwls_tol: Optional[float] = None,
    iwls_maxiter: Optional[int] = None,
    collin_tol: Optional[float] = None,
    separation_check: Optional[list[str]] = None,
    solver: Optional[SolverOptions] = None,
    drop_intercept: Optional[bool] = None,
    copy_data: Optional[bool] = None,
    store_data: Optional[bool] = None,
    lean: Optional[bool] = None,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
) -> Union[Feols, Fepois, FixestMulti]:
    """
    Estimate GLM regression models (currently without fixed effects, this is work in progress).
    This feature is currently experimental, full support will be released with pyfixest 0.29.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        Syntax: "Y ~ X1 + X2 | FE1 + FE2". "|" separates left-hand side and fixed
        effects.
        Special syntax includes:
        - Stepwise regressions (sw, sw0)
        - Cumulative stepwise regression (csw, csw0)
        - Multiple dependent variables (Y1 + Y2 ~ X)
        - Interaction of variables (i(X1,X2))
        - Interacted fixed effects (fe1^fe2)
        Compatible with formula parsing via the formulaic module.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    family : str
        The family of the GLM model. Options include "gaussian", "logit" and "probit".

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid",
        "hetero", "HC1", "HC2", "HC3", or a dictionary for CRV1/CRV3 inference.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : FixedRmOptions
        Specifies whether to drop singleton fixed effects.
        Options: "none" (default), "singleton".

    fixef_tol: float, optional
        Tolerance for the fixed effects demeaning algorithm. Defaults to 1e-08.

    iwls_tol : Optional[float], optional
        Tolerance for IWLS convergence, by default 1e-08.

    iwls_maxiter : Optional[float], optional
        Maximum number of iterations for IWLS convergence, by default 25.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

    separation_check: list[str], optional
        Methods to identify and drop separated observations.
        Either "fe" or "ir". Executes "fe" by default (when None).

    solver : SolverOptions, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".
        Defaults to "scipy.linalg.solve".

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `fepois`.
        For example, the input data set is re-index within the function.
        As far as I know, the only other relevant case is
        when using interacted fixed effects, in which case you'll find
        a column with interacted fixed effects in the data set.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object. This has
        impact on post-estimation capabilities that rely on the data, e.g. `predict()`
        or `vcov()`.

    lean: bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
        to obtain the appropriate standard-errors at estimation time,
        since obtaining different SEs won't be possible afterwards.

    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    split: Optional[str]
        A character string, i.e. 'split = var'. If provided, the sample is split according to the
        variable and one estimation is performed for each value of that variable. If you also want
        to include the estimation for the full sample, use the argument fsplit instead.

    fsplit: Optional[str]
        This argument is the same as split but also includes the full sample as the first estimation.

    Returns
    -------
    object
        An instance of the `Fepois` class or an instance of class `FixestMulti`
        for multiple models specified via `fml`.

    Examples
    --------
    The `fepois()` function can be used to estimate a simple Poisson regression
    model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for
    `f1` and `f2`: fixed effects are specified after the `|` symbol.

    ```{python}
    import pyfixest as pf
    import numpy as np

    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > 0, 1, 0)
    data["f1"] = np.where(data["f1"] > data["f1"].median(), "group1", "group2")

    fit_probit = pf.feglm("Y ~ X1*f1", data, family = "probit")
    fit_logit = pf.feglm("Y ~ X1*f1", data, family = "logit")
    fit_gaussian = pf.feglm("Y ~ X1*f1", data, family = "gaussian")

    pf.etable([fit_probit, fit_logit, fit_gaussian])
    ```

    `PyFixest` integrates with the [marginaleffects](https://marginaleffects.com/bonus/python.html) package. For example, to compute average marginal effects
    for the probit model above, you can use the following code:

    ```{python}
    # we load polars as marginaleffects outputs pl.DataFrame's
    import polars as pl
    from marginaleffects import avg_slopes
    pl.concat([avg_slopes(model, variables  = "X1") for model in [fit_probit, fit_logit, fit_gaussian]])
    ```

    We can also compute marginal effects by group (group average marginal effects):

    ```{python}
    avg_slopes(fit_probit, variables  = "X1", by = "f1")
    ```

    We find homogeneous effects by "f1" in the probit model.

    For more examples of other function arguments, please take a look at the documentation of the [feols()](https://py-econometrics.github.io/pyfixest/reference/estimation.estimation.feols.html#pyfixest.estimation.estimation.feols)
    function.

    """
    if family not in ["logit", "probit", "gaussian"]:
        raise ValueError(
            f"Only families 'gaussian', 'logit' and 'probit'are supported but you asked for {family}."
        )

    # WLS currently not supported for GLM regression
    weights = None
    weights_type = "aweights"

    context = {} if context is None else capture_context(context)

    args = locals()
    filtered_args = {
        k: args[k] for k in EstimationInputs.__dataclass_fields__ if k in args
    }
    EstimationInputs(**filtered_args).validate()

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        context=context,
    )

    # same checks as for Poisson regression
    fixest._prepare_estimation(
        f"feglm-{family}", fml, vcov, weights, ssc, fixef_rm, drop_intercept
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
        solver=solver,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)
