from collections.abc import Mapping
from typing import Any, Optional, Union

import pandas as pd

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    FixedRmOptions,
    QuantregMethodOptions,
    SolverOptions,
    VcovTypeOptions,
    WeightsTypeOptions,
)
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import capture_context
from pyfixest.utils.utils import ssc as ssc_func


def feols(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    weights: Union[None, str] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: FixedRmOptions = "none",
    fixef_tol=1e-08,
    fixef_maxiter: int = 100_000,
    collin_tol: float = 1e-10,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    weights_type: WeightsTypeOptions = "aweights",
    solver: SolverOptions = "scipy.linalg.solve",
    demeaner_backend: DemeanerBackendOptions = "numba",
    use_compression: bool = False,
    reps: int = 100,
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

    fixef_maxiter: int, optional
        Maximum number of iterations for the demeaning algorithm. Defaults to 100,000.

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
    if ssc is None:
        ssc = ssc_func()
    context = {} if context is None else capture_context(context)

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        fixef_maxiter=fixef_maxiter,
        collin_tol=collin_tol,
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
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
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


def fepois(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: FixedRmOptions = "none",
    fixef_tol: float = 1e-08,
    fixef_maxiter: int = 100_000,
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

    fixef_maxiter: int, optional
        Maximum number of iterations for the demeaning algorithm. Defaults to 100,000.

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
    if separation_check is None:
        separation_check = ["fe"]
    if ssc is None:
        ssc = ssc_func()
    context = {} if context is None else capture_context(context)

    # WLS currently not supported for Poisson regression
    weights = None
    weights_type = "aweights"

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        separation_check=separation_check,
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
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


def feglm(
    fml: str,
    data: DataFrameType,  # type: ignore
    family: str,
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    fixef_rm: FixedRmOptions = "none",
    fixef_tol: float = 1e-08,
    fixef_maxiter: int = 100_000,
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-10,
    separation_check: Optional[list[str]] = None,
    solver: SolverOptions = "scipy.linalg.solve",
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
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

    fixef_maxiter: int, optional
        Maximum iterations for the demeaning algorithm.

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

    if separation_check is None:
        separation_check = ["fe"]
    if ssc is None:
        ssc = ssc_func()
    # WLS currently not supported for GLM regression
    weights = None
    weights_type = "aweights"

    context = {} if context is None else capture_context(context)

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=None,
        split=split,
        fsplit=fsplit,
        separation_check=separation_check,
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
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


def quantreg(
    fml: str,
    data: DataFrameType,  # type: ignore
    vcov: Optional[Union[VcovTypeOptions, dict[str, str]]] = "nid",
    quantile: float = 0.5,
    method: QuantregMethodOptions = "fn",
    tol: float = 1e-06,
    maxiter: Optional[int] = None,
    ssc: Optional[dict[str, Union[str, bool]]] = None,
    collin_tol: float = 1e-10,
    separation_check: Optional[list[str]] = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
    split: Optional[str] = None,
    fsplit: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997).
    Note that the interior point algorithm assumes independent observations.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax.
        In contrast to `feols()` and `feglm()`, no fixed effects formula syntax is supported.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    quantile : float
        The quantile to estimate. Must be between 0 and 1.

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

    tol : float, optional
        The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the
        algorithm stops when the relative change in the duality gap is less than tol.

    maxiter : int, optional
        The maximum number of iterations. If None, maxiter = the number of observations in the model
        (as in R's quantreg package via nit(3) = n).

    vcov : Union[VcovTypeOptions, dict[str, str]]
        Type of variance-covariance matrix for inference. Currently supported are "nid" and cluster robust errors.
        The "nid" method implements the robust sandwich estimator proposed in Hendricks and Koenker (1993).
        Any of "hetero" / HC1 / HC2 / HC3 also works and is equivalent to nid. Alternatively, cluster robust inference
        following Parente and Santos Silva (2016) can be specified via a dictionary with the keys "type" and "cluster".
        Only one-way clustering is supported.

    ssc : dict[str, Union[str, bool]], optional
        A dictionary specifying the small sample correction for inference.
        If None, uses default settings from `ssc_func()`. Note that by default, R's quantreg and Stata's qreg2 do not use
        small sample corrections. To match their behavior, set `ssc = pf.ssc(adj = False, cluster_adj = False)`.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-10.

    separation_check : list[str], optional
        Methods to identify and drop separated observations. Not used in quantile regression.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    copy_data : bool, optional
        Whether to copy the data before estimation, by default True.
        If set to False, the data is not copied, which can save memory but
        may lead to unintended changes in the input data outside of `quantreg`.

    store_data : bool, optional
        Whether to store the data in the model object, by default True.
        If set to False, the data is not stored in the model object, which can
        improve performance and save memory. However, it will no longer be possible
        to access the data via the `data` attribute of the model object.

    lean : bool, optional
        False by default. If True, then all large objects are removed from the
        returned result: this will save memory but will block the possibility
        to use many methods. It is recommended to use the argument vcov
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
        An instance of the Quantreg class or FixestMulti class for multiple models specified via `fml`.

    Examples
    --------
    The following example regresses `Y` on `X1` and `X2` at the median (0.5 quantile):

    ```{python}
    import pyfixest as pf
    import pandas as pd
    import numpy as np

    data = pf.get_data()

    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.summary()
    ```

    The employed type of inference can be specified via the `vcov` argument. Currently,
    only "nid" (non-IID) and cluster robust errors as in Parente and Santos Silva (2016) are supported.

    ```{python}
    fit_nid = pf.quantreg("Y ~ X1 + X2", data.dropna(), quantile=0.5, vcov="nid")
    fit_crv = pf.quantreg("Y ~ X1 + X2", data.dropna(), quantile=0.5, vcov = {"CRV1": "f1"})
    pf.etable([fit_nid, fit_crv])
    ```

    After fitting a model via `quantreg()`, you can use the `predict()` method to
    get the predicted values:

    ```{python}
    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.predict()[0:5]
    ```

    The `predict()` method also supports a `newdata` argument to predict on new data:

    ```{python}
    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.predict(newdata=data)[0:5]
    ```

    Last, you can plot the results of a model via the `coefplot()` method:

    ```{python}
    fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
    fit.coefplot()
    ```

    You can visualize the quantile regression process via the `qplot()` function:

    ```{python}
    fit_process = [pf.quantreg("Y ~ X1 + X2", data, quantile=q) for q in [0.1, 0.25, 0.5, 0.75, 0.9]]
    pf.qplot(fit_process)
    ```
    """
    # WLS currently not supported for quantile regression
    weights = None
    weights_type = "aweights"
    solver: SolverOptions = "np.linalg.solve"

    if ssc is None:
        ssc = ssc_func()

    context = {} if context is None else capture_context(context)

    fixef_rm = "none"
    fixef_tol = 1e-08
    iwls_tol = 1e-08
    iwls_maxiter = 25

    if isinstance(vcov, str) and vcov in ["hetero", "HC1", "HC2", "HC3"]:
        vcov = "nid"

    _quantreg_input_checks(quantile, tol, maxiter)

    _estimation_input_checks(
        fml=fml,
        data=data,
        vcov=vcov,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
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
        separation_check=separation_check,
    )

    fixest = FixestMulti(
        data=data,
        copy_data=copy_data,
        store_data=store_data,
        lean=lean,
        fixef_tol=fixef_tol,
        weights_type=weights_type,
        use_compression=False,
        reps=None,
        seed=seed,
        split=split,
        fsplit=fsplit,
        context=context,
        quantreg_method=method,
    )

    # same checks as for Poisson regression
    fixest._prepare_estimation(
        "quantreg", fml, vcov, weights, ssc, fixef_rm, drop_intercept
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Quantile Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
        separation_check=separation_check,
        solver=solver,
        quantile=quantile,
        quantile_tol=tol,
        quantile_maxiter=maxiter,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)


def _estimation_input_checks(
    fml: str,
    data: DataFrameType,
    vcov: Optional[Union[str, dict[str, str]]],
    weights: Union[None, str],
    ssc: dict[str, Union[str, bool]],
    fixef_rm: str,
    collin_tol: float,
    copy_data: bool,
    store_data: bool,
    lean: bool,
    fixef_tol: float,
    fixef_maxiter: int,
    weights_type: str,
    use_compression: bool,
    reps: Optional[int],
    seed: Optional[int],
    split: Optional[str],
    fsplit: Optional[str],
    separation_check: Optional[list[str]] = None,
):
    if not isinstance(fml, str):
        raise TypeError("fml must be a string")
    if not isinstance(data, pd.DataFrame):
        data = _narwhals_to_pandas(data)
    if not isinstance(vcov, (str, dict, type(None))):
        raise TypeError("vcov must be a string, dictionary, or None")
    if not isinstance(fixef_rm, str):
        raise TypeError("fixef_rm must be a string")
    if not isinstance(collin_tol, float):
        raise TypeError("collin_tol must be a float")

    if fixef_rm not in ["none", "singleton"]:
        raise ValueError("fixef_rm must be either 'none' or 'singleton'")
    if collin_tol <= 0:
        raise ValueError("collin_tol must be greater than zero")
    if collin_tol >= 1:
        raise ValueError("collin_tol must be less than one")

    if not (isinstance(weights, str) or weights is None):
        raise ValueError(
            f"weights must be a string or None but you provided weights = {weights}."
        )
    if weights is not None:
        assert weights in data.columns, "weights must be a column in data"

    bool_args = [copy_data, store_data, lean]
    for arg in bool_args:
        if not isinstance(arg, bool):
            raise TypeError(f"The function argument {arg} must be of type bool.")

    if not isinstance(fixef_tol, float):
        raise TypeError(
            """The function argument `fixef_tol` needs to be of
            type float.
            """
        )
    if fixef_tol <= 0:
        raise ValueError(
            """
            The function argument `fixef_tol` needs to be of
            strictly larger than 0.
            """
        )
    if fixef_tol >= 1:
        raise ValueError(
            """
            The function argument `fixef_tol` needs to be of
            strictly smaller than 1.
            """
        )

    if not isinstance(fixef_maxiter, int):
        raise TypeError(
            """The function argument `fixef_maxiter` needs to be of
            type int.
            """
        )
    if fixef_maxiter <= 0:
        raise ValueError(
            """
            The function argument `fixef_maxiter` needs to be of
            strictly larger than 0.
            """
        )

    if weights_type not in ["aweights", "fweights"]:
        raise ValueError(
            f"""
            The `weights_type` argument must be of type `aweights`
            (for analytical / precision weights) or `fweights`
            (for frequency weights) but it is {weights_type}.
            """
        )

    if not isinstance(use_compression, bool):
        raise TypeError("The function argument `use_compression` must be of type bool.")
    if use_compression and weights is not None:
        raise NotImplementedError(
            "Compressed regression is not supported with weights."
        )

    if reps is not None:
        if not isinstance(reps, int):
            raise TypeError("The function argument `reps` must be of type int.")

        if reps <= 0:
            raise ValueError("The function argument `reps` must be strictly positive.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("The function argument `seed` must be of type int.")

    if split is not None and not isinstance(split, str):
        raise TypeError("The function argument split needs to be of type str.")

    if fsplit is not None and not isinstance(fsplit, str):
        raise TypeError("The function argument fsplit needs to be of type str.")

    if split is not None and fsplit is not None and split != fsplit:
        raise ValueError(
            f"""
                        Arguments split and fsplit are both specified, but not identical.
                        split is specified as {split}, while fsplit is specified as {fsplit}.
                        """
        )

    if isinstance(split, str) and split not in data.columns:
        raise KeyError(f"Column '{split}' not found in data.")

    if isinstance(fsplit, str) and fsplit not in data.columns:
        raise KeyError(f"Column '{fsplit}' not found in data.")

    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError(
                "The function argument `separation_check` must be of type list."
            )

        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                "The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'."
            )


def _quantreg_input_checks(quantile: float, tol: float, maxiter: Optional[int]):
    "Run custom input checks for quantreg."
    if not 0 < tol < 1:
        raise ValueError("tol must be in (0, 1)")
    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be greater than 0")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
