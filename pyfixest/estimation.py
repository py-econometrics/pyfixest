from typing import Optional, Union, Dict
from pyfixest.utils import ssc
from pyfixest.FixestMulti import FixestMulti
from pyfixest.fepois import Fepois
from pyfixest.feols import Feols
from pyfixest.dev_utils import DataFrameType

import pandas as pd


def feols(
    fml: str,
    data: DataFrameType,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
    collin_tol: float = 1e-10,
    drop_intercept: bool = False,
    i_ref1: Optional[Union[list, str]] = None,
    i_ref2: Optional[Union[list, str]] = None,
) -> Union[Feols, FixestMulti]:
    """
    Estimate linear regression models with fixed effects using fixest formula syntax.

    This method accommodates complex models with stepwise regressions, multiple dependent variables, interaction of variables,
    interacted fixed effects, and instruments. It's compatible with various syntax elements from the formulaic module.

    Parameters
    ----------
    fml : str
        A three-sided formula string using fixest formula syntax. Syntax: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1". "|" separates dependent variable,
        fixed effects, and instruments. Special syntax includes stepwise regressions, cumulative stepwise regression, multiple dependent variables,
        interaction of variables (i(X1,X2)), and interacted fixed effects (fe1^fe2).

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : Union[str, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid", "hetero", "HC1", "HC2", "HC3", or a dictionary for CRV1/CRV3 inference.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : str
        Specifies whether to drop singleton fixed effects. Options: "none" (default), "singleton".

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-06.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    i_ref1 : Optional[Union[list, str]], optional
        Reference category for the first set of categorical variables interacted via "i()", by default None.

    i_ref2 : Optional[Union[list, str]], optional
        Reference category for the second set of categorical variables interacted via "i()", by default None.

    Returns
    -------
    object
        An instance of the [Feols(/docs/reference/Feols.qmd) class or `FixestMulti` class for multiple models specified via `fml`.

    Examples
    --------

    As in `fixest`, the [Feols(/docs/reference/Feols.qmd) function can be used to estimate a simple linear regression model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for `f1` and `f2`: fixed effects are specified
    after the `|` symbol.

    ```{python}
    from pyfixest.estimation import feols
    from pyfixest.utils import get_data
    from pyfixest.summarize import etable

    data = get_data()

    fit = feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```

    Calling `feols()` returns an instance of the [Feols(/docs/reference/Feols.qmd) class. The `summary()` method can be used to print the results.

    An alternative way to retrieve model results is via the `tidy()` method, which returns a pandas dataframe with the
    estimated coefficients, standard errors, t-statistics, and p-values.

    ```{python}
    fit.tidy()
    ```

    You can also access all elements in the tidy data frame by dedicated methods, e.g. `fit.coef()` for the coefficients, `fit.se()` for the standard errors,
    `fit.tstat()` for the t-statistics, and `fit.pval()` for the p-values, and `fit.confint()` for the confidence intervals.

    The employed type of inference can be specified via the `vcov` argument. If vcov is not provided, `PyFixest` employs the `fixest` default of iid inference,
    unless there are fixed effects in the model, in which case `feols()` clusters the standard error by the first fixed effect (CRV1 inference).

    ```{python}
    fit1 = feols("Y ~ X1 + X2 | f1 + f2", data, vcov="iid")
    fit2 = feols("Y ~ X1 + X2 | f1 + f2", data, vcov="hetero")
    fit3 = feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
    ```

    Supported inference types are "iid", "hetero", "HC1", "HC2", "HC3", and "CRV1"/"CRV3". Clustered standard errors are specified via a dictionary, e.g. `{"CRV1": "f1"}`
    for CRV1 inference with clustering by `f1` or `{"CRV3": "f1"}` for CRV3 inference with clustering by `f1`. For two-way clustering, you can provide a formula string, e.g.
    `{"CRV1": "f1 + f2"}` for CRV1 inference with clustering by `f1`.

    ```{python}
    fit4 = feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1 + f2"})
    ```

    Inference can be adjusted post estimation via the `vcov` method:

    ```{python}
    fit.summary()
    fit.vcov("iid").summary()
    ```

    The `ssc` argument specifies the small sample correction for inference. In general, `feols()` uses all of `fixest::feols()` defaults, but sets the
    `fixef.K` argument to `"none"` whereas the `fixest::feols()` default is `"nested"`. See here for more details: [link to github](https://github.com/s3alfisc/pyfixest/issues/260).

    `feols()` supports a range of multiple estimation syntax, i.e. you can estimate multiple models in one call. The following example estimates two models, one with
    fixed effects for `f1` and one with fixed effects for `f2` using the `sw()` syntax.

    ```{python}
    fit = feols("Y ~ X1 + X2 | sw(f1, f2)", data)
    type(fit)
    ```

    The returned object is an instance of the `FixestMulti` class. You can access the results of the first model via `fit.fetch_model(0)` and the results of the second model
    via `fit.fetch_model(1)`. You can compare the model results via the `etable()` function:

    ```{python}
    etable([fit.fetch_model(0), fit.fetch_model(1)])
    ```

    Other supported multiple estimation syntax include `sw0()`, `csw()` and `csw0()`. While `sw()` adds variables in a "stepwise" fashinon, `csw()`
    does so cumulatively.

    ```{python}
    fit = feols("Y ~ X1 + X2 | csw(f1, f2)", data)
    etable([fit.fetch_model(0), fit.fetch_model(1)])
    ```

    The `sw0()` and `csw0()` syntax are similar to `sw()` and `csw()`, but start with a model that exludes the variables specified in `sw()` and `csw()`:

    ```{python}
    fit = feols("Y ~ X1 + X2 | sw0(f1, f2)", data)
    etable([fit.fetch_model(0), fit.fetch_model(1), fit.fetch_model(2)])
    ```

    The `feols()` function also supports multiple dependent variables. The following example estimates two models, one with `Y1` as the dependent variable and one with `Y2` as the dependent variable.

    ```{python}
    fit = feols("Y + Y2 ~ X1 | f1 + f2", data)
    etable([fit.fetch_model(0), fit.fetch_model(1)])
    ```

    It is possible to combine different multiple estimation operators:

    ```{python}
    fit = feols("Y + Y2 ~ X1 | sw(f1, f2)", data)
    etable([fit.fetch_model(0), fit.fetch_model(1), fit.fetch_model(2), fit.fetch_model(3)])
    ```

    In general, using muliple estimation syntax can improve the estimation time as covariates that are demeaned in one model and are used
    in another model do not need to be demeaned again: `feols()` implements a caching mechanism that stores the demeaned covariates.

    Besides OLS, `feols()` also supports IV estimation via three part formulas:

    ```{python}
    fit = feols("Y ~  X2 | f1 + f2 | X1 ~ Z1", data)
    fit.tidy()
    ```
    Here, `X1` is the endogenous variable and `Z1` is the instrument. `f1` and `f2` are the fixed effects, as before. To estimate
    IV models without fixed effects, simply omit the fixed effects part of the formula:

    ```{python}
    fit = feols("Y ~  X2 | X1 ~ Z1", data)
    fit.tidy()
    ```

    Last, `feols()` supports interaction of variables via the `i()` syntax. Documentation on this is tba.

    After fitting a model via `feols()`, you can use the `predict()` method to get the predicted values:

    ```{python}
    fit = feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.predict()[0:5]
    ```

    The `predict()` method also supports a `newdata` argument to predict on new data, which returns a numpy array of the predicted values:

    ```{python}
    fit = feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.predict(newdata=data)[0:5]
    ```

    Last, you can plot the results of a model via the `coefplot()` method:

    ```{python}
    fit = feols("Y ~ X1 + X2 | f1 + f2", data)
    fit.coefplot()
    ```

    """

    assert i_ref2 is None, "The function argument i_ref2 is not yet supported."

    _estimation_input_checks(fml, data, vcov, ssc, fixef_rm, collin_tol, i_ref1)

    fixest = FixestMulti(data=data)
    fixest._prepare_estimation(
        "feols", fml, vcov, ssc, fixef_rm, drop_intercept, i_ref1, i_ref2
    )

    # demean all models: based on fixed effects x split x missing value combinations
    fixest._estimate_all_models(vcov, fixest._fixef_keys, collin_tol=collin_tol)

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)


def fepois(
    fml: str,
    data: DataFrameType,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-10,
    drop_intercept: bool = False,
    i_ref1: Optional[Union[list, str]] = None,
    i_ref2: Optional[Union[list, str]] = None,
) -> Union[Fepois, FixestMulti]:
    """
    Estimate Poisson regression models with fixed effects using the `pplmhdfe` algorithm.

    This method is based on the Stata package of the same name and supports various features like stepwise regressions,
    cumulative stepwise regression, multiple dependent variables, interaction of variables, and interacted fixed effects.

    Parameters
    ----------
    fml : str
        A two-sided formula string using fixest formula syntax. Syntax: "Y ~ X1 + X2 | FE1 + FE2". "|" separates left-hand side and fixed effects.
        Special syntax includes:
        - Stepwise regressions (sw, sw0)
        - Cumulative stepwise regression (csw, csw0)
        - Multiple dependent variables (Y1 + Y2 ~ X)
        - Interaction of variables (i(X1,X2))
        - Interacted fixed effects (fe1^fe2)
        Compatible with formula parsing via the formulaic module.

    data : DataFrameType
        A pandas or polars dataframe containing the variables in the formula.

    vcov : Union[str, dict[str, str]]
        Type of variance-covariance matrix for inference. Options include "iid", "hetero", "HC1", "HC2", "HC3", or a dictionary for CRV1/CRV3 inference.

    ssc : str
        A ssc object specifying the small sample correction for inference.

    fixef_rm : str
        Specifies whether to drop singleton fixed effects. Options: "none" (default), "singleton".

    iwls_tol : Optional[float], optional
        Tolerance for IWLS convergence, by default 1e-08.

    iwls_maxiter : Optional[float], optional
        Maximum number of iterations for IWLS convergence, by default 25.

    collin_tol : float, optional
        Tolerance for collinearity check, by default 1e-06.

    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default False.

    i_ref1 : Optional[Union[list, str]], optional
        Reference category for the first set of categorical variables interacted via "i()", by default None.

    i_ref2 : Optional[Union[list, str]], optional
        Reference category for the second set of categorical variables interacted via "i()", by default None.

    Returns
    -------
    object
        An instance of the `Fepois` class or an instance of class `FixestMulti` for multiple models specified via `fml`.

    Examples
    --------
    The `fepois()` function can be used to estimate a simple Poisson regression model with fixed effects.
    The following example regresses `Y` on `X1` and `X2` with fixed effects for `f1` and `f2`: fixed effects are specified
    after the `|` symbol.

    ```{python}
    from pyfixest.estimation import fepois
    from pyfixest.utils import get_data
    from pyfixest.summarize import etable

    data = get_data(model = "Fepois")
    fit = fepois("Y ~ X1 + X2 | f1 + f2", data)
    fit.summary()
    ```
    For more examples, please take a look at the documentation of the `feols()` function.
    """

    assert i_ref2 is None, "The function argument i_ref2 is not yet supported."

    _estimation_input_checks(fml, data, vcov, ssc, fixef_rm, collin_tol, i_ref1)

    fixest = FixestMulti(data=data)

    fixest._prepare_estimation(
        "fepois", fml, vcov, ssc, fixef_rm, drop_intercept, i_ref1, i_ref2
    )
    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(
        vcov=vcov,
        fixef_keys=fixest._fixef_keys,
        iwls_tol=iwls_tol,
        iwls_maxiter=iwls_maxiter,
        collin_tol=collin_tol,
    )

    if fixest._is_multiple_estimation:
        return fixest
    else:
        return fixest.fetch_model(0, print_fml=False)


def _estimation_input_checks(fml, data, vcov, ssc, fixef_rm, collin_tol, i_ref1):
    if not isinstance(fml, str):
        raise ValueError("fml must be a string")
    if not isinstance(data, pd.DataFrame):
        try:
            import polars as pl

            if not isinstance(data, pl.DataFrame):
                raise ValueError("data must be a pandas or polars dataframe")
        except ImportError:
            raise ValueError("data must be a pandas or polars dataframe")
    if not isinstance(vcov, (str, dict, type(None))):
        raise ValueError("vcov must be a string, dictionary, or None")
    if not isinstance(fixef_rm, str):
        raise ValueError("fixef_rm must be a string")
    if not isinstance(collin_tol, float):
        raise ValueError("collin_tol must be a float")

    if not fixef_rm in ["none", "singleton"]:
        raise ValueError("fixef_rm must be either 'none' or 'singleton'")
    if not collin_tol > 0:
        raise ValueError("collin_tol must be greater than zero")
    if not collin_tol < 1:
        raise ValueError("collin_tol must be less than one")

    assert i_ref1 is None or isinstance(
        i_ref1, (list, str, int, bool, float)
    ), "i_ref1 must be either None, a list, string, int, bool, or float"
    # check that if i_ref1 is a list, all elements are of the same type
    if isinstance(i_ref1, list):
        assert len(i_ref1) > 0, "i_ref1 must not be an empty list"
        assert all(
            isinstance(x, type(i_ref1[0])) for x in i_ref1
        ), "i_ref1 must be a list of elements of the same type"
