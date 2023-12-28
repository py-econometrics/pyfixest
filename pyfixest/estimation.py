from typing import Optional, Union, Dict
from pyfixest.utils import ssc
from pyfixest.FixestMulti import FixestMulti
from pyfixest.fepois import Fepois
from pyfixest.feols import Feols
import pandas as pd
import polars as pl


def feols(
    fml: str,
    data: Union[pd.DataFrame, pl.DataFrame],
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
    collin_tol: float = 1e-10,
    drop_intercept: bool = False,
    i_ref1: Optional[Union[list, str]] = None,
    i_ref2: Optional[Union[list, str]] = None,
) -> Union[Feols, FixestMulti]:
    """

    # feols

    Method for estimating linear regression models with fixed effects.

    Args:
        fml (str): A three-sided formula string using fixest formula syntax.

            The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1" where:

            - Y: Dependent variable
            - X1, X2: Independent variables
            - FE1, FE2: Fixed effects
            - Z1, Z2: Instruments

            In short, "|" separates left-hand side, fixed effects, and instruments

            - If no fixed effects and instruments are specified, the formula can be simplified to "Y ~ X1 + X2".
            - If no instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | FE1 + FE2".
            - If no fixed effects are specified but instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | X1 ~ Z1".

            Supported multiple estimation syntax includes:

            - Stepwise regressions (sw, sw0)
            - Cumulative stepwise regression (csw, csw0)
            - Multiple dependent variables (Y1 + Y2 ~ X)

            Other special syntax includes:

            - i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
              Using i() is required to use with some custom methods, e.g. iplot(). In contrast to r-fixest, reference levels cannot be
              set in the formula, but must be specified via the i_ref1 and i_ref2 arguments. The first variable - in the example, X1 -
              will always be treated as a categorical.
            - ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic tools such as "C", "I", ":", "*", "np.log", "np.power", etc.

        data (pd.DataFrame): A pandas dataframe containing the variables in the formula.

        vcov (Union(str, dict[str, str])): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
            For twoway clustering, combine the cluster variables via a "+", i.e. dict(".CRV1":"clustervar1+clustervar2").

        ssc (str): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.

        fixef_rm (str): A string to specify whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.

        collin_tol (float): tolerance for collinearity check. 1e-06 by default. If collinear variables are detected, they will be dropped from the model. The performed check is
                            via the diagonal cholesky decomposition of the correlation matrix of the variables.
                            If the tolerance is higher, more variables will be dropped.

        drop_intercept (bool): Whether to drop the intercept from the model. False by default. If True, the intercept will be dropped **after** creating the model matrix via formulaic.
                               This implies that reference levels for categorical variables will be dropped as well and are not recovered.

        i_ref1 (Optional[Union[list, str]]): A list of strings or a string specifying the reference category for the first set of categorical variables in the formula, interacted via "i()".
        i_ref2 (Optional[Union[list, str]]): A list of strings or a string specifying the reference category for the second set of categorical variables in the formula, interacted via "i()".

    Returns:
        An instance of the `Feols` class or an instance of class `FixestMulti` if multiple models are specified via the `fml` argument.

    Examples:
        >>> from pyfixest.estimation import feols
        >>> from pyfixest.utils import get_data, ssc
        >>> from pyfixest.summarize import summary
        >>> import pandas as pd
        >>> data = get_data()
        >>> data["f1"] = pd.Categorical(data["f1"])

        >>> ## basic usage
        >>> fit = feols("Y ~ X1 + X2 | f1 + f2", data=data)

        >>> ## Inference
        >>> fit2 = feols("Y ~ X1 + X2 | f1 + f2", data=data, vcov = "hetero")
        >>> fit3 = feols("Y ~ X1 + X2 | f1 + f2", data=data, vcov = {"CRV1":"group_id"})
        >>> fit4 = feols("Y ~ X1 + X2", data=data, vcov = {"CRV3":"group_id"}) # currently only for models without fixed effects
        >>> fit5 = feols("Y ~ X1 + X2", data=data, vcov = {"CRV3":"group_id"}).wildboottest(param = "X1", B = 999) # wild bootstrap currently only for models without fixed effects


        >>> ## iv estimation
        >>> fit6 = feols("Y ~  X2 | f1 + f2 | X1 ~ Z1", data=data)
        >>> fit7 = feols("Y ~ X2 | f1 + f2 | X1 ~ Z1 + Z2", data=data, vcov = "hetero")
        >>> fit8 = feols("Y ~ 1 | f1 + f2 | X1 ~ Z1 + Z2", data=data, vcov = {"CRV1":"group_id"})

        >>> ## multiple estimation
        >>> fit9 = feols("Y + Y2 ~ X1 + X2 | f1 + f2", data=data)
        >>> fit10 = feols("Y ~ X1 + X2 | sw(f1, f2)", data=data, fixef_rm = "singleton")
        >>> fit11 = feols("Y ~ sw(X1, X2) | csw(f1, f2)", data=data, ssc = ssc(adj = False))

        >>> ## `i()` syntax
        >>> fit12 = feols("Y ~ i(f1, X1) | f1 + f2", data = data)

        >>> ## interact fixed effects
        >>> fit13 = feols("Y ~ X1 + X2 | f1^f2", data = data)

        >>> ## Fetching results
        >>> fit.summary()
        >>> fit.tidy()
        >>> fit.coef()
        >>> fit.se()
        >>> fit.confint()
        >>> mod = fit9.fetch_model(0)
        >>> summary(fit)
        >>> summary([fit, fit2, mod])

        >>> ## Plotting
        >>> fit.coefplot(yintercept=0, figsize = (3,3))
        >>> fit12.iplot(yintercept=0, figsize = (14,4))

        >>> # Update inference post estimation
        >>> fit.vcov({"CRV1":"group_id"}).summary()


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
    data: Union[pd.DataFrame, pl.DataFrame],
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
    # fepois

    Method for estimating Poisson regression models with fixed effects. Implements the `pplmhdfe` algorithm from the
    Stata package of the same name.

    Args:
        fml (str): A two-sided formula string using fixest formula syntax.

            The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2" where:

            - Y: Dependent variable
            - X1, X2: Independent variables
            - FE1, FE2: Fixed effects

            In short, "|" separates left-hand side and fixed effects. If no fixed effects are specified,
            the formula can be simplified to "Y ~ X1 + X2".

            Supported multiple estimation syntax includes:

            - Stepwise regressions (sw, sw0)
            - Cumulative stepwise regression (csw, csw0)
            - Multiple dependent variables (Y1 + Y2 ~ X)

            Other special syntax includes:

            - i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
              Using i() is required to use with some custom methods, e.g. iplot(). In contrast to r-fixest, reference levels cannot be
              set in the formula, but must be specified via the i_ref1 and i_ref2 arguments. The first variable - in the example, X1 -
              will always be treated as a categorical.
            - ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic tools such as "C", "I", ":", "*", "np.log", "np.power", etc.

        data (pd.DataFrame): A pandas dataframe containing the variables in the formula.

        vcov (Union(str, dict[str, str])): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
            For twoway clustering, combine the cluster variables via a "+", i.e. dict(".CRV1":"clustervar1+clustervar2").

        ssc (string): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.

        fixef_rm (string): A string specifying whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.

        iwls_tol (Optional[float]): tolerance for IWLS convergence. 1e-08 by default.

        iwls_maxiter (Optional[float]): maximum number of iterations for IWLS convergence. 25 by default.

        collin_tol (float): tolerance for collinearity check. 1e-06 by default. If collinear variables are detected, they will be dropped from the model. The performed check is
                            via the diagonal cholesky decomposition of the correlation matrix of the variables. If the tolerance is higher, more variables will be dropped.

        drop_intercept (bool): Whether to drop the intercept from the model. False by default. If True, the intercept will be dropped **after** creating the model matrix via formulaic.
                               This implies that reference levels for categorical variables will be dropped as well and are not recovered.

        i_ref1 (Optional[Union[list, str]]): A list of strings or a string specifying the reference category for the first set of categorical variables in the formula, interacted via "i()".
        i_ref2 (Optional[Union[list, str]]): A list of strings or a string specifying the reference category for the second set of categorical variables in the formula, interacted via "i()".

    Returns:
        An instance of the `Fepois` class or an instance of class `FixestMulti` if multiple models are specified via the `fml` argument.

    Examples:
        >>> from pyfixest.estimation import fepois
        >>> from pyfixest.utils import get_data, ssc
        >>> from pyfixest.summarize import summary
        >>> import pandas as pd
        >>> data = get_data(model = "Fepois")
        >>> data["f1"] = pd.Categorical(data["f1"])

        >>> ## basic usage
        >>> fit = fepois("Y ~ X1 + X2 | f1 + f2", data=data)
        >>> fit2 = fepois("Y ~ X1 + X2 | f1 + f2", data=data, vcov = "hetero")
        >>> fit3 = fepois("Y ~ X1 + X2 | f1 + f2", data=data, vcov = {"CRV1":"group_id"})

        >>> ## multiple estimation
        >>> fit4 = fepois("Y + Y2 ~ X1 + X2 | f1 + f2", data=data)
        >>> fit5 = fepois("Y ~ X1 + X2 | sw(f1, f2)", data=data, fixef_rm = "singleton")
        >>> fit6 = fepois("Y ~ X1 | sw(f1, f2)", data=data, ssc = ssc(adj = False))

        >>> ## `i()` syntax
        >>> fit7 = fepois("Y ~ i(f1, X1) | f1 + f2", data = data)

        >>> ## interact fixed effects
        >>> fit8 = fepois("Y ~ X1 + X2 | f1^f2", data = data)

        >>> ## Fetching results
        >>> fit.summary()
        >>> fit.tidy()
        >>> fit.coef()
        >>> fit.se()
        >>> fit.confint()
        >>> summary(fit)
        >>> summary([fit, fit2])

        >>> ## Plotting
        >>> fit.coefplot(yintercept=0, figsize=(3, 3))
        >>> fit7.iplot(yintercept=0, figsize=(14, 4))

        >>> # Update inference post estimation
        >>> fit.vcov({"CRV1":"group_id"}).summary()

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
    # check if data is a pd.DataFrame or pl.DataFrame
    if not isinstance(data, (pd.DataFrame, pl.DataFrame)):
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
