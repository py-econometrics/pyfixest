from typing import Optional, Union, Dict
from pyfixest.utils import ssc
from pyfixest.fixest import Fixest
from pyfixest.fepois import Fepois
from pyfixest.feols import Feols
import pandas as pd


def feols(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
) -> Union[Feols, Fixest]:
    """
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
              Using i() is required to use with some custom methods, e.g. iplot().
            - ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic tools such as "C", "I", ":", "*", "np.log", "np.power", etc.

        data (pd.DataFrame): A pandas dataframe containing the variables in the formula.

        vcov (Union(str, dict[str, str])): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.

        ssc (str): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.

        fixef_rm (str): A string to specify whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.

    Returns:
        An instance of the Feols class or a dictionary of Feols classes if multiple models are specified via the `fml` argument.

    Examples:
        >>> from pyfixest.estimation import feols
        >>> from pyfixest.utils import get_data
        >>> import pandas as pd
        >>> data = get_data()
        >>> data["f1"] = pd.Categorical(data["f1"])

        >>> ## basic usage
        >>> fit = feols("Y ~ X1 + X2 | f1 + f2", data=data)

        >>> ## Inference
        >>> fit2 = feols("Y ~ X1 + X2 | f1 + f2", data=data, vcov = "hetero")
        >>> fit3 = feols("Y ~ X1 + X2 | f1 + f2", data=data, vcov = {"CRV1":"group"})
        >>> feols("Y ~ X1 + X2", data=data, vcov = {"CRV3":"group"}) # currently only for models without fixed effects
        >>> feols("Y ~ X1 + X2", data=data, vcov = {"CRV3":"group", "CRV1":"group"}).wildboottest(param = "X1", B = 999) # wild bootstrap currently only for models without fixed effects


        >>> ## iv estimation
        >>> fit4 = feols("Y ~ X1 + X2 | f1 + f2 | X1 ~ Z1", data=data)
        >>> fit5 = feols("Y ~ X1 + X2 | f1 + f2 | X1 ~ Z1 + Z2", data=data, vcov = "hetero")
        >>> fit6 = feols("Y ~ 1 | f1 + f2 | X1 ~ Z1 + Z2", data=data, vcov = {"CRV1":"group"})

        >>> ## multiple estimation
        >>> fit7 = feols("Y + Y2 ~ X1 + X2 | f1 + f2", data=data)
        >>> fit8 = feols("Y ~ X1 + X2 | sw(f1, f2)", data=data, fixef_rm = "singleton")
        >>> fit9 = feols("Y ~ csw0(f1, f2) | sw(f1, f2)", data=data, ssc = ssc(adj = False))

        >>> ## `i()` syntax
        >>> fit10 = feols("Y ~ i(f1, X1) | f1 + f2")

        >>> ## interact fixed effects
        >>> fit11 = feols("Y ~ X1 + X2 | f1^f2")

        >>> ## Fetching results
        >>> fit.summary()
        >>> fit.tidy()
        >>> fit.coef()
        >>> fit.se()
        >>> fit.confint()
        >>> mod = fit7.fetch_model(0)
        >>> summary(fit)
        >>> summary([fit, fit2, mod])

        >>> ## Plotting
        >>> fit.coefplot()
        >>> fit7.iplot()

        >>> # Update inference post estimation
        >>> fit.vcov({"CRV1":"group"}}).summary()

    """

    fixest = Fixest(data=data)
    fixest._prepare_estimation("feols", fml, vcov, ssc, fixef_rm)

    # demean all models: based on fixed effects x split x missing value combinations
    fixest._estimate_all_models(vcov, fixest._fixef_keys)

    if fixest._is_fixef_multi:
        return fixest
    else:
        return fixest.fetch_model(0)


def fepois(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
) -> Union[Fepois, Fixest]:

    """
    Method for estimating Poisson regression models with fixed effects.

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
              Using i() is required to use with some custom methods, e.g. iplot().
            - ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic tools such as "C", "I", ":", "*", "np.log", "np.power", etc.

        data (pd.DataFrame): A pandas dataframe containing the variables in the formula.

        vcov (Union(str, dict[str, str])): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.

        ssc (string): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.

        fixef_rm (string): A string specifying whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.

        iwls_tol (Optional[float]): tolerance for IWLS convergence. 1e-08 by default.

        iwls_maxiter (Optional[float]): maximum number of iterations for IWLS convergence. 25 by default.

    Returns:
        An instance of a Fepois class or an object of type Fixest if more than one model is estimated.

    Examples:
        >>> from pyfixest.estimation import fepois
        >>> from pyfixest.utils import get_data
        >>> import pandas as pd
        >>> data = get_data()
        >>> data["f1"] = pd.Categorical(data["f1"])

        >>> ## basic usage
        >>> fit = fepois("Y ~ X1 + X2 | f1 + f2", data=data)
        >>> fit2 = fepois("Y ~ X1 + X2 | f1 + f2", data=data, vcov = "hetero")
        >>> fit3 = fepois("Y ~ X1 + X2 | f1 + f2", data=data, vcov = {"CRV1":"group"})

        >>> ## multiple estimation
        >>> fit4 = fepois("Y + Y2 ~ X1 + X2 | f1 + f2", data=data)
        >>> fit5 = fepois("Y ~ X1 + X2 | sw(f1, f2)", data=data, fixef_rm = "singleton")
        >>> fit6 = fepois("Y ~ csw0(f1, f2) | sw(f1, f2)", data=data, ssc = ssc(adj = False))

        >>> ## `i()` syntax
        >>> fit7 = fepois("Y ~ i(f1, X1) | f1 + f2")

        >>> ## interact fixed effects
        >>> fit8 = fepois("Y ~ X1 + X2 | f1^f2")

        >>> ## Fetching results
        >>> fit.summary()
        >>> fit.tidy()
        >>> fit.coef()
        >>> fit.se()
        >>> fit.confint()
        >>> summary(fit)
        >>> summary([fit, fit2])

        >>> ## Plotting
        >>> fit.coefplot()
        >>> fit7.iplot()

        >>> # Update inference post estimation
        >>> fit.vcov({"CRV1":"group"}}).summary()
    """

    fixest = Fixest(data=data)

    fixest._prepare_estimation(
        estimation="fepois", fml=fml, vcov=vcov, ssc=ssc, fixef_rm=fixef_rm
    )

    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(vcov = vcov, fixef_keys = fixest._fixef_keys, iwls_tol = iwls_tol, iwls_maxiter = iwls_maxiter)

    if fixest._is_fixef_multi:
        return fixest
    else:
        return fixest.fetch_model(0)

