from typing import Optional, Union, Dict
from pyfixest.utils import ssc
from pyfixest.exceptions import (
    MultiEstNotSupportedError,
)
from pyfixest.fixest import Fixest
import pandas as pd


def feols(
    data: pd.DataFrame,
    fml: str,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
) -> None:
    """
    Method for fixed effects regression modeling using the PyHDFE package for projecting out fixed effects.
    Args:
        fml (str): A three-sided formula string using fixest formula syntax. Supported syntax includes:
            The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1" where:

            Y: Dependent variable
            X1, X2: Independent variables
            FE1, FE2: Fixed effects
            Z1, Z2: Instruments
            |: Separates left-hand side, fixed effects, and instruments

            If no fixed effects and instruments are specified, the formula can be simplified to "Y ~ X1 + X2".
            If no instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | FE1 + FE2".
            If no fixed effects are specified but instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | X1 ~ Z1".

            Supported multiple estimation syntax includes:

            Stepwise regressions (sw, sw0)
            Cumulative stepwise regression (csw, csw0)
            Multiple dependent variables (Y1 + Y2 ~ X)

            Other special syntax includes:
            i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
            Using i() is required to use with some custom methods, e.g. iplot().
            ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic functionaloty such as "C", "I", ":",, "*", "np.log", "np.power", etc.

        vcov (Union(str, dict)): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
        ssc (ssc): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.
        fixef_rm: A string specifiny whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.
    Returns:
        An instance of the Feols class.
    Examples:
        Standard formula:
            fml = 'Y ~ X1 + X2'
            fit = feols(fml, data, vcov='iid')
        With fixed effects:
            fml = 'Y ~ X1 + X2 | fe1 + fe2'
        With interacted fixed effects:
            fml = 'Y ~ X1 + X2 | fe1^fe2'
        Multiple dependent variables:
            fml = 'Y1 + Y2 ~ X1 + X2'
        Stepwise regressions (sw and sw0):
            fml = 'Y1 + Y2 ~ sw(X1, X2, X3)'
        Cumulative stepwise regressions (csw and csw0):
            fml = 'Y1 + Y2 ~ csw(X1, X2, X3) '
        Combinations:
            fml = 'Y1 + Y2 ~ csw(X1, X2, X3) | sw(X4, X5) + X6'
        With instruments:
            fml = 'Y ~ X1 + X2 | X1 ~ Z1'
        With instruments and fixed effects:
            fml = 'Y ~ X1 + X2 | X1 ~ Z1  | fe1 + fe2'

    Attributes:
        - attributes set via _prepare_estimation():
            _fml: the provided formula string.
            _method: the estimation method. Either "feols" or "fepois".
            _is_iv: boolean indicating whether the model is an IV model.
            _fml_dict: a dictionary of deparsed formulas.
            _fml_dict_iv: a dictionary of deparsed formulas for IV models. None if no IV models. Basically, the same as
                            `_fml_dict` but with instruments.
            _ivars: a list of interaction variables. None if no interaction variables.
            _ssc_dict: a dictionary with information on small sample corrections.
            _drop_singletons: boolean indicating whether singleton fixed effects are dropped in the estimation.
            _fixef_keys: a list of fixed effects combinations.
            _drop_ref: a list of dropped reference categories for `i()` interactions.
            _split: the split variable if split estimation is used, else None.
            _splitvar: the split variable if split estimation is used, else None.
            _estimate_split_model: boolean indicating whether the split model is estimated.
            _estimate_full_model: boolean indicating whether the full model is estimated.
        - attributes set via _model_matrix_fixest():
            icovars: a list of interaction variables. None if no interaction variables via `i()` provided.
        - attributes set via _estimate_all_models():

        - attributes set via _is_multiple_estimation():
            is_fixef_multi: boolean indicating whether multiple regression models will be estimated

    """

    fixest = Fixest(data=data)

    fixest._prepare_estimation("feols", fml, vcov, ssc, fixef_rm)

    # demean all models: based on fixed effects x split x missing value combinations
    fixest._estimate_all_models(vcov, fixest._fixef_keys)

    # create self._is_fixef_multi flag
    fixest._is_multiple_estimation()

    if fixest._is_fixef_multi and fixest._is_iv:
        raise MultiEstNotSupportedError(
            "Multiple Estimations is currently not supported with IV."
            "This is mostly due to insufficient testing and will be possible with the next release of PyFixest."
        )

    return fixest


def fepois(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    ssc=ssc(),
    fixef_rm: str = "none",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
) -> None:
    """
    Method for fixed effects regression modeling using the PyHDFE package for projecting out fixed effects.
    Args:
        fml (str): A three-sided formula string using fixest formula syntax. Supported syntax includes:
            The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1" where:

            Y: Dependent variable
            X1, X2: Independent variables
            FE1, FE2: Fixed effects
            Z1, Z2: Instruments
            |: Separates left-hand side, fixed effects, and instruments

            If no fixed effects and instruments are specified, the formula can be simplified to "Y ~ X1 + X2".
            If no instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | FE1 + FE2".
            If no fixed effects are specified but instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | X1 ~ Z1".

            Supported multiple estimation syntax includes:

            Stepwise regressions (sw, sw0)
            Cumulative stepwise regression (csw, csw0)
            Multiple dependent variables (Y1 + Y2 ~ X)

            Other special syntax includes:
            i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
            Using i() is required to use with some custom methods, e.g. iplot().
            ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

            All other parts of the formula must be compatible with formula parsing via the formulaic module.
            You can use formulaic functionaloty such as "C", "I", ":",, "*", "np.log", "np.power", etc.

        vcov (Union(str, dict)): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
        ssc (ssc): A ssc object specifying the small sample correction to use for inference. See the documentation for sscc() for more information.
        fixef_rm: A string specifiny whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.
        iwls_tol: tolerance for IWLS convergence. 1e-08 by default.
        iwls_maxiter: maximum number of iterations for IWLS convergence. 25 by default.

    Returns:
        An instance of class Fixest.
    Examples:
        Standard formula:
            fml = 'Y ~ X1 + X2'
            fit = fepois(fml, data, vcov='iid')
        With fixed effects:
            fml = 'Y ~ X1 + X2 | fe1 + fe2'
        With interacted fixed effects:
            fml = 'Y ~ X1 + X2 | fe1^fe2'
        Multiple dependent variables:
            fml = 'Y1 + Y2 ~ X1 + X2'
        Stepwise regressions (sw and sw0):
            fml = 'Y1 + Y2 ~ sw(X1, X2, X3)'
        Cumulative stepwise regressions (csw and csw0):
            fml = 'Y1 + Y2 ~ csw(X1, X2, X3) '
        Combinations:
            fml = 'Y1 + Y2 ~ csw(X1, X2, X3) | sw(X4, X5) + X6'
        With instruments:
            fml = 'Y ~ X1 + X2 | X1 ~ Z1'
        With instruments and fixed effects:
            fml = 'Y ~ X1 + X2 | X1 ~ Z1  | fe1 + fe2'

    Returns:
        An instance of a Fixest class.



    """

    fixest = Fixest(data=data)
    fixest._iwls_maxiter = iwls_maxiter
    fixest._iwls_tol = iwls_tol

    fixest._prepare_estimation(
        estimation="fepois", fml=fml, vcov=vcov, ssc=ssc, fixef_rm=fixef_rm
    )

    if fixest._is_iv:
        raise NotImplementedError(
            "IV Estimation is not supported for Poisson Regression"
        )

    fixest._estimate_all_models(vcov, fixest._fixef_keys)

    # create self._is_fixef_multi flag
    fixest._is_multiple_estimation()

    return fixest
