import re
import warnings

import numpy as np
import pandas as pd

from typing import Union, Dict, Optional, List

from pyfixest.feols import Feols
from pyfixest.fepois import Fepois, _check_for_separation
from pyfixest.feiv import Feiv
from pyfixest.model_matrix_fixest import model_matrix_fixest
from pyfixest.demean import demean_model
from pyfixest.FormulaParser import FixestFormulaParser
from pyfixest.utils import ssc
from pyfixest.exceptions import MatrixNotFullRankError, MultiEstNotSupportedError
from pyfixest.visualize import iplot, coefplot


class FixestMulti:

    """
    # FixestMulti:

    A class to estimate multiple regression models with fixed effects.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize a class for multiple fixed effect estimations.

        Args:
            data (pd.DataFrame): The input DataFrame for the object.

        Returns:
            None
        """

        self._data = None
        self._all_fitted_models = None

        # assert that data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame")

        self._data = data.copy()
        # reindex: else, potential errors when pd.DataFrame.dropna()
        # -> drops indices, but formulaic model_matrix starts from 0:N...
        self._data.index = range(data.shape[0])
        self.all_fitted_models = dict()

    def _prepare_estimation(
        self,
        estimation: str,
        fml: str,
        vcov: Union[None, str, Dict[str, str]] = None,
        ssc: Dict[str, str] = {},
        fixef_rm: str = "none",
        drop_intercept: bool = False,
        i_ref1: Optional[Union[List, str]] = None,
        i_ref2: Optional[Union[List, str]] = None,
    ) -> None:
        """
        Utility function to prepare estimation via the `feols()` or `fepois()` methods. The function is called by both methods.
        Mostly deparses the fml string.

        Args:
            estimation (str): Type of estimation. Either "feols" or "fepois".
            fml (str): A three-sided formula string using fixest formula syntax. Supported syntax includes: see `feols()` or `fepois()`.
            vcov (Union[None, str, Dict[str, str]], optional): A string or dictionary specifying the type of variance-covariance matrix to use for inference. See `feols()` or `fepois()`.
            ssc (Dict[str, str], optional): A dictionary specifying the type of standard errors to use for inference. See `feols()` or `fepois()`.
            fixef_rm (str, optional): A string specifying whether singleton fixed effects should be dropped.
                Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.
            drop_intercept (bool, optional): Whether to drop the intercept. Default is False.
            i_ref1 (Optional[Union[List, str]], optional): A list or string specifying the reference category for the first interaction variable.
            i_ref2 (Optional[Union[List, str]], optional): A list or string specifying the reference category for the second interaction variable.

        Returns:
            None
        """

        self._method = None
        self._is_iv = None
        self._fml_dict = None
        self._fml_dict_iv = None
        self._ssc_dict = None
        self._drop_singletons = None
        self._fixef_keys = None
        self._is_multiple_estimation = None
        self._i_ref1 = None
        self._i_ref2 = None
        self._drop_intercept = None

        # set i_ref1 and i_ref2 to list if not None
        if i_ref1 is not None:
            if not isinstance(i_ref1, list):
                i_ref1 = [i_ref1]
        if i_ref2 is not None:
            if not isinstance(i_ref2, list):
                i_ref2 = [i_ref2]

        fxst_fml = FixestFormulaParser(fml)
        fxst_fml.get_fml_dict()  # fxst_fml._fml_dict might look like this: {'0': {'Y': ['Y~X1'], 'Y2': ['Y2~X1']}}. Hence {FE: {DEPVAR: [FMLS]}}
        if fxst_fml._is_iv:
            _is_iv = True
            fxst_fml.get_fml_dict(iv=True)
        else:
            _is_iv = False

        self._method = estimation
        self._is_iv = _is_iv
        self._fml_dict = fxst_fml._fml_dict
        if _is_iv:
            self._fml_dict_iv = fxst_fml._fml_dict_iv
        self._ssc_dict = ssc
        self._drop_singletons = _drop_singletons(fixef_rm)
        self._fixef_keys = list(self._fml_dict.keys())

        self._i_ref1 = i_ref1
        self._i_ref2 = i_ref2
        self._drop_intercept = drop_intercept

    def _estimate_all_models(
        self,
        vcov: Union[str, Dict[str, str], None],
        fixef_keys: Union[List[str], None],
        collin_tol: float = 1e-6,
        iwls_maxiter: int = 25,
        iwls_tol: float = 1e-08,
    ) -> None:
        """
        Estimate multiple regression models.

        Args:
            vcov (Union[str, Dict[str, str]]): A string or dictionary specifying the type of variance-covariance
                matrix to use for inference.
                - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                - If a dictionary, it should have the format {"CRV1": "clustervar"} for CRV1 inference
                  or {"CRV3": "clustervar"} for CRV3 inference.
            fixef_keys (List[str]): A list of fixed effects combinations.
            collin_tol (float, optional): The tolerance level for the multicollinearity check. Default is 1e-6.
            iwls_maxiter (int, optional): The maximum number of iterations for the IWLS algorithm. Default is 25.
                Only relevant for non-linear estimation strategies.
            iwls_tol (float, optional): The tolerance level for the IWLS algorithm. Default is 1e-8.
                Only relevant for non-linear estimation strategies.

        Returns:
            None
        """

        _fml_dict = self._fml_dict
        _is_iv = self._is_iv
        _data = self._data
        _method = self._method
        _drop_singletons = self._drop_singletons
        _ssc_dict = self._ssc_dict
        _drop_intercept = self._drop_intercept
        _i_ref1 = self._i_ref1
        _i_ref2 = self._i_ref2

        for _, fval in enumerate(fixef_keys):
            dict2fe = _fml_dict.get(fval)

            # dictionary to cache demeaned data with index: na_index_str,
            # only relevant for `.feols()`
            lookup_demeaned_data = dict()

            # loop over both dictfe and dictfe_iv (if the latter is not None)
            for depvar in dict2fe.keys():
                for _, fml_linear in enumerate(dict2fe.get(depvar)):
                    covar = fml_linear.split("~")[1]
                    endogvars, instruments = None, None
                    if _is_iv:
                        endogvars, instruments = _get_endogvars_instruments(
                            fml_dict_iv=self._fml_dict_iv,
                            fval=fval,
                            depvar=depvar,
                            covar=covar,
                        )
                    # stitch formula back together
                    fml = get_fml(depvar, covar, fval, endogvars, instruments)

                    # get Y, X, Z, fe, NA indices for model
                    (
                        Y,
                        X,
                        fe,
                        endogvar,
                        Z,
                        na_index,
                        na_index_str,
                        _icovars,
                        X_is_empty,
                    ) = model_matrix_fixest(
                        fml=fml,
                        data=_data,
                        drop_singletons=_drop_singletons,
                        drop_intercept=_drop_intercept,
                        i_ref1=_i_ref1,
                        i_ref2=_i_ref2,
                    )

                    weights = np.ones((Y.shape[0], 1))

                    self._X_is_empty = False
                    if X_is_empty:
                        self._X_is_empty = True

                    coefnames = X.columns.tolist()

                    if fe is not None:
                        _k_fe = fe.nunique(axis=0)
                    else:
                        _k_fe = None

                    if _method == "feols":
                        # demean Y, X, Z, if not already done in previous estimation

                        Yd, Xd = demean_model(
                            Y, X, fe, weights, lookup_demeaned_data, na_index_str
                        )

                        if _is_iv:
                            endogvard, Zd = demean_model(
                                endogvar,
                                Z,
                                fe,
                                weights,
                                lookup_demeaned_data,
                                na_index_str,
                            )
                        else:
                            endogvard, Zd = None, None

                        if not _is_iv:
                            Zd = Xd

                        Yd, Xd, Zd, endogvard = [
                            x.to_numpy() if x is not None else x
                            for x in [Yd, Xd, Zd, endogvard]
                        ]

                        has_weights = False
                        if has_weights:
                            w = np.sqrt(weights.to_numpy())
                            Yd *= np.sqrt(w)
                            Zd *= np.sqrt(w)
                            Xd *= np.sqrt(w)

                        if _is_iv:
                            FIT = Feiv(
                                Y=Yd,
                                X=Xd,
                                Z=Zd,
                                weights=weights,
                                coefnames=coefnames,
                                collin_tol=collin_tol,
                            )
                        else:
                            # initiate OLS class
                            FIT = Feols(
                                Y=Yd,
                                X=Xd,
                                weights=weights,
                                coefnames=coefnames,
                                collin_tol=collin_tol,
                            )

                        # special case: sometimes it is useful to fit models as "Y ~ 0 | f1 + f2" to demean Y and to use the predict() method
                        if FIT._X_is_empty:
                            FIT._u_hat = Y.to_numpy() - Yd
                        else:
                            FIT.get_fit()

                    elif _method == "fepois":
                        # check for separation and drop separated variables

                        na_separation = []
                        if fe is not None:
                            na_separation = _check_for_separation(
                                Y=Y, fe=fe, check="fe"
                            )
                            if na_separation:
                                warnings.warn(
                                    f"{str(len(na_separation))} observations removed because of separation."
                                )

                                Y.drop(na_separation, axis=0, inplace=True)
                                X.drop(na_separation, axis=0, inplace=True)
                                fe.drop(na_separation, axis=0, inplace=True)

                        Y, X = [x.to_numpy() for x in [Y, X]]
                        N = X.shape[0]

                        if fe is not None:
                            fe = fe.to_numpy()
                            if fe.ndim == 1:
                                fe = fe.reshape((N, 1))

                        # initiate OLS class
                        FIT = Fepois(
                            Y=Y,
                            X=X,
                            fe=fe,
                            weights=weights,
                            coefnames=coefnames,
                            drop_singletons=_drop_singletons,
                            maxiter=iwls_maxiter,
                            tol=iwls_tol,
                            collin_tol=collin_tol,
                        )

                        FIT.get_fit()

                        FIT.na_index = na_index
                        FIT.n_separation_na = None
                        if na_separation:
                            FIT.na_index += na_separation
                            FIT.n_separation_na = len(na_separation)

                    else:
                        raise ValueError(
                            "Estimation method not supported. Please use 'feols' or 'fepois'."
                        )

                    # some bookkeeping
                    FIT._fml = fml
                    FIT._depvar = depvar
                    FIT._Y_untransformed = Y
                    FIT._data = _data.iloc[~_data.index.isin(na_index)]
                    FIT._ssc_dict = _ssc_dict
                    FIT._k_fe = _k_fe
                    if fval != "0":
                        FIT._has_fixef = True
                        FIT._fixef = fval
                    else:
                        FIT._has_fixef = False
                        FIT._fixef = None

                    # if X is empty: no inference (empty X only as shorthand for demeaning)
                    if not FIT._X_is_empty:
                        # inference
                        vcov_type = _get_vcov_type(vcov, fval)
                        FIT.vcov(vcov=vcov_type)
                        FIT.get_inference()

                        # other regression stats
                        if _method == "feols":
                            if not FIT._is_iv:
                                FIT.get_performance()

                        if _icovars is not None:
                            FIT._icovars = _icovars
                        else:
                            FIT._icovars = None

                    # store fitted model
                    self.all_fitted_models[fml] = FIT

        if len(self.all_fitted_models) > 1:
            self._is_multiple_estimation = True
            if self._is_iv:
                raise MultiEstNotSupportedError(
                    f"""
                        Multiple Estimations is currently not supported with IV.
                        This is mostly due to insufficient testing and will be possible with a future release of PyFixest.
                        """
                )

    def vcov(self, vcov: Union[str, Dict[str, str]]):
        """
        Update regression inference "on the fly".

        By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance-covariance matrix specified via the method.

        Args:
            vcov (Union[str, Dict[str, str]]): A string or dictionary specifying the type of variance-covariance
                matrix to use for inference.
                - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                - If a dictionary, it should have the format {"CRV1": "clustervar"} for CRV1 inference
                  or {"CRV3": "clustervar"} for CRV3 inference.

        Returns:
            An instance of the "Fixest" class with updated inference.f
        """

        for model in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[model]
            fxst._vcov_type = vcov

            fxst.vcov(vcov=vcov)
            fxst.get_inference()

        return self

    def tidy(self) -> pd.DataFrame:
        """
        Returns the results of an estimation using `feols()` as a tidy Pandas DataFrame.
        Returns:
            pd.DataFrame or str
                A tidy DataFrame with the following columns:
                - fml: the formula used to generate the results
                - Coefficient: the names of the coefficients
                - Estimate: the estimated coefficients
                - Std. Error: the standard errors of the estimated coefficients
                - t value: the t-values of the estimated coefficients
                - Pr(>|t|): the p-values of the estimated coefficients
                - 2.5 %: the lower bound of the 95% confidence interval
                - 97.5 %: the upper bound of the 95% confidence interval
                If `type` is set to "markdown", the resulting DataFrame will be returned as a
                markdown-formatted string with three decimal places.
        """

        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst._fml
            res.append(df)

        res = pd.concat(res, axis=0).set_index(["fml", "Coefficient"])

        return res

    def summary(self, digits: int = 3) -> None:
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            fxst.summary(digits=digits)

    def etable(self, digits: int = 3) -> pd.DataFrame:
        return self.tidy().T.round(digits)

    def coef(self) -> pd.Series:
        """
        Obtain the coefficients of the fitted models.
        Returns:
            A pd.Series with coefficient names and Estimates. The key indicates which models the estimated statistic derives from.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Obtain the standard errors of the fitted models.

        Returns:
            A pd.Series with coefficient names and standard error estimates. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Obtain the t-statistics of the fitted models.

         Returns:
            A pd.Series with coefficient names and estimated t-statistics. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Obtain the p-values of the fitted models.

        Returns:
            A pd.Series with coefficient names and p-values. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.Series:
        """'
        Obtain confidence intervals for the fitted models.

        Returns:
            A pd.Series with coefficient names and confidence intervals. The key indicates which models the estimated statistic derives from.
        """

        return self.tidy()[["2.5 %", "97.5 %"]]

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple = (500, 300),
        yintercept: Union[int, str, None] = None,
        xintercept: Union[int, str, None] = None,
        rotate_xticks: int = 0,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Plot model coefficients with confidence intervals for variable interactions specified via the `i()` syntax.

        Args:
            alpha (float, optional): The significance level for the confidence intervals. Default is 0.05.
            figsize (tuple, optional): The size of the figure. Default is (10, 10).
            yintercept (Union[int, str, None], optional): The value at which to draw a horizontal line.
            xintercept (Union[int, str, None], optional): The value at which to draw a vertical line.
            rotate_xticks (int, optional): The rotation angle for x-axis tick labels. Default is 0.
            title (str, optional): The title of the plot. Default is None.
            coord_flip (bool, optional): Whether to flip the coordinates of the plot. Default is True.

        Returns:
            A lets-plot figure of coefficients (and respective CIs) interacted via the `i()` syntax.
        """

        models = self.all_fitted_models
        # get a list, not a dict, as iplot only works with lists
        models = [models[x] for x in list(self.all_fitted_models.keys())]

        plot = iplot(
            models=models,
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

        return plot

    def coefplot(
        self,
        alpha: float = 0.05,
        figsize: tuple = (500, 300),
        yintercept: int = 0,
        rotate_xticks: int = 0,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Plot estimation results. The plot() method is only defined for single regressions.
        Args:
            alpha (float): the significance level for the confidence intervals. Default is 0.05.
            figsize (tuple): the size of the figure. Default is (5, 2).
            yintercept (float): the value of the y-intercept. Default is 0.
            figtitle (str, optional): The title of the figure. Default is None.
            figtext (str, optional): The text at the bottom of the figure. Default is None.
            title (str, optional): The title of the plot. Default is None.
            coord_flip (bool, optional): Whether to flip the coordinates of the plot. Default is True.
        Returns:
            A lets-plot figure of regression coefficients.
        """

        # get a list, not a dict, as iplot only works with lists
        models = self.all_fitted_models
        models = [models[x] for x in list(self.all_fitted_models.keys())]

        plot = coefplot(
            models=models,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=None,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

        return plot

    def wildboottest(
        self,
        B: int,
        cluster: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        param: Optional[str] = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: Optional[str] = None,
        adj: bool = True,
        cluster_adj: bool = True,
    ) -> pd.DataFrame:
        """
        Run a wild cluster bootstrap for all regressions in the Fixest object.

        Args:
            B (int): The number of bootstrap iterations to run.
            param (Union[str, None], optional): A string of length one, containing the test parameter of interest. Default is None.
            cluster: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
            weights_type (str, optional): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb', or 'normal'.
                Default is 'rademacher'.
            impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                Default is True.
            bootstrap_type (str, optional): A string of length one. Allows choosing the bootstrap type
                to be run. Either '11', '31', '13', or '33'. Default is '11'.
            seed (Union[str, None], optional): Option to provide a random seed. Default is None.
            adj (bool, optional): Whether to adjust the original coefficients with the bootstrap distribution.
                Default is True.
            cluster_adj (bool, optional): Whether to adjust standard errors for clustering in the bootstrap.
                Default is True.

        Returns:
            A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated
            statistic derives from.
        """

        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]

            boot_res = fxst.wildboottest(
                B,
                cluster,
                param,
                weights_type,
                impose_null,
                bootstrap_type,
                seed,
                adj,
                cluster_adj,
            )

            pvalue = boot_res["Pr(>|t|)"]
            tstat = boot_res["t value"]

            res.append(
                pd.Series(
                    {"fml": x, "param": param, "t value": tstat, "Pr(>|t|)": pvalue}
                )
            )

        res = pd.concat(res, axis=1).T.set_index("fml")

        return res

    def fetch_model(
        self, i: Union[int, str], print_fml: Optional[bool] = True
    ) -> Union[Feols, Fepois]:
        """
        Utility method to fetch a model of class Feols from the Fixest class.
        Args:
            i (int or str): The index of the model to fetch.
            print_fml (bool, optional): Whether to print the formula of the model. Default is True.
        Returns:
            A Feols object.
        """

        if isinstance(i, str):
            i = int(i)

        keys = list(self.all_fitted_models.keys())
        if i >= len(keys):
            raise IndexError(f"Index {i} is larger than the number of fitted models.")
        key = keys[i]
        if print_fml:
            print("Model: ", key)
        model = self.all_fitted_models[key]
        return model


def get_fml(
    depvar: str, covar: str, fval: str, endogvars: str = None, instruments: str = None
) -> str:
    """
    Stitches together the formula string for the regression.

    Args:
        depvar (str): The dependent variable.
        covar (str): The covariates. E.g. "X1+X2+X3"
        fval (str): The fixed effects. E.g. "X1+X2". "0" if no fixed effects.
        endogvars (str, optional): The endogenous variables.
        instruments (str, optional): The instruments. E.g. "Z1+Z2+Z3"

    Returns:
        str: The formula string for the regression.
    """

    fml = f"{depvar} ~ {covar}"

    if endogvars is not None:
        fml_iv = f"| {endogvars} ~ {instruments}"
    else:
        fml_iv = None

    if fval != "0":
        fml_fval = f"| {fval}"
    else:
        fml_fval = None

    if fml_fval is not None:
        fml += fml_fval

    if fml_iv is not None:
        fml += fml_iv

    fml = fml.replace(" ", "")

    return fml


def _get_vcov_type(vcov, fval):
    """
    Passes the specified vcov type. If no vcov type specified, sets the default vcov type as iid if no fixed effect
    is included in the model, and CRV1 clustered by the first fixed effect if a fixed effect is included in the model.
    Args:
        vcov (str): The specified vcov type.
        fval (str): The specified fixed effects. (i.e. "X1+X2")
    Returns:
        vcov_type (str): The specified vcov type.
    """

    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}
    else:
        vcov_type = vcov

    return vcov_type


def _drop_singletons(fixef_rm: bool) -> bool:
    """
    Checks if the fixef_rm argument is set to "singleton". If so, returns True, else False.
    Args:
        fixef_rm (str): The fixef_rm argument.
    Returns:
        drop_singletons (bool): Whether to drop singletons.
    """

    if fixef_rm == "singleton":
        return True
    else:
        return False


def _get_endogvars_instruments(
    fml_dict_iv: dict, fval: str, depvar: str, covar: str
) -> tuple:
    """
    Fetch the endogenous variables and instruments from the fml_dict_iv dictionary.

    Args:
        fml_dict_iv (dict): The dictionary of formulas for the IV estimation.
        fval (str): The fixed effects. E.g. "X1+X2". "0" if no fixed effects.
        depvar (str): The dependent variable.
        covar (str): The covariates. E.g. "X1+X2+X3"
    Returns:
        endogvars (str): The endogenous variables.
        instruments (str): The instruments. E.g. "Z1+Z2+Z3"
    """

    dict2fe_iv = fml_dict_iv.get(fval)
    instruments2 = dict2fe_iv.get(depvar)[0].split("~")[1]
    endogvar_list = list(set(covar.split("+")) - set(instruments2.split("+")))
    instrument_list = list(set(instruments2.split("+")) - set(covar.split("+")))
    endogvars = endogvar_list[0]
    instruments = "+".join(instrument_list)

    return endogvars, instruments
