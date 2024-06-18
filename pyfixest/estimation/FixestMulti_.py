import functools
import warnings
from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd

from pyfixest.errors import MultiEstNotSupportedError
from pyfixest.estimation.demean_ import demean_model
from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols, _check_vcov_input, _deparse_vcov_input
from pyfixest.estimation.fepois_ import Fepois, _check_for_separation
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.estimation.model_matrix_fixest_ import model_matrix_fixest
from pyfixest.utils.dev_utils import DataFrameType, _drop_cols, _polars_to_pandas


class FixestMulti:
    """A class to estimate multiple regression models with fixed effects."""

    def __init__(
        self,
        data: DataFrameType,
        copy_data: bool,
        store_data: bool,
        fixef_tol: float,
        weights_type: str,
    ) -> None:
        """
        Initialize a class for multiple fixed effect estimations.

        Parameters
        ----------
        data : panda.DataFrame
            The input DataFrame for the object.
        copy_data : bool
            Whether to copy the data or not.
        store_data : bool
            Whether to store the data in the resulting model object or not.
        fixef_tol: float
            The tolerance for the convergence of the demeaning algorithm.
        weights_type: str
            The type of weights employed in the estimation. Either analytical /
            precision weights are employed (`aweights`) or
            frequency weights (`fweights`).

        Returns
        -------
            None
        """
        self._copy_data = copy_data
        self._store_data = store_data
        self._fixef_tol = fixef_tol
        self._weights_type = weights_type

        data = _polars_to_pandas(data)

        if self._copy_data:
            self._data = data.copy()
        else:
            self._data = data
        # reindex: else, potential errors when pd.DataFrame.dropna()
        # -> drops indices, but formulaic model_matrix starts from 0:N...
        self._data.reset_index(drop=True, inplace=True)
        self.all_fitted_models: dict[str, Union[Feols, Fepois, Feiv]] = {}

        # set functions inherited from other modules
        _module = import_module("pyfixest.report")
        _tmp = getattr(_module, "coefplot")
        self.coefplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.coefplot.__doc__ = _tmp.__doc__
        _tmp = getattr(_module, "iplot")
        self.iplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.iplot.__doc__ = _tmp.__doc__
        _tmp = getattr(_module, "summary")
        self.summary = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.summary.__doc__ = _tmp.__doc__
        _tmp = getattr(_module, "etable")
        self.etable = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.etable.__doc__ = _tmp.__doc__

    def _prepare_estimation(
        self,
        estimation: str,
        fml: str,
        vcov: Union[None, str, dict[str, str]] = None,
        weights: Union[None, str] = None,
        ssc: dict[str, Union[str, bool]] = {},
        fixef_rm: str = "none",
        drop_intercept: bool = False,
    ) -> None:
        """
        Prepare model for estimation.

        Utility function to prepare estimation via the `feols()` or `fepois()` methods.
        The function is called by both methods. Mostly deparses the fml string.

        Parameters
        ----------
        estimation : str
            Type of estimation. Either "feols" or "fepois".
        fml : str
            A three-sided formula string using fixest formula syntax.
            Supported syntax includes: see `feols()` or `fepois()`.
        vcov : Union[None, str, dict[str, str]], optional
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference.
            See `feols()` or `fepois()`.
        weights : Union[None, np.ndarray], optional
            An array of weights.
            Either None or a 1D array of length N. Default is None.
        ssc : dict[str, str], optional
            A dictionary specifying the type of standard errors to use for inference.
            See `feols()` or `fepois()`.
        fixef_rm : bool, optional
            A string specifying whether singleton fixed effects should be dropped.
            Options are "none" (default) and "singleton". If "singleton",
            singleton fixed effects are dropped.
        drop_intercept : bool, optional
            Whether to drop the intercept. Default is False.

        Returns
        -------
            None
        """
        self._method = None
        self._is_iv = None
        self._fml_dict = None
        self._fml_dict_iv = None
        self._ssc_dict: dict[str, Union[str, bool]] = {}
        self._drop_singletons = False
        self._is_multiple_estimation = None
        self._drop_intercept = None
        self._weights = weights
        self._has_weights = False
        if weights is not None:
            self._has_weights = True

        self._drop_intercept = drop_intercept

        FML = FixestFormulaParser(fml)
        self.FixestFormulaDict = FML.FixestFormulaDict
        self._method = estimation
        self._is_iv = FML.is_iv
        # self._fml_dict = fxst_fml.condensed_fml_dict
        # self._fml_dict_iv = fxst_fml.condensed_fml_dict_iv
        self._ssc_dict = ssc
        self._drop_singletons = _drop_singletons(fixef_rm)

    def _estimate_all_models(
        self,
        vcov: Union[str, dict[str, str], None],
        collin_tol: float = 1e-6,
        iwls_maxiter: int = 25,
        iwls_tol: float = 1e-08,
    ) -> None:
        """
        Estimate multiple regression models.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]]
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference.
            - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            - If a dictionary, it should have the format {"CRV1": "clustervar"}
            for CRV1 inference or {"CRV3": "clustervar"} for CRV3 inference.
        collin_tol : float, optional
            The tolerance level for the multicollinearity check. Default is 1e-6.
        iwls_maxiter : int, optional
            The maximum number of iterations for the IWLS algorithm. Default is 25.
            Only relevant for non-linear estimation strategies.
        iwls_tol : float, optional
            The tolerance level for the IWLS algorithm. Default is 1e-8.
            Only relevant for non-linear estimation strategies.

        Returns
        -------
            None
        """
        _is_iv = self._is_iv
        _data = self._data
        _method = self._method
        _drop_singletons = self._drop_singletons
        _ssc_dict = self._ssc_dict
        _drop_intercept = self._drop_intercept
        _weights = self._weights
        _has_fixef = False
        _fixef_tol = self._fixef_tol
        _weights_type = self._weights_type

        FixestFormulaDict = self.FixestFormulaDict
        _fixef_keys = list(FixestFormulaDict.keys())

        for _, fval in enumerate(_fixef_keys):
            fixef_key_models = FixestFormulaDict.get(fval)

            # dictionary to cache demeaned data with index: na_index_str,
            # only relevant for `.feols()`
            lookup_demeaned_data: dict[str, pd.DataFrame] = {}

            for FixestFormula in fixef_key_models:  # type: ignore
                # loop over both dictfe and dictfe_iv (if the latter is not None)
                # get Y, X, Z, fe, NA indices for model

                mm_dict = model_matrix_fixest(
                    # fml=fml,
                    FixestFormula=FixestFormula,
                    data=_data,
                    drop_singletons=_drop_singletons,
                    drop_intercept=_drop_intercept,
                    weights=_weights,
                )

                mm_dict_keys = [
                    "Y",
                    "X",
                    "fe",
                    "endogvar",
                    "Z",
                    "weights_df",
                    "na_index",
                    "na_index_str",
                    "_icovars",
                    "X_is_empty",
                ]
                (
                    Y,
                    X,
                    fe,
                    endogvar,
                    Z,
                    weights_df,
                    na_index,
                    na_index_str,
                    _icovars,
                    X_is_empty,
                ) = (mm_dict[key] for key in mm_dict_keys)

                if _weights is not None:
                    weights = weights_df.to_numpy()
                else:
                    weights = np.ones(Y.shape[0])

                weights = weights.reshape((weights.shape[0], 1))

                self._X_is_empty = False
                if X_is_empty:
                    self._X_is_empty = True

                coefnames = X.columns.tolist()

                _k_fe = fe.nunique(axis=0) if fe is not None else None

                FIT: Union[Feols, Feiv, Fepois]

                if _method == "feols":
                    # demean Y, X, Z, if not already done in previous estimation

                    if fe is not None:
                        _has_fixef = True

                    Yd, Xd = demean_model(
                        Y,
                        X,
                        fe,
                        weights,
                        lookup_demeaned_data,
                        na_index_str,
                        _fixef_tol,
                    )

                    if _is_iv:
                        endogvard, Zd = demean_model(
                            endogvar,
                            Z,
                            fe,
                            weights,
                            lookup_demeaned_data,
                            na_index_str,
                            _fixef_tol,
                        )
                    else:
                        endogvard, Zd = None, None

                    if not _is_iv:
                        Zd = Xd

                    Yd_array, Xd_array, Zd_array, endogvard_array = (
                        x.to_numpy() if x is not None else np.array([])
                        for x in [Yd, Xd, Zd, endogvard]
                    )

                    if _is_iv:
                        coefnames_z = Z.columns.tolist()
                        FIT = Feiv(
                            Y=Yd_array,
                            X=Xd_array,
                            Z=Zd_array,
                            weights=weights,
                            coefnames_x=coefnames,
                            coefnames_z=coefnames_z,
                            collin_tol=collin_tol,
                            weights_name=_weights,
                            weights_type=_weights_type,
                        )
                    else:
                        # initiate OLS class

                        FIT = Feols(
                            Y=Yd_array,
                            X=Xd_array,
                            fe=None,  # hack, demeaning happens outside!
                            weights=weights,
                            coefnames=coefnames,
                            collin_tol=collin_tol,
                            weights_name=_weights,
                            weights_type=_weights_type,
                        )

                    FIT.na_index = na_index

                    # special case: sometimes it is useful to fit models as
                    # "Y ~ 0 | f1 + f2" to demean Y and to use the predict() method
                    if FIT._X_is_empty:
                        FIT._u_hat = Y.to_numpy() - Yd_array
                    else:
                        FIT.fit()

                elif _method == "fepois":
                    # check for separation and drop separated variables

                    na_separation: list[int] = []
                    if fe is not None:
                        na_separation = _check_for_separation(Y=Y, fe=fe)
                        if na_separation:
                            warnings.warn(
                                f"{str(len(na_separation))} observations removed because of separation."
                            )

                            Y.drop(na_separation, axis=0, inplace=True)
                            X.drop(na_separation, axis=0, inplace=True)
                            fe.drop(na_separation, axis=0, inplace=True)

                    Y_array, X_array = (x.to_numpy() for x in [Y, X])
                    N = X_array.shape[0]

                    fe_array = None
                    if fe is not None:
                        _has_fixef = True
                        fe_array = fe.to_numpy()
                        if fe_array.ndim == 1:
                            fe_array = fe_array.reshape((N, 1))

                    # initiate OLS class
                    FIT = Fepois(
                        Y=Y_array,
                        X=X_array,
                        fe=fe_array,
                        weights=weights,
                        coefnames=coefnames,
                        drop_singletons=_drop_singletons,
                        maxiter=iwls_maxiter,
                        tol=iwls_tol,
                        collin_tol=collin_tol,
                        weights_name=None,
                        fixef_tol=_fixef_tol,
                        weights_type=_weights_type,
                    )

                    FIT.fit()

                    FIT.na_index = na_index
                    if na_separation:
                        FIT.na_index = np.concatenate(
                            [FIT.na_index, np.array(na_separation)]
                        )
                        FIT.n_separation_na = len(na_separation)

                else:
                    raise ValueError(
                        "Estimation method not supported. Please use 'feols' or 'fepois'."
                    )

                    # enrich FIT with model info obtained outside of the model class

                _data_clean = _drop_cols(_data, FIT.na_index)

                FIT.add_fixest_multi_context(
                    fml=FixestFormula.fml,
                    depvar=FixestFormula._depvar,
                    Y=Y,
                    _data=_data_clean,
                    _ssc_dict=_ssc_dict,
                    _k_fe=_k_fe,
                    fval=fval,
                    store_data=self._store_data,
                )

                # if X is empty: no inference (empty X only as shorthand for demeaning)  # noqa: W505
                if not FIT._X_is_empty:
                    # inference
                    vcov_type = _get_vcov_type(vcov, fval)
                    FIT.vcov(vcov=vcov_type, data=_data_clean)
                    FIT.inference()

                    # other regression stats
                    if _method == "feols" and not FIT._is_iv:
                        FIT.performance()
                    if _icovars is not None:
                        FIT._icovars = _icovars
                    else:
                        FIT._icovars = None

                    # store fitted model
                self.all_fitted_models[FixestFormula.fml] = FIT

        self.set_fixest_multi_flag()

    def set_fixest_multi_flag(self):
        """
        Set a flag to indicate whether multiple estimations are being performed or not.

        Simple check if `all_fitted_models` has length greater than 1.
        Throws an error if multiple estimations are being performed with IV estimation.
        Args:
            None
        Returns:
            None
        """
        if len(self.all_fitted_models) > 1:
            self._is_multiple_estimation = True
            if self._is_iv:
                raise MultiEstNotSupportedError(
                    """
                    Multiple Estimations is currently not supported with IV.
                    This is mostly due to insufficient testing and will be possible
                    with a future release of PyFixest.
                    """
                )
        else:
            self._is_multiple_estimation = False

    def to_list(self):
        """
        Return a list of all fitted models.

        Parameters
        ----------
            None

        Returns
        -------
            A list of all fitted models of types Feols or Fepois.
        """
        return list(self.all_fitted_models.values())

    def vcov(self, vcov: Union[str, dict[str, str]]):
        """
        Update regression inference "on the fly".

        By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance-covariance matrix
        specified via the method.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]])
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference.
            - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            - If a dictionary, it should have the format {"CRV1": "clustervar"}
            for CRV1 inference or {"CRV3": "clustervar"} for CRV3 inference.

        Returns
        -------
            An instance of the "Fixest" class with updated inference.
        """
        for model in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[model]
            _data = fxst._data

            _check_vcov_input(vcov, _data)
            (
                fxst._vcov_type,
                fxst._vcov_type_detail,
                _,
                _,
            ) = _deparse_vcov_input(vcov, False, False)

            fxst.vcov(vcov=vcov)
            fxst.inference()

        return self

    def tidy(self) -> pd.DataFrame:
        """
        Return the results of an estimation using `feols()` as a tidy Pandas DataFrame.

        Returns
        -------
        pandas.DataFrame or str
                A tidy DataFrame with the following columns:
                - fml: the formula used to generate the results
                - Coefficient: the names of the coefficients
                - Estimate: the estimated coefficients
                - Std. Error: the standard errors of the estimated coefficients
                - t value: the t-values of the estimated coefficients
                - Pr(>|t|): the p-values of the estimated coefficients
                - 2.5%: the lower bound of the 95% confidence interval
                - 97.5%: the upper bound of the 95% confidence interval
                If `type` is set to "markdown", the resulting DataFrame will be
                returned as a markdown-formatted string with three decimal places.
        """
        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst._fml
            res.append(df)

        res_df = pd.concat(res, axis=0)
        res_df.set_index(["fml", "Coefficient"], inplace=True)

        return res_df

    def coef(self) -> pd.Series:
        """
        Obtain the coefficients of the fitted models.

        Returns
        -------
        pandas.Series
            A pd.Series with coefficient names and Estimates. The key indicates
            which models the estimated statistic derives from.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Obtain the standard errors of the fitted models.

        Returns
        -------
        pandas.Series
            A pd.Series with coefficient names and standard error estimates.
            The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Obtain the t-statistics of the fitted models.

        Returns
        -------
            A pd.Series with coefficient names and estimated t-statistics.
            The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Obtain the p-values of the fitted models.

        Returns
        -------
        pandas.Series
            A pd.Series with coefficient names and p-values.
            The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        """
        Obtain confidence intervals for the fitted models.

        Returns
        -------
        pandas.Series
            A pd.Series with coefficient names and confidence intervals.
            The key indicates which models the estimated statistic derives from.
        """
        return self.tidy()[["2.5%", "97.5%"]]

    def wildboottest(
        self,
        reps: int,
        cluster: Optional[str] = None,
        param: Optional[str] = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: Optional[int] = None,
        adj: bool = True,
        cluster_adj: bool = True,
    ) -> pd.DataFrame:
        """
        Run a wild cluster bootstrap for all regressions in the Fixest object.

        Parameters
        ----------
        B : int
            The number of bootstrap iterations to run.
        param : Union[str, None], optional
            A string of length one, containing the test parameter of interest.
            Default is None.
        cluster : Union[str, None], optional
            The name of the cluster variable. Default is None. If None, uses
            the `self._clustervar` attribute as the cluster variable. If the
            `self._clustervar` attribute is None, a heteroskedasticity-robust
            wild bootstrap is run.
        weights_type : str, optional
            The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb',
            or 'normal'. Default is 'rademacher'.
        impose_null : bool, optional
            Should the null hypothesis be imposed on the bootstrap dgp, or not?
            Default is True.
        bootstrap_type : str, optional
            A string of length one. Allows choosing the bootstrap type to be run.
            Either '11', '31', '13', or '33'. Default is '11'.
        seed : Union[str, None], optional
            Option to provide a random seed. Default is None.
        adj:bool, optional
            Whether to adjust the original coefficients with the bootstrap distribution.
            Default is True.
        cluster_adj : bool, optional
            Whether to adjust standard errors for clustering in the bootstrap.
            Default is True.

        Returns
        -------
        pandas.DataFrame
            A pd.DataFrame with bootstrapped t-statistic and p-value.
            The index indicates which model the estimated statistic derives from.
        """
        res_df = pd.DataFrame()
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]

            boot_res = fxst.wildboottest(
                reps,
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

            result_dict = {
                "fml": x,
                "param": param,
                "t value": tstat,
                "Pr(>|t|)": pvalue,
            }

            res_df = pd.concat([res_df, pd.DataFrame(result_dict)], axis=1)

        res_df = res_df.T.set_index("fml")

        return res_df

    def fetch_model(
        self, i: Union[int, str], print_fml: Optional[bool] = True
    ) -> Union[Feols, Fepois]:
        """
        Fetch a model of class Feols from the Fixest class.

        Parameters
        ----------
        i : int or str
            The index of the model to fetch.
        print_fml : bool, optional
            Whether to print the formula of the model. Default is True.

        Returns
        -------
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


def _get_vcov_type(
    vcov: Union[str, dict[str, str], None], fval: str
) -> Union[str, dict[str, str]]:
    """
    Pass the specified vcov type.

    Passes the specified vcov type. If no vcov type specified, sets the default
    vcov type as iid if no fixed effect is included in the model, and CRV1
    clustered by the first fixed effect if a fixed effect is included in the model.

    Parameters
    ----------
    vcov : Union[str, dict[str, str], None]
        The specified vcov type.
    fval : str
        The specified fixed effects. (i.e. "X1+X2")

    Returns
    -------
    str
        vcov_type (str) : The specified vcov type.
    """
    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"  # type: ignore
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}  # type: ignore
    else:
        vcov_type = vcov  # type: ignore

    return vcov_type  # type: ignore


def _drop_singletons(fixef_rm: str) -> bool:
    """
    Drop singleton fixed effects.

    Checks if the fixef_rm argument is set to "singleton". If so, returns True,
    else False.

    Parameters
    ----------
    fixef_rm : str
        The fixef_rm argument. Either "none" or "singleton".

    Returns
    -------
    bool
        drop_singletons (bool) : Whether to drop singletons.
    """
    return fixef_rm == "singleton"
