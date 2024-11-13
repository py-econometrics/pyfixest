import functools
from importlib import import_module
from typing import Optional, Union

import pandas as pd

from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.feols_ import Feols, _check_vcov_input, _deparse_vcov_input
from pyfixest.estimation.feols_compressed_ import FeolsCompressed
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.utils.dev_utils import DataFrameType, _polars_to_pandas


class FixestMulti:
    """A class to estimate multiple regression models with fixed effects."""

    def __init__(
        self,
        data: DataFrameType,
        copy_data: bool,
        store_data: bool,
        lean: bool,
        fixef_tol: float,
        weights_type: str,
        use_compression: bool,
        reps: Optional[int],
        seed: Optional[int],
        split: Optional[str],
        fsplit: Optional[str],
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
        lean: bool
            Whether to store large-memory objects in the resulting model object or not.
        fixef_tol: float
            The tolerance for the convergence of the demeaning algorithm.
        weights_type: str
            The type of weights employed in the estimation. Either analytical /
            precision weights are employed (`aweights`) or
            frequency weights (`fweights`).
        use_compression: bool
            Whether to use sufficient statistics to losslessly fit the regression model
            on compressed data. False by default.
        reps : int
            The number of bootstrap iterations to run. Only relevant for wild cluster
            bootstrap for use_compression=True.
        seed : Optional[int]
            Option to provide a random seed. Default is None.
            Only relevant for wild cluster bootstrap for use_compression=True.

        Returns
        -------
            None
        """
        self._copy_data = copy_data
        self._store_data = store_data
        self._lean = lean
        self._fixef_tol = fixef_tol
        self._weights_type = weights_type
        self._use_compression = use_compression
        self._reps = reps if use_compression else None
        self._seed = seed if use_compression else None

        self._run_split = split is not None or fsplit is not None
        self._run_full = not (split and not fsplit)

        self._splitvar: Optional[str] = None
        if self._run_split:
            if split:
                self._splitvar = split
            else:
                self._splitvar = fsplit
        else:
            self._splitvar = None

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
        self._is_iv = False
        self._fml_dict = None
        self._fml_dict_iv = None
        self._ssc_dict: dict[str, Union[str, bool]] = {}
        self._drop_singletons = False
        self._is_multiple_estimation = False
        self._drop_intercept = False
        self._weights = weights
        self._has_weights = False
        if weights is not None:
            self._has_weights = True

        self._drop_intercept = drop_intercept

        FML = FixestFormulaParser(fml)
        FML.set_fixest_multi_flag()
        self._is_multiple_estimation = FML._is_multiple_estimation or self._run_split
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
        solver: str = "np.linalg.solve",
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
        solver: str, default is 'np.linalg.solve'.
            Solver to use for the estimation. Alternative is 'np.linalg.lstsq'.
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
        _fixef_tol = self._fixef_tol
        _weights_type = self._weights_type
        _lean = self._lean
        _store_data = self._store_data
        _copy_data = self._copy_data
        _run_split = self._run_split
        _run_full = self._run_full
        _splitvar = self._splitvar

        FixestFormulaDict = self.FixestFormulaDict
        _fixef_keys = list(FixestFormulaDict.keys())

        all_splits = (["all"] if _run_full else []) + (
            _data[_splitvar].dropna().unique().tolist() if _run_split else []
        )

        for sample_split_value in all_splits:
            for _, fval in enumerate(_fixef_keys):
                fixef_key_models = FixestFormulaDict.get(fval)

                # dictionary to cache demeaned data with index: na_index_str,
                # only relevant for `.feols()`
                lookup_demeaned_data: dict[str, pd.DataFrame] = {}

                for FixestFormula in fixef_key_models:  # type: ignore
                    # loop over both dictfe and dictfe_iv (if the latter is not None)
                    # get Y, X, Z, fe, NA indices for model

                    FIT: Union[Feols, Feiv, Fepois]

                    if _method == "feols" and not _is_iv:
                        FIT = Feols(
                            FixestFormula=FixestFormula,
                            data=_data,
                            ssc_dict=_ssc_dict,
                            drop_singletons=_drop_singletons,
                            drop_intercept=_drop_intercept,
                            weights=_weights,
                            weights_type=_weights_type,
                            solver=solver,
                            collin_tol=collin_tol,
                            fixef_tol=_fixef_tol,
                            lookup_demeaned_data=lookup_demeaned_data,
                            store_data=_store_data,
                            copy_data=_copy_data,
                            lean=_lean,
                            sample_split_value=sample_split_value,
                            sample_split_var=_splitvar,
                        )
                        FIT.prepare_model_matrix()
                        FIT.demean()
                        FIT.to_array()
                        FIT.drop_multicol_vars()
                        FIT.wls_transform()
                    elif _method == "feols" and _is_iv:
                        FIT = Feiv(
                            FixestFormula=FixestFormula,
                            data=_data,
                            ssc_dict=_ssc_dict,
                            drop_singletons=_drop_singletons,
                            drop_intercept=_drop_intercept,
                            weights=_weights,
                            weights_type=_weights_type,
                            solver=solver,
                            collin_tol=collin_tol,
                            fixef_tol=_fixef_tol,
                            lookup_demeaned_data=lookup_demeaned_data,
                            store_data=_store_data,
                            copy_data=_copy_data,
                            lean=_lean,
                            sample_split_value=sample_split_value,
                            sample_split_var=_splitvar,
                        )
                        FIT.prepare_model_matrix()
                        FIT.demean()
                        FIT.to_array()
                        FIT.drop_multicol_vars()
                        FIT.wls_transform()
                    elif _method == "fepois":
                        FIT = Fepois(
                            FixestFormula=FixestFormula,
                            data=_data,
                            ssc_dict=_ssc_dict,
                            drop_singletons=_drop_singletons,
                            drop_intercept=_drop_intercept,
                            weights=_weights,
                            weights_type=_weights_type,
                            solver=solver,
                            collin_tol=collin_tol,
                            fixef_tol=_fixef_tol,
                            lookup_demeaned_data=lookup_demeaned_data,
                            tol=iwls_tol,
                            maxiter=iwls_maxiter,
                            store_data=_store_data,
                            copy_data=_copy_data,
                            lean=_lean,
                            sample_split_value=sample_split_value,
                            sample_split_var=_splitvar,
                            # solver=_solver
                        )
                        FIT.prepare_model_matrix()
                        FIT.to_array()
                        FIT.drop_multicol_vars()

                    elif _method == "compression":
                        FIT = FeolsCompressed(
                            FixestFormula=FixestFormula,
                            data=_data,
                            ssc_dict=_ssc_dict,
                            drop_singletons=_drop_singletons,
                            drop_intercept=_drop_intercept,
                            weights=_weights,
                            weights_type=_weights_type,
                            solver=solver,
                            collin_tol=collin_tol,
                            fixef_tol=_fixef_tol,
                            lookup_demeaned_data=lookup_demeaned_data,
                            store_data=_store_data,
                            copy_data=_copy_data,
                            lean=_lean,
                            reps=self._reps,
                            seed=self._seed,
                            sample_split_value=sample_split_value,
                            sample_split_var=_splitvar,
                        )
                        FIT.prepare_model_matrix()
                        FIT.to_array()
                        FIT.drop_multicol_vars()
                        FIT.wls_transform()

                    FIT.get_fit()
                    # if X is empty: no inference (empty X only as shorthand for demeaning)  # noqa: W505
                    if not FIT._X_is_empty:
                        # inference
                        vcov_type = _get_vcov_type(vcov, fval)
                        FIT.vcov(vcov=vcov_type, data=FIT._data)

                        FIT.get_inference()
                        # other regression stats
                        if _method == "feols" and not FIT._is_iv:
                            FIT.get_performance()
                        if isinstance(FIT, Feiv):
                            FIT.first_stage()

                    # delete large attributescl
                    FIT._clear_attributes()
                    FIT._sample_split_value = sample_split_value

                    # store fitted model
                    if sample_split_value != "all":
                        FIT._model_name = f"{FixestFormula.fml} (sample: {FIT._sample_split_var} = {FIT._sample_split_value})"
                    else:
                        FIT._model_name = FixestFormula.fml

                    self.all_fitted_models[FIT._model_name] = FIT

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
            fxst.get_inference()

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
