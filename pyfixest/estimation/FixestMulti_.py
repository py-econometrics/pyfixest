import functools
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Optional, Union

import pandas as pd

from pyfixest.estimation.fegaussian_ import Fegaussian
from pyfixest.estimation.feiv_ import Feiv
from pyfixest.estimation.felogit_ import Felogit
from pyfixest.estimation.feols_ import Feols, _check_vcov_input, _deparse_vcov_input
from pyfixest.estimation.feols_compressed_ import FeolsCompressed
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.feprobit_ import Feprobit
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    QuantregMethodOptions,
    SolverOptions,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.estimation.vcov_utils import _get_vcov_type
from pyfixest.utils.dev_utils import DataFrameType, _narwhals_to_pandas
from pyfixest.utils.utils import capture_context


class FixestMulti:
    """A class to estimate multiple regression models with fixed effects."""

    def __init__(
        self,
        data: DataFrameType,
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
        context: Union[int, Mapping[str, Any]] = 0,
        quantreg_method: QuantregMethodOptions = "fn",
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
        fixef_maxiter: int
             The maximum iterations for the demeaning algorithm.
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
        separation_check: list[str], optional
            Only used in "fepois". Methods to identify and drop separated observations.
            Either "fe" or "ir". Executes both by default.
        context : int or Mapping[str, Any]
            A dictionary containing additional context variables to be used by
            formulaic during the creation of the model matrix. This can include
            custom factorization functions, transformations, or any other
            variables that need to be available in the formula environment.
        quantreg_method: QuantregMethodOptions, optional
            The method to use for the quantile regression. Currently, only "fn" is
            supported, which implements the Frisch-Newton Interior Point solver.
            See `quantreg` for more details.

        Returns
        -------
            None
        """
        self._copy_data = copy_data
        self._store_data = store_data
        self._lean = lean
        self._fixef_tol = fixef_tol
        self._fixef_maxiter = fixef_maxiter
        self._weights_type = weights_type
        self._use_compression = use_compression
        self._reps = reps
        self._seed = seed
        self._separation_check = separation_check
        self._context = capture_context(context)
        self._quantreg_method = quantreg_method

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

        data = _narwhals_to_pandas(data)

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
        _tmp = _module.coefplot
        self.coefplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.coefplot.__doc__ = _tmp.__doc__
        _tmp = _module.iplot
        self.iplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.iplot.__doc__ = _tmp.__doc__
        _tmp = _module.summary
        self.summary = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.summary.__doc__ = _tmp.__doc__
        _tmp = _module.etable
        self.etable = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.etable.__doc__ = _tmp.__doc__

    def _prepare_estimation(
        self,
        estimation: str,
        fml: str,
        vcov: Union[None, str, dict[str, str]] = None,
        vcov_kwargs: dict[str, Any]= None,
        weights: Union[None, str] = None,
        ssc: Optional[dict[str, Union[str, bool]]] = None,
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
        vcov_kwargs : dict[str, Any], optional
            Additional keyword arguments for the variance-covariance matrix.
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
        self._ssc_dict = ssc if ssc is not None else {}
        self._drop_singletons = _drop_singletons(fixef_rm)

    def _estimate_all_models(
        self,
        vcov: Union[str, dict[str, str], None],
        vcov_kwargs: dict[str, Any],
        solver: SolverOptions,
        demeaner_backend: DemeanerBackendOptions = "numba",
        collin_tol: float = 1e-6,
        iwls_maxiter: int = 25,
        iwls_tol: float = 1e-08,
        separation_check: Optional[list[str]] = None,
        quantile: Optional[float] = None,
        quantile_tol: float = 1e-06,
        quantile_maxiter: Optional[int] = None,
    ) -> None:
        """
        Estimate multiple regression models.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]]
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference.
            - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3", "NW", "DK".
            - If a dictionary, it should have the format {"CRV1": "clustervar"}
            for CRV1 inference or {"CRV3": "clustervar"} for CRV3 inference.
        vcov_kwargs : dict[str, Any]
            Additional keyword arguments for the variance-covariance matrix.
        solver: SolverOptions
            Solver to use for the estimation.
        demeaner_backend: DemeanerBackendOptions, optional
            The backend to use for demeaning. Can be either "numba" or "jax".
            Defaults to "numba".
        collin_tol : float, optional
            The tolerance level for the multicollinearity check. Default is 1e-6.
        iwls_maxiter : int, optional
            The maximum number of iterations for the IWLS algorithm. Default is 25.
            Only relevant for non-linear estimation strategies.
        iwls_tol : float, optional
            The tolerance level for the IWLS algorithm. Default is 1e-8.
            Only relevant for non-linear estimation strategies.
        separation_check: list[str], optional
            Only used in "fepois". Methods to identify and drop separated observations.
            Either "fe" or "ir". Executes both by default.
        quantile: float
            The quantile to use for quantile regression. Default is None.
            Only relevant for "quantreg" estimation.
        quantile_tol: float, optional
            The tolerance for the quantile regression FN algorithm.
            Default is 1e-06.
        quantile_maxiter: int, optional
            The maximum number of iterations for the quantile regression FN algorithm.
            If None, maxiter = the number of observations in the model
            (as in R's quantreg package via nit(3) = n).


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
        _fixef_maxiter = self._fixef_maxiter
        _weights_type = self._weights_type
        _lean = self._lean
        _store_data = self._store_data
        _copy_data = self._copy_data
        _run_split = self._run_split
        _run_full = self._run_full
        _splitvar = self._splitvar
        _context = self._context
        _quantreg_method = self._quantreg_method
        _quantiles = quantile
        _quantile_tol = quantile_tol
        _quantile_maxiter = quantile_maxiter

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

                    FIT: Union[
                        Feols,
                        Feiv,
                        Fepois,
                        Fegaussian,
                        Felogit,
                        Feprobit,
                        FeolsCompressed,
                        Quantreg,
                    ]

                    model_kwargs = {
                        "FixestFormula": FixestFormula,
                        "data": _data,
                        "ssc_dict": _ssc_dict,
                        "drop_singletons": _drop_singletons,
                        "drop_intercept": _drop_intercept,
                        "weights": _weights,
                        "weights_type": _weights_type,
                        "solver": solver,
                        "collin_tol": collin_tol,
                        "fixef_tol": _fixef_tol,
                        "fixef_maxiter": _fixef_maxiter,
                        "store_data": _store_data,
                        "copy_data": _copy_data,
                        "lean": _lean,
                        "context": _context,
                        "sample_split_value": sample_split_value,
                        "sample_split_var": _splitvar,
                        "lookup_demeaned_data": lookup_demeaned_data,
                    }

                    if _method in {"feols", "fepois"}:
                        model_kwargs.update(
                            {
                                "demeaner_backend": demeaner_backend,
                            }
                        )

                    if _method in {
                        "fepois",
                        "feglm-logit",
                        "feglm-probit",
                        "feglm-gaussian",
                    }:
                        model_kwargs.update(
                            {
                                "separation_check": separation_check,
                                "tol": iwls_tol,
                                "maxiter": iwls_maxiter,
                            }
                        )

                    if _method == "quantreg":
                        model_kwargs.update(
                            {
                                "quantile": _quantiles,
                                "method": _quantreg_method,
                                "quantile_tol": _quantile_tol,
                                "quantile_maxiter": _quantile_maxiter,
                                "seed": self._seed,
                            }
                        )

                    model_map = {
                        ("feols", False): Feols,
                        ("feols", True): Feiv,
                        ("fepois", None): Fepois,
                        ("feglm-logit", None): Felogit,
                        ("feglm-probit", None): Feprobit,
                        ("feglm-gaussian", None): Fegaussian,
                        ("compression", None): FeolsCompressed,
                        ("quantreg", None): Quantreg,
                    }

                    if _method == "compression":
                        model_kwargs.update(
                            {
                                "reps": self._reps,
                                "seed": self._seed,
                            }
                        )

                    model_key = (
                        (_method, _is_iv) if _method == "feols" else (_method, None)
                    )
                    ModelClass = model_map[model_key]  # type: ignore
                    FIT = ModelClass(**model_kwargs)

                    FIT.prepare_model_matrix()
                    if type(FIT) in [Feols, Feiv]:
                        FIT.demean()
                    FIT.to_array()
                    if isinstance(FIT, (Felogit, Feprobit, Fegaussian)):
                        FIT._check_dependent_variable()
                    FIT.drop_multicol_vars()
                    if isinstance(FIT, (Feols, Feiv, FeolsCompressed)):
                        FIT.wls_transform()

                    FIT.get_fit()
                    # if X is empty: no inference (empty X only as shorthand for demeaning)
                    if not FIT._X_is_empty:
                        # inference
                        vcov_type = _get_vcov_type(vcov, fval)
                        FIT.vcov(vcov=vcov_type, vcov_kwargs=vcov_kwargs, data=FIT._data)

                        FIT.get_inference()
                        if _method == "feols" and not FIT._is_iv:
                            FIT.get_performance()
                        if isinstance(FIT, Feiv):
                            FIT.first_stage()
                    # delete large attributes
                    FIT._clear_attributes()

                    self.all_fitted_models[FIT._model_name] = FIT

    def to_list(self) -> list[Union[Feols, Fepois, Feiv]]:
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
