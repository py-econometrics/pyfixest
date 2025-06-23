import functools
import gc
import re
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.sparse import csc_matrix, diags, spmatrix
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2, f, norm, t

from pyfixest.errors import EmptyVcovError, VcovTypeNotSupportedError
from pyfixest.estimation.backends import BACKENDS
from pyfixest.estimation.decomposition import GelbachDecomposition, _decompose_arg_check
from pyfixest.estimation.demean_ import demean_model
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.literals import (
    DemeanerBackendOptions,
    PredictionErrorOptions,
    PredictionType,
    SolverOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.model_matrix_fixest_ import model_matrix_fixest
from pyfixest.estimation.prediction import (
    _compute_prediction_error,
    _get_fixed_effects_prediction_component,
    get_design_matrix_and_yhat,
)
from pyfixest.estimation.ritest import (
    _decode_resampvar,
    _get_ritest_pvalue,
    _get_ritest_stats_fast,
    _get_ritest_stats_slow,
    _plot_ritest_pvalue,
)
from pyfixest.estimation.solvers import solve_ols
from pyfixest.estimation.vcov_utils import (
    _check_cluster_df,
    _compute_bread,
    _count_G_for_ssc_correction,
    _get_cluster_df,
    _prepare_twoway_clustering,
)
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _drop_cols,
    _extract_variable_level,
    _narwhals_to_pandas,
    _select_order_coefs,
)
from pyfixest.utils.utils import (
    capture_context,
    get_ssc,
    simultaneous_crit_val,
)

decomposition_type = Literal["gelbach"]
prediction_type = Literal["response", "link"]


class Feols:
    """
    Non user-facing class to estimate a linear regression via OLS.

    Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.feols.qmd) function. Note that
    no demeaning is performed in this class: demeaning is performed in the
    [FixestMulti](/reference/estimation.fixest_multi.qmd) class (to allow for caching
    of demeaned variables for multiple estimation).

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable, a two-dimensional numpy array.
    X : np.ndarray
        Independent variables, a two-dimensional numpy array.
    weights : np.ndarray
        Weights, a one-dimensional numpy array.
    collin_tol : float
        Tolerance level for collinearity checks.
    coefnames : list[str]
        Names of the coefficients (of the design matrix X).
    weights_name : Optional[str]
        Name of the weights variable.
    weights_type : Optional[str]
        Type of the weights variable. Either "aweights" for analytic weights or
        "fweights" for frequency weights.
    solver : str, optional.
        The solver to use for the regression. Can be "np.linalg.lstsq",
        "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax".
        Defaults to "scipy.linalg.solve".
    context : int or Mapping[str, Any]
        A dictionary containing additional context variables to be used by
        formulaic during the creation of the model matrix. This can include
        custom factorization functions, transformations, or any other
        variables that need to be available in the formula environment.

    Attributes
    ----------
    _method : str
        Specifies the method used for regression, set to "feols".
    _is_iv : bool
        Indicates whether instrumental variables are used, initialized as False.

    _Y : np.ndarray
        The demeaned dependent variable, a two-dimensional numpy array.
    _X : np.ndarray
        The demeaned independent variables, a two-dimensional numpy array.
    _X_is_empty : bool
        Indicates whether the X array is empty.
    _collin_tol : float
        Tolerance level for collinearity checks.
    _coefnames : list
        Names of the coefficients (of the design matrix X).
    _collin_vars : list
        Variables identified as collinear.
    _collin_index : list
        Indices of collinear variables.
    _Z : np.ndarray
        Alias for the _X array, used for calculations.
    _solver: str
        The solver used for the regression.
    _weights : np.ndarray
        Array of weights for each observation.
    _N : int
        Number of observations.
    _k : int
        Number of independent variables (or features).
    _support_crv3_inference : bool
        Indicates support for CRV3 inference.
    _data : Any
        Data used in the regression, to be enriched outside of the class.
    _fml : Any
        Formula used in the regression, to be enriched outside of the class.
    _has_fixef : bool
        Indicates whether fixed effects are used.
    _fixef : Any
        Fixed effects used in the regression.
    _icovars : Any
        Internal covariates, to be enriched outside of the class.
    _ssc_dict : dict
        dictionary for sum of squares and cross products matrices.
    _tZX : np.ndarray
        Transpose of Z multiplied by X, set in get_fit().
    _tXZ : np.ndarray
        Transpose of X multiplied by Z, set in get_fit().
    _tZy : np.ndarray
        Transpose of Z multiplied by Y, set in get_fit().
    _tZZinv : np.ndarray
        Inverse of the transpose of Z multiplied by Z, set in get_fit().
    _beta_hat : np.ndarray
        Estimated regression coefficients.
    _Y_hat_link : np.ndarray
        Prediction at the level of the explanatory variable, i.e., the linear predictor X @ beta.
    _Y_hat_response : np.ndarray
        Prediction at the level of the response variable, i.e., the expected predictor E(Y|X).
    _u_hat : np.ndarray
        Residuals of the regression model.
    _scores : np.ndarray
        Scores used in the regression analysis.
    _hessian : np.ndarray
        Hessian matrix used in the regression.
    _bread : np.ndarray
        Bread matrix, used in calculating the variance-covariance matrix.
    _vcov_type : Any
        Type of variance-covariance matrix used.
    _vcov_type_detail : Any
        Detailed specification of the variance-covariance matrix type.
    _is_clustered : bool
        Indicates if clustering is used in the variance-covariance calculation.
    _clustervar : Any
        Variable used for clustering in the variance-covariance calculation.
    _G : Any
        Group information used in clustering.
    _ssc : Any
        Sum of squares and cross products matrix.
    _vcov : np.ndarray
        Variance-covariance matrix of the estimated coefficients.
    _se : np.ndarray
        Standard errors of the estimated coefficients.
    _tstat : np.ndarray
        T-statistics of the estimated coefficients.
    _pvalue : np.ndarray
        P-values associated with the t-statistics.
    _conf_int : np.ndarray
        Confidence intervals for the estimated coefficients.
    _F_stat : Any
        F-statistic for the model, set in get_Ftest().
    _fixef_dict : dict
        dictionary containing fixed effects estimates.
    _sumFE : np.ndarray
        Sum of all fixed effects for each observation.
    _rmse : float
        Root mean squared error of the model.
    _r2 : float
        R-squared value of the model.
    _r2_within : float
        R-squared value computed on demeaned dependent variable.
    _adj_r2 : float
        Adjusted R-squared value of the model.
    _adj_r2_within : float
        Adjusted R-squared value computed on demeaned dependent variable.
    _solver: Literal["np.linalg.lstsq", "np.linalg.solve", "scipy.linalg.solve",
        "scipy.sparse.linalg.lsqr", "jax"],
        default is "scipy.linalg.solve". Solver to use for the estimation.
    _demeaner_backend: DemeanerBackendOptions
    _data: pd.DataFrame
        The data frame used in the estimation. None if arguments `lean = True` or
        `store_data = False`.
    _model_name: str
        The name of the model. Usually just the formula string. If split estimation is used,
        the model name will include the split variable and value.
    _model_name_plot: str
        The name of the model used when plotting and summarizing models. Usually identical to
        `_model_name`. This might be different when pf.summary() or pf.coefplot() are called
        and models with identical _model_name attributes are passed. In this case,
        the _model_name_plot attribute will be modified.
    _quantile: Optional[float]
        The quantile used for quantile regression. None if not a quantile regression.

    # special for did
    _res_cohort_eventtime_dict: Optional[dict[str, Any]]
    _yname: Optional[str]
    _gname: Optional[str]
    _tname: Optional[str]
    _idname: Optional[str]
    _att: Optional[Any]
    test_treatment_heterogeneity: Callable[..., Any]
    aggregate: Callable[..., Any]
    iplot_aggregate: Callable[..., Any]

    """

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, Union[str, bool]],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: Optional[str],
        weights_type: Optional[str],
        collin_tol: float,
        fixef_tol: float,
        fixef_maxiter: int,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        solver: SolverOptions = "np.linalg.solve",
        demeaner_backend: DemeanerBackendOptions = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int, float]] = None,
    ) -> None:
        self._sample_split_value = sample_split_value
        self._sample_split_var = sample_split_var
        self._model_name = (
            FixestFormula.fml
            if self._sample_split_var is None
            else f"{FixestFormula.fml} (Sample: {self._sample_split_var} = {self._sample_split_value})"
        )
        self._model_name_plot = self._model_name
        self._method = "feols"
        self._is_iv = False
        self.FixestFormula = FixestFormula

        if sample_split_value == "all":
            data_split = data.copy()
        else:
            data_split = data[data[sample_split_var] == sample_split_value].copy()
        data_split.reset_index(drop=True, inplace=True)  # set index to 0:N

        self._data = data_split.copy() if copy_data else data_split
        self._ssc_dict = ssc_dict
        self._drop_singletons = drop_singletons
        self._drop_intercept = drop_intercept
        self._weights_name = weights
        self._weights_type = weights_type
        self._has_weights = weights is not None
        self._collin_tol = collin_tol
        self._fixef_tol = fixef_tol
        self._fixef_maxiter = fixef_maxiter
        self._solver = solver
        self._demeaner_backend = demeaner_backend
        self._lookup_demeaned_data = lookup_demeaned_data
        self._store_data = store_data
        self._copy_data = copy_data
        self._lean = lean
        self._use_mundlak = False
        self._context = capture_context(context)

        self._support_crv3_inference = True
        if self._weights_name is not None:
            self._supports_wildboottest = False
        self._supports_wildboottest = True
        self._supports_cluster_causal_variance = True
        if self._has_weights or self._is_iv:
            self._supports_wildboottest = False
        self._support_decomposition = True

        # attributes that have to be enriched outside of the class -
        # not really optimal code change later
        self._fml = FixestFormula.fml
        self._has_fixef = False
        self._fixef = FixestFormula._fval
        # self._coefnames = None
        self._icovars = None

        try:
            impl = BACKENDS[demeaner_backend]
        except KeyError:
            raise ValueError(f"Unknown backend {demeaner_backend!r}")

        self._demean_func = impl["demean"]
        self._find_collinear_variables_func = impl["collinear"]
        self._crv1_meat_func = impl["crv1_meat"]
        self._cound_nested_fixef_func = impl["nested"]

        # set in get_fit()
        self._tZX = np.array([])
        # self._tZXinv = None
        self._tXZ = np.array([])
        self._tZy = np.array([])
        self._tZZinv = np.array([])
        self._beta_hat = np.array([])
        self._Y_hat_link = np.array([])
        self._Y_hat_response = np.array([])
        self._u_hat = np.array([])
        self._scores = np.array([])
        self._hessian = np.array([])
        self._bread = np.array([])

        # set in vcov()
        self._vcov_type = ""
        self._vcov_type_detail = ""
        self._is_clustered = False
        self._clustervar: list[str] = []
        self._G: list[int] = []
        self._ssc = np.array([], dtype=np.float64)
        self._vcov = np.array([])
        self.na_index = np.array([])  # initiated outside of the class
        self.n_separation_na = 0

        # set in get_inference()
        self._se = np.array([])
        self._tstat = np.array([])
        self._pvalue = np.array([])
        self._conf_int = np.array([])

        # set in get_Ftest()
        self._F_stat = None

        # set in fixef()
        self._fixef_dict: dict[str, dict[str, float]] = {}
        self._sumFE = None

        # set in get_performance()
        self._rmse = np.nan
        self._r2 = np.nan
        self._r2_within = np.nan
        self._adj_r2 = np.nan
        self._adj_r2_within = np.nan

        # special for poisson
        self.deviance = None

        # special for did
        self._res_cohort_eventtime_dict: Optional[dict[str, Any]] = None
        self._yname: Optional[str] = None
        self._gname: Optional[str] = None
        self._tname: Optional[str] = None
        self._idname: Optional[str] = None
        self._att: Optional[bool] = None

        # set functions inherited from other modules
        _module = import_module("pyfixest.report")
        _tmp = _module.coefplot
        self.coefplot = functools.partial(_tmp, models=[self])
        self.coefplot.__doc__ = _tmp.__doc__
        _tmp = _module.iplot
        self.iplot = functools.partial(_tmp, models=[self])
        self.iplot.__doc__ = _tmp.__doc__
        _tmp = _module.summary
        self.summary = functools.partial(_tmp, models=[self])
        self.summary.__doc__ = _tmp.__doc__

        # DiD methods - assign placeholder functions
        def _not_implemented_did(*args, **kwargs):
            raise NotImplementedError(
                "This method is only available for DiD models, not for vanilla 'feols'."
            )

        self.test_treatment_heterogeneity = _not_implemented_did
        self.aggregate = _not_implemented_did
        self.iplot_aggregate = _not_implemented_did

    def prepare_model_matrix(self):
        "Prepare model matrices for estimation."
        mm_dict = model_matrix_fixest(
            FixestFormula=self.FixestFormula,
            data=self._data,
            drop_singletons=self._drop_singletons,
            drop_intercept=self._drop_intercept,
            weights=self._weights_name,
            context=self._context,
        )

        self._Y = mm_dict.get("Y")
        self._Y_untransformed = mm_dict.get("Y").copy()
        self._X = mm_dict.get("X")
        self._fe = mm_dict.get("fe")
        self._endogvar = mm_dict.get("endogvar")
        self._Z = mm_dict.get("Z")
        self._weights_df = mm_dict.get("weights_df")
        self._na_index = mm_dict.get("na_index")
        self._na_index_str = mm_dict.get("na_index_str")
        self._icovars = mm_dict.get("icovars")
        self._X_is_empty = mm_dict.get("X_is_empty")
        self._model_spec = mm_dict.get("model_spec")

        self._coefnames = self._X.columns.tolist()
        self._coefnames_z = self._Z.columns.tolist() if self._Z is not None else None
        self._depvar = self._Y.columns[0]

        self._has_fixef = self._fe is not None
        self._fixef = self.FixestFormula._fval

        self._k_fe = self._fe.nunique(axis=0) if self._has_fixef else None
        self._n_fe = len(self._k_fe) if self._has_fixef else 0

        # update data:
        self._data = _drop_cols(self._data, self._na_index)

        self._weights = self._set_weights()
        self._N, self._N_rows = self._set_nobs()

    def _set_nobs(self) -> tuple[int, int]:
        """
        Fetch the number of observations used in fitting the regression model.

        Returns
        -------
        tuple[int, int]
            A tuple containing the total number of observations and the number of rows
            in the dependent variable array.
        """
        _N_rows = len(self._Y)
        if self._weights_type == "aweights":
            _N = _N_rows
        elif self._weights_type == "fweights":
            _N = np.sum(self._weights)

        return _N, _N_rows

    def _set_weights(self) -> np.ndarray:
        """
        Return the weights used in the regression model.

        Returns
        -------
        np.ndarray
            The weights used in the regression model.
            If no weights are used, returns an array of ones
            with the same length as the dependent variable array.
        """
        N = len(self._Y)

        if self._weights_df is not None:
            _weights = self._weights_df.to_numpy()
        else:
            _weights = np.ones(N)

        return _weights.reshape((N, 1))

    def demean(self):
        "Demean the dependent variable and covariates by the fixed effect(s)."
        if self._has_fixef:
            self._Yd, self._Xd = demean_model(
                self._Y,
                self._X,
                self._fe,
                self._weights.flatten(),
                self._lookup_demeaned_data,
                self._na_index_str,
                self._fixef_tol,
                self._fixef_maxiter,
                self._demean_func,
                # self._demeaner_backend,
            )
        else:
            self._Yd, self._Xd = self._Y, self._X

    def to_array(self):
        "Convert estimation data frames to np arrays."
        self._Y, self._X = (
            self._Yd.to_numpy(),
            self._Xd.to_numpy(),
        )

    def wls_transform(self):
        "Transform model matrices for WLS Estimation."
        self._X_untransformed = self._X.copy()
        if self._has_weights:
            w = np.sqrt(self._weights)
            self._Y = self._Y * w
            self._X = self._X * w

    def drop_multicol_vars(self):
        "Detect and drop multicollinear variables."
        if self._X.shape[1] > 0:
            (
                self._X,
                self._coefnames,
                self._collin_vars,
                self._collin_index,
            ) = _drop_multicollinear_variables(
                self._X,
                self._coefnames,
                self._collin_tol,
                backend_func=self._find_collinear_variables_func,
            )
        # update X_is_empty
        self._X_is_empty = self._X.shape[1] == 0
        self._k = self._X.shape[1] if not self._X_is_empty else 0

    def _get_predictors(self) -> None:
        self._Y_hat_link = self._Y_untransformed.values.flatten() - self.resid()
        self._Y_hat_response = self._Y_hat_link

    def get_fit(self) -> None:
        """
        Fit an OLS model.

        Returns
        -------
        None
        """
        if self._X_is_empty:
            self._u_hat = self._Y
        else:
            _X = self._X
            _Y = self._Y
            self._Z = self._X
            _Z = self._Z
            _solver = self._solver
            self._tZX = _Z.T @ _X
            self._tZy = _Z.T @ _Y

            self._beta_hat = solve_ols(self._tZX, self._tZy, _solver)

            self._u_hat = self._Y.flatten() - (self._X @ self._beta_hat).flatten()

            self._scores = _X * self._u_hat[:, None]
            self._hessian = self._tZX.copy()

            # IV attributes, set to None for OLS, Poisson
            self._tXZ = np.array([])
            self._tZZinv = np.array([])

        self._get_predictors()

    def vcov(
        self, vcov: Union[str, dict[str, str]], data: Optional[DataFrameType] = None
    ) -> "Feols":
        """
        Compute covariance matrices for an estimated regression model.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]]
            A string or dictionary specifying the type of variance-covariance matrix
            to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format {"CRV1": "clustervar"} for
            CRV1 inference or {"CRV3": "clustervar"}
            for CRV3 inference. Note that CRV3 inference is currently not supported
            for IV estimation.
        data: Optional[DataFrameType], optional
            The data used for estimation. If None, tries to fetch the data from the
            model object. Defaults to None.


        Returns
        -------
        Feols
            An instance of class [Feols(/reference/Feols.qmd) with updated inference.
        """
        # Assuming `data` is the DataFrame in question

        _data = data if data is not None else self._data
        try:
            _data = _narwhals_to_pandas(_data)
        except TypeError as e:
            raise TypeError(
                f"The data set must be a DataFrame type. Received: {type(data)}"
            ) from e

        _data = self._data
        _has_fixef = self._has_fixef
        _is_iv = self._is_iv
        _method = self._method
        _support_crv3_inference = self._support_crv3_inference

        _tXZ = self._tXZ
        _tZZinv = self._tZZinv
        _tZX = self._tZX
        _hessian = self._hessian

        # assign estimated fixed effects, and fixed effects nested within cluster.

        # deparse vcov input
        _check_vcov_input(vcov, _data)
        (
            self._vcov_type,
            self._vcov_type_detail,
            self._is_clustered,
            self._clustervar,
        ) = _deparse_vcov_input(vcov, _has_fixef, _is_iv)

        self._bread = _compute_bread(_is_iv, _tXZ, _tZZinv, _tZX, _hessian)

        # compute vcov

        ssc_kwargs = {
            "ssc_dict": self._ssc_dict,
            "N": self._N,
            "k": self._k,
            "k_fe": self._k_fe.sum() if self._has_fixef else 0,
            "n_fe": self._n_fe,
        }

        if self._vcov_type == "iid":
            ssc_kwargs_iid = {
                "k_fe_nested": 0,
                "n_fe_fully_nested": 0,
                "vcov_sign": 1,
                "vcov_type": "iid",
                "G": 1,
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_iid}
            self._ssc, self._dof_k, self._df_t = get_ssc(**all_kwargs)

            self._vcov = self._ssc * self._vcov_iid()

        elif self._vcov_type == "hetero":
            # this is what fixest does internally: see fixest:::vcov_hetero_internal:
            # adj = ifelse(ssc$cluster.adj, n/(n - 1), 1)

            ssc_kwargs_hetero = {
                "k_fe_nested": 0,
                "n_fe_fully_nested": 0,
                "vcov_sign": 1,
                "vcov_type": "hetero",
                "G": self._N,
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_hetero}
            self._ssc, self._dof_k, self._df_t = get_ssc(**all_kwargs)
            self._vcov = self._ssc * self._vcov_hetero()

        elif self._vcov_type == "nid":
            ssc_kwargs_hetero = {
                "k_fe_nested": 0,
                "n_fe_fully_nested": 0,
                "vcov_sign": 1,
                "vcov_type": "hetero",
                "G": self._N,
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_hetero}
            self._ssc, self._dof_k, self._df_t = get_ssc(**all_kwargs)
            self._vcov = self._ssc * self._vcov_nid()

        elif self._vcov_type == "CRV":
            if data is not None:
                # use input data set
                self._cluster_df = _get_cluster_df(
                    data=data,
                    clustervar=self._clustervar,
                )
                _check_cluster_df(cluster_df=self._cluster_df, data=data)
            else:
                # use stored data
                self._cluster_df = _get_cluster_df(
                    data=self._data, clustervar=self._clustervar
                )
                _check_cluster_df(cluster_df=self._cluster_df, data=self._data)

            if self._cluster_df.shape[1] > 1:
                self._cluster_df = _prepare_twoway_clustering(
                    clustervar=self._clustervar, cluster_df=self._cluster_df
                )

            self._G = _count_G_for_ssc_correction(
                cluster_df=self._cluster_df, ssc_dict=self._ssc_dict
            )

            # loop over columns of cluster_df
            vcov_sign_list = [1, 1, -1]
            df_t_full = np.zeros(self._cluster_df.shape[1])

            cluster_arr_int = np.column_stack(
                [
                    pd.factorize(self._cluster_df[col])[0]
                    for col in self._cluster_df.columns
                ]
            )

            k_fe_nested = 0
            n_fe_fully_nested = 0
            if self._has_fixef and self._ssc_dict["fixef_k"] == "nested":
                k_fe_nested_flag, n_fe_fully_nested = self._cound_nested_fixef_func(
                    all_fixef_array=np.array(
                        self._fixef.replace("^", "_").split("+"), dtype=str
                    ),
                    cluster_colnames=np.array(self._cluster_df.columns, dtype=str),
                    cluster_data=cluster_arr_int.astype(np.uintp),
                    fe_data=self._fe.to_numpy().astype(np.uintp)
                    if isinstance(self._fe, pd.DataFrame)
                    else self._fe.astype(np.uintp),
                )

                k_fe_nested = (
                    np.sum(self._k_fe[k_fe_nested_flag]) if n_fe_fully_nested > 0 else 0
                )

            self._vcov = np.zeros((self._k, self._k))

            for x, _ in enumerate(self._cluster_df.columns):
                cluster_col = cluster_arr_int[:, x]
                clustid = np.unique(cluster_col)

                ssc_kwargs_crv = {
                    "k_fe_nested": k_fe_nested,
                    "n_fe_fully_nested": n_fe_fully_nested,
                    "G": self._G[x],
                    "vcov_sign": vcov_sign_list[x],
                    "vcov_type": "CRV",
                }

                all_kwargs = {**ssc_kwargs, **ssc_kwargs_crv}
                ssc, dof_k, df_t = get_ssc(**all_kwargs)

                self._ssc = np.array([ssc]) if x == 0 else np.append(self._ssc, ssc)
                self._dof_k = dof_k  # the same across all vcov's

                # update. take min(df_t) ad the end of loop
                df_t_full[x] = df_t

                if self._vcov_type_detail == "CRV1":
                    self._vcov += self._ssc[x] * self._vcov_crv1(
                        clustid=clustid, cluster_col=cluster_col
                    )

                elif self._vcov_type_detail == "CRV3":
                    # check: is fixed effect cluster fixed effect?
                    # if not, either error or turn fixefs into dummies
                    # for now: don't allow for use with fixed effects

                    if not _support_crv3_inference:
                        raise VcovTypeNotSupportedError(
                            f"CRV3 inference is not for models of type '{self._method}'."
                        )

                    if (
                        (_has_fixef is False)
                        and (_method == "feols")
                        and (_is_iv is False)
                    ):
                        self._vcov += self._ssc[x] * self._vcov_crv3_fast(
                            clustid=clustid, cluster_col=cluster_col
                        )
                    else:
                        self._vcov += self._ssc[x] * self._vcov_crv3_slow(
                            clustid=clustid, cluster_col=cluster_col
                        )
            # take minimum cluster for dof for multiway clustering
            self._df_t = np.min(df_t_full)
        # update p-value, t-stat, standard error, confint
        self.get_inference()

        return self

    def _vcov_iid(self):
        _N = self._N
        _u_hat = self._u_hat
        _bread = self._bread
        sigma2 = np.sum(_u_hat.flatten() ** 2) / (_N - 1)
        _vcov = _bread * sigma2

        return _vcov

    def _vcov_hetero(self):
        _scores = self._scores
        _vcov_type_detail = self._vcov_type_detail
        _tXZ = self._tXZ
        _tZZinv = self._tZZinv
        _tZX = self._tZX
        _X = self._X
        _is_iv = self._is_iv
        _bread = self._bread

        if _vcov_type_detail in ["hetero", "HC1"]:
            transformed_scores = _scores
        elif _vcov_type_detail in ["HC2", "HC3"]:
            leverage = np.sum(_X * (_X @ np.linalg.inv(_tZX)), axis=1)
            transformed_scores = (
                _scores / np.sqrt(1 - leverage)[:, None]
                if _vcov_type_detail == "HC2"
                else _scores / (1 - leverage)[:, None]
            )

        Omega = transformed_scores.T @ transformed_scores

        _meat = _tXZ @ _tZZinv @ Omega @ _tZZinv @ _tZX if _is_iv else Omega
        _vcov = _bread @ _meat @ _bread

        return _vcov

    def _vcov_nid(self):
        raise NotImplementedError(
            "Only models of type Quantreg support a variance-covariance matrix of type 'nid'."
        )

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):
        _is_iv = self._is_iv
        _tXZ = self._tXZ
        _tZZinv = self._tZZinv
        _tZX = self._tZX
        _bread = self._bread

        _scores = self._scores

        k = _scores.shape[1]
        meat = np.zeros((k, k))

        meat = self._crv1_meat_func(
            scores=_scores.astype(np.float64),
            clustid=clustid.astype(np.uintp),
            cluster_col=cluster_col.astype(np.uintp),
        )

        meat = _tXZ @ _tZZinv @ meat @ _tZZinv @ _tZX if _is_iv else meat
        vcov = _bread @ meat @ _bread

        return vcov

    def _vcov_crv3_fast(self, clustid, cluster_col):
        _k = self._k
        _Y = self._Y
        _X = self._X
        _beta_hat = self._beta_hat

        beta_jack = np.zeros((len(clustid), _k))

        # inverse hessian precomputed?
        tXX = np.transpose(_X) @ _X
        tXy = np.transpose(_X) @ _Y

        # compute leave-one-out regression coefficients (aka clusterjacks')
        for ixg, g in enumerate(clustid):
            Xg = _X[np.equal(g, cluster_col)]
            Yg = _Y[np.equal(g, cluster_col)]
            tXgXg = np.transpose(Xg) @ Xg
            # jackknife regression coefficient
            beta_jack[ixg, :] = (
                np.linalg.pinv(tXX - tXgXg) @ (tXy - np.transpose(Xg) @ Yg)
            ).flatten()

        # optional: beta_bar in MNW (2022)
        # center = "estimate"
        # if center == 'estimate':
        #    beta_center = beta_hat
        # else:
        #    beta_center = np.mean(beta_jack, axis = 0)
        beta_center = _beta_hat

        vcov_mat = np.zeros((_k, _k))
        for ixg, _ in enumerate(clustid):
            beta_centered = beta_jack[ixg, :] - beta_center
            vcov_mat += np.outer(beta_centered, beta_centered)

        _vcov = vcov_mat

        return _vcov

    def _vcov_crv3_slow(self, clustid, cluster_col):
        _k = self._k
        _method = self._method
        _fml = self._fml
        _data = self._data
        _weights_name = self._weights_name
        _weights_type = self._weights_type
        _beta_hat = self._beta_hat

        beta_jack = np.zeros((len(clustid), _k))

        # lazy loading to avoid circular import
        fixest_module = import_module("pyfixest.estimation")
        fit_ = fixest_module.feols if _method == "feols" else fixest_module.fepois

        for ixg, g in enumerate(clustid):
            # direct leave one cluster out implementation
            data = _data[~np.equal(g, cluster_col)]
            fit = fit_(
                fml=_fml,
                data=data,
                vcov="iid",
                weights=_weights_name,
                weights_type=_weights_type,
            )
            beta_jack[ixg, :] = fit.coef().to_numpy()

        # optional: beta_bar in MNW (2022)
        # center = "estimate"
        # if center == 'estimate':
        #    beta_center = beta_hat
        # else:
        #    beta_center = np.mean(beta_jack, axis = 0)
        beta_center = _beta_hat

        vcov_mat = np.zeros((_k, _k))
        for ixg, _ in enumerate(clustid):
            beta_centered = beta_jack[ixg, :] - beta_center
            vcov_mat += np.outer(beta_centered, beta_centered)

        _vcov = vcov_mat

        return _vcov

    def get_inference(self, alpha: float = 0.05) -> None:
        """
        Compute standard errors, t-statistics, and p-values for the regression model.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05, which
            produces a 95% confidence interval.

        Returns
        -------
        None

        Details
        -------
        relevant fixest functions:
        - fixest_CI_factor: https://github.com/lrberge/fixest/blob/5523d48ef4a430fa2e82815ca589fc8a47168fe7/R/miscfuns.R#L5614
        -
        """
        if len(self._vcov) == 0:
            raise EmptyVcovError()
        _beta_hat = self._beta_hat

        _method = self._method

        self._se = np.sqrt(np.diagonal(self._vcov))
        self._tstat = _beta_hat / self._se
        # use t-dist for linear models, but normal for non-linear models
        if _method in ["fepois", "feglm-probit", "feglm-logit", "feglm-gaussian"]:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf(alpha / 2))
        else:
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), self._df_t))
            z = np.abs(t.ppf(alpha / 2, self._df_t))

        z_se = z * self._se
        self._conf_int = np.array([_beta_hat - z_se, _beta_hat + z_se])

    def add_fixest_multi_context(
        self,
        depvar: str,
        Y: pd.Series,
        _data: pd.DataFrame,
        _ssc_dict: dict[str, Union[str, bool]],
        _k_fe: int,
        fval: str,
        store_data: bool,
    ) -> None:
        """
        Enrich Feols object.

        Enrich an instance of `Feols` Class with additional
        attributes set in the `FixestMulti` class.

        Parameters
        ----------
        FixestFormula : FixestFormula
            The formula(s) used for estimation encoded in a `FixestFormula` object.
        depvar : str
            The dependent variable of the regression model.
        Y : pd.Series
            The dependent variable of the regression model.
        _data : pd.DataFrame
            The data used for estimation.
        _ssc_dict : dict
            A dictionary with the sum of squares and cross products matrices.
        _k_fe : int
            The number of fixed effects.
        fval : str
            The fixed effects formula.
        store_data : bool
            Indicates whether to save the data used for estimation in the object

        Returns
        -------
        None
        """
        # some bookkeeping
        self._fml = self.FixestFormula.fml
        self._depvar = depvar
        self._Y_untransformed = Y
        self._data = pd.DataFrame()

        if store_data:
            self._data = _data

        self._ssc_dict = _ssc_dict
        self._k_fe = _k_fe
        if fval != "0":
            self._has_fixef = True
            self._fixef = fval
        else:
            self._has_fixef = False

    def _clear_attributes(self):
        attributes = []

        if not self._store_data:
            attributes += ["_data"]

        if self._lean:
            attributes += [
                "_data",
                "_X",
                "_Y",
                "_Z",
                "_Xd",
                "_Yd",
                "_Zd",
                "_cluster_df",
                "_tXZ",
                "_tZy",
                "_tZX",
                "_weights",
                "_scores",
                "_tZZinv",
                "_u_hat",
                "_Y_hat_link",
                "_Y_hat_response",
                "_Y_untransformed",
            ]

        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def wald_test(self, R=None, q=None, distribution="F"):
        """
        Conduct Wald test.

        Compute a Wald test for a linear hypothesis of the form R * β = q.
        where R is m x k matrix, β is a k x 1 vector of coefficients,
        and q is m x 1 vector.
        By default, tests the joint null hypothesis that all coefficients are zero.

        This method producues the following attriutes

        _dfd : int
            degree of freedom in denominator
        _dfn : int
            degree of freedom in numerator
        _wald_statistic : scalar
            Wald-statistics computed for hypothesis testing
        _f_statistic : scalar
            Wald-statistics(when R is an indentity matrix, and q being zero vector)
            computed for hypothesis testing
        _p_value : scalar
            corresponding p-value for statistics

        Parameters
        ----------
        R : array-like, optional
            The matrix R of the linear hypothesis.
            If None, defaults to an identity matrix.
        q : array-like, optional
            The vector q of the linear hypothesis.
            If None, defaults to a vector of zeros.
        distribution : str, optional
            The distribution to use for the p-value. Can be either "F" or "chi2".
            Defaults to "F".

        Returns
        -------
        pd.Series
            A pd.Series with the Wald statistic and p-value.

        Examples
        --------
        ```{python}
        import numpy as np
        import pandas as pd
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(adj=False))

        R = np.array([[1,-1]] )
        q = np.array([0.0])

        # Wald test
        fit.wald_test(R=R, q=q, distribution = "chi2")
        f_stat = fit._f_statistic
        p_stat = fit._p_value

        print(f"Python f_stat: {f_stat}")
        print(f"Python p_stat: {p_stat}")
        ```
        """
        _vcov = self._vcov
        _N = self._N
        _k = self._k
        _beta_hat = self._beta_hat
        _k_fe = np.sum(self._k_fe.values) if self._has_fixef else 0

        # If R is None, default to the identity matrix
        if R is None:
            R = np.eye(_k)

        # Ensure R is two-dimensional
        if R.ndim == 1:
            R = R.reshape((1, len(R)))

        if R.shape[1] != _k:
            raise ValueError(
                "The number of columns of R must be equal to the number of coefficients."
            )

        # If q is None, default to a vector of zeros
        if q is None:
            q = np.zeros(R.shape[0])
        else:
            if not isinstance(q, (int, float, np.ndarray)):
                raise ValueError("q must be a numeric scalar or array.")
            if isinstance(q, np.ndarray):
                if q.ndim != 1:
                    raise ValueError("q must be a one-dimensional array or a scalar.")
                if q.shape[0] != R.shape[0]:
                    raise ValueError("q must have the same number of rows as R.")

        n_restriction = R.shape[0]
        self._dfn = n_restriction

        if self._is_clustered:
            self._dfd = np.min(np.array(self._G)) - 1
        else:
            self._dfd = _N - _k - _k_fe

        bread = R @ _beta_hat - q
        meat = np.linalg.inv(R @ _vcov @ R.T)
        W = bread.T @ meat @ bread
        self._wald_statistic = W

        # Check if distribution is "F" and R is not identity matrix
        # or q is not zero vector
        if distribution == "F" and (
            not np.array_equal(R, np.eye(_k)) or not np.all(q == 0)
        ):
            warnings.warn(
                "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector."
            )
            distribution = "chi2"

        if distribution == "F":
            self._f_statistic = W / self._dfn
            self._p_value = 1 - f.cdf(self._f_statistic, dfn=self._dfn, dfd=self._dfd)
            res = pd.Series({"statistic": self._f_statistic, "pvalue": self._p_value})
        elif distribution == "chi2":
            self._f_statistic = W / self._dfn
            self._p_value = chi2.sf(self._wald_statistic, self._dfn)
            res = pd.Series(
                {"statistic": self._wald_statistic, "pvalue": self._p_value}
            )
        else:
            raise ValueError("Distribution must be F or chi2")

        return res

    def wildboottest(
        self,
        reps: int,
        cluster: Optional[str] = None,
        param: Optional[str] = None,
        weights_type: Optional[str] = "rademacher",
        impose_null: Optional[bool] = True,
        bootstrap_type: Optional[str] = "11",
        seed: Optional[int] = None,
        adj: Optional[bool] = True,
        cluster_adj: Optional[bool] = True,
        parallel: Optional[bool] = False,
        return_bootstrapped_t_stats=False,
    ):
        """
        Run a wild cluster bootstrap based on an object of type "Feols".

        Parameters
        ----------
        reps : int
            The number of bootstrap iterations to run.
        cluster : Union[str, None], optional
            The variable used for clustering. Defaults to None. If None, then
            uses the variable specified in the model's `clustervar` attribute.
            If no `_clustervar` attribute is found, runs a heteroskedasticity-
            robust bootstrap.
        param : Union[str, None], optional
            A string of length one, containing the test parameter of interest.
            Defaults to None.
        weights_type : str, optional
            The type of bootstrap weights. Options are 'rademacher', 'mammen',
            'webb', or 'normal'. Defaults to 'rademacher'.
        impose_null : bool, optional
            Indicates whether to impose the null hypothesis on the bootstrap DGP.
            Defaults to True.
        bootstrap_type : str, optional
            A string of length one to choose the bootstrap type.
            Options are '11', '31', '13', or '33'. Defaults to '11'.
        seed : Union[int, None], optional
            An option to provide a random seed. Defaults to None.
        adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of observations and covariates. Defaults to True.
        cluster_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of clusters. Defaults to True.
        parallel : bool, optional
            Indicates whether to run the bootstrap in parallel. Defaults to False.
        seed : Union[str, None], optional
            An option to provide a random seed. Defaults to None.
        return_bootstrapped_t_stats : bool, optional:
            If True, the method returns a tuple of the regular output and the
            bootstrapped t-stats. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the original, non-bootstrapped t-statistic and
            bootstrapped p-value, along with the bootstrap type, inference type
            (HC vs CRV), and whether the null hypothesis was imposed on the
            bootstrap DGP. If `return_bootstrapped_t_stats` is True, the method
            returns a tuple of the regular output and the bootstrapped t-stats.

        Examples
        --------
        ```{python}
        #| echo: true
        #| results: asis
        #| include: true

        import re
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2 | f1", data)

        fit.wildboottest(
            param = "X1",
            reps=1000,
            seed = 822
        )

        fit.wildboottest(
            param = "X1",
            reps=1000,
            seed = 822,
            bootstrap_type = "31"
        )

        ```
        """
        _is_iv = self._is_iv
        _has_fixef = self._has_fixef
        _xnames = self._coefnames
        _data = self._data
        _clustervar = self._clustervar
        _supports_wildboottest = self._supports_wildboottest

        if param is not None and param not in _xnames:
            raise ValueError(
                f"Parameter {param} not found in the model's coefficients."
            )

        if not _supports_wildboottest:
            if self._is_iv:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for IV estimation."
                )
            if self._has_weights:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for WLS estimation."
                )

        cluster_list = []

        if cluster is not None and isinstance(cluster, str):
            cluster_list = [cluster]
        if cluster is not None and isinstance(cluster, list):
            cluster_list = cluster

        if cluster is None and _clustervar is not None:
            if isinstance(_clustervar, str):
                cluster_list = [_clustervar]
            else:
                cluster_list = _clustervar

        run_heteroskedastic = not cluster_list

        if not run_heteroskedastic and not len(cluster_list) == 1:
            raise NotImplementedError(
                "Multiway clustering is currently not supported with the wild cluster bootstrap."
            )

        if not run_heteroskedastic and cluster_list[0] not in _data.columns:
            raise ValueError(
                f"Cluster variable {cluster_list[0]} not found in the data."
            )

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
            )

        if _is_iv:
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported with IV estimation."
            )

        if self._method == "fepois":
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported for Poisson regression."
            )

        _Y, _X, _xnames = self._model_matrix_one_hot()

        # later: allow r <> 0 and custom R
        R = np.zeros(len(_xnames))
        if param is not None:
            R[_xnames.index(param)] = 1
        r = 0

        if run_heteroskedastic:
            inference = "HC"

            boot = WildboottestHC(X=_X, Y=_Y, R=R, r=r, B=reps, seed=seed)
            boot.get_adjustments(bootstrap_type=bootstrap_type)
            boot.get_uhat(impose_null=impose_null)
            boot.get_tboot(weights_type=weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            full_enumeration_warn = False

        else:
            inference = f"CRV({cluster_list[0]})"

            cluster_array = _data[cluster_list[0]].to_numpy().flatten()

            boot = WildboottestCL(
                X=_X,
                Y=_Y,
                cluster=cluster_array,
                R=R,
                B=reps,
                seed=seed,
                parallel=parallel,
            )
            boot.get_scores(
                bootstrap_type=bootstrap_type,
                impose_null=impose_null,
                adj=adj,
                cluster_adj=cluster_adj,
            )
            _, _, full_enumeration_warn = boot.get_weights(weights_type=weights_type)
            boot.get_numer()
            boot.get_denom()
            boot.get_tboot()
            boot.get_vcov()
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")

            if full_enumeration_warn:
                warnings.warn(
                    "2^G < the number of boot iterations, setting full_enumeration to True."
                )

        if np.isscalar(boot.t_stat):
            boot.t_stat = np.asarray(boot.t_stat)
        else:
            boot.t_stat = boot.t_stat[0]

        res = {
            "param": param,
            "t value": boot.t_stat.astype(np.float64),
            "Pr(>|t|)": np.asarray(boot.pvalue).astype(np.float64),
            "bootstrap_type": bootstrap_type,
            "inference": inference,
            "impose_null": impose_null,
            "ssc": boot.small_sample_correction if run_heteroskedastic else boot.ssc,
        }

        res_df = pd.Series(res)

        if return_bootstrapped_t_stats:
            return res_df, boot.t_boot
        else:
            return res_df

    def ccv(
        self,
        treatment,
        cluster: Optional[str] = None,
        seed: Optional[int] = None,
        n_splits: int = 8,
        pk: float = 1,
        qk: float = 1,
    ) -> pd.DataFrame:
        """
        Compute the Causal Cluster Variance following Abadie et al (QJE 2023).

        Parameters
        ----------
        treatment: str
            The name of the treatment variable.
        cluster : str
            The name of the cluster variable. None by default.
            If None, uses the cluster variable from the model fit.
        seed : int, optional
            An integer to set the random seed. Defaults to None.
        n_splits : int, optional
            The number of splits to use in the cross-fitting procedure. Defaults to 8.
        pk: float, optional
            The proportion of sampled clusters. Defaults to 1, which
            corresponds to all clusters of the population being sampled.
        qk: float, optional
            The proportion of sampled observations within each cluster.
            Defaults to 1, which corresponds to all observations within
            each cluster being sampled.

        Returns
        -------
        pd.DataFrame
            A DataFrame with inference based on the "Causal Cluster Variance"
            and "regular" CRV1 inference.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        import numpy as np

        data = pf.get_data()
        data["D"] = np.random.choice([0, 1], size=data.shape[0])

        fit = pf.feols("Y ~ D", data=data, vcov={"CRV1": "group_id"})
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=8, seed=123).head()
        ```
        """
        assert self._supports_cluster_causal_variance, (
            "The model does not support the causal cluster variance estimator."
        )
        assert isinstance(treatment, str), "treatment must be a string."
        assert isinstance(cluster, str) or cluster is None, (
            "cluster must be a string or None."
        )
        assert isinstance(seed, int) or seed is None, "seed must be an integer or None."
        assert isinstance(n_splits, int), "n_splits must be an integer."
        assert isinstance(pk, (int, float)) and 0 <= pk <= 1
        assert isinstance(qk, (int, float)) and 0 <= qk <= 1

        if self._has_fixef:
            raise NotImplementedError(
                "The causal cluster variance estimator is currently not supported for models with fixed effects."
            )

        if treatment not in self._coefnames:
            raise ValueError(
                f"Variable {treatment} not found in the model's coefficients."
            )

        if cluster is None:
            if self._clustervar is None:
                raise ValueError("No cluster variable found in the model fit.")
            elif len(self._clustervar) > 1:
                raise ValueError(
                    "Multiway clustering is currently not supported with the causal cluster variance estimator."
                )
            else:
                cluster = self._clustervar[0]

        # check that cluster is in data
        if cluster not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster} not found in the data used for the model fit."
            )

        if not self._is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object."
            )
            self.vcov({"CRV1": cluster})

        if seed is None:
            seed = np.random.randint(1, 100_000_000)
        rng = np.random.default_rng(seed)

        depvar = self._depvar
        fml = self._fml
        xfml_list = fml.split("~")[1].split("+")
        xfml_list = [x for x in xfml_list if x != treatment]
        xfml = "" if not xfml_list else "+".join(xfml_list)

        data = self._data
        Y = self._Y.flatten()
        W = data[treatment].to_numpy()
        assert np.all(np.isin(W, [0, 1])), (
            "Treatment variable must be binary with values 0 and 1"
        )
        X = self._X
        cluster_vec = data[cluster].to_numpy()
        unique_clusters = np.unique(cluster_vec)

        tau_full = np.array(self.coef().xs(treatment))

        N = self._N
        G = len(unique_clusters)

        ccv_module = import_module("pyfixest.estimation.ccv")
        _compute_CCV = ccv_module._compute_CCV

        vcov_splits = 0.0
        for _ in range(n_splits):
            vcov_ccv = _compute_CCV(
                fml=fml,
                Y=Y,
                X=X,
                W=W,
                rng=rng,
                data=data,
                treatment=treatment,
                cluster_vec=cluster_vec,
                pk=pk,
                tau_full=tau_full,
            )
            vcov_splits += vcov_ccv

        vcov_splits /= n_splits
        vcov_splits /= N

        crv1_idx = self._coefnames.index(treatment)
        vcov_crv1 = self._vcov[crv1_idx, crv1_idx]
        vcov_ccv = qk * vcov_splits + (1 - qk) * vcov_crv1

        se = np.sqrt(vcov_ccv)
        tstat = tau_full / se
        df = G - 1
        pvalue = 2 * (1 - t.cdf(np.abs(tstat), df))
        alpha = 0.95
        z = np.abs(t.ppf((1 - alpha) / 2, df))
        z_se = z * se
        conf_int = np.array([tau_full - z_se, tau_full + z_se])

        res_ccv_dict: dict[str, Union[float, np.ndarray]] = {
            "Estimate": tau_full,
            "Std. Error": se,
            "t value": tstat,
            "Pr(>|t|)": pvalue,
            "2.5%": conf_int[0],
            "97.5%": conf_int[1],
        }

        res_ccv = pd.Series(res_ccv_dict)

        res_ccv.name = "CCV"

        res_crv1 = self.tidy().xs(treatment)
        res_crv1.name = "CRV1"

        return pd.concat([res_ccv, res_crv1], axis=1).T

        ccv_module = import_module("pyfixest.estimation.ccv")
        _ccv = ccv_module._ccv

        return _ccv(
            data=data,
            depvar=depvar,
            treatment=treatment,
            cluster=cluster,
            xfml=xfml,
            seed=seed,
            pk=pk,
            qk=qk,
            n_splits=n_splits,
        )

    def _model_matrix_one_hot(
        self, output="numpy"
    ) -> tuple[np.ndarray, Union[np.ndarray, spmatrix], list[str]]:
        """
        Transform a model matrix with fixed effects into a one-hot encoded matrix.

        Parameters
        ----------
        output : str, optional
            The type of output. Defaults to "numpy", in which case the returned matrices
            Y and X are numpy arrays. If set to "sparse", the returned design matrix X will
            be sparse.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list[str]]
            A tuple with the dependent variable, the model matrix, and the column names.
        """
        if self._has_fixef:
            fml_linear, fixef = self._fml.split("|")
            fixef_vars = fixef.split("+")
            fixef_vars_C = [f"C({x})" for x in fixef_vars]
            fixef_fml = "+".join(fixef_vars_C)
            fml_dummies = f"{fml_linear} + {fixef_fml}"
            # output = "pandas" as Y, X need to be np.arrays for parallel processing
            # if output = "numpy", type of Y, X is not np.ndarray but a formulaic object
            # which cannot be pickled by joblib

            Y, X = Formula(fml_dummies).get_model_matrix(self._data, output=output)
            xnames = X.model_spec.column_names
            Y = Y.toarray().flatten() if output == "sparse" else Y.flatten()
            X = csc_matrix(X) if output == "sparse" else X

        else:
            Y = self._Y.flatten()
            X = self._X
            xnames = self._coefnames

        X = csc_matrix(X) if output == "sparse" else X

        return Y, X, xnames

    def decompose(
        self,
        param: str,
        type: decomposition_type = "gelbach",
        cluster: Optional[str] = None,
        combine_covariates: Optional[dict[str, list[str]]] = None,
        reps: int = 1000,
        seed: Optional[int] = None,
        nthreads: Optional[int] = None,
        agg_first: Optional[bool] = None,
        only_coef: bool = False,
        digits=4,
    ) -> pd.DataFrame:
        """
        Implement the Gelbach (2016) decomposition method for mediation analysis.

        Compares a short model `depvar on param` with the long model
        specified in the original feols() call.

        For details, take a look at
        "When do covariates matter?" by Gelbach (2016, JoLe). You can find
        an ungated version of the paper on SSRN under the following link:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737 .

        Parameters
        ----------
        param : str
            The name of the focal covariate whose effect is to be decomposed into direct
            and indirect components with respect to the rest of the right-hand side.
        type : str, optional
            The type of decomposition method to use. Defaults to "gelbach", which
            currently is the only supported option.
        cluster: Optional
            The name of the cluster variable. If None, uses the cluster variable
            from the model fit. Defaults to None.
        combine_covariates: Optional.
            A dictionary that specifies which covariates to combine into groups.
            See the example for how to use this argument. Defaults to None.
        reps : int, optional
            The number of bootstrap iterations to run. Defaults to 1000.
        seed : int, optional
            An integer to set the random seed. Defaults to None.
        nthreads : int, optional
            The number of threads to use for the bootstrap. Defaults to None.
            If None, uses all available threads minus one.
        agg_first : bool, optional
            If True, use the 'aggregate first' algorithm described in Gelbach (2016).
            Recommended in cases with many (potentially high-dimensional) covariates.
            False by default if the 'combine_covariates' argument is None, True otherwise.
        only_coef : bool, optional
            Indicates whether to compute inference for the decomposition. Defaults to False.
            If True, skips the inference step and only returns the decomposition results.
        digits : int, optional
            The number of digits to round the results to. Defaults to 4.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the decomposition results.

        Examples
        --------
        ```{python}
        import re
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        data = gelbach_data(nobs = 1000)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

        # simple decomposition
        res = fit.decompose(param = "x1")
        pf.make_table(res)

        # group covariates via "combine_covariates" argument
        res = fit.decompose(param = "x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]})
        pf.make_table(res)

        # group covariates via regex
        res = fit.decompose(param="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
        ```
        """
        _decompose_arg_check(
            type=type,
            has_weights=self._has_weights,
            is_iv=self._is_iv,
            method=self._method,
        )

        nthreads_int = -1 if nthreads is None else nthreads

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        if agg_first is None:
            agg_first = combine_covariates is not None

        cluster_df: Optional[pd.Series] = None
        if cluster is not None:
            cluster_df = self._data[cluster]
        elif self._is_clustered:
            cluster_df = self._data[self._clustervar[0]]
        else:
            cluster_df = None

        Y, X, xnames = self._model_matrix_one_hot(output="sparse")

        if combine_covariates is not None:
            for key, value in combine_covariates.items():
                if isinstance(value, re.Pattern):
                    matched = [x for x in xnames if value.search(x)]
                    if len(matched) == 0:
                        raise ValueError(f"No covariates match the regex {value}.")
                    combine_covariates[key] = matched

        med = GelbachDecomposition(
            param=param,
            coefnames=xnames,
            cluster_df=cluster_df,
            nthreads=nthreads_int,
            combine_covariates=combine_covariates,
            agg_first=agg_first,
            only_coef=only_coef,
            atol=1e-12,
            btol=1e-12,
        )

        med.fit(
            X=X,
            Y=Y,
        )

        if not only_coef:
            med.bootstrap(rng=rng, B=reps)

        med.summary(digits=digits)

        self.GelbachDecompositionResults = med

        return med.summary_table.T

    def fixef(
        self, atol: float = 1e-06, btol: float = 1e-06
    ) -> dict[str, dict[str, float]]:
        """
        Compute the coefficients of (swept out) fixed effects for a regression model.

        This method creates the following attributes:
        - `alphaDF` (pd.DataFrame): A DataFrame with the estimated fixed effects.
        - `sumFE` (np.array): An array with the sum of fixed effects for each
        observation (i = 1, ..., N).

        Returns
        -------
        None
        """
        _has_fixef = self._has_fixef
        _is_iv = self._is_iv
        _method = self._method
        _fml = self._fml
        _data = self._data
        _weights_sqrt = np.sqrt(self._weights).flatten()

        blocked_transforms = ["i(", "^", "poly("]
        for bt in blocked_transforms:
            if bt in _fml:
                raise NotImplementedError(
                    f"The fixef() method is currently not supported for models with '{bt}' transformations."
                )

        if not _has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        if _is_iv:
            raise NotImplementedError(
                "The fixef() method is currently not supported for IV models."
            )

        # fixef_vars = self._fixef.split("+")[0]

        depvars, rhs = _fml.split("~")
        covars, fixef_vars = rhs.split("|")

        fixef_vars_list = fixef_vars.split("+")
        fixef_vars_C = [f"C({x})" for x in fixef_vars_list]
        fixef_fml = "+".join(fixef_vars_C)

        fml_linear = f"{depvars} ~ {covars}"
        Y, X = Formula(fml_linear).get_model_matrix(
            _data, output="pandas", context=self._context
        )
        if self._X_is_empty:
            Y = Y.to_numpy()
            uhat = Y.flatten()

        else:
            X = X[self._coefnames]  # drop intercept, potentially multicollinear vars
            Y = Y.to_numpy().flatten().astype(np.float64)
            X = X.to_numpy()
            uhat = (Y - X @ self._beta_hat).flatten()

        D2 = Formula("-1+" + fixef_fml).get_model_matrix(_data, output="sparse")
        cols = D2.model_spec.column_names

        if self._has_weights:
            uhat *= _weights_sqrt
            weights_diag = diags(_weights_sqrt, 0)
            D2 = weights_diag.dot(D2)

        alpha = lsqr(D2, uhat, atol=atol, btol=btol)[0]

        res: dict[str, dict[str, float]] = {}
        for i, col in enumerate(cols):
            variable, level = _extract_variable_level(col)
            # check if res already has a key variable
            if variable not in res:
                res[variable] = dict()
                res[variable][level] = alpha[i]
                continue
            else:
                if level not in res[variable]:
                    res[variable][level] = alpha[i]

        self._fixef_dict = res
        self._alpha = alpha
        self._sumFE = D2.dot(alpha)

        return self._fixef_dict

    def predict(
        self,
        newdata: Optional[DataFrameType] = None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: PredictionType = "link",
        se_fit: Optional[bool] = False,
        interval: Optional[PredictionErrorOptions] = None,
        alpha: float = 0.05,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict values of the model on new data.

        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations will be set to NaN.

        Parameters
        ----------
        newdata : Optional[DataFrameType], optional
            A narwhals compatible DataFrame (polars, pandas, duckdb, etc).
            If None (default), the data used for fitting the model is used.
        type : str, optional
            The type of prediction to be computed.
            Can be either "response" (default) or "link". For linear models, both are
            identical.
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        type:
            The type of prediction to be made. Can be either 'link' or 'response'.
             Defaults to 'link'. 'link' and 'response' lead
            to identical results for linear models.
        se_fit: Optional[bool], optional
            If True, the standard error of the prediction is computed. Only feasible
            for models without fixed effects. GLMs are not supported. Defaults to False.
        interval: str, optional
            The type of interval to compute. Can be either 'prediction' or None.
        alpha: float, optional
            The alpha level for the confidence interval. Defaults to 0.05. Only
            used if interval = "prediction" is not None.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            Returns a pd.Dataframe with columns "fit", "se_fit" and CIs if argument "interval=prediction".
            Otherwise, returns a np.ndarray with the predicted values of the model or the prediction
            standard errors if argument "se_fit=True".
        """
        if self._is_iv:
            raise NotImplementedError(
                "The predict() method is currently not supported for IV models."
            )

        if interval == "prediction" or se_fit:
            if self._has_fixef:
                raise NotImplementedError(
                    "Prediction errors are currently not supported for models with fixed effects."
                )

            if self._has_weights:
                raise NotImplementedError(
                    "Prediction errors are currently not supported for models with weights."
                )

        _validate_literal_argument(type, PredictionType)
        if interval is not None:
            _validate_literal_argument(interval, PredictionErrorOptions)

        if newdata is None:
            # note: no need to worry about fixed effects, as not supported with
            # prediction errors; will throw error later;
            # divide by sqrt(weights) as self._X is "weighted"

            X = self._X
            X_index = np.arange(self._N)

            yhat = (
                self._Y_hat_link
                if type == "link" or self._method == "feols"
                else self._Y_hat_response
            )
            if not se_fit and interval != "prediction":
                return yhat
            else:
                prediction_df = _compute_prediction_error(
                    model=self,
                    nobs=self._N,
                    yhat=yhat,
                    X=X,
                    X_index=X_index,
                    alpha=alpha,
                )

                if interval == "prediction":
                    return prediction_df
                else:
                    return prediction_df["se_fit"].to_numpy()

        else:
            y_hat, X, X_index = get_design_matrix_and_yhat(
                model=self,
                newdata=newdata if newdata is not None else None,
                context=self._context,
            )

            y_hat += _get_fixed_effects_prediction_component(
                model=self, newdata=newdata, atol=atol, btol=btol
            )

            if not se_fit and interval != "prediction":
                return y_hat
            else:
                prediction_df = _compute_prediction_error(
                    model=self,
                    nobs=newdata.shape[0],
                    yhat=y_hat,
                    X=X,
                    X_index=X_index,
                    alpha=alpha,
                )

                if interval == "prediction":
                    return prediction_df
                else:
                    return prediction_df["se_fit"].to_numpy()

    def get_performance(self) -> None:
        """
        Get Goodness-of-Fit measures.

        Compute multiple additional measures commonly reported with linear
        regression output, including R-squared and adjusted R-squared. Note that
        variables with the suffix _within use demeaned dependent variables Y,
        while variables without do not or are invariant to demeaning.

        Returns
        -------
        None

        Creates the following instances:
        - r2 (float): R-squared of the regression model.
        - adj_r2 (float): Adjusted R-squared of the regression model.
        - r2_within (float): R-squared of the regression model, computed on
        demeaned dependent variable.
        - adj_r2_within (float): Adjusted R-squared of the regression model,
        computed on demeaned dependent variable.
        """
        _Y_within = self._Y
        _Y = self._Y_untransformed.to_numpy()

        _u_hat = self._u_hat
        _N = self._N
        _k = self._k
        _has_intercept = not self._drop_intercept
        _has_fixef = self._has_fixef
        _weights = self._weights

        if _has_fixef:
            _k_fe = np.sum(self._k_fe - 1) + 1
            _adj_factor = (_N - _has_intercept) / (_N - _k - _k_fe)
            _adj_factor_within = (_N - _k_fe) / (_N - _k - _k_fe)
        else:
            _adj_factor = (_N - _has_intercept) / (_N - _k)

        ssu = np.sum(_u_hat**2)
        ssy = np.sum(_weights * (_Y - np.average(_Y, weights=_weights)) ** 2)
        self._rmse = np.sqrt(ssu / _N)
        self._r2 = 1 - (ssu / ssy)
        self._adj_r2 = 1 - (ssu / ssy) * _adj_factor

        if _has_fixef:
            ssy_within = np.sum(_Y_within**2)
            self._r2_within = 1 - (ssu / ssy_within)
            self._adj_r2_within = 1 - (ssu / ssy_within) * _adj_factor_within

    def tidy(
        self,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Tidy model outputs.

        Return a tidy pd.DataFrame with the point estimates, standard errors,
        t-statistics, and p-values.

        Parameters
        ----------
        alpha: Optional[float]
            The significance level for the confidence intervals. If None,
            computes a 95% confidence interval (`alpha = 0.05`).

        Returns
        -------
        tidy_df : pd.DataFrame
            A tidy pd.DataFrame containing the regression results, including point
            estimates, standard errors, t-statistics, and p-values.
        """
        ub, lb = 1 - alpha / 2, alpha / 2
        try:
            self.get_inference(alpha=alpha)
        except EmptyVcovError:
            warnings.warn(
                "Empty variance-covariance matrix detected",
                UserWarning,
            )

        tidy_df = pd.DataFrame(
            {
                "Coefficient": self._coefnames,
                "Estimate": self._beta_hat,
                "Std. Error": self._se,
                "t value": self._tstat,
                "Pr(>|t|)": self._pvalue,
                # use slice because self._conf_int might be empty
                f"{lb * 100:.1f}%": self._conf_int[:1].flatten(),
                f"{ub * 100:.1f}%": self._conf_int[1:2].flatten(),
            }
        )

        return tidy_df.set_index("Coefficient")

    def coef(self) -> pd.Series:
        """
        Fitted model coefficents.

        Returns
        -------
        pd.Series
            A pd.Series with the estimated coefficients of the regression model.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Fitted model standard errors.

        Returns
        -------
        pd.Series
            A pd.Series with the standard errors of the estimated regression model.
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Fitted model t-statistics.

        Returns
        -------
        pd.Series
            A pd.Series with t-statistics of the estimated regression model.
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Fitted model p-values.

        Returns
        -------
        pd.Series
            A pd.Series with p-values of the estimated regression model.
        """
        return self.tidy()["Pr(>|t|)"]

    def confint(
        self,
        alpha: float = 0.05,
        keep: Optional[Union[list, str]] = None,
        drop: Optional[Union[list, str]] = None,
        exact_match: Optional[bool] = False,
        joint: bool = False,
        seed: Optional[int] = None,
        reps: int = 10_000,
    ) -> pd.DataFrame:
        r"""
        Fitted model confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05.
            keep: str or list of str, optional
        joint : bool, optional
            Whether to compute simultaneous confidence interval for joint null
            of parameters selected by `keep` and `drop`. Defaults to False. See
            https://www.causalml-book.org/assets/chapters/CausalML_chap_4.pdf,
            Remark 4.4.1 for details.
        keep: str or list of str, optional
            The pattern for retaining coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Default is keeping all coefficients.
            You should use regular expressions to select coefficients.
                "age",            # would keep all coefficients containing age
                r"^tr",           # would keep all coefficients starting with tr
                r"\\d$",          # would keep all coefficients ending with number
            Output will be in the order of the patterns.
        drop: str or list of str, optional
            The pattern for excluding coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
            Default is keeping all coefficients. Parameter `keep` and `drop` can be
            used simultaneously.
        exact_match: bool, optional
            Whether to use exact match for `keep` and `drop`. Default is False.
            If True, the pattern will be matched exactly to the coefficient name
            instead of using regular expressions.
        reps : int, optional
            The number of bootstrap iterations to run for joint confidence intervals.
            Defaults to 10_000. Only used if `joint` is True.
        seed : int, optional
            The seed for the random number generator. Defaults to None. Only used if
            `joint` is True.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame with confidence intervals of the estimated regression model
            for the selected coefficients.

        Examples
        --------
        ```{python}
        #| echo: true
        #| results: asis
        #| include: true

        from pyfixest.utils import get_data
        from pyfixest.estimation import feols

        data = get_data()
        fit = feols("Y ~ C(f1)", data=data)
        fit.confint(alpha=0.10).head()
        fit.confint(alpha=0.10, joint=True, reps=9999).head()
        ```
        """
        if keep is None:
            keep = []
        if drop is None:
            drop = []

        tidy_df = self.tidy()
        if keep or drop:
            if isinstance(keep, str):
                keep = [keep]
            if isinstance(drop, str):
                drop = [drop]
            idxs = _select_order_coefs(tidy_df.index.tolist(), keep, drop, exact_match)
            coefnames = tidy_df.loc[idxs, :].index.tolist()
        else:
            coefnames = self._coefnames

        joint_indices = [i for i, x in enumerate(self._coefnames) if x in coefnames]
        if not joint_indices:
            raise ValueError("No coefficients match the keep/drop patterns.")

        if not joint:
            if self._method == "feols":
                crit_val = np.abs(t.ppf(alpha / 2, self._df_t))
            else:
                crit_val = np.abs(norm.ppf(alpha / 2))
        else:
            D_inv = 1 / self._se[joint_indices]
            V = self._vcov[np.ix_(joint_indices, joint_indices)]
            C_coefs = (D_inv * V).T * D_inv
            crit_val = simultaneous_crit_val(C_coefs, reps, alpha=alpha, seed=seed)

        ub = pd.Series(
            self._beta_hat[joint_indices] + crit_val * self._se[joint_indices]
        )
        lb = pd.Series(
            self._beta_hat[joint_indices] - crit_val * self._se[joint_indices]
        )

        df = pd.DataFrame(
            {
                f"{alpha / 2 * 100:.1f}%": lb,
                f"{(1 - alpha / 2) * 100:.1f}%": ub,
            }
        )
        # df = pd.DataFrame({f"{alpha / 2}%": lb, f"{1-alpha / 2}%": ub})
        df.index = coefnames

        return df

    def resid(self) -> np.ndarray:
        """
        Fitted model residuals.

        Returns
        -------
        np.ndarray
            A np.ndarray with the residuals of the estimated regression model.
        """
        return self._u_hat.flatten() / np.sqrt(self._weights.flatten())

    def ritest(
        self,
        resampvar: str,
        cluster: Optional[str] = None,
        reps: int = 100,
        type: str = "randomization-c",
        rng: Optional[np.random.Generator] = None,
        choose_algorithm: str = "auto",
        store_ritest_statistics: bool = False,
        level: float = 0.95,
    ) -> pd.Series:
        """
        Conduct Randomization Inference (RI) test against a null hypothesis of
        `resampvar = 0`.

        Parameters
        ----------
        resampvar : str
            The name of the variable to be resampled.
        cluster : str, optional
            The name of the cluster variable in case of cluster random assignment.
            If provided, `resampvar` is held constant within each `cluster`.
            Defaults to None.
        reps : int, optional
            The number of randomization iterations. Defaults to 100.
        type: str
            The type of the randomization inference test.
            Can be "randomization-c" or "randomization-t". Note that
            the "randomization-c" is much faster, while the
            "randomization-t" is recommended by Wu & Ding (JASA, 2021).
        rng : np.random.Generator, optional
            A random number generator. Defaults to None.
        choose_algorithm: str, optional
            The algorithm to use for the computation. Defaults to "auto".
            The alternative is "fast" and "slow", and should only be used
            for running CI tests. Ironically, this argument is not tested
            for any input errors from the user! So please don't use it =)
        include_plot: bool, optional
            Whether to include a plot of the distribution p-values. Defaults to False.
        store_ritest_statistics: bool, optional
            Whether to store the simulated statistics of the RI procedure.
            Defaults to False. If True, stores the simulated statistics
            in the model object via the `ritest_statistics` attribute as a
            numpy array.
        level: float, optional
            The level for the confidence interval of the randomization inference
            p-value. Defaults to 0.95.

        Returns
        -------
        A pd.Series with the regression coefficient of `resampvar` and the p-value
        of the RI test. Additionally, reports the standard error and the confidence
        interval of the p-value.

        Examples
        --------
        ```{python}

        #| echo: true
        #| results: asis
        #| include: true

        import pyfixest as pf
        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data)

        # Conduct a randomization inference test for the coefficient of X1
        fit.ritest("X1", reps=1000)

        # use randomization-t instead of randomization-c
        fit.ritest("X1", reps=1000, type="randomization-t")

        # store statistics for plotting
        fit.ritest("X1", reps=1000, store_ritest_statistics=True)
        ```
        """
        _fml = self._fml
        _data = self._data
        _method = self._method
        _is_iv = self._is_iv
        _coefnames = self._coefnames
        _has_fixef = self._has_fixef

        resampvar = resampvar.replace(" ", "")
        resampvar_, h0_value, hypothesis, test_type = _decode_resampvar(resampvar)

        if _is_iv:
            raise NotImplementedError(
                "Randomization Inference is not supported for IV models."
            )

        # check that resampvar in _coefnames
        if resampvar_ not in _coefnames:
            raise ValueError(f"{resampvar_} not found in the model's coefficients.")

        if cluster is not None and cluster not in _data:
            raise ValueError(f"The variable {cluster} is not found in the data.")

        clustervar_arr = _data[cluster].to_numpy().reshape(-1, 1) if cluster else None

        if clustervar_arr is not None and np.any(np.isnan(clustervar_arr)):
            raise ValueError(
                """
            The cluster variable contains missing values. This is not allowed
            for randomization inference via `ritest()`.
            """
            )

        # update vcov if cluster provided but not in model
        if cluster is not None and not self._is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object."
            )
            self.vcov({"CRV1": cluster})

        rng = np.random.default_rng() if rng is None else rng

        sample_coef = np.array(self.coef().xs(resampvar_))
        sample_tstat = np.array(self.tstat().xs(resampvar_))
        sample_stat = sample_tstat if type == "randomization-t" else sample_coef

        if type not in ["randomization-t", "randomization-c"]:
            raise ValueError("type must be 'randomization-t' or 'randomization-c.")

        # always run slow algorithm for randomization-t
        choose_algorithm = "slow" if type == "randomization-t" else choose_algorithm

        assert isinstance(reps, int) and reps > 0, "reps must be a positive integer."

        if self._has_weights:
            raise NotImplementedError(
                """
                Regression Weights are not supported with Randomization Inference.
                """
            )

        if choose_algorithm == "slow" or _method == "fepois":
            vcov_input: Union[str, dict[str, str]]
            if cluster is not None:
                vcov_input = {"CRV1": cluster}
            else:
                # "iid" for models without controls, else HC1
                vcov_input = (
                    "hetero"
                    if (_has_fixef and len(_coefnames) > 1) or len(_coefnames) > 2
                    else "iid"
                )

            # for performance reasons
            if type == "randomization-c":
                vcov_input = "iid"

            ri_stats = _get_ritest_stats_slow(
                data=_data,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                fml=_fml,
                reps=reps,
                vcov=vcov_input,
                type=type,
                rng=rng,
                model=_method,
            )

        else:
            _Y = self._Y
            _X = self._X
            _coefnames = self._coefnames

            _weights = self._weights.flatten()
            _data = self._data
            _fval_df = _data[self._fixef.split("+")] if _has_fixef else None

            _D = self._data[resampvar_].to_numpy()

            ri_stats = _get_ritest_stats_fast(
                Y=_Y,
                X=_X,
                D=_D,
                coefnames=_coefnames,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                reps=reps,
                rng=rng,
                fval_df=_fval_df,
                weights=_weights,
            )

        ri_pvalue, se_pvalue, ci_pvalue = _get_ritest_pvalue(
            sample_stat=sample_stat,
            ri_stats=ri_stats[1:],
            method=test_type,
            h0_value=h0_value,
            level=level,
        )

        if store_ritest_statistics:
            self._ritest_statistics = ri_stats
            self._ritest_pvalue = ri_pvalue
            self._ritest_sample_stat = sample_stat - h0_value

        res = pd.Series(
            {
                "H0": hypothesis,
                "ri-type": type,
                "Estimate": sample_coef,
                "Pr(>|t|)": ri_pvalue,
                "Std. Error (Pr(>|t|))": se_pvalue,
            }
        )

        alpha = 1 - level
        ci_lower_name = str(f"{alpha / 2 * 100:.1f}% (Pr(>|t|))")
        ci_upper_name = str(f"{(1 - alpha / 2) * 100:.1f}% (Pr(>|t|))")
        res[ci_lower_name] = ci_pvalue[0]
        res[ci_upper_name] = ci_pvalue[1]

        if cluster is not None:
            res["Cluster"] = cluster

        return res

    def plot_ritest(self, plot_backend="lets_plot"):
        """
        Plot the distribution of the Randomization Inference Statistics.

        Parameters
        ----------
        plot_backend : str, optional
            The plotting backend to use. Defaults to "lets_plot". Alternatively,
            "matplotlib" is available.

        Returns
        -------
        A lets_plot or matplotlib figure with the distribution of the Randomization
        Inference Statistics.
        """
        if not hasattr(self, "_ritest_statistics"):
            raise ValueError(
                """
                            The randomization inference statistics have not been stored
                            in the model object. Please set `store_ritest_statistics=True`
                            when calling `ritest()`
                            """
            )

        ri_stats = self._ritest_statistics
        sample_stat = self._ritest_sample_stat

        return _plot_ritest_pvalue(
            ri_stats=ri_stats, sample_stat=sample_stat, plot_backend=plot_backend
        )

    def update(
        self, X_new: np.ndarray, y_new: np.ndarray, inplace: bool = False
    ) -> np.ndarray:
        """
        Update coefficients for new observations using Sherman-Morrison formula.

        Parameters
        ----------
            X : np.ndarray
                Covariates for new data points. Users expected to ensure conformability
                with existing data.
            y : np.ndarray
                Outcome values for new data points
            inplace : bool, optional
                Whether to update the model object in place. Defaults to False.

        Returns
        -------
        np.ndarray
            Updated coefficients
        """
        if self._has_fixef:
            raise NotImplementedError(
                "The update() method is currently not supported for models with fixed effects."
            )
        if not np.all(X_new[:, 0] == 1):
            X_new = np.column_stack((np.ones(len(X_new)), X_new))
        X_n_plus_1 = np.vstack((self._X, X_new))
        epsi_n_plus_1 = y_new - X_new @ self._beta_hat
        gamma_n_plus_1 = np.linalg.inv(X_n_plus_1.T @ X_n_plus_1) @ X_new.T
        beta_n_plus_1 = self._beta_hat + gamma_n_plus_1 @ epsi_n_plus_1
        if inplace:
            self._X = X_n_plus_1
            self._Y = np.append(self._Y, y_new)
            self._beta_hat = beta_n_plus_1
            self._u_hat = self._Y - self._X @ self._beta_hat
            self._N += X_new.shape[0]

        return beta_n_plus_1


def _feols_input_checks(Y: np.ndarray, X: np.ndarray, weights: np.ndarray):
    """
    Perform basic checks on the input matrices Y and X for the FEOLS.

    Parameters
    ----------
    Y : np.ndarray
        FEOLS input matrix Y.
    X : np.ndarray
        FEOLS input matrix X.
    weights : np.ndarray
        FEOLS weights.

    Returns
    -------
    None
    """
    if not isinstance(Y, (np.ndarray)):
        raise TypeError("Y must be a numpy array.")
    if not isinstance(X, (np.ndarray)):
        raise TypeError("X must be a numpy array.")
    if not isinstance(weights, (np.ndarray)):
        raise TypeError("weights must be a numpy array.")

    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")


def _get_vcov_type(vcov: str, fval: str):
    """
    Get variance-covariance matrix type.

    Passes the specified vcov type and sets the default vcov type based on the
    inclusion of fixed effects in the model.

    Parameters
    ----------
    vcov : str
        The specified vcov type.
    fval : str
        The specified fixed effects, formatted as a string (e.g., "X1+X2").

    Returns
    -------
    vcov_type : str
        The specified or default vcov type. Defaults to 'iid' if no fixed effect
        is included in the model, and 'CRV1' clustered by the first fixed effect
        if a fixed effect is included.
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


def _drop_multicollinear_variables(
    X: np.ndarray,
    names: list[str],
    collin_tol: float,
    backend_func: Callable,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Check for multicollinearity in the design matrices X and Z.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.
    backend_func: Callable
        Which backend function to use for the multicollinearity check.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    names : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : numpy.ndarray
        Logical array, where True indicates that the variable is collinear.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = X.T @ X
    id_excl, n_excl, all_removed = backend_func(tXX, collin_tol)

    collin_vars = []
    collin_index = []

    if all_removed:
        raise ValueError(
            """
            All variables are collinear. Maybe your model specification introduces multicollinearity? If not, please reach out to the package authors!.
            """
        )

    names_array = np.array(names)
    if n_excl > 0:
        collin_vars = names_array[id_excl].tolist()
        if len(collin_vars) > 5:
            indent = "    "
            formatted_collinear_vars = (
                f"\n{indent}" + f"\n{indent}".join(collin_vars[:5]) + f"\n{indent}..."
            )
        else:
            formatted_collinear_vars = str(collin_vars)

        warnings.warn(
            f"""
            {len(collin_vars)} variables dropped due to multicollinearity.
            The following variables are dropped: {formatted_collinear_vars}.
            """
        )

        X = np.delete(X, id_excl, axis=1)
        if X.ndim == 2 and X.shape[1] == 0:
            raise ValueError(
                """
                All variables are collinear. Please check your model specification.
                """
            )

        names_array = np.delete(names_array, id_excl)
        collin_index = id_excl.tolist()

    return X, list(names_array), collin_vars, collin_index


def _check_vcov_input(vcov: Union[str, dict[str, str]], data: pd.DataFrame):
    """
    Check the input for the vcov argument in the Feols class.

    Parameters
    ----------
    vcov : Union[str, dict[str, str]]
        The vcov argument passed to the Feols class.
    data : pd.DataFrame
        The data passed to the Feols class.

    Returns
    -------
    None
    """
    assert isinstance(vcov, (dict, str, list)), "vcov must be a dict, string or list"
    if isinstance(vcov, dict):
        assert next(iter(vcov.keys())) in [
            "CRV1",
            "CRV3",
        ], "vcov dict key must be CRV1 or CRV3"
        assert isinstance(next(iter(vcov.values())), str), (
            "vcov dict value must be a string"
        )
        deparse_vcov = next(iter(vcov.values())).split("+")
        assert len(deparse_vcov) <= 2, "not more than twoway clustering is supported"

    if isinstance(vcov, list):
        assert all(isinstance(v, str) for v in vcov), "vcov list must contain strings"
        assert all(v in data.columns for v in vcov), (
            "vcov list must contain columns in the data"
        )
    if isinstance(vcov, str):
        assert vcov in [
            "iid",
            "hetero",
            "HC1",
            "HC2",
            "HC3",
            "nid",
        ], "vcov string must be iid, hetero, HC1, HC2, or HC3"


def _deparse_vcov_input(vcov: Union[str, dict[str, str]], has_fixef: bool, is_iv: bool):
    """
    Deparse the vcov argument passed to the Feols class.

    Parameters
    ----------
    vcov : Union[str, dict[str, str]]
        The vcov argument passed to the Feols class.
    has_fixef : bool
        Whether the regression has fixed effects.
    is_iv : bool
        Whether the regression is an IV regression.

    Returns
    -------
    vcov_type : str
        The type of vcov to be used. Either "iid", "hetero", or "CRV".
    vcov_type_detail : str or list
        The type of vcov to be used, with more detail. Options include "iid",
        "hetero", "HC1", "HC2", "HC3", "CRV1", or "CRV3".
    is_clustered : bool
        Indicates whether the vcov is clustered.
    clustervar : str
        The name of the cluster variable.
    """
    if isinstance(vcov, dict):
        vcov_type_detail = next(iter(vcov.keys()))
        deparse_vcov = next(iter(vcov.values())).split("+")
        if isinstance(deparse_vcov, str):
            deparse_vcov = [deparse_vcov]
        deparse_vcov = [x.replace(" ", "") for x in deparse_vcov]
    elif isinstance(vcov, (list, str)):
        vcov_type_detail = vcov
    else:
        raise TypeError("arg vcov needs to be a dict, string or list")

    if vcov_type_detail == "iid":
        vcov_type = "iid"
        is_clustered = False
    elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
        vcov_type = "hetero"
        is_clustered = False
        if vcov_type_detail in ["HC2", "HC3"]:
            if has_fixef:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for regressions with fixed effects."
                )
            if is_iv:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for IV regressions."
                )
    elif vcov_type_detail in ["CRV1", "CRV3"]:
        vcov_type = "CRV"
        is_clustered = True

    elif vcov_type_detail == "nid":
        vcov_type = "nid"
        is_clustered = False

    clustervar = deparse_vcov if is_clustered else None

    # loop over clustervar to change "^" to "_"
    if clustervar and "^" in clustervar:
        clustervar = [x.replace("^", "_") for x in clustervar]
        warnings.warn(
            f"""
            The '^' character in the cluster variable name is replaced by '_'.
            In consequence, the clustering variable(s) is (are) named {clustervar}.
            """
        )

    return vcov_type, vcov_type_detail, is_clustered, clustervar
