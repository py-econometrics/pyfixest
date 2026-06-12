from __future__ import annotations

import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner
from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.formula import model_matrix as model_matrix_fixest
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanCache
from pyfixest.estimation.internals.fit_ import (
    _drop_multicollinear_variables,
    fit_ols,
)
from pyfixest.estimation.internals.literals import (
    PredictionErrorOptions,
    PredictionType,
    SolverOptions,
)
from pyfixest.estimation.internals.vcov_ import (
    _jackknife_vcov,
    vcov_crv1,
    vcov_crv3_fast,
    vcov_hac,
    vcov_hetero,
    vcov_iid,
)
from pyfixest.estimation.internals.vcov_utils import (
    _check_cluster_df,
    _compute_bread,
    _count_G_for_ssc_correction,
    _get_cluster_df,
    _prepare_twoway_clustering,
)
from pyfixest.estimation.models._result_accessor_mixin import ResultAccessorMixin
from pyfixest.estimation.post_estimation.ccv import _ccv_impl
from pyfixest.estimation.post_estimation.decomposition import (
    GelbachDecomposition,
    _decompose_impl,
)
from pyfixest.estimation.post_estimation.fixef import _fixef_impl
from pyfixest.estimation.post_estimation.prediction import _predict_impl
from pyfixest.estimation.post_estimation.ritest import (
    _plot_ritest_impl,
    _ritest_impl,
)
from pyfixest.estimation.post_estimation.wald import _wald_test_impl
from pyfixest.estimation.post_estimation.wild_bootstrap import _wildboottest_impl
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _narwhals_to_pandas,
)
from pyfixest.utils.utils import (
    capture_context,
    get_ssc,
)

decomposition_type = Literal["gelbach"]
prediction_type = Literal["response", "link"]


class Feols(ResultAccessorMixin):
    """
    Non user-facing class to estimate a linear regression via OLS.

    Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.api.feols.feols.qmd) function. Note that
    no demeaning is performed in this class: demeaning is performed in the
    FixestMulti class (to allow for caching of demeaned variables for multiple
    estimation).

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
        "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr".
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

    _Y_df : pd.DataFrame
        The dependent variable model matrix (raw, never mutated).
    _X_df : pd.DataFrame
        The independent variable model matrix (raw, never mutated).
    _Y_demeaned : np.ndarray
        The demeaned dependent variable, unweighted, set in demean().
    _X_demeaned : np.ndarray
        The demeaned, collinearity-pruned independent variables, unweighted.
    _Y_wls : np.ndarray
        sqrt(weights)-scaled demeaned dependent variable of the final solve.
    _X_wls : np.ndarray
        sqrt(weights)-scaled demeaned design matrix of the final solve.
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
    _solver: str
        The solver used for the regression.
    _weights : np.ndarray
        Array of user weights for each observation; never overwritten.
    _weights_irls : np.ndarray or None
        Final IRLS weights, set by GLM models only; None for OLS/IV.
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
        Response-scale residuals of the regression model.
    _u_hat_wls : np.ndarray
        Solve-scale residuals (sqrt(weights)-scaled); equal to _u_hat
        for unweighted models.
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
    _alpha : pd.DataFrame
        A DataFrame with the estimated fixed effects.
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
        "scipy.sparse.linalg.lsqr"],
        default is "scipy.linalg.solve". Solver to use for the estimation.
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
        ssc_dict: dict[str, str | bool],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: str | None,
        weights_type: str | None,
        collin_tol: float,
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
        solver: SolverOptions = "np.linalg.solve",
        demeaner: AnyDemeaner | None = None,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: int | Mapping[str, Any] = 0,
        sample_split_var: str | None = None,
        sample_split_value: str | int | float | _AllSampleSentinel | None = None,
    ) -> None:
        self._sample_split_value = sample_split_value
        self._sample_split_var = sample_split_var
        self._model_name = (
            FixestFormula.formula
            if self._sample_split_var is None
            else f"{FixestFormula.formula} (Sample: {self._sample_split_var} = {self._sample_split_value})"
        )
        self._model_name_plot = self._model_name
        self._method = "feols"
        self._is_iv = False
        self.FixestFormula = FixestFormula

        if self._sample_split_var is None:
            pass
        elif self._sample_split_value is _ALL_SAMPLE:
            data = data.loc[data[sample_split_var].notnull()]
        else:
            data = data.loc[data[self._sample_split_var] == sample_split_value]

        data = data.reset_index(drop=True)

        self._data = data.copy() if copy_data else data
        self._ssc_dict = ssc_dict
        self._drop_singletons = drop_singletons
        self._drop_intercept = drop_intercept
        self._weights_name = weights
        self._weights_type = weights_type
        self._has_weights = weights is not None
        self._offset_name: str | None = None
        self._offset: np.ndarray | None = None
        self._collin_tol = collin_tol
        self._solver = solver
        if demeaner is None:
            demeaner = MapDemeaner()
        self._demeaner = demeaner
        if isinstance(demeaner, LsmrDemeaner):
            self._fixef_tol = max(demeaner.fixef_atol, demeaner.fixef_btol)
        else:
            self._fixef_tol = demeaner.fixef_tol
        self._fixef_maxiter = demeaner.fixef_maxiter
        self._demean_cache = DemeanCache(lookup_demeaned_data)
        self._store_data = store_data
        self._copy_data = copy_data
        self._lean = lean
        self._use_mundlak = False
        self._context = capture_context(context)

        self._support_crv3_inference = True
        self._support_hac_inference = True
        if self._weights_name is not None:
            self._supports_wildboottest = False
        self._supports_wildboottest = True
        self._supports_cluster_causal_variance = True
        if self._has_weights or self._is_iv:
            self._supports_wildboottest = False
        self._support_decomposition = True

        # attributes that have to be enriched outside of the class -
        # not really optimal code change later
        self._fml = FixestFormula.formula
        self._has_fixef = False
        self._fixef = FixestFormula.fixed_effects
        # self._coefnames = None
        self._icovars = None

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
        self._u_hat_wls = np.array([])
        self._scores = np.array([])
        self._hessian = np.array([])
        self._bread = np.array([])
        # final IRLS weights; set by GLM models, None for OLS/IV
        self._weights_irls: np.ndarray | None = None

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

        # set in fixef()
        self._fixef_dict: dict[str, dict[str, float]] = {}
        self._alpha = None
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
        self._res_cohort_eventtime_dict: dict[str, Any] | None = None
        self._yname: str | None = None
        self._gname: str | None = None
        self._tname: str | None = None
        self._idname: str | None = None
        self._att: bool | None = None

        # set functions inherited from other modules
        self._bind_report_methods()

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
        model_matrix = model_matrix_fixest.create_model_matrix(
            formula=self.FixestFormula,
            data=self._data,
            drop_singletons=self._drop_singletons,
            drop_intercept=self._drop_intercept,
            weights=self._weights_name,
            offset=self._offset_name,
            context=self._context,
        )

        self._Y_df = model_matrix.dependent
        self._X_df = model_matrix.independent
        self._fe = model_matrix.fixed_effects
        self._endogvar_df = model_matrix.endogenous
        self._Z_df = model_matrix.instruments
        self._weights_df = model_matrix.weights
        self._offset_df = model_matrix.offset
        self._na_index = model_matrix.na_index
        # TODO: set dynamically based on naming set in pyfixest.estimation.formula.factor_interaction._encode_i
        is_icovar = (
            self._X_df.columns.str.contains(r"^.+::.+$")
            if not self._X_df.empty
            else None
        )
        self._icovars = (
            self._X_df.columns[is_icovar].tolist()
            if is_icovar is not None and is_icovar.any()
            else None
        )
        self._X_is_empty = not model_matrix.independent.shape[0] > 0
        self._model_spec = model_matrix.model_spec

        self._coefnames = self._X_df.columns.tolist()
        self._coefnames_z = (
            self._Z_df.columns.tolist() if self._Z_df is not None else None
        )
        self._depvar = self._Y_df.columns[0]

        self._has_fixef = self._fe is not None
        self._fixef = self.FixestFormula.fixed_effects

        self._k_fe = self._fe.nunique(axis=0) if self._has_fixef else None
        self._n_fe = len(self._k_fe) if self._has_fixef else 0

        # update data
        self._data.drop(
            self._data.index[~self._data.index.isin(model_matrix.dependent.index)],
            inplace=True,
        )

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
        N_rows = len(self._Y_df)
        if self._weights_type == "aweights":
            N = N_rows
        elif self._weights_type == "fweights":
            N = np.sum(self._weights)

        return N, N_rows

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
        N = len(self._Y_df)

        if self._weights_df is not None:
            _weights = self._weights_df.to_numpy()
        else:
            _weights = np.ones(N)

        return _weights.reshape((N, 1))

    def demean(self):
        """Demean the dependent variable and covariates by the fixed effect(s).

        Sets ``_Y_demeaned`` / ``_X_demeaned``: unweighted numpy arrays. When
        the model has no fixed effects, these are plain array versions of
        ``_Y_df`` / ``_X_df``.
        """
        if self._has_fixef:
            Yd, Xd, _ = self._demean_cache.demean_yx(
                self._Y_df,
                self._X_df,
                self._fe,
                self._weights.flatten(),
                self._na_index,
                self._demeaner,
            )
        else:
            Yd, Xd = self._Y_df, self._X_df

        self._Y_demeaned = Yd.to_numpy()
        self._X_demeaned = Xd.to_numpy()

    @property
    def preconditioner(self) -> Preconditioner | None:
        """The within preconditioner used during demeaning, if any.

        ``None`` when no preconditioner participated in the solve —
        ``preconditioner='off'``, single-FE designs (MAP fallback), or any
        non-within backend. Otherwise the instance built on the first solve.
        Pass it back via
        ``LsmrDemeaner(backend='within', preconditioner=...)`` to skip the
        setup phase on a later fit over the same design.
        """
        return self._demean_cache.preconditioner

    def drop_multicol_vars(self):
        "Detect and drop multicollinear variables."
        if self._X_demeaned.shape[1] > 0:
            (
                self._X_demeaned,
                self._coefnames,
                self._collin_vars,
                self._collin_index,
            ) = _drop_multicollinear_variables(
                self._X_demeaned,
                self._coefnames,
                self._collin_tol,
            )
        # update X_is_empty
        self._X_is_empty = self._X_demeaned.shape[1] == 0
        self._k = self._X_demeaned.shape[1] if not self._X_is_empty else 0

    def _get_predictors(self) -> None:
        self._Y_hat_link = self._Y_df.to_numpy().flatten() - self.resid()
        self._Y_hat_response = self._Y_hat_link

    @property
    def _solve_weights(self) -> np.ndarray:
        """Weights of the final least-squares solve.

        User weights for OLS/IV; final IRLS weights for GLM models.
        """
        return self._weights if self._weights_irls is None else self._weights_irls

    def get_fit(self) -> None:
        """
        Fit an OLS model.

        Returns
        -------
        None
        """
        self.demean()
        self.drop_multicol_vars()

        if self._X_is_empty:
            self._Y_wls = np.sqrt(self._weights) * self._Y_demeaned
            self._X_wls = self._X_demeaned
            self._u_hat = self._Y_demeaned
            self._u_hat_wls = self._Y_wls
        else:
            fit = fit_ols(
                X=self._X_demeaned,
                Y=self._Y_demeaned,
                weights=self._weights,
                solver=self._solver,
            )

            self._X_wls = fit.X_wls
            self._Y_wls = fit.Y_wls
            self._tZX = fit.tZX
            self._tZy = fit.tZy
            self._beta_hat = fit.beta
            self._u_hat = fit.residuals
            self._u_hat_wls = fit.residuals_wls
            self._scores = fit.scores
            self._hessian = fit.hessian

            # IV attributes, set to None for OLS, Poisson
            self._tXZ = np.array([])
            self._tZZinv = np.array([])

        self._get_predictors()

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
        data: DataFrameType | None = None,
    ) -> Feols:
        """
        Compute covariance matrices for an estimated regression model.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]]
            A string or dictionary specifying the type of variance-covariance matrix
            to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3", "NW", "DK".
            If a dictionary, it should have the format {"CRV1": "clustervar"} for
            CRV1 inference or {"CRV3": "clustervar"}
            for CRV3 inference. Note that CRV3 inference is currently not supported
            for IV estimation.
        vcov_kwargs : Optional[dict[str, any]]
             Additional keyword arguments for the variance-covariance matrix.
        data: Optional[DataFrameType], optional
            The data used for estimation. If None, tries to fetch the data from the
            model object. Defaults to None.


        Returns
        -------
        Feols
            An instance of class [Feols](/reference/estimation.models.feols_.Feols.qmd) with updated inference.
        """
        # Assuming `data` is the DataFrame in question

        data_to_check = data if data is not None else self._data
        try:
            data_to_check = _narwhals_to_pandas(data_to_check)
        except TypeError as e:
            raise TypeError(
                f"The data set must be a DataFrame type. Received: {type(data)}"
            ) from e

        # assign estimated fixed effects, and fixed effects nested within cluster.

        # deparse vcov input
        _check_vcov_input(vcov=vcov, vcov_kwargs=vcov_kwargs, data=self._data)

        (
            self._vcov_type,
            self._vcov_type_detail,
            self._is_clustered,
            self._clustervar,
        ) = _deparse_vcov_input(vcov, self._has_fixef, self._is_iv)

        self._bread = _compute_bread(
            self._is_iv, self._tXZ, self._tZZinv, self._tZX, self._hessian
        )

        # HAC attributes
        self._lag = vcov_kwargs.get("lag", None) if vcov_kwargs is not None else None
        self._time_id = (
            vcov_kwargs.get("time_id", None) if vcov_kwargs is not None else None
        )
        self._panel_id = (
            vcov_kwargs.get("panel_id", None) if vcov_kwargs is not None else None
        )
        self._is_sorted = (
            vcov_kwargs.get("is_sorted", None) if vcov_kwargs is not None else None
        )

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
            self._ssc, self._df_k, self._df_t = get_ssc(**all_kwargs)

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
            self._ssc, self._df_k, self._df_t = get_ssc(**all_kwargs)
            self._vcov = self._ssc * self._vcov_hetero()

        elif self._vcov_type == "HAC":
            ssc_kwargs_hac = {
                "k_fe_nested": 0,  # nesting ignored / irrelevant for HAC SEs
                "n_fe_fully_nested": 0,  # nesting ignored / irrelevant for HAC SEs
                "vcov_sign": 1,
                "vcov_type": "HAC",
                "G": np.unique(self._data[self._time_id]).shape[
                    0
                ],  # number of unique time periods T used
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_hac}
            self._ssc, self._df_k, self._df_t = get_ssc(**all_kwargs)

            self._vcov = self._ssc * self._vcov_hac()

        elif self._vcov_type == "nid":
            ssc_kwargs_hetero = {
                "k_fe_nested": 0,
                "n_fe_fully_nested": 0,
                "vcov_sign": 1,
                "vcov_type": "hetero",
                "G": self._N,
            }

            all_kwargs = {**ssc_kwargs, **ssc_kwargs_hetero}
            self._ssc, self._df_k, self._df_t = get_ssc(**all_kwargs)
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
            if self._fixef is not None and self._ssc_dict["k_fixef"] == "nonnested":
                k_fe_nested_flag, n_fe_fully_nested = count_fixef_fully_nested_all(
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
                ssc, df_k, df_t = get_ssc(**all_kwargs)

                self._ssc = np.array([ssc]) if x == 0 else np.append(self._ssc, ssc)
                self._df_k = df_k  # the same across all vcov's

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

                    if not self._support_crv3_inference:
                        raise VcovTypeNotSupportedError(
                            f"CRV3 inference is not for models of type '{self._method}'."
                        )

                    if (
                        (self._has_fixef is False)
                        and (self._method == "feols")
                        and (self._is_iv is False)
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
        return vcov_iid(residuals_wls=self._u_hat_wls, bread=self._bread, N=self._N)

    def _vcov_hetero(self):
        return vcov_hetero(
            scores=self._scores,
            X_wls=self._X_wls,
            tZX=self._tZX,
            weights=self._solve_weights,
            weights_type=self._weights_type,
            vcov_type_detail=self._vcov_type_detail,
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
        )

    def _vcov_hac(self):
        if not self._support_hac_inference:
            raise NotImplementedError(
                "HAC inference is not supported for this model type."
            )

        # some data checks on input pandas df
        # time needs to be numeric or date else we cannot sort by time
        if not np.issubdtype(
            self._data[self._time_id], np.number
        ) and not np.issubdtype(self._data[self._time_id], np.datetime64):
            raise ValueError(
                "The time variable must be numeric or date, else we cannot sort by time."
            )

        time_arr = self._data[self._time_id].to_numpy()
        panel_arr = (
            self._data[self._panel_id].to_numpy()
            if self._panel_id is not None
            else None
        )

        return vcov_hac(
            scores=self._scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            lag=self._lag,
            vcov_type_detail=self._vcov_type_detail,
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
            tZX=self._tZX,
        )

    def _vcov_nid(self):
        raise NotImplementedError(
            "Only models of type Quantreg support a variance-covariance matrix of type 'nid'."
        )

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):
        return vcov_crv1(
            scores=self._scores,
            clustid=clustid,
            cluster_col=cluster_col,
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
            tZX=self._tZX,
        )

    def _vcov_crv3_fast(self, clustid, cluster_col):
        return vcov_crv3_fast(
            X=self._X_wls,
            Y=self._Y_wls,
            beta_hat=self._beta_hat,
            clustid=clustid,
            cluster_col=cluster_col,
        )

    def _vcov_crv3_slow(self, clustid, cluster_col):
        beta_jack = np.zeros((len(clustid), self._k))

        # lazy loading to avoid circular import
        fixest_module = import_module("pyfixest.estimation")
        fit_ = fixest_module.feols if self._method == "feols" else fixest_module.fepois

        for ixg, g in enumerate(clustid):
            # direct leave one cluster out implementation
            data = self._data[~np.equal(g, cluster_col)]
            fit = fit_(
                fml=self._fml,
                data=data,
                vcov="iid",
                weights=self._weights_name,
                weights_type=self._weights_type,
            )
            beta_jack[ixg, :] = fit.coef().to_numpy()

        # optional: beta_bar in MNW (2022)
        # center = "estimate"
        # if center == 'estimate':
        #    beta_center = beta_hat
        # else:
        #    beta_center = np.mean(beta_jack, axis = 0)
        return _jackknife_vcov(beta_jack=beta_jack, beta_center=self._beta_hat)

    def add_fixest_multi_context(
        self,
        depvar: str,
        Y: pd.Series,
        _data: pd.DataFrame,
        _ssc_dict: dict[str, str | bool],
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
        self._fml = self.FixestFormula.formula
        self._depvar = depvar
        self._Y_df = Y
        self._data = pd.DataFrame()

        if store_data:
            self._data = _data

        self._ssc_dict = _ssc_dict
        self._k_fe = _k_fe  # type: ignore[assignment]
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
                "_X_df",
                "_Y_df",
                "_Z_df",
                "_endogvar_df",
                "_X_demeaned",
                "_Y_demeaned",
                "_Z_demeaned",
                "_endogvar_demeaned",
                "_X_wls",
                "_Y_wls",
                "_Z_wls",
                "_cluster_df",
                "_tXZ",
                "_tZy",
                "_tZX",
                "_weights",
                "_weights_irls",
                "_scores",
                "_tZZinv",
                "_u_hat",
                "_u_hat_wls",
                "_Y_hat_link",
                "_Y_hat_response",
            ]

        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    def wald_test(self, R=None, q=None, distribution="F"):
        """
        Conduct Wald test.

        Compute a Wald test for a linear hypothesis of the form R * beta = q.
        where R is m x k matrix, beta is a k x 1 vector of coefficients,
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
        fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False))

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
        return _wald_test_impl(model=self, R=R, q=q, distribution=distribution)

    def wildboottest(
        self,
        reps: int,
        cluster: str | None = None,
        param: str | None = None,
        weights_type: str | None = "rademacher",
        impose_null: bool | None = True,
        bootstrap_type: str | None = "11",
        seed: int | None = None,
        k_adj: bool | None = True,
        G_adj: bool | None = True,
        parallel: bool | None = False,
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
        k_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of observations and covariates. Defaults to True.
        G_adj : bool, optional
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
        return _wildboottest_impl(
            model=self,
            reps=reps,
            cluster=cluster,
            param=param,
            weights_type=weights_type,
            impose_null=impose_null,
            bootstrap_type=bootstrap_type,
            seed=seed,
            k_adj=k_adj,
            G_adj=G_adj,
            parallel=parallel,
            return_bootstrapped_t_stats=return_bootstrapped_t_stats,
        )

    def ccv(
        self,
        treatment,
        cluster: str | None = None,
        seed: int | None = None,
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
        return _ccv_impl(
            model=self,
            treatment=treatment,
            cluster=cluster,
            seed=seed,
            n_splits=n_splits,
            pk=pk,
            qk=qk,
        )

    def decompose(
        self,
        param: str | None = None,
        x1_vars: list[str] | str | None = None,
        decomp_var: str | None = None,
        type: decomposition_type = "gelbach",
        cluster: str | None = None,
        combine_covariates: dict[str, list[str]] | None = None,
        reps: int = 1000,
        seed: int | None = None,
        nthreads: int | None = None,
        agg_first: bool | None = None,
        only_coef: bool = False,
        digits=4,
    ) -> GelbachDecomposition:
        """
        Implement the Gelbach (2016) decomposition method for mediation analysis.

        Compares a short model `depvar on param` with the long model
        specified in the original feols() call.

        For details, take a look at
        "When do covariates matter?" by Gelbach (2016, JoLe). You can find
        an ungated version of the paper on SSRN under the following link:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737 .

        When the initial regression is weighted, weights are interpreted as frequency
        weights. Inference is not yet supported for weighted models.

        Parameters
        ----------
        param : str
            The name of the focal covariate whose effect is to be decomposed into direct
            and indirect components with respect to the rest of the right-hand side.
        x1_vars : list[str]
            A list of covariates that are included in both the baseline and the full
            regressions.
        decomp_var : str
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
            False by default, unless combine_covariates is provided.
            Recommended to set to True if combine_covariates is argument is provided.
            As a rule of thumb, the more covariates are combined, the larger the performance
            improvement.
        only_coef : bool, optional
            Indicates whether to compute inference for the decomposition. Defaults to False.
            If True, skips the inference step and only returns the decomposition results.
        digits : int, optional
            The number of digits to round the results to. Defaults to 4.

        Returns
        -------
        GelbachDecomposition
            A GelbachDecomposition object with the decomposition results.
            Use `tidy()` and `etable()` to access the estimation results.

        Examples
        --------
        ```{python}
        import re
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        data = gelbach_data(nobs = 1000)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

        # simple decomposition
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1)
        type(gb)

        gb.tidy()
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, x1_vars = ["x21"])
        # combine covariates
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]})
        # supress inference
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]}, only_coef = True)
        # print results
        gb.etable()

        # group covariates via regex
        res = fit.decompose(decomp_var="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
        ```
        """
        return _decompose_impl(
            model=self,
            param=param,
            x1_vars=x1_vars,
            decomp_var=decomp_var,
            type=type,
            cluster=cluster,
            combine_covariates=combine_covariates,
            reps=reps,
            seed=seed,
            nthreads=nthreads,
            agg_first=agg_first,
            only_coef=only_coef,
            digits=digits,
        )

    def fixef(
        self, atol: float = 1e-06, btol: float = 1e-06
    ) -> dict[str, dict[str, float]]:
        """
        Compute the coefficients of (swept out) fixed effects for a regression model.

        This method creates the following attributes:
        - `_alpha` (pd.DataFrame): A DataFrame with the estimated fixed effects.
        - `_sumFE` (np.array): An array with the sum of fixed effects for each
        observation (i = 1, ..., N).

        Parameters
        ----------
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html

        Returns
        -------
        dict[str, dict[str, float]]
            A dictionary with the estimated fixed effects.
        """
        return _fixef_impl(model=self, atol=atol, btol=btol)

    def predict(
        self,
        newdata: DataFrameType | None = None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: PredictionType = "link",
        se_fit: bool | None = False,
        interval: PredictionErrorOptions | None = None,
        alpha: float = 0.05,
    ) -> np.ndarray | pd.DataFrame:
        """
        Predict values of the model on new data.

        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations will be set to NaN.

        Parameters
        ----------
        newdata : DataFrameType, optional
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
        return _predict_impl(
            model=self,
            newdata=newdata,
            atol=atol,
            btol=btol,
            type=type,
            se_fit=se_fit,
            interval=interval,
            alpha=alpha,
        )

    def ritest(
        self,
        resampvar: str,
        cluster: str | None = None,
        reps: int = 100,
        type: str = "randomization-c",
        rng: np.random.Generator | None = None,
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
            The alternatives are "fast" and "slow". The fast algorithm requires
            the optional `numba` extra (install via `pip install pyfixest[numba]`);
            without it, the fast path raises an `ImportError`. The slow path
            does not require numba.
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
        return _ritest_impl(
            model=self,
            resampvar=resampvar,
            cluster=cluster,
            reps=reps,
            type=type,
            rng=rng,
            choose_algorithm=choose_algorithm,
            store_ritest_statistics=store_ritest_statistics,
            level=level,
        )

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
        return _plot_ritest_impl(model=self, plot_backend=plot_backend)

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
        X_n_plus_1 = np.vstack((self._X_wls, X_new))
        epsi_n_plus_1 = y_new - X_new @ self._beta_hat
        gamma_n_plus_1 = np.linalg.inv(X_n_plus_1.T @ X_n_plus_1) @ X_new.T
        beta_n_plus_1 = self._beta_hat + gamma_n_plus_1 @ epsi_n_plus_1
        if inplace:
            self._X_wls = X_n_plus_1
            self._Y_wls = np.append(self._Y_wls, y_new)
            self._beta_hat = beta_n_plus_1
            self._u_hat = self._Y_wls - self._X_wls @ self._beta_hat
            self._u_hat_wls = self._u_hat
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


def _check_vcov_input(
    vcov: str | dict[str, str],
    vcov_kwargs: dict[str, Any] | None,
    data: pd.DataFrame,
):
    """
    Check the input for the vcov argument in the Feols class.

    Parameters
    ----------
    vcov : Union[str, dict[str, str]]
        The vcov argument passed to the Feols class.
    vcov_kwargs : Optional[dict[str, Any]]
        The vcov_kwargs argument passed to the Feols class.
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
            "NW",
            "DK",
            "nid",
        ], (
            "vcov string must be iid, hetero, HC1, HC2, HC3, NW, or DK, or for quantile regression, 'nid'."
        )

        # check that time_id is provided if vcov is NW or DK
        if (
            vcov in {"NW", "DK"}
            and vcov_kwargs is not None
            and "time_id" not in vcov_kwargs
        ):
            raise ValueError("Missing required 'time_id' for NW/DK vcov")


def _deparse_vcov_input(vcov: str | dict[str, str], has_fixef: bool, is_iv: bool):
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
    elif vcov_type_detail in ["NW", "DK"]:
        vcov_type = "HAC"
        is_clustered = False

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
