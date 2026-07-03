from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal, cast

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser import DefaultFormulaParser
from scipy.sparse import csc_matrix, diags, spmatrix
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2, f, t

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner
from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula import model_matrix as model_matrix_fixest
from pyfixest.estimation.formula.model_matrix import _ModelMatrixKey
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanCache
from pyfixest.estimation.internals.families import T_DIST, InferenceDist
from pyfixest.estimation.internals.fit_ import fit_ols
from pyfixest.estimation.internals.literals import (
    PredictionErrorOptions,
    PredictionType,
    SolverOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.vcov_ import (
    vcov_crv1,
    vcov_crv3_fast,
    vcov_hac,
    vcov_hetero,
    vcov_iid_ols,
)
from pyfixest.estimation.internals.vcov_utils import (
    _compute_bread,
    prepare_cluster_state,
    run_crv_loop,
)
from pyfixest.estimation.models._result_accessor_mixin import ResultAccessorMixin
from pyfixest.estimation.post_estimation.decomposition import (
    GelbachDecomposition,
    _decompose_arg_check,
)
from pyfixest.estimation.post_estimation.prediction import (
    _compute_prediction_error,
    _rows_with_unseen_categories,
)
from pyfixest.estimation.post_estimation.ritest import (
    _HAS_NUMBA,
    _decode_resampvar,
    _get_ritest_pvalue,
    _get_ritest_stats_fast,
    _get_ritest_stats_slow,
    _plot_ritest_pvalue,
)
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _extract_variable_level,
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
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
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
        self._inference_dist: InferenceDist = T_DIST
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
        self._demean_cache = DemeanCache(lookup_demeaned_data, lookup_preconditioner)
        self._store_data = store_data
        self._copy_data = copy_data
        self._lean = lean
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
        self._fixef = (
            str(FixestFormula.fixed_effects).replace(" ", "").replace(":", "^")
            if FixestFormula.is_fixed_effects
            else None
        )
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

        # special for poisson / glm
        self.deviance: float | None = None

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

        self._Y = model_matrix.dependent
        self._Y_untransformed = model_matrix.dependent.copy()
        self._X = model_matrix.independent
        self._fe = model_matrix.fixed_effects
        self._endogvar = model_matrix.endogenous
        self._Z = model_matrix.instruments
        self._weights_df = model_matrix.weights
        self._offset_df = model_matrix.offset
        self._na_index = model_matrix.na_index
        # TODO: set dynamically based on naming set in pyfixest.estimation.formula.factor_interaction._encode_i
        is_icovar = (
            self._X.columns.str.contains(r"^.+::.+$") if not self._X.empty else None
        )
        self._icovars = (
            self._X.columns[is_icovar].tolist()
            if is_icovar is not None and is_icovar.any()
            else None
        )
        self._X_is_empty = not model_matrix.independent.shape[0] > 0
        self._model_spec = model_matrix.model_spec

        self._coefnames = self._X.columns.tolist()
        self._coefnames_z = self._Z.columns.tolist() if self._Z is not None else None
        self._depvar = self._Y.columns[0]

        self._has_fixef = self._fe is not None
        self._fixef = (
            str(self.FixestFormula.fixed_effects).replace(" ", "").replace(":", "^")
            if self.FixestFormula.is_fixed_effects
            else None
        )

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
        N_rows = len(self._Y)
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
        N = len(self._Y)

        if self._weights_df is not None:
            _weights = self._weights_df.to_numpy()
        else:
            _weights = np.ones(N)

        return _weights.reshape((N, 1))

    def demean(self):
        "Demean the dependent variable and covariates by the fixed effect(s)."
        if self._has_fixef:
            self._Yd, self._Xd, _ = self._demean_cache.demean_yx(
                self._Y,
                self._X,
                self._fe,
                self._weights.flatten(),
                self._na_index,
                self._demeaner,
            )
        else:
            self._Yd, self._Xd = self._Y, self._X

    @property
    def preconditioner(self) -> Preconditioner | None:
        """The within preconditioner used during demeaning, if any.

        ``None`` when no preconditioner participated in the solve —
        ``preconditioner='off'``, single-FE designs (MAP fallback), or any
        non-within backend. Otherwise the instance built on the first solve
        for this model's row sample. Pass it back via
        ``LsmrDemeaner(backend='within', preconditioner=...)`` to skip the
        setup phase on a later fit over the same design.
        """
        return self._demean_cache.lookup_preconditioner.get(self._na_index)

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
            ) = drop_multicollinear_variables(
                self._X,
                self._coefnames,
                self._collin_tol,
            )
        # update X_is_empty
        self._X_is_empty = self._X.shape[1] == 0
        self._k = self._X.shape[1] if not self._X_is_empty else 0

    def _get_predictors(self) -> None:
        self._Y_hat_link = self._Y_untransformed.to_numpy().flatten() - self.resid()
        self._Y_hat_response = self._Y_hat_link

    def get_fit(self) -> None:
        """
        Fit an OLS model.

        Returns
        -------
        None
        """
        self.demean()
        self.to_array()
        self.drop_multicol_vars()
        self.wls_transform()

        if self._X_is_empty:
            self._u_hat = self._Y
        else:
            fit = fit_ols(X=self._X, Y=self._Y, solver=self._solver)

            self._Z = self._X
            self._tZX = fit.tZX
            self._tZy = fit.tZy
            self._beta_hat = fit.beta
            self._u_hat = fit.residuals
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

        if self._vcov_type == "iid":
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(vcov_type="iid", G=1)
            )
            self._vcov = self._ssc * self._vcov_iid()

        elif self._vcov_type == "hetero":
            # fixest:::vcov_hetero_internal: adj = ifelse(ssc$cluster.adj, n/(n - 1), 1)
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(vcov_type="hetero", G=self._N)
            )
            self._vcov = self._ssc * self._vcov_hetero()

        elif self._vcov_type == "HAC":
            kw = vcov_kwargs or {}
            self._lag = kw.get("lag")
            self._time_id = kw.get("time_id")
            self._panel_id = kw.get("panel_id")
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(
                    vcov_type="HAC",
                    G=np.unique(self._data[self._time_id]).shape[0],
                )  # number of unique time periods T used
            )
            self._vcov = self._ssc * self._vcov_hac()

        elif self._vcov_type == "nid":
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(vcov_type="hetero", G=self._N)
            )
            self._vcov = self._ssc * self._vcov_nid()

        elif self._vcov_type == "CRV":
            prep = prepare_cluster_state(
                data=data if data is not None else self._data,
                clustervar=self._clustervar,
                ssc_dict=self._ssc_dict,
                fixef=self._fixef,
                fe=self._fe,
                k_fe=self._k_fe,
            )
            self._cluster_df = prep.cluster_df
            self._G = prep.G
            self._vcov, self._ssc, self._df_k, self._df_t = run_crv_loop(
                prep=prep,
                k=self._k,
                make_ssc_kwargs=self._make_ssc_kwargs,
                cluster_vcov=self._vcov_crv_cluster,
            )
        # update p-value, t-stat, standard error, confint
        self.get_inference()

        return self

    def _make_ssc_kwargs(
        self,
        *,
        vcov_type: str,
        G: int | list[int],
        vcov_sign: int = 1,
        k_fe_nested: int = 0,
        n_fe_fully_nested: int = 0,
    ) -> dict:
        "Bundle model-level and vcov-type-specific args for get_ssc()."
        return {
            "ssc_dict": self._ssc_dict,
            "N": self._N,
            "k": self._k,
            "k_fe": self._k_fe.sum() if self._has_fixef else 0,
            "n_fe": self._n_fe,
            "vcov_type": vcov_type,
            "G": G,
            "vcov_sign": vcov_sign,
            "k_fe_nested": k_fe_nested,
            "n_fe_fully_nested": n_fe_fully_nested,
        }

    def _vcov_crv_cluster(
        self, clustid: np.ndarray, cluster_col: np.ndarray
    ) -> np.ndarray:
        "Pick CRV1 / CRV3-fast / CRV3-slow for one cluster column."
        if self._vcov_type_detail == "CRV1":
            return self._vcov_crv1(clustid=clustid, cluster_col=cluster_col)

        if not self._support_crv3_inference:
            raise VcovTypeNotSupportedError(
                f"CRV3 inference is not for models of type '{self._method}'."
            )
        use_fast = not self._has_fixef and self._method == "feols" and not self._is_iv
        crv3 = self._vcov_crv3_fast if use_fast else self._vcov_crv3_slow
        return crv3(clustid=clustid, cluster_col=cluster_col)

    def _vcov_iid(self):
        return vcov_iid_ols(residuals=self._u_hat, bread=self._bread, N=self._N)

    def _vcov_hetero(self):
        return vcov_hetero(
            scores=self._scores,
            X=self._X,
            tZX=self._tZX,
            weights=self._weights,
            weights_type=self._weights_type,
            vcov_type_detail=self._vcov_type_detail,
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
        )

    def _vcov_hac(self):
        _time_id = self._time_id
        _panel_id = self._panel_id
        _data = self._data

        if not self._support_hac_inference:
            raise NotImplementedError(
                "HAC inference is not supported for this model type."
            )

        # some data checks on input pandas df
        # time needs to be numeric or date else we cannot sort by time
        if not np.issubdtype(_data[_time_id], np.number) and not np.issubdtype(
            _data[_time_id], np.datetime64
        ):
            raise ValueError(
                "The time variable must be numeric or date, else we cannot sort by time."
            )

        _time_arr = _data[_time_id].to_numpy()
        _panel_arr = _data[_panel_id].to_numpy() if _panel_id is not None else None

        return vcov_hac(
            scores=self._scores,
            time_arr=_time_arr,
            panel_arr=_panel_arr,
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
            X=self._X,
            Y=self._Y,
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
        beta_center = self._beta_hat

        vcov_mat = np.zeros((self._k, self._k))
        for ixg, _ in enumerate(clustid):
            beta_centered = beta_jack[ixg, :] - beta_center
            vcov_mat += np.outer(beta_centered, beta_centered)

        return vcov_mat

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
        self._Y_untransformed = Y
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
        k_fe = np.sum(self._k_fe.values) if self._has_fixef else 0

        # If R is None, default to the identity matrix
        if R is None:
            R = np.eye(self._k)

        # Ensure R is two-dimensional
        if R.ndim == 1:
            R = R.reshape((1, len(R)))

        if R.shape[1] != self._k:
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
            self._dfd = self._N - self._k - k_fe

        bread = R @ self._beta_hat - q
        meat = np.linalg.pinv(R @ self._vcov @ R.T)
        W = bread.T @ meat @ bread
        self._wald_statistic = W

        # Check if distribution is "F" and R is not identity matrix
        # or q is not zero vector
        if distribution == "F" and (
            not np.array_equal(R, np.eye(self._k)) or not np.all(q == 0)
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
        if param is not None and param not in self._coefnames:
            raise ValueError(
                f"Parameter {param} not found in the model's coefficients."
            )

        if not self._supports_wildboottest:
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

        if cluster is None and self._clustervar is not None:
            if isinstance(self._clustervar, str):
                cluster_list = [self._clustervar]
            else:
                cluster_list = self._clustervar

        run_heteroskedastic = not cluster_list

        if not run_heteroskedastic and not len(cluster_list) == 1:
            raise NotImplementedError(
                "Multiway clustering is currently not supported with the wild cluster bootstrap."
            )

        if not run_heteroskedastic and cluster_list[0] not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster_list[0]} not found in the data."
            )

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
            )

        if self._is_iv:
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

            cluster_array = self._data[cluster_list[0]].to_numpy().flatten()

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
                adj=k_adj,
                cluster_adj=G_adj,
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

        ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
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

        res_ccv_dict: dict[str, float | np.ndarray] = {
            "Estimate": tau_full,
            "Std. Error": se,
            "t value": tstat,
            "Pr(>|t|)": pvalue,
            "2.5%": conf_int[0],
            "97.5%": conf_int[1],
        }

        res_ccv = pd.Series(res_ccv_dict)

        res_ccv.name = "CCV"

        res_crv1 = cast(pd.Series, self.tidy().xs(treatment))
        res_crv1.name = "CRV1"

        return pd.concat([res_ccv, res_crv1], axis=1).T

        ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
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
    ) -> tuple[np.ndarray, np.ndarray | spmatrix, list[str]]:
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

            Y, X = formulaic.Formula(fml_dummies).get_model_matrix(
                self._data,
                output=output,
                context=FORMULAIC_TRANSFORMS | {**self._context},
            )
            xnames = X.model_spec.column_names
            Y = Y.toarray().flatten() if output == "sparse" else Y.flatten()
            X = csc_matrix(X) if output == "sparse" else X

        else:
            Y = self._Y.flatten() / np.sqrt(self._weights.flatten())
            X = self._X / np.sqrt(self._weights)
            xnames = self._coefnames

        X = csc_matrix(X) if output == "sparse" else X

        return Y, X, xnames

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
        has_param = param is not None
        has_decomp = decomp_var is not None

        if not has_param and not has_decomp:
            raise ValueError("Either 'param' or 'decomp_var' must be provided.")

        if has_param and has_decomp:
            raise ValueError(
                "The 'param' and 'decomp_var' arguments cannot be provided at the same time."
            )

        if has_param:
            warnings.warn(
                "The 'param' argument is deprecated. Please use 'decomp_var' instead.",
                UserWarning,
            )
            decomp_var = param

        if x1_vars is not None:
            if isinstance(x1_vars, str):
                x1_vars = [x.strip() for x in x1_vars.split("+")]
            else:
                x1_vars = list(x1_vars)

        _decompose_arg_check(
            type=type,
            has_weights=self._has_weights,
            weights_type=self._weights_type,
            is_iv=self._is_iv,
            method=self._method,
            only_coef=only_coef,
        )

        nthreads_int = -1 if nthreads is None else nthreads

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        if agg_first is None:
            agg_first = combine_covariates is not None

        cluster_df: pd.Series | None = None
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
            decomp_var=cast(str, decomp_var),
            x1_vars=x1_vars,
            coefnames=xnames,
            depvarname=self._depvar,
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
            weights=self._weights,
            store=True,
        )

        if not only_coef:
            med.bootstrap(rng=rng, B=reps)

        self.GelbachDecompositionResults = med

        return med

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
        weights_sqrt = np.sqrt(self._weights).flatten()

        if not self._has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        if self._is_iv:
            raise NotImplementedError(
                "The fixef() method is currently not supported for IV models."
            )

        Y, X = self._model_spec[_ModelMatrixKey.main].get_model_matrix(
            self._data,
            output="pandas",
            context=FORMULAIC_TRANSFORMS | {**self._context},
        )
        Y = Y.to_numpy().flatten().astype(np.float64)
        if self._X_is_empty:
            uhat = Y.flatten()
        else:
            # drop intercept, potentially multicollinear vars
            X = X[self._coefnames].to_numpy()
            if self._method == "fepois" or self._method.startswith("feglm"):
                # determine residuals from estimated linear predictor
                # equation (5.2) in Stammann (2018) http://arxiv.org/abs/1707.01815
                Y = self._Y_hat_link
                # _Y_hat_link contains the offset as part of eta; subtract it so
                # that _sumFE represents the pure FE contribution and predict()
                # can add the offset back from newdata without double-counting.
                if self._offset_name is not None:
                    assert self._offset is not None
                    Y = Y - self._offset.flatten()
            uhat = (Y - X @ self._beta_hat).flatten()
        # one-hot encoding of fixed effects (treatment coding: reference level dropped)
        D2 = formulaic.Formula(
            [f"C({fe})" for fe in self.FixestFormula.fixed_effects_wrapped],
            _parser=DefaultFormulaParser(include_intercept=False),
        ).get_model_matrix(
            self._data,
            output="sparse",
            ensure_full_rank=False,
            context=FORMULAIC_TRANSFORMS,
            transform_state=self._model_spec[
                _ModelMatrixKey.fixed_effects
            ].transform_state,
        )
        one_hot_encoded_fixed_effects = D2.model_spec.column_names

        if self._has_weights:
            uhat *= weights_sqrt
            weights_diag = diags(weights_sqrt, 0)
            D2 = weights_diag.dot(D2)

        alpha = lsqr(D2, uhat, atol=atol, btol=btol)[0]

        res: dict[str, dict[str, float]] = {}
        for i, col in enumerate(one_hot_encoded_fixed_effects):
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
            y_hat = (
                self._Y_hat_link
                if type == "link" or self._method == "feols"
                else self._Y_hat_response
            )
            n_observations = self._N
        else:
            newdata = _narwhals_to_pandas(newdata).reset_index(drop=True)
            n_observations = newdata.shape[0]
            context = FORMULAIC_TRANSFORMS | {**self._context}
            # Use na_action="drop" on each sub-spec separately because dependent variable
            # may not be available in newdata, then intersect indices so a NaN in *any* variable
            # (covariate or FE) marks the whole row as NaN in the output.
            X_mm = self._model_spec[_ModelMatrixKey.main].rhs.get_model_matrix(
                newdata, context=context, na_action="drop"
            )
            valid_idx = X_mm.index.to_numpy()
            # rows with a categorical level unseen during fitting (in C()/i()) would
            # be silently encoded as the reference level -> drop them to NaN instead,
            # matching how unseen fixed-effect levels are handled below.
            unseen = _rows_with_unseen_categories(
                self._model_spec[_ModelMatrixKey.main].rhs, newdata
            )
            valid_idx = valid_idx[~unseen[valid_idx]]
            if self._has_fixef:
                fe_mm = self._model_spec[
                    _ModelMatrixKey.fixed_effects
                ].get_model_matrix(newdata, context=context, na_action="drop")
                valid_idx = np.intersect1d(valid_idx, fe_mm.index.to_numpy())

                if self._sumFE is None:
                    self.fixef(atol, btol)
                fe_hat = (
                    pd.concat(
                        [
                            fe_mm.loc[valid_idx, column].map(
                                {
                                    float(level): coefficient
                                    for level, coefficient in self._fixef_dict[
                                        column
                                    ].items()
                                }
                            )
                            for column in fe_mm.columns
                        ],
                        axis=1,
                    )
                    .sum(axis=1, skipna=False)
                    .to_numpy()
                )
                # fixed effects estimates are nan if singletons or unseen levels
                valid_idx = valid_idx[~np.isnan(fe_hat)]
                fe_hat = fe_hat[~np.isnan(fe_hat)]

            X_coef = X_mm.loc[valid_idx, self._coefnames].to_numpy()
            y_hat = np.full(n_observations, np.nan)
            y_hat[valid_idx] = X_coef @ self._beta_hat
            if self._has_fixef:
                y_hat[valid_idx] += fe_hat
            # Pad X to full size; NaN rows yield NaN SE/CI via einsum propagation.
            X = np.full((n_observations, X_coef.shape[1]), np.nan)
            X[valid_idx] = X_coef
            if self._offset_name is not None:
                if self._offset_name not in newdata.columns:
                    raise ValueError(
                        f"Offset variable '{self._offset_name}' not found in newdata."
                    )
                offset = pd.to_numeric(
                    newdata[self._offset_name], errors="coerce"
                ).to_numpy()
                if np.isnan(offset).any():
                    raise ValueError(
                        f"Offset column '{self._offset_name}' in newdata contains "
                        "NaN or non-numeric values."
                    )
                y_hat = y_hat + offset
            if type == "response" and self._method == "fepois":
                y_hat = np.exp(y_hat)

        if se_fit or interval == "prediction":
            prediction_df = _compute_prediction_error(
                model=self,
                nobs=n_observations,
                yhat=y_hat,
                X=X,
                alpha=alpha,
            )
            if interval == "prediction":
                return prediction_df
            else:
                return prediction_df["se_fit"].to_numpy()
        else:
            return y_hat

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
        resampvar = resampvar.replace(" ", "")
        resampvar_, h0_value, hypothesis, test_type = _decode_resampvar(resampvar)

        if self._is_iv:
            raise NotImplementedError(
                "Randomization Inference is not supported for IV models."
            )

        # check that resampvar in _coefnames
        if resampvar_ not in self._coefnames:
            raise ValueError(f"{resampvar_} not found in the model's coefficients.")

        if cluster is not None and cluster not in self._data:
            raise ValueError(f"The variable {cluster} is not found in the data.")

        clustervar_arr = (
            self._data[cluster].to_numpy().reshape(-1, 1) if cluster else None
        )

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

        if choose_algorithm == "auto":
            choose_algorithm = "fast" if _HAS_NUMBA else "slow"

        assert isinstance(reps, int) and reps > 0, "reps must be a positive integer."

        if self._has_weights:
            raise NotImplementedError(
                """
                Regression Weights are not supported with Randomization Inference.
                """
            )

        if choose_algorithm == "slow" or self._method == "fepois":
            vcov_input: str | dict[str, str]
            if cluster is not None:
                vcov_input = {"CRV1": cluster}
            else:
                # "iid" for models without controls, else HC1
                vcov_input = (
                    "hetero"
                    if (self._has_fixef and len(self._coefnames) > 1)
                    or len(self._coefnames) > 2
                    else "iid"
                )

            # for performance reasons
            if type == "randomization-c":
                vcov_input = "iid"

            ri_stats = _get_ritest_stats_slow(
                data=self._data,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                fml=self._fml,
                reps=reps,
                vcov=vcov_input,
                type=type,
                rng=rng,
                model=self._method,
            )

        else:
            weights = self._weights.flatten()
            fval_df = (
                self._data[self._fixef.split("+")] if self._fixef is not None else None
            )
            D = self._data[resampvar_].to_numpy()

            ri_stats = _get_ritest_stats_fast(
                Y=self._Y,
                X=self._X,
                D=D,
                coefnames=self._coefnames,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                reps=reps,
                rng=rng,
                fval_df=fval_df,
                weights=weights,
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
