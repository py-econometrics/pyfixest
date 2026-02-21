import functools
import gc
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2, f, norm, t

from pyfixest.estimation.formula import model_matrix as model_matrix_fixest
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.backends import BACKENDS
from pyfixest.estimation.internals.demean_ import demean_model
from pyfixest.estimation.internals.literals import (
    DemeanerBackendOptions,
    PredictionErrorOptions,
    PredictionType,
    SolverOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.solvers import solve_ols
from pyfixest.estimation.models._output_mixin import OutputMixin
from pyfixest.estimation.models._post_estimation_mixin import PostEstimationMixin
from pyfixest.estimation.post_estimation.prediction import (
    _compute_prediction_error,
    _get_fixed_effects_prediction_component,
    get_design_matrix_and_yhat,
)
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _extract_variable_level,
)
from pyfixest.utils.utils import (
    capture_context,
)

prediction_type = Literal["response", "link"]


class Feols(PostEstimationMixin, OutputMixin):
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
        lookup_demeaned_data: dict[frozenset[int], pd.DataFrame],
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
            FixestFormula.formula
            if self._sample_split_var is None
            else f"{FixestFormula.formula} (Sample: {self._sample_split_var} = {self._sample_split_value})"
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

        try:
            impl = BACKENDS[demeaner_backend]
        except KeyError:
            raise ValueError(f"Unknown backend {demeaner_backend!r}")

        self._demean_func = impl["demean"]
        self._find_collinear_variables_func = impl["collinear"]
        self._crv1_meat_func = impl["crv1_meat"]
        self._count_nested_fixef_func = impl["nonnested"]

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

    def prepare_model_matrix(self):
        "Prepare model matrices for estimation."
        model_matrix = model_matrix_fixest.create_model_matrix(
            formula=self.FixestFormula,
            data=self._data,
            drop_singletons=self._drop_singletons,
            drop_intercept=self._drop_intercept,
            weights=self._weights_name,
            context=self._context,
        )

        self._Y = model_matrix.dependent
        self._Y_untransformed = model_matrix.dependent.copy()
        self._X = model_matrix.independent
        self._fe = model_matrix.fixed_effects
        self._endogvar = model_matrix.endogenous
        self._Z = model_matrix.instruments
        self._weights_df = model_matrix.weights
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
            self._Yd, self._Xd = demean_model(
                self._Y,
                self._X,
                self._fe,
                self._weights.flatten(),
                self._lookup_demeaned_data,
                self._na_index,
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
        self.demean()
        self.to_array()
        self.drop_multicol_vars()
        self.wls_transform()

        if self._X_is_empty:
            self._u_hat = self._Y
        else:
            self._Z = self._X
            self._tZX = self._Z.T @ self._X
            self._tZy = self._Z.T @ self._Y

            self._beta_hat = solve_ols(self._tZX, self._tZy, self._solver)

            self._u_hat = self._Y.flatten() - (self._X @ self._beta_hat).flatten()

            self._scores = self._X * self._u_hat[:, None]
            self._hessian = self._tZX.copy()

            # IV attributes, set to None for OLS, Poisson
            self._tXZ = np.array([])
            self._tZZinv = np.array([])

        self._get_predictors()

    def vcov(
        self,
        vcov: Union[str, dict[str, str]],
        vcov_kwargs: Optional[dict[str, Union[str, int]]] = None,
        data: Optional[DataFrameType] = None,
    ) -> "Feols":
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
        from pyfixest.estimation.internals.vcov_computation import compute_vcov

        compute_vcov(self, vcov=vcov, vcov_kwargs=vcov_kwargs, data=data)
        self.get_inference()
        return self

    def _vcov_iid(self):
        from pyfixest.estimation.internals.vcov_computation import vcov_iid

        return vcov_iid(self)

    def _vcov_hetero(self):
        from pyfixest.estimation.internals.vcov_computation import vcov_hetero

        return vcov_hetero(self)

    def _vcov_hac(self):
        from pyfixest.estimation.internals.vcov_computation import vcov_hac

        return vcov_hac(self)

    def _vcov_nid(self):
        from pyfixest.estimation.internals.vcov_computation import vcov_nid

        return vcov_nid(self)

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):
        from pyfixest.estimation.internals.vcov_computation import vcov_crv1

        return vcov_crv1(self, clustid=clustid, cluster_col=cluster_col)

    def _vcov_crv3_fast(self, clustid, cluster_col):
        from pyfixest.estimation.internals.vcov_computation import vcov_crv3_fast

        return vcov_crv3_fast(self, clustid=clustid, cluster_col=cluster_col)

    def _vcov_crv3_slow(self, clustid, cluster_col):
        from pyfixest.estimation.internals.vcov_computation import vcov_crv3_slow

        return vcov_crv3_slow(self, clustid=clustid, cluster_col=cluster_col)

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

        self._se = np.sqrt(np.diagonal(self._vcov))
        self._tstat = self._beta_hat / self._se
        # use t-dist for linear models, but normal for non-linear models
        if self._method in ["fepois", "feglm-probit", "feglm-logit", "feglm-gaussian"]:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf(alpha / 2))
        else:
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), self._df_t))
            z = np.abs(t.ppf(alpha / 2, self._df_t))

        z_se = z * self._se
        self._conf_int = np.array([self._beta_hat - z_se, self._beta_hat + z_se])

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
        self._fml = self.FixestFormula.formula
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
        meat = np.linalg.inv(R @ self._vcov @ R.T)
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

        blocked_transforms = ["i(", "^", "poly("]
        for bt in blocked_transforms:
            if bt in self._fml:
                raise NotImplementedError(
                    f"The fixef() method is currently not supported for models with '{bt}' transformations."
                )

        if not self._has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        if self._is_iv:
            raise NotImplementedError(
                "The fixef() method is currently not supported for IV models."
            )

        depvars, rhs = self._fml.split("~")
        covars, fixef_vars = rhs.split("|")

        fixef_vars_list = fixef_vars.split("+")
        fixef_vars_C = [f"C({x})" for x in fixef_vars_list]
        fixef_fml = "+".join(fixef_vars_C)

        Y, X = Formula(f"{depvars} ~ {covars}").get_model_matrix(
            self._data, output="pandas", context=self._context
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
            uhat = (Y - X @ self._beta_hat).flatten()
        D2 = Formula("-1+" + fixef_fml).get_model_matrix(self._data, output="sparse")
        cols = D2.model_spec.column_names

        if self._has_weights:
            uhat *= weights_sqrt
            weights_diag = diags(weights_sqrt, 0)
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
            X_index = np.arange(self._N)
            y_hat = (
                self._Y_hat_link
                if type == "link" or self._method == "feols"
                else self._Y_hat_response
            )
            n_observations = self._N
        else:
            y_hat, X, X_index = get_design_matrix_and_yhat(
                model=self,
                newdata=newdata,
                context=self._context,
            )
            y_hat += _get_fixed_effects_prediction_component(
                model=self, newdata=newdata, atol=atol, btol=btol
            )
            n_observations = newdata.shape[0]
            if type == "response" and self._method == "fepois":
                y_hat = np.exp(y_hat)

        if se_fit or interval == "prediction":
            prediction_df = _compute_prediction_error(
                model=self,
                nobs=n_observations,
                yhat=y_hat,
                X=X,
                X_index=X_index,
                alpha=alpha,
            )
            if interval == "prediction":
                return prediction_df
            else:
                return prediction_df["se_fit"].to_numpy()
        else:
            return y_hat

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
