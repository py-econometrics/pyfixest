from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2, f

from pyfixest.core.demean import Preconditioner, WithinPreconditionerName
from pyfixest.demeaners import AnyDemeaner, LsmrDemeaner, MapDemeaner
from pyfixest.errors import EmptyVcovError, VcovTypeNotSupportedError
from pyfixest.estimation.api.utils import _ALL_SAMPLE, _AllSampleSentinel
from pyfixest.estimation.capabilities import (
    DID_METHOD_LABELS,
    Capabilities,
    Feature,
    FitFeatures,
    require_support,
)
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula import model_matrix as model_matrix_fixest
from pyfixest.estimation.formula.formulaic_compat import (
    materialize_model_spec_with_unseen_mask,
)
from pyfixest.estimation.formula.model_matrix import _ModelMatrixKey
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanCache, DemeanedData
from pyfixest.estimation.internals.families import T_DIST, InferenceDist
from pyfixest.estimation.internals.literals import (
    EstimatorKind,
    InferenceType,
    PredictionErrorOptions,
    PredictionType,
    SolverOptions,
    WeightsTypeOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.model_state import (
    ObservationWeights,
    WithinLinearData,
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
from pyfixest.estimation.models._tidy_accessors import TidyColumnAccessors
from pyfixest.estimation.post_estimation.fixed_effects import (
    FixedEffect,
    build_fixed_effects,
    check_fe_dtype_compatibility,
    contrast_code_fixed_effects,
    fixed_effects_to_frame,
    predict_fixed_effects,
    warn_on_unseen_fixed_effect_levels,
)
from pyfixest.estimation.post_estimation.prediction import _compute_prediction_error
from pyfixest.estimation.post_estimation.wald import _wald_statistic
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _narwhals_to_pandas,
    _select_coefnames_and_indices,
)
from pyfixest.utils.utils import (
    capture_context,
    get_ssc,
    simultaneous_crit_val,
)


class BaseRegression(TidyColumnAccessors):
    """
    Shared implementation behind every pyfixest regression result.

    `BaseRegression` owns what the estimators have in common: configuration and
    formula preparation, canonical observation weights and within-scale arrays,
    the storage-policy guards, the capability contract, covariance dispatch,
    coefficient-level inference, and the post-estimation methods that every
    estimator can run. Estimator-specific numerical work lives in the leaf
    classes: `get_fit()` is abstract here, and the covariance, prediction, and
    fixed-effect hooks below are the seams a leaf overrides.

    Users never construct this class. It is the base of
    [Feols](/reference/estimation.models.feols_.Feols.qmd),
    [Feglm](/reference/estimation.models.feglm_.Feglm.qmd), and
    [Quantreg](/reference/estimation.quantreg.quantreg_.Quantreg.qmd), and the
    type a post-estimation consumer can rely on.

    Parameters
    ----------
    FixestFormula : pyfixest.estimation.formula.parse.Formula
        Parsed fixest formula for this model.
    data : pandas.DataFrame
        Estimation data, before formula-level row filtering.
    ssc_dict : dict
        Small-sample-correction options, as built by `pf.ssc()`.
    drop_singletons : bool
        Whether singleton fixed-effect levels are removed.
    drop_intercept : bool
        Whether the intercept is dropped from the design.
    weights : str or None
        Name of the observation-weight column, or None for an unweighted fit.
    weights_type : str or None
        Either `"aweights"` for analytic or `"fweights"` for frequency weights.
    collin_tol : float
        Tolerance of the collinearity check on the within design.
    lookup_demeaned_data : dict
        Demeaning cache shared across the models of one cache block.
    solver : SolverOptions, optional
        Linear solver used by the estimator's fit primitive.
    demeaner : AnyDemeaner, optional
        Resolved typed demeaner configuration. Defaults to `MapDemeaner()`.
    lookup_preconditioner : dict, optional
        Within-preconditioner cache shared across the models of one cache block.
    store_data : bool, optional
        Whether the estimation data are retained on the result.
    copy_data : bool, optional
        Whether the input frame is copied before filtering.
    lean : bool, optional
        Whether the large fit arrays are discarded after estimation.
    context : int or Mapping[str, Any], optional
        Additional evaluation scope for formulaic.
    sample_split_var : str or None, optional
        Name of the sample-split variable, or None outside split estimation.
    sample_split_value : optional
        Value of the sample-split variable this result was fitted on.

    Attributes
    ----------
    _method : str
        Mutable label of the estimation method. Difference-in-differences entry
        points relabel it after fitting; use `_estimator` for the model class.
    _estimator : EstimatorKind
        The estimator that produced the fit, declared by the model class.
    _capabilities : Capabilities
        Which post-estimation features the model class supports.
    _is_iv : bool
        Whether an endogenous regressor is instrumented.
    _Y : numpy.ndarray
        Within-scale response, in response units.
    _X : numpy.ndarray
        Within-scale design, not premultiplied by weights.
    _Z : numpy.ndarray
        Within-scale instruments; the design itself outside IV models.
    _weights : numpy.ndarray
        Observation weights as a column vector.
    _X_is_empty : bool
        Whether the design has no columns.
    _collin_tol : float
        Tolerance level for collinearity checks.
    _coefnames : list[str]
        Names of the retained design columns.
    _collin_vars : list
        Variables identified as collinear.
    _collin_index : list
        Indices of collinear variables.
    _solver : SolverOptions
        The solver used for the estimation.
    _N : int or float
        Effective sample size: retained rows, or the sum of frequency weights.
    _N_rows : int
        Number of retained rows.
    _k : int
        Number of retained design columns.
    _data : pandas.DataFrame
        Estimation data. Deleted when `lean=True` or `store_data=False`.
    _fml : str
        The formula string of the model.
    _has_fixef : bool
        Whether fixed effects are absorbed.
    _fixef : str or None
        The fixed-effect specification, or None.
    _icovars : list[str] or None
        Columns generated by the `i()` operator, or None.
    _ssc_dict : dict
        Small-sample-correction options.
    _tZX, _tXZ, _tZy, _tZZinv : numpy.ndarray
        Cross-products retained by the fit primitive.
    _beta_hat : numpy.ndarray
        Estimated coefficients.
    _Y_hat_link : numpy.ndarray
        Linear predictor.
    _Y_hat_response : numpy.ndarray
        Prediction on the response scale, i.e. `E(Y|X)`.
    _u_hat : numpy.ndarray
        Residuals retained by the fit primitive.
    _scores : numpy.ndarray
        Weighted scores used by the covariance estimators.
    _hessian : numpy.ndarray
        Hessian of the estimating equation.
    _bread : numpy.ndarray
        Bread matrix of the sandwich covariance.
    _vcov_type : str
        Covariance family: `"iid"`, `"hetero"`, `"HAC"`, `"nid"`, or `"CRV"`.
    _vcov_type_detail : str
        The requested covariance type, such as `"HC3"` or `"CRV1"`.
    _is_clustered : bool
        Whether the covariance is clustered.
    _clustervar : list[str]
        Cluster variables of a clustered covariance.
    _G : list[int]
        Number of clusters per cluster dimension.
    _ssc : numpy.ndarray
        The applied small-sample correction.
    _vcov : numpy.ndarray
        Covariance matrix of the estimated coefficients.
    _se, _tstat, _pvalue, _conf_int : numpy.ndarray
        Coefficient-level inference, set by `get_inference()`.
    _fixef_coefficients : dict
        Fixed-effect estimates grouped by fixed effect, set by `fixef()`.
    _alpha : numpy.ndarray
        Stacked fixed-effect coefficients, set by `fixef()`.
    _sumFE : numpy.ndarray
        Sum of the fixed effects for each observation, set by `fixef()`.
    _rmse, _r2, _r2_within, _adj_r2, _adj_r2_within : float
        Goodness-of-fit measures, set by `get_performance()`.
    _model_name : str
        Name of the model, usually the formula string, extended by the sample
        split and, for quantile regression, the quantile.
    _model_name_plot : str
        Name used when summarizing and plotting; disambiguated on collision.
    """

    # Declared by each leaf class; see `pyfixest.estimation.capabilities`.
    _estimator: ClassVar[EstimatorKind]
    # A base result supports nothing until its class declares otherwise.
    _capabilities: ClassVar[Capabilities] = Capabilities()

    # Fields populated across the fit lifecycle. They are declared here, rather
    # than only assigned, because `prepare_model_matrix()` and the fit
    # primitives publish several of them from untyped code paths.
    _method: str
    _is_iv: bool
    _inference_dist: InferenceDist
    _drop_intercept: bool
    _has_fixef: bool
    _has_weights: bool
    _coefnames: list[str]
    _k_fe: pd.Series
    _response: NDArray[np.float64]
    _observation_weights: ObservationWeights
    _within_data: WithinLinearData
    _N: int | float
    _N_rows: int
    _k: int
    _df_t: int
    _vcov: NDArray[np.float64]
    _beta_hat: NDArray[np.float64]
    _u_hat: NDArray[np.float64]
    _se: NDArray[np.float64]
    _tstat: NDArray[np.float64]
    _pvalue: NDArray[np.float64]
    _conf_int: NDArray[np.float64]
    _vcov_type: str
    _rmse: float
    _r2: float
    _adj_r2: float
    _r2_within: float
    _adj_r2_within: float

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
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
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
        self._method = self._estimator
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
        # Storage cleanup runs at the very end of the fit, so `lean` alone does
        # not say whether the fit arrays are still there. Estimation itself goes
        # through the same guarded methods.
        self._fit_state_discarded = False
        self._context = capture_context(context)

        # attributes that have to be enriched outside of the class -
        # not really optimal code change later
        self._fml = FixestFormula.formula
        self._has_fixef = False
        self._fixef = (
            str(FixestFormula.fixed_effects).replace(" ", "")
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
        self.n_separation_na = 0

        # set in get_inference()
        self._se = np.array([])
        self._tstat = np.array([])
        self._pvalue = np.array([])
        self._conf_int = np.array([])

        # set in fixef()
        self._fixef_coefficients: dict[str, FixedEffect] = {}
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
        self._bind_estimator_methods()

    def _bind_estimator_methods(self) -> None:
        """Bind estimator-specific instance methods; the base binds none."""

    def _bind_report_methods(self):
        """Bind summary, coefplot, iplot, and etable from pyfixest.report as instance methods."""
        _module = import_module("pyfixest.report")

        _tmp = _module.summary
        self.summary = functools.partial(_tmp, models=[self])
        self.summary.__doc__ = _tmp.__doc__

        _tmp = _module.coefplot
        self.coefplot = functools.partial(_tmp, models=[self])
        self.coefplot.__doc__ = _tmp.__doc__

        _tmp = _module.iplot
        self.iplot = functools.partial(_tmp, models=[self])
        self.iplot.__doc__ = _tmp.__doc__

        _tmp = _module.etable
        self.etable = functools.partial(_tmp, models=[self])
        self.etable.__doc__ = _tmp.__doc__

    def prepare_model_matrix(self):
        """Build and retain the canonical formula-derived estimator inputs."""
        model_matrix = model_matrix_fixest.create_model_matrix(
            formula=self.FixestFormula,
            data=self._data,
            drop_singletons=self._drop_singletons,
            drop_intercept=self._drop_intercept,
            weights=self._weights_name,
            offset=self._offset_name,
            context=self._context,
        )
        self._publish_model_matrix(model_matrix)

        # an empty drop still rebuilds the whole frame, so guard it
        if self._na_index:
            self._data.drop(index=list(self._na_index), inplace=True)

        return model_matrix

    # Deliberately untyped: an annotation makes mypy check this body, whose published
    # attribute types are still transitional or loose; see the typing follow-up.
    def _publish_model_matrix(self, model_matrix):
        """Publish structurally immutable formula and observation-weight state."""
        self._model_matrix = model_matrix
        self._response = model_matrix.dependent.to_numpy(dtype=np.float64).flatten()
        self._fe = model_matrix.fixed_effects
        self._na_index = model_matrix.na_index
        # TODO: set dynamically based on naming set in pyfixest.estimation.formula.factor_interaction._encode_i
        independent = model_matrix.independent
        is_icovar = (
            independent.columns.str.contains(r"^.+::.+$")
            if not independent.empty
            else None
        )
        self._icovars = (
            independent.columns[is_icovar].tolist()
            if is_icovar is not None and is_icovar.any()
            else None
        )
        self._X_is_empty = independent.shape[1] == 0
        self._model_spec = model_matrix.model_spec

        self._coefnames = independent.columns.tolist()
        self._coefnames_z = (
            model_matrix.instruments.columns.tolist()
            if model_matrix.instruments is not None
            else None
        )
        self._depvar = model_matrix.dependent.columns[0]

        self._has_fixef = self._fe is not None
        self._fixef = (
            str(self.FixestFormula.fixed_effects).replace(" ", "")
            if self.FixestFormula.is_fixed_effects
            else None
        )

        self._k_fe = self._fe.nunique(axis=0) if self._has_fixef else None
        self._n_fe = len(self._k_fe) if self._has_fixef else 0

        self._observation_weights = self._set_observation_weights()
        self._N = self._observation_weights.n_effective
        self._N_rows = self._observation_weights.n_rows

    def _validate_response(self) -> None:
        """Validate estimator-specific response constraints, if any."""

    def _set_observation_weights(self) -> ObservationWeights:
        """Build canonical user-scale observation weights for this row sample."""
        n_rows = len(self._model_matrix.dependent)
        if self._model_matrix.weights is None:
            return ObservationWeights.unweighted(n_rows=n_rows)

        assert self._weights_type in ("aweights", "fweights")
        weights_kind = cast(WeightsTypeOptions, self._weights_type)
        return ObservationWeights.from_values(
            self._model_matrix.weights.to_numpy().reshape(-1),
            kind=weights_kind,
        )

    def _prepare_within_data(self) -> WithinLinearData:
        """Return fixed-effect-residualized arrays in their original units."""
        response_frame = self._model_matrix.dependent
        design_frame = self._model_matrix.independent
        response = response_frame.to_numpy(dtype=np.float64)
        design = design_frame.to_numpy(dtype=np.float64)

        if self._model_matrix.fixed_effects is not None:
            response, design, _ = self._demean_cache.demean_yx(
                response,
                design,
                y_names=response_frame.columns,
                x_names=design_frame.columns,
                fe=self._model_matrix.fixed_effects.to_numpy(),
                weights=self._observation_weights.values,
                na_index=self._na_index,
                demeaner=self._demeaner,
            )

        return WithinLinearData(response=response, design=design)

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
        self._require_fit_arrays("preconditioner", arrays="the demeaning caches")
        return self._demean_cache.lookup_preconditioner.get(self._na_index)

    def _drop_multicollinear_within_data(
        self, within_data: WithinLinearData
    ) -> WithinLinearData:
        """Return within data after the established unweighted rank check."""
        design = within_data.design
        if design.shape[1] > 0:
            (
                design,
                self._coefnames,
                self._collin_vars,
                self._collin_index,
            ) = drop_multicollinear_variables(
                design,
                self._coefnames,
                self._collin_tol,
            )

        return WithinLinearData(
            response=within_data.response,
            design=design,
            instruments=within_data.instruments,
            endogenous=within_data.endogenous,
        )

    def _set_within_data(self, within_data: WithinLinearData) -> None:
        """Publish canonical within data for this fit."""
        self._within_data = within_data
        self._X_is_empty = within_data.design.shape[1] == 0
        self._k = within_data.design.shape[1]

    # Read-only views on the canonical state objects. They exist so that long
    # established attribute names keep working, but there is exactly one stored
    # representation: assigning or deleting them is a programming error.

    @property
    def _Y(self) -> NDArray[np.float64]:
        """Within-scale dependent variable, in response units."""
        return self._within_data.response

    @property
    def _X(self) -> NDArray[np.float64]:
        """Within-scale design matrix, not premultiplied by weights."""
        return self._within_data.design

    @property
    def _Z(self) -> NDArray[np.float64]:
        """Within-scale instruments; the design itself outside IV models."""
        within_data = self._within_data
        return (
            within_data.design
            if within_data.instruments is None
            else within_data.instruments
        )

    @property
    def _weights(self) -> NDArray[np.float64]:
        """Observation weights as a column vector.

        Unweighted fits store no weight vector, so the ones column is built on
        access rather than kept alive for the lifetime of the result.
        """
        values = self._observation_weights.values
        if values is None:
            return np.ones((self._observation_weights.n_rows, 1), dtype=np.float64)
        return values.reshape((-1, 1))

    def get_fit(self) -> None:
        """
        Estimate the model parameters.

        Returns
        -------
        None
            The estimates are written onto the result object.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_fit().")

    def _finalize_fit(self) -> None:
        """Run estimator-specific post-fit work; nothing is shared."""

    def _iter_fitted_models(self) -> tuple[BaseRegression, ...]:
        """Yield this fitted result to the result container."""
        return (self,)

    @property
    def _fit_features(self) -> FitFeatures:
        """Fit-level properties the capability rules read.

        Rebuilt on every access: `vcov()` and the storage options mutate a
        fitted result in place, so a cached snapshot could outlive the state it
        describes.
        """
        family = getattr(self, "_family", None)
        return FitFeatures(
            estimator=self._estimator,
            family=None if family is None else family.name,
            is_iv=self._is_iv,
            has_fixef=self._has_fixef,
            has_weights=self._has_weights,
            weights_kind=(
                cast(WeightsTypeOptions, self._weights_type)
                if self._has_weights
                else None
            ),
            is_did=self._method in DID_METHOD_LABELS,
        )

    def _require_support(self, feature: Feature, *, subject: str | None = None) -> None:
        """Reject a feature this fit does not support.

        Parameters
        ----------
        feature : Feature
            The feature the caller is about to run.
        subject : str, optional
            How the error message names the feature. Defaults to
            `"<feature>()"`.
        """
        require_support(
            capabilities=self._capabilities,
            feature=feature,
            features=self._fit_features,
            subject=f"{feature}()" if subject is None else subject,
        )

    def capabilities(self) -> pd.DataFrame:
        """
        Report which post-estimation methods this fitted model supports.

        Support depends on the estimator and on the fit: weights, absorbed
        fixed effects, and instruments each withdraw methods whose derivation
        does not cover them. Calling an unsupported method raises an error
        carrying the same reason.

        Returns
        -------
        pandas.DataFrame
            Indexed by feature, with a boolean `supported` column and a
            `reason` column that is None wherever the feature is available.
            See [Which Methods Does Each Estimator Support?](/how-to/supported-methods.qmd)
            for what each feature name refers to.

        Notes
        -----
        Restrictions that depend on the arguments rather than on the fit, such
        as the covariance types `evalue()` accepts or `update(inplace=True)`,
        are documented with the individual methods.
        [`pf.estimation.support_matrix()`](/reference/estimation.capabilities.support_matrix.qmd)
        gives the same table across all estimators.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2", pf.get_data(), weights="weights")
        fit.capabilities()
        ```
        """
        reasons = self._capabilities.evaluate(self._fit_features)
        index = pd.Index(list(reasons), name="feature")
        return pd.DataFrame(
            {
                "supported": pd.Series(
                    [reason is None for reason in reasons.values()], index=index
                ),
                # object dtype so a supported feature keeps its None reason
                # instead of being coerced to a missing string value.
                "reason": pd.Series(list(reasons.values()), index=index, dtype=object),
            }
        )

    def _require_fit_arrays(
        self,
        method: str,
        *,
        arrays: str,
        remedy: str = "Refit with lean=False.",
    ) -> None:
        """Reject a call whose input arrays `lean=True` discarded."""
        if self._lean and self._fit_state_discarded:
            raise RuntimeError(
                f"{method}() is unavailable after fitting with lean=True because "
                f"{arrays} were discarded. {remedy}"
            )

    def _require_estimation_data(
        self,
        method: str,
        *,
        remedy: str = "Refit with store_data=True.",
    ) -> None:
        """Reject a call whose estimation data the storage options discarded."""
        if not hasattr(self, "_data"):
            raise RuntimeError(
                f"{method}() is unavailable when store_data=False or lean=True "
                f"because the estimation data were discarded. {remedy}"
            )

    def _clear_attributes(self):
        attributes = []

        if not self._store_data:
            attributes += ["_data", "_model_matrix"]

        if self._lean:
            # The array aliases are read-only views, so the state objects
            # backing them are what has to go.
            attributes += [
                "_data",
                "_model_matrix",
                "_within_data",
                "_observation_weights",
                "_demean_cache",
                "_fe",
                "_response",
                "_cluster_df",
                "_tXZ",
                "_tZy",
                "_tZX",
                "_tZZinv",
                "_scores",
                "_u_hat",
                "_Y_hat_link",
                "_Y_hat_response",
            ]

        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

        if self._lean:
            self._fit_state_discarded = True

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
        data: DataFrameType | None = None,
    ) -> BaseRegression:
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
            The already-filtered estimation sample in its original estimation
            order. This is required for data-dependent covariance updates when
            the fitted model does not retain its input data. If None, tries to
            fetch the data from the model object. Defaults to None.


        Returns
        -------
        BaseRegression
            The fitted result itself, with the covariance matrix and every
            derived inference quantity replaced.

        Examples
        --------
        Updates the variance estimator of a fitted model without refitting it.
        The model is modified in place and returned.

        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.vcov("iid").tidy()
        ```

        ```{python}
        # switch to cluster-robust inference
        fit.vcov({"CRV1": "f1"}).tidy()
        ```

        See [On Small Sample Corrections](/explanation/ssc.qmd) for how the
        `ssc` adjustments interact with each estimator.
        """
        self._require_fit_arrays(
            "vcov",
            arrays="the required estimation arrays",
            remedy="Set vcov at estimation time or refit with lean=False.",
        )

        data_to_check: pd.DataFrame | None
        if data is None:
            data_to_check = getattr(self, "_data", None)
        else:
            try:
                data_to_check = _narwhals_to_pandas(data)
            except TypeError as e:
                raise TypeError(
                    f"The data set must be a DataFrame type. Received: {type(data)}"
                ) from e
            if len(data_to_check) != self._N_rows:
                raise ValueError(
                    "`data` passed to vcov() must contain exactly the already-filtered "
                    "estimation sample in its original estimation order; expected "
                    f"{self._N_rows} rows, received {len(data_to_check)}."
                )

        # assign estimated fixed effects, and fixed effects nested within cluster.

        # deparse vcov input
        _check_vcov_input(vcov=vcov, vcov_kwargs=vcov_kwargs, data=data_to_check)

        vcov_type, vcov_type_detail, is_clustered, clustervar = _deparse_vcov_input(
            vcov, self._has_fixef, self._is_iv
        )

        # Reject before any inference state is overwritten, so a failed update
        # leaves the previous covariance estimate intact.
        if vcov_type in {"HAC", "CRV"} and data_to_check is None:
            self._require_estimation_data(
                "vcov",
                remedy="Pass the estimation sample via data= or refit with store_data=True.",
            )

        self._vcov_type = vcov_type
        self._vcov_type_detail = vcov_type_detail
        self._is_clustered = is_clustered
        self._clustervar = clustervar

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
            assert data_to_check is not None
            kw = vcov_kwargs or {}
            self._lag = kw.get("lag")
            self._time_id = kw.get("time_id")
            self._panel_id = kw.get("panel_id")
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(
                    vcov_type="HAC",
                    G=np.unique(data_to_check[self._time_id]).shape[0],
                )  # number of unique time periods T used
            )
            self._vcov = self._ssc * self._vcov_hac(data=data_to_check)

        elif self._vcov_type == "nid":
            self._ssc, self._df_k, self._df_t = get_ssc(
                **self._make_ssc_kwargs(vcov_type="hetero", G=self._N)
            )
            self._vcov = self._ssc * self._vcov_nid()

        elif self._vcov_type == "CRV":
            assert data_to_check is not None
            prep = prepare_cluster_state(
                data=data_to_check,
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
                cluster_vcov=partial(self._vcov_crv_cluster, data=data_to_check),
            )
        # update p-value, t-stat, standard error, confint
        self.get_inference()

        return self

    def _make_ssc_kwargs(
        self,
        *,
        vcov_type: str,
        G: int | float | list[int],
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
        self,
        clustid: np.ndarray,
        cluster_col: np.ndarray,
        *,
        data: pd.DataFrame,
    ) -> np.ndarray:
        "Pick CRV1 / CRV3-fast / CRV3-slow for one cluster column."
        if self._vcov_type_detail == "CRV1":
            return self._vcov_crv1(clustid=clustid, cluster_col=cluster_col)

        self._require_support("crv3", subject="CRV3 inference")
        use_fast = not self._has_fixef and self._method == "feols" and not self._is_iv
        if use_fast:
            return self._vcov_crv3_fast(clustid=clustid, cluster_col=cluster_col)
        return self._vcov_crv3_slow(
            clustid=clustid,
            cluster_col=cluster_col,
            data=data,
        )

    def _vcov_iid(self):
        return vcov_iid_ols(
            residuals=self._u_hat,
            bread=self._bread,
            N=self._N,
            weights=self._observation_weights.values,
        )

    def _leverage_weights(self) -> np.ndarray | None:
        """Return weights used by the fitted normal equations."""
        return self._observation_weights.values

    def _fixef_weights(self) -> np.ndarray | None:
        """Return weights used by fixed-effect coefficient recovery."""
        return self._observation_weights.values

    def _vcov_hetero(self):
        observation_weights = self._observation_weights.values
        return vcov_hetero(
            scores=self._scores,
            X=self._X,
            tZX=self._tZX,
            # Only the frequency-weight correction divides by the weights.
            weights=(
                observation_weights.reshape((-1, 1))
                if observation_weights is not None and self._weights_type == "fweights"
                else None
            ),
            leverage_weights=self._leverage_weights(),
            weights_type=self._weights_type,
            vcov_type_detail=self._vcov_type_detail,
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
        )

    def _vcov_hac(self, *, data: pd.DataFrame):
        _time_id = self._time_id
        _panel_id = self._panel_id

        self._require_support("hac", subject="HAC inference (NW, DK)")

        # some data checks on input pandas df
        # time needs to be numeric or date else we cannot sort by time
        if not np.issubdtype(data[_time_id], np.number) and not np.issubdtype(
            data[_time_id], np.datetime64
        ):
            raise ValueError(
                "The time variable must be numeric or date, else we cannot sort by time."
            )

        _time_arr = data[_time_id].to_numpy()
        _panel_arr = data[_panel_id].to_numpy() if _panel_id is not None else None

        return vcov_hac(
            scores=self._scores,
            time_arr=_time_arr,
            panel_arr=_panel_arr,
            lag=cast(int | None, self._lag),
            vcov_type_detail=cast(Literal["NW", "DK"], self._vcov_type_detail),
            bread=self._bread,
            is_iv=self._is_iv,
            tXZ=self._tXZ,
            tZZinv=self._tZZinv,
            tZX=self._tZX,
        )

    def _vcov_nid(self):
        "Reject 'nid' covariance for estimators without a conditional density."
        self._require_support("nid", subject="'nid' inference")
        raise NotImplementedError(
            f"'nid' inference is declared supported for models of type "
            f"'{self._estimator}', but the class provides no implementation."
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
            weights=self._observation_weights.values,
            beta_hat=self._beta_hat,
            clustid=clustid,
            cluster_col=cluster_col,
        )

    def _estimation_refit_kwargs(self) -> dict[str, Any]:
        """Return options needed to replay this estimator on modified data."""
        demeaner = self._demeaner
        if isinstance(demeaner, LsmrDemeaner) and isinstance(
            demeaner.preconditioner, Preconditioner
        ):
            # A prebuilt factorization belongs to the original FE design. Keep
            # its algorithmic variant, but rebuild it for the changed row set.
            preconditioner = cast(
                WithinPreconditionerName,
                demeaner.preconditioner.variant.lower(),
            )
            demeaner = replace(demeaner, preconditioner=preconditioner)

        return {
            "weights": self._weights_name,
            "weights_type": self._weights_type,
            "ssc": dict(self._ssc_dict),
            "fixef_rm": "singleton" if self._drop_singletons else "none",
            "solver": self._solver,
            "demeaner": demeaner,
            "drop_intercept": self._drop_intercept,
            "collin_tol": self._collin_tol,
            "context": self._context,
        }

    def _crv3_refit(self, data: pd.DataFrame) -> BaseRegression:
        """Replay this estimator for one leave-one-cluster-out sample.

        Only estimators that declare `crv3` support reach this method, and each
        of them replays its own public entry point so the refit keeps the
        original family and solver configuration.
        """
        raise NotImplementedError(
            f"CRV3 inference is declared supported for models of type "
            f"'{self._estimator}', but the class provides no leave-one-cluster-out "
            "refit."
        )

    def _vcov_crv3_slow(
        self,
        clustid: np.ndarray,
        cluster_col: np.ndarray,
        *,
        data: pd.DataFrame,
    ) -> np.ndarray:
        beta_jack = np.zeros((len(clustid), self._k))

        for ixg, g in enumerate(clustid):
            # direct leave one cluster out implementation
            refit_data = data[~np.equal(g, cluster_col)]
            fit = self._crv3_refit(data=refit_data)
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
        self._pvalue = self._inference_dist.pvalue(self._tstat, self._df_t)
        z = self._inference_dist.crit_val(alpha, self._df_t)

        z_se = z * self._se
        self._conf_int = np.array([self._beta_hat - z_se, self._beta_hat + z_se])

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
            The measures are stored on the model object rather than returned.

        Notes
        -----
        Sets the attributes `_rmse`, `_r2`, `_adj_r2`, `_r2_within`, and
        `_adj_r2_within`. The `_within` variants are computed on the demeaned
        dependent variable and are only defined for models with fixed effects.

        Examples
        --------
        The estimation functions call this during fitting, so the measures are
        available on any fitted model.

        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.get_performance()

        fit._r2, fit._adj_r2, fit._r2_within
        ```
        """
        self._require_fit_arrays(
            "get_performance", arrays="the response and within arrays"
        )
        Y_within = self._within_data.response.flatten()
        Y = self._response
        observation_weights = self._observation_weights.values

        has_intercept = not self._drop_intercept

        if self._has_fixef:
            k_fe = np.sum(self._k_fe - 1) + 1
            adj_factor = (self._N - has_intercept) / (self._N - self._k - k_fe)
            adj_factor_within = (self._N - k_fe) / (self._N - self._k - k_fe)
        else:
            adj_factor = (self._N - has_intercept) / (self._N - self._k)

        if observation_weights is None:
            ssu = np.sum(self._u_hat**2)
            y_center = np.mean(Y)
            ssy = np.sum((Y - y_center) ** 2)
        else:
            weights = observation_weights
            ssu = np.sum(weights * self._u_hat**2)
            y_center = np.average(Y, weights=weights)
            ssy = np.sum(weights * (Y - y_center) ** 2)
        self._rmse = np.sqrt(ssu / self._N)
        self._r2 = 1 - (ssu / ssy)
        self._adj_r2 = 1 - (ssu / ssy) * adj_factor

        if self._has_fixef:
            ssy_within = (
                np.sum(Y_within**2)
                if observation_weights is None
                else np.sum(weights * Y_within**2)
            )
            self._r2_within = 1 - (ssu / ssy_within)
            self._adj_r2_within = 1 - (ssu / ssy_within) * adj_factor_within

    def tidy(
        self,
        alpha: float = 0.05,
        inference_type: InferenceType = "regular",
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
        inference_type : {"regular"}, optional
            Type of coefficient-wise inference to report. Only `"regular"` is
            currently available. Defaults to `"regular"`.

        Returns
        -------
        tidy_df : pd.DataFrame
            A tidy pd.DataFrame containing the regression results, including point
            estimates, standard errors, t-statistics, and p-values.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.tidy()
        ```

        Changing the variance estimator changes the standard errors, t-values
        and p-values reported by `tidy()`.

        ```{python}
        fit.vcov("hetero").tidy()
        ```
        """
        inference_type = self._normalize_inference_type(inference_type)
        if inference_type == "simult":
            raise ValueError(
                "tidy() does not support inference_type='simult'. Use "
                "confint(inference_type='simult') for simultaneous intervals."
            )
        if inference_type == "savi":
            raise NotImplementedError(
                "inference_type='savi' is not available in tidy() yet."
            )

        ub, lb = 1 - alpha / 2, alpha / 2
        try:
            self.get_inference(alpha=alpha)
        except EmptyVcovError:
            warnings.warn(
                "Empty variance-covariance matrix detected",
                UserWarning,
            )

        data = {
            "Coefficient": self._coefnames,
            "Estimate": self._beta_hat,
            "Std. Error": self._se,
            "t value": self._tstat,
            "Pr(>|t|)": self._pvalue,
            # use slice because self._conf_int might be empty
            f"{lb * 100:.1f}%": self._conf_int[:1].flatten(),
            f"{ub * 100:.1f}%": self._conf_int[1:2].flatten(),
        }
        if (
            getattr(self, "_sample_split_var", None) is not None
            and (sample := getattr(self, "_sample_split_value", None)) is not None
        ):
            data["Sample"] = sample
        return pd.DataFrame(data).set_index("Coefficient")

    def _normalize_inference_type(
        self, inference_type: InferenceType, joint: bool = False
    ) -> InferenceType:
        """Validate `inference_type` and fold the deprecated `joint` flag into it."""
        _validate_literal_argument(inference_type, InferenceType)

        if joint:
            warnings.warn(
                "joint=True is deprecated. Use inference_type='simult' instead.",
                FutureWarning,
                stacklevel=3,
            )
            if inference_type not in ("regular", "simult"):
                raise ValueError(
                    "joint=True cannot be combined with "
                    f"inference_type={inference_type!r}."
                )
            inference_type = "simult"

        return inference_type

    def confint(
        self,
        alpha: float = 0.05,
        keep: list | str | None = None,
        drop: list | str | None = None,
        exact_match: bool | None = False,
        joint: bool = False,
        seed: int | None = None,
        reps: int = 10_000,
        *,
        inference_type: InferenceType = "regular",
        mixture_precision: float = 1.0,
    ) -> pd.DataFrame:
        r"""
        Fitted model confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05.
            keep: str or list of str, optional
        joint : bool, optional
            Deprecated. Use `inference_type="simult"` instead. Whether to
            compute simultaneous confidence intervals for the joint null of the
            parameters selected by `keep` and `drop`. Defaults to False. See
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
            The seed for the random number generator. Defaults to None. Only used
            when `inference_type="simult"`.
        inference_type : {"regular", "simult", "savi"}, optional
            Type of confidence interval to compute. "regular" returns pointwise
            intervals; "simult" returns simultaneous (joint) intervals for the
            coefficients selected by `keep` and `drop`; "savi" returns
            coefficient-wise asymptotic SAVI confidence sequences. Defaults to
            "regular". Supersedes the deprecated `joint` argument.
        mixture_precision: float, optional
            Only relevant for `inference_type="savi"`. Controls the mixing weight of the
            prior in the SAVI e-value. Larger values produce wider confidence
            sequences early on but narrow faster as the sample grows. Defaults to 1. Use
            `pyfixest.optimal_mixture_precision()`
            to minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame with confidence intervals of the estimated regression model
            for the selected coefficients.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). With
        `HC2`/`HC3`, pyfixest's default small-sample correction scales the
        variance by `n / (n - k)`. You need to pass `ssc(k_adj=False)` to reproduce `avlm`,
        which applies no such correction.

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
        fit.confint(alpha=0.10, inference_type="simult", reps=9999).head()

        savi_fit = feols("Y ~ X1 + X2", data=data, vcov="hetero")
        savi_fit.confint(alpha=0.10, inference_type="savi").head()
        ```
        """
        inference_type = self._normalize_inference_type(inference_type, joint=joint)
        if inference_type == "savi":
            from pyfixest.estimation.post_estimation.savi import _confint

            return _confint(
                model=self,
                alpha=alpha,
                mixture_precision=mixture_precision,
                keep=keep,
                drop=drop,
                exact_match=exact_match,
            )

        coefnames, coef_indices = _select_coefnames_and_indices(
            self._coefnames, keep, drop, exact_match
        )

        if inference_type == "regular":
            crit_val = self._inference_dist.crit_val(alpha, self._df_t)
        else:
            joint_indices = sorted(coef_indices)
            D_inv = 1 / self._se[joint_indices]
            V = self._vcov[np.ix_(joint_indices, joint_indices)]
            C_coefs = (D_inv * V).T * D_inv
            crit_val = simultaneous_crit_val(C_coefs, reps, alpha=alpha, seed=seed)

        ub = pd.Series(self._beta_hat[coef_indices] + crit_val * self._se[coef_indices])
        lb = pd.Series(self._beta_hat[coef_indices] - crit_val * self._se[coef_indices])

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

        Residuals are stored and returned on the scale of the original dependent
        variable. Observation weights are applied only where an estimating
        equation or diagnostic requires them.

        Returns
        -------
        np.ndarray
            A np.ndarray with the residuals of the estimated regression model.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.resid()[:5]
        ```
        """
        self._require_fit_arrays("resid", arrays="the residual arrays")
        return self._u_hat.flatten()

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
        R = np.eye(self._k) if R is None else np.atleast_2d(np.asarray(R, dtype=float))

        W, self._dfn = _wald_statistic(
            beta_hat=self._beta_hat,
            vcov=self._vcov,
            R=R,
            q=q,
        )

        if self._is_clustered:
            self._dfd = np.min(np.array(self._G)) - 1
        else:
            self._dfd = self._N - self._k - k_fe

        self._wald_statistic = W

        # The F distribution is only used for the joint test that all
        # coefficients are zero (R identity, q zero).
        if distribution == "F" and (
            not np.array_equal(R, np.eye(self._k)) or (q is not None and np.any(q))
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

    def fixef(self, atol: float = 1e-06, btol: float = 1e-06) -> pd.DataFrame:
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
        pd.DataFrame
            A tidy DataFrame with columns `variable`, `code`, `level`, and
            `coefficient` containing the estimated fixed effects.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fixed_effects = fit.fixef()
        fixed_effects.head()
        ```
        """
        if not self._has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        self._require_support("fixef", subject="The fixef() method")

        self._require_fit_arrays("fixef", arrays="the fitted arrays")
        self._require_estimation_data("fixef")

        fixef_weights = self._fixef_weights()

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
        # one-hot encoding of fixed effects (treatment coding: reference level
        # dropped for the second and subsequent FEs via ensure_full_rank=True).
        contrast_coding = contrast_code_fixed_effects(
            fixed_effects=self.FixestFormula.fixed_effects_wrapped,
            fixed_effect_names=self._model_spec[
                _ModelMatrixKey.fixed_effects
            ].column_names,
            data=self._data,
            context=FORMULAIC_TRANSFORMS | {**self._context},
            transform_state=self._model_spec[
                _ModelMatrixKey.fixed_effects
            ].transform_state,
        )
        fixed_effect_design = contrast_coding.matrix
        solve_design = fixed_effect_design
        if fixef_weights is not None:
            weights_sqrt = np.sqrt(fixef_weights).flatten()
            uhat *= weights_sqrt
            weights_diag = diags(weights_sqrt, 0)
            solve_design = weights_diag.dot(fixed_effect_design)

        alpha = lsqr(solve_design, uhat, atol=atol, btol=btol)[0]

        self._fixef_coefficients = build_fixed_effects(
            fixed_effect_coefficients=alpha,
            contrast_coding=contrast_coding,
            transform_state=self._model_spec[
                _ModelMatrixKey.fixed_effects
            ].transform_state,
        )
        self._alpha = alpha
        self._sumFE = fixed_effect_design.dot(alpha)

        return fixed_effects_to_frame(self._fixef_coefficients)

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

        Examples
        --------
        In-sample predictions:

        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2 | f1", data)
        fit.predict()[:5]
        ```

        Pass `newdata` to predict out of sample. Fixed effect levels that do not
        appear in the estimation sample return missing values.

        ```{python}
        fit.predict(newdata=data.head())
        ```
        """
        self._require_support("predict", subject="The predict() method")

        if interval == "prediction" or se_fit:
            self._require_support(
                "prediction_errors", subject="Prediction with standard errors"
            )

        _validate_literal_argument(type, PredictionType)
        if interval is not None:
            _validate_literal_argument(interval, PredictionErrorOptions)

        if newdata is None or se_fit or interval == "prediction":
            self._require_fit_arrays(
                "predict", arrays="the fitted design and residual arrays"
            )

        if newdata is None:
            # note: no need to worry about fixed effects, as not supported with
            # prediction errors; will throw error later;
            X = self._X
            y_hat = (
                self._Y_hat_link
                if type == "link" or self._method == "feols"
                else self._Y_hat_response
            )
            n_observations = self._N_rows
        else:
            newdata = _narwhals_to_pandas(newdata).reset_index(drop=True)
            n_observations = newdata.shape[0]
            context = FORMULAIC_TRANSFORMS | {**self._context}
            # Use na_action="drop" on each sub-spec separately because dependent variable
            # may not be available in newdata, then intersect indices so a NaN in *any* variable
            # (covariate or FE) marks the whole row as NaN in the output.
            rhs_spec = self._model_spec[_ModelMatrixKey.main].rhs
            X_mm, unseen = materialize_model_spec_with_unseen_mask(
                rhs_spec, newdata, context
            )
            valid_idx = X_mm.index.to_numpy()
            # rows with a categorical level unseen during fitting (in C()/i()) would
            # be silently encoded as the reference level -> drop them to NaN instead,
            # matching how unseen fixed-effect levels are handled below.
            valid_idx = valid_idx[~unseen[valid_idx]]
            if self._has_fixef:
                # Fixed-effect levels are recovered from the estimation sample.
                self._require_fit_arrays(
                    "predict", arrays="the fitted design and residual arrays"
                )
                self._require_estimation_data("predict")
                fe_spec = self._model_spec[_ModelMatrixKey.fixed_effects]
                check_fe_dtype_compatibility(fe_spec, newdata)
                # na_action="ignore" keeps unseen-level rows as NaN codes
                fe_mm = fe_spec.get_model_matrix(
                    newdata, context=context, na_action="ignore"
                )
                warn_on_unseen_fixed_effect_levels(fe_mm, fe_spec, newdata)
                valid_fixed_effects = fe_mm.notna().all(axis="columns").to_numpy()
                valid_idx = valid_idx[valid_fixed_effects[valid_idx]]
                if self._sumFE is None:
                    self.fixef(atol, btol)
                fe_hat = predict_fixed_effects(
                    model_matrix=fe_mm.loc[valid_idx],
                    coefficients=self._fixef_coefficients,
                )

            X_coef = X_mm.loc[valid_idx, self._coefnames].to_numpy()
            y_hat = np.full(n_observations, np.nan)
            y_hat[valid_idx] = X_coef @ self._beta_hat
            if self._has_fixef:
                y_hat[valid_idx] += fe_hat
            # Pad X to full size; NaN rows yield NaN SE/CI via einsum propagation.
            X = np.full((n_observations, X_coef.shape[1]), np.nan)
            X[valid_idx] = X_coef
            if self._offset_name is not None:
                offset_mm = self._model_spec[_ModelMatrixKey.offset].get_model_matrix(
                    newdata,
                    context=context,
                    na_action="drop",
                    output="pandas",
                )
                if not offset_mm.index.equals(newdata.index):
                    raise ValueError(
                        f"Offset expression '{self._offset_name}' evaluates to missing "
                        "values in `newdata`."
                    )

                y_hat += offset_mm.iloc[:, 0].to_numpy()

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
        from pyfixest.estimation.post_estimation.ritest import (
            _HAS_NUMBA,
            _decode_resampvar,
            _get_ritest_pvalue,
            _get_ritest_stats_fast,
            _get_ritest_stats_slow,
        )

        resampvar = resampvar.replace(" ", "")
        resampvar_, h0_value, hypothesis, test_type = _decode_resampvar(resampvar)

        self._require_support("ritest", subject="Randomization inference")

        # check that resampvar in _coefnames
        if resampvar_ not in self._coefnames:
            raise ValueError(f"{resampvar_} not found in the model's coefficients.")

        self._require_fit_arrays("ritest", arrays="the fitted arrays")
        self._require_estimation_data("ritest")

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
                refit_kwargs=self._estimation_refit_kwargs(),
            )

        else:
            observation_weights = self._observation_weights.values
            # The demeaning kernel behind the fast path always needs a vector.
            weights = (
                np.ones(self._N_rows, dtype=np.float64)
                if observation_weights is None
                else observation_weights
            )
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
        from pyfixest.estimation.post_estimation.ritest import _plot_ritest_pvalue

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

    def evalue(
        self,
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """Compute coefficient-wise SAVI e-values.

        Parameters
        ----------
        mixture_precision : float, optional
            Positive mixture precision fixed before sequential monitoring.
            Defaults to 1. Use `pyfixest.optimal_mixture_precision()` to
            minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.Series
            One e-value per coefficient.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). Note that
        for `HC2`/`HC3`, pyfixest's default small-sample correction scales the
        variance by `n / (n - k)` while the R implementation in `avlm` does not.
        Inference is pointwise / by coefficient.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
        fit.evalue()
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _evalue

        return _evalue(model=self, mixture_precision=mixture_precision)

    def pvalue_savi(
        self,
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """Compute coefficient-wise SAVI sequential p-values.

        The sequential-p-value analogue of `evalue`. See `evalue` for the
        `mixture_precision` argument and the supported-model restrictions.

        Returns
        -------
        pd.Series
            One sequential p-value per coefficient.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
        fit.pvalue_savi()
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _pvalue_savi

        return _pvalue_savi(model=self, mixture_precision=mixture_precision)


def _check_vcov_input(
    vcov: str | dict[str, str],
    vcov_kwargs: dict[str, Any] | None,
    data: pd.DataFrame | None,
):
    """
    Check the input for the vcov argument of a fitted result.

    Parameters
    ----------
    vcov : Union[str, dict[str, str]]
        The vcov argument passed to the fitted result.
    vcov_kwargs : Optional[dict[str, Any]]
        The vcov_kwargs argument passed to the fitted result.
    data : pd.DataFrame or None
        The estimation sample, or None when the fitted model retains none.

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
        if data is None:
            raise RuntimeError(
                "A vcov column list requires estimation data. Pass data= or fit "
                "with store_data=True."
            )
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
    Deparse the vcov argument passed to a fitted result.

    Parameters
    ----------
    vcov : Union[str, dict[str, str]]
        The vcov argument passed to the fitted result.
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
