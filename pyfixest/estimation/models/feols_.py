from __future__ import annotations

import re
import warnings
from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import ClassVar, Literal, cast, overload

import formulaic
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import lsqr
from scipy.stats import t

from pyfixest.core.demean import Preconditioner
from pyfixest.errors import VcovTypeNotSupportedError
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.formulaic_compat import (
    i_term_columns,
    materialize_model_spec_with_unseen_mask,
)
from pyfixest.estimation.formula.model_matrix import (
    ModelMatrix,
    _ModelMatrixKey,
    create_model_matrix,
)
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.demean_ import DemeanCache, DemeanedData
from pyfixest.estimation.internals.families import T_DIST
from pyfixest.estimation.internals.fit_ import fit_ols
from pyfixest.estimation.internals.fit_statistics import (
    FitStatistics,
    linear_fit_statistics,
)
from pyfixest.estimation.internals.literals import (
    HacVcovTypeOptions,
    HeteroVcovTypeOptions,
    PredictionErrorOptions,
    PredictionType,
    WaldDistributionOptions,
    WeightsTypeOptions,
    _validate_literal_argument,
)
from pyfixest.estimation.internals.model_state import (
    Capabilities,
    CollinearityCheck,
    EstimationOptions,
    EstimationSample,
    FittedValues,
    FixedEffectCounts,
    ModelDescription,
    ObservationWeights,
    RitestStatistics,
    SampleSplit,
    SandwichComponents,
    VarianceCovariance,
    VcovSpec,
    WaldTest,
    WithinLinearData,
)
from pyfixest.estimation.internals.retention import (
    omitted_attributes,
    require_retained,
)
from pyfixest.estimation.internals.vcov_ import (
    meat_crv1,
    meat_hac,
    meat_hetero,
    vcov_crv3_fast,
    vcov_iid_ols,
)
from pyfixest.estimation.internals.vcov_utils import (
    ClusterSmallSampleCorrection,
    VcovTerm,
    combine_terms,
    get_ssc_cluster,
    prepare_cluster_state,
)
from pyfixest.estimation.models._result_accessor_mixin import ResultAccessorMixin
from pyfixest.estimation.post_estimation.decomposition import (
    GelbachDecomposition,
    _decompose_arg_check,
)
from pyfixest.estimation.post_estimation.fixed_effects import (
    FixedEffectEstimates,
    build_fixed_effect_dummies,
    build_fixed_effects,
    fixed_effects_to_frame,
    sum_fixed_effects,
)
from pyfixest.estimation.post_estimation.prediction import _compute_prediction_error
from pyfixest.estimation.post_estimation.wald import wald_test
from pyfixest.estimation.refit import refit
from pyfixest.utils.dev_utils import (
    DataFrameType,
    _find_stack_level,
    _narwhals_to_pandas,
)
from pyfixest.utils.utils import (
    DegreesOfFreedomCounts,
    SmallSampleCorrection,
    get_ssc,
)

decomposition_type = Literal["gelbach"]
prediction_type = Literal["response", "link"]


def _fixed_effect_names(model_matrix: ModelMatrix) -> tuple[str, ...]:
    """Name each absorbed term in materialized fixed-effect column order.

    Empty when the materialized model matrix carries no fixed-effect block.
    """
    fixed_effects = model_matrix.fixed_effects
    if fixed_effects is None:
        return ()
    return tuple(fixed_effects.columns)


class Feols(ResultAccessorMixin):
    """Fitted linear-regression result returned by `pf.feols()`.

    Access this object as ``fit = pf.feols(formula, data)``. Use the estimation
    function to construct it. Fit several specifications in one call to obtain a
    `FixestMulti` container instead. An IV formula returns the `Feiv` subclass.

    Use ``fit.tidy()`` for labeled coefficient results, ``fit.predict()`` for
    predictions, and ``fit.fitstat`` for goodness-of-fit statistics. The
    [fitted-result overview](/reference-guide/fitted-results.qmd) groups the
    available operations by task.

    Attributes
    ----------
    model : ModelDescription
        Access as ``fit.model``: formula, estimator, dependent variable,
        absorbed fixed effects, interaction names, and sample-split metadata.
    options : EstimationOptions
        Access as ``fit.options``: estimation settings and storage policy.
    coeftable : CoefficientTable
        Coefficient arrays and confidence bounds, refreshed during estimation,
        by ``vcov()``, and by ``tidy()``. Prefer ``tidy()`` for labeled output.
    fitstat : FitStatistics
        Goodness-of-fit measures; unavailable measures are ``NaN``.
    sample_info : EstimationSample
        Retained observation counts and dropped row positions/counts.
    collinearity : CollinearityCheck
        Retained coefficient names and columns removed by the rank check.
    fixef_counts : FixedEffectCounts
        Counts of absorbed fixed-effect levels used in inference.
    variance_covariance : VarianceCovariance
        Current coefficient covariance matrix, its specification, adjustments,
        and degrees of freedom. Published during estimation and by ``vcov()``.
    capabilities : Capabilities
        Features supported by this estimator. Individual operations can also
        require particular weights, fixed effects, or retained data.
    fitted_values : FittedValues
        In-sample fitted values, in retained-row order. Removed by ``lean=True``.
    model_matrix : ModelMatrix
        Formula-derived model frames. Removed by ``store_data=False`` or
        ``lean=True``.
    within_data : WithinLinearData
        Demeaned response and regressors after collinear columns are removed;
        not multiplied by square-root weights. Removed by ``lean=True``.
    observation_weights : ObservationWeights
        User-scale weights and their interpretation. Removed by ``lean=True``.
    sandwich : SandwichComponents
        Scores, Hessian, and bread for covariance construction. Not available
        for a fixed-effects-only fit; removed by ``lean=True``.
    wald : WaldTest
        Latest Wald test. Non-IV OLS fits with coefficients run the joint test
        automatically; ``wald_test()`` publishes a new value.
    fixef_estimates : FixedEffectEstimates
        Created by ``fixef()``: recovered fixed effects and their contribution
        to each observation.
    ritest_statistics : RitestStatistics
        Created by ``ritest(store_ritest_statistics=True)`` for later plotting.
    preconditioner : Preconditioner or None
        Within-backend preconditioner used in demeaning, when available. It can
        be reused on a later fit over the same design.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
    fit.tidy()
    fit.fitstat.r2_within
    ```

    Notes
    -----
    Storage options remove attributes rather than replacing them with ``None``.
    See [model storage](/reference-guide/model-storage.qmd) before discarding
    state needed by a later operation. Subclasses share result methods but do
    not support every operation; see
    [model support](/reference-guide/model-support.qmd).

    See Also
    --------
    [feols()](/reference/estimation.api.feols.feols.qmd),
    [FixestMulti](/reference/estimation.FixestMulti_.FixestMulti.qmd),
    [fitted-result overview](/reference-guide/fitted-results.qmd).
    """

    # Set in get_fit().
    collinearity: CollinearityCheck
    sandwich: SandwichComponents
    fitted_values: FittedValues
    # Set in _publish_fit_statistics() right after get_fit().
    fitstat: FitStatistics
    # Set in vcov().
    variance_covariance: VarianceCovariance
    # Set in wald_test().
    wald: WaldTest
    # Set in ritest() when store_ritest_statistics is True.
    ritest_statistics: RitestStatistics
    # Set in fixef().
    fixef_estimates: FixedEffectEstimates

    # The fit is a single least-squares solve, so the fast CRV3 jackknife and
    # the fast ritest algorithm apply. Subclasses with other fits override it.
    _closed_form_ols: bool = True

    # Features this model class supports; __init__ copies them to
    # `capabilities`. Each subclass declares its own value in full.
    _declared_capabilities: ClassVar[Capabilities] = Capabilities(
        covariance_update=True,
        crv3_inference=True,
        hac_inference=True,
        multiway_clustering=True,
        wildboottest=True,
        cluster_causal_variance=True,
        decomposition=True,
        prediction=True,
        fixed_effect_recovery=True,
        randomization_inference=True,
        sherman_morrison_update=True,
        anytime_valid_inference=True,
    )

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: EstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split: SampleSplit | None = None,
    ) -> None:
        self.options = options
        self.model = self._describe_model(
            parsed_formula=FixestFormula, sample_split=sample_split
        )
        self._model_name_plot = self.model.model_name

        if sample_split is None:
            pass
        elif sample_split.value is None:
            data = data.loc[data[sample_split.var].notnull()]
        else:
            data = data.loc[data[sample_split.var] == sample_split.value]

        data = data.reset_index(drop=True)

        self._data = data.copy() if options.copy_data else data
        self._demean_cache = DemeanCache(lookup_demeaned_data, lookup_preconditioner)

        self.capabilities = self._declared_capabilities
        if self.options.has_weights:
            self.capabilities = replace(self.capabilities, wildboottest=False)

    def _describe_model(
        self,
        *,
        parsed_formula: FixestFormula,
        sample_split: SampleSplit | None,
    ) -> ModelDescription:
        """Describe the model this class fits, before its model matrix exists.

        Subclasses override this to name their estimation function and its
        inference distribution. The matrix-time fields stay empty until
        `_publish_model_matrix()` republishes the description with them. An
        unsplit fit publishes ``None`` as its split variable and value.
        """
        model_name = parsed_formula.render()
        if sample_split is not None:
            sample_label = (
                "all" if sample_split.value is None else repr(sample_split.value)
            )
            model_name += f" (Sample: {sample_split.var} = {sample_label})"

        return ModelDescription(
            parsed_formula=parsed_formula,
            method="feols",
            is_iv=False,
            model_name=model_name,
            sample_split=sample_split,
            inference_dist=T_DIST,
        )

    def prepare_model_matrix(self):
        """Build and retain the canonical formula-derived estimator inputs."""
        model_matrix = create_model_matrix(
            formula=self.model.parsed_formula,
            data=self._data,
            drop_singletons=self.options.drop_singletons,
            drop_intercept=self.options.drop_intercept,
            weights=self.options.weights,
            offset=self.options.offset,
            context=self.options.context,
        )
        self._publish_model_matrix(model_matrix)

        # an empty drop still rebuilds the whole frame, so guard it
        if model_matrix.dropped_row_index:
            self._data.drop(index=list(model_matrix.dropped_row_index), inplace=True)

        return model_matrix

    # Deliberately untyped: an annotation makes mypy check this body, whose published
    # attribute types are still transitional or loose; see the typing follow-up.
    def _publish_model_matrix(self, model_matrix):
        """Publish structurally immutable formula, sample, and weight state."""
        self.model_matrix = model_matrix
        independent = model_matrix.independent
        self.model = replace(
            self.model,
            depvar=model_matrix.dependent.columns[0],
            fixed_effects=_fixed_effect_names(model_matrix=model_matrix),
            interacted_covariates=(
                tuple(i_term_columns(model_matrix.model_spec[_ModelMatrixKey.main].rhs))
            ),
            model_spec=model_matrix.model_spec,
        )
        self._X_is_empty = independent.shape[1] == 0

        self._coefnames = independent.columns.tolist()

        fixed_effects = model_matrix.fixed_effects
        self.fixef_counts = (
            FixedEffectCounts(
                n_levels_by_fe=tuple(
                    int(size) for size in fixed_effects.nunique(axis=0)
                )
            )
            if fixed_effects is not None
            else FixedEffectCounts()
        )

        self.observation_weights = self._set_observation_weights()
        weights = self.observation_weights
        n_rows = model_matrix.n_rows
        # Frequency weights count repeated observations; other fits count rows.
        if weights.weights_type == "fweights":
            assert weights.values is not None
            n_obs: int | float = float(weights.values.sum())
        else:
            n_obs = n_rows
        self.sample_info = EstimationSample(
            dropped_row_index=model_matrix.dropped_row_index,
            n_rows=n_rows,
            n_obs=n_obs,
            dropped_by_stage=model_matrix.dropped_by_stage,
        )

    def _validate_response(self) -> None:
        """Validate estimator-specific response constraints, if any."""

    def _set_observation_weights(self) -> ObservationWeights:
        """Build canonical user-scale observation weights for this row sample."""
        weights = self.model_matrix.weights
        if weights is None:
            return ObservationWeights.unweighted()

        # `weights_type` is validated at the API boundary (estimation/api/utils.py).
        weights_type = cast(WeightsTypeOptions, self.options.weights_type)
        return ObservationWeights.from_values(
            weights.to_numpy().reshape(-1),
            weights_type=weights_type,
        )

    def _demean(self) -> WithinLinearData:
        """Convert model frames to arrays and residualize them on the fixed effects.

        The returned arrays are in the units of the data; they are not
        multiplied by square-root weights.
        """
        response_frame = self.model_matrix.dependent
        design_frame = self.model_matrix.independent
        response = response_frame.to_numpy(dtype=np.float64)
        design = design_frame.to_numpy(dtype=np.float64)

        fixed_effects = self.model_matrix.fixed_effects
        if fixed_effects is not None:
            response, design, _ = self._demean_cache.demean_yx(
                response,
                design,
                y_keys=self.model_matrix.column_keys("dependent"),
                x_keys=self.model_matrix.column_keys("independent"),
                fe=fixed_effects.to_numpy(),
                weights=self.observation_weights.values,
                na_index=self.sample_info.dropped_row_index,
                demeaner=self.options.demeaner,
            )
        return WithinLinearData(response=response, design=design)

    @property
    def preconditioner(self) -> Preconditioner | None:
        """The within preconditioner used during demeaning, if any.

        ``None`` when no preconditioner participated in the solve —
        ``preconditioner='off'``, single-FE designs (MAP fallback), or any
        non-within backend. Otherwise the instance built on the first solve
        for this model's row sample, or the additive Schwarz preconditioner
        that replaced it when a later adaptive solve escalated. Pass it back via
        ``LsmrDemeaner(backend='within', preconditioner=...)`` to skip the
        setup phase on a later fit over the same design.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        return self._demean_cache.lookup_preconditioner.get(
            self.sample_info.dropped_row_index
        )

    def _drop_multicollinear_within_data(
        self, within_data: WithinLinearData
    ) -> WithinLinearData:
        """Return within data after the established unweighted rank check."""
        design = within_data.design
        if design.shape[1] == 0:
            # Fixed-effects-only model: nothing to check, but the attribute is
            # published for every fitted model.
            self.collinearity = CollinearityCheck(
                dropped_coef_names=(),
                mask=tuple(False for _ in self._coefnames),
                coefnames=tuple(self._coefnames),
            )
            return within_data

        design, collinearity = drop_multicollinear_variables(
            design,
            self._coefnames,
            self.options.collin_tol,
        )
        self.collinearity = collinearity
        self._coefnames = list(collinearity.coefnames)

        return replace(within_data, design=design)

    def _set_within_data(self, within_data: WithinLinearData) -> None:
        """Publish canonical within data after column selection."""
        self.within_data = within_data
        self._X_is_empty = within_data.design.shape[1] == 0
        self._k = within_data.design.shape[1]

    def _prediction_design(self) -> np.ndarray:
        """Return the coefficient-ordered design for the shared predict() method.

        Linear and quantile models retain it in within_data. The GLM override
        reads the corresponding final IRLS design from working_state.
        """
        require_retained(self, "predict", "within_data")
        return self.within_data.design

    def _fixef_dependent(self) -> np.ndarray:
        """Return the dependent variable of the fixed-effect regression in `fixef()`.

        `fixef()` regresses this variable minus `X @ beta_hat` on the fixed
        effects. Linear models use the observed `Y` of the estimation sample;
        the GLM override uses the linear predictor net of the offset.
        """
        return self.model_matrix.dependent.to_numpy().flatten().astype(np.float64)

    def get_fit(self) -> None:
        """
        Fit an OLS model.

        Returns
        -------
        None
        """
        within_data = self._drop_multicollinear_within_data(self._demean())
        self._set_within_data(within_data)

        if self._X_is_empty:
            # Fixed-effects-only model: no coefficients, residuals are the
            # within-transformed response, and the plan skips vcov().
            self._beta_hat = np.empty(0)
            self._u_hat = within_data.response.flatten()
        else:
            fit = fit_ols(
                X=within_data.design,
                Y=within_data.response,
                weights=self.observation_weights.values,
                solver=self.options.solver,
            )

            self._beta_hat = fit.beta
            self._u_hat = fit.residuals
            self.sandwich = fit.sandwich

        # The response minus the residual carries the fixed-effect
        # contribution, which `design @ beta_hat` alone would omit.
        fitted = self.model_matrix.dependent.to_numpy().flatten() - self.resid()
        self.fitted_values = FittedValues(link=fitted, response=fitted)

    def _publish_fit_statistics(self) -> None:
        """Publish `_fit_statistics()` as `fitstat`."""
        self.fitstat = self._fit_statistics()

    def _fit_statistics(self) -> FitStatistics:
        """Compute the goodness-of-fit measures of the fitted model.

        `_publish_fit_statistics()` stores the result as `fitstat` right after
        `get_fit()`, before `lean=True` clears the arrays read here.
        Subclasses override this hook to compute their own measures; an
        override returns ``FitStatistics()`` where the estimator defines none.
        """
        # Empty designs are used only for demeaning and may have no residual
        # degrees of freedom. Leave their fit statistics undefined.
        if self._X_is_empty:
            return FitStatistics()
        return linear_fit_statistics(
            Y=self.model_matrix.dependent.to_numpy(),
            Y_within=self.within_data.response,
            residuals=self._u_hat,
            weights=self.observation_weights.values,
            N=self.sample_info.n_obs,
            k=self._k,
            k_fe=self.fixef_counts.fixef_dof,
            has_intercept=not self.options.drop_intercept,
            has_fixef=self.model.has_fixef,
        )

    def _finalize_fit(self) -> None:
        """Run the OLS Wald test that all coefficients are zero."""
        self.wald_test()

    def _iter_fitted_models(self) -> tuple[Feols, ...]:
        """Yield this fitted result to the result container."""
        return (self,)

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
        data: DataFrameType | None = None,
    ) -> Feols:
        """Update a fitted model's covariance estimate and inference in place.

        The same fitted result is returned, without refitting the coefficients.
        The update refreshes ``variance_covariance`` and ``coeftable``.

        Parameters
        ----------
        vcov : str or dict[str, str]
            A string or dictionary specifying the type of variance-covariance matrix
            to use for inference.
            If a string, it can be one of `iid`, `hetero`, `HC1`, `HC2`, `HC3`,
            `NW` (Newey-West) or `DK` (Driscoll-Kraay); quantile regression
            fits also accept `nid`.
            If a dictionary, it should have the format `{"CRV1": "clustervar"}` for
            CRV1 inference or `{"CRV3": "clustervar"}` for CRV3 inference. Use
            "f1 + f2" for two-way clustering.
            Not every model supports every estimator; unsupported choices raise
            an error. `HC2` and `HC3` are rejected for models with fixed effects
            and for IV models. `CRV3` is rejected for IV, non-Poisson GLM and
            quantile regression fits. HAC is rejected for quantile regression
            and with `weights_type="fweights"`. See `fit.capabilities`.
        vcov_kwargs : dict[str, str | int], optional
            Additional arguments for the HAC estimators: "lag" (the number of
            lags; time-series Newey-West currently requires it), "`time_id`" (the
            time variable, required for NW and DK) and "`panel_id`" (the panel
            identifier, required for DK and for panel Newey-West).
        data : DataFrameType, optional
            The data used for estimation. If `None`, tries to fetch the data from the
            model object. Defaults to `None`.


        Returns
        -------
        Feols
            The same fitted result (including its estimator-specific subclass),
            with updated inference.

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

        Notes
        -----
        Requires retained estimation data even when ``data`` is supplied.
        Refit with ``store_data=True, lean=False`` if needed; see
        [model storage](/reference-guide/model-storage.qmd) and
        [model support](/reference-guide/model-support.qmd).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        self._require_capability(
            capability="covariance_update",
            method="vcov",
            exception_type=VcovTypeNotSupportedError,
        )
        spec = VcovSpec.from_user_input(vcov, vcov_kwargs)
        self._check_vcov_support(spec)
        return self._vcov_from_spec(spec, data=data)

    def _check_vcov_support(self, spec: VcovSpec) -> None:
        """Reject a covariance estimator this model cannot compute.

        Reads only the model description, options, and capabilities, so the
        estimation pipeline runs it before fitting.
        """
        if spec.vcov_type_detail in ("HC2", "HC3"):
            if self.model.has_fixef:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for regressions with fixed effects."
                )
            if self.model.is_iv:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for IV regressions."
                )
        if spec.vcov_type == "CRV":
            if len(spec.clustervar) > 1 and not self.capabilities.multiway_clustering:
                raise NotImplementedError(
                    f"Multiway clustering is not (yet) supported for {type(self).__name__} models."
                )
            if spec.vcov_type_detail == "CRV3" and not self.capabilities.crv3_inference:
                raise VcovTypeNotSupportedError(
                    f"CRV3 inference is not for models of type '{self.model.method}'."
                )
        if spec.vcov_type == "HAC":
            if not self.capabilities.hac_inference:
                raise NotImplementedError(
                    "HAC inference is not supported for this model type."
                )
            if self.options.has_weights and self.options.weights_type == "fweights":
                raise NotImplementedError(
                    "HAC inference (NW, DK) is not supported with `weights_type='fweights'`."
                )
        if spec.vcov_type == "nid":
            raise NotImplementedError(
                "Only models of type Quantreg support a variance-covariance matrix of type 'nid'."
            )

    def _vcov_from_spec(
        self, spec: VcovSpec, data: DataFrameType | None = None
    ) -> Feols:
        """Compute and publish the covariance of a parsed, supported `spec`.

        `vcov()` parses and checks the user's input first; the estimation
        pipeline passes the spec it parsed before fitting.
        """
        require_retained(self, "vcov", "_data")

        data_to_check = data if data is not None else self._data
        try:
            data_to_check = _narwhals_to_pandas(data_to_check)
        except TypeError as e:
            raise TypeError(
                f"The data set must be a DataFrame type. Received: {type(data)}"
            ) from e

        vcov_type = spec.vcov_type

        # Every estimator follows the same three steps: small-sample factors,
        # one unadjusted term per cluster dimension, and their combination.
        G: tuple[int, ...] = ()
        df_t: int | float
        correction: SmallSampleCorrection | ClusterSmallSampleCorrection
        if vcov_type == "CRV":
            prep = prepare_cluster_state(
                data=data_to_check,
                clustervar=list(spec.clustervar),
                ssc=self.options.ssc,
                fixef=self.model.fixed_effects,
                fe=self.model_matrix.fixed_effects,
                n_levels_by_fe=self.fixef_counts.n_levels_by_fe,
            )
            G = tuple(int(g) for g in prep.G)
            correction = get_ssc_cluster(
                prep=prep, ssc_options=self.options.ssc, dof_counts=self._dof_counts
            )
            ssc, df_k, df_t = correction.adj, correction.df_k, correction.df_t
            terms = [
                self._vcov_crv_cluster(
                    clustid=clustid,
                    cluster_col=cluster_col,
                    vcov_type_detail=spec.vcov_type_detail,
                )
                for clustid, cluster_col in prep.dimensions()
            ]
        else:
            ssc_G: int | float
            if vcov_type == "iid":
                ssc_vcov_type, ssc_G = "iid", 1
                term = self._vcov_iid()
            elif vcov_type == "hetero":
                # fixest:::vcov_hetero_internal: adj = ifelse(ssc$cluster.adj, n/(n - 1), 1)
                ssc_vcov_type, ssc_G = "hetero", self.sample_info.n_obs
                term = self._vcov_hetero(vcov_type_detail=spec.vcov_type_detail)
            elif vcov_type == "HAC":
                # G is the number of unique time periods T used
                ssc_vcov_type = "HAC"
                ssc_G = np.unique(self._data[spec.time_id]).shape[0]
                term = self._vcov_hac(spec)
            elif vcov_type == "nid":
                ssc_vcov_type, ssc_G = "hetero", self.sample_info.n_obs
                term = self._vcov_nid()
            else:
                raise ValueError(f"Unknown vcov type {vcov_type!r}.")
            correction = get_ssc(
                self.options.ssc, self._dof_counts(G=ssc_G), vcov_type=ssc_vcov_type
            )
            ssc = np.array([correction.adj])
            df_k, df_t = correction.df_k, correction.df_t
            terms = [term]
        term = combine_terms(terms, ssc)

        self.variance_covariance = VarianceCovariance(
            vcov=term.vcov,
            meat=term.meat,
            ssc=ssc,
            df_k=df_k,
            df_t=df_t,
            spec=spec,
            G=G,
        )
        # update p-value, t-stat, standard error, confint
        self.get_inference()

        return self

    def _dof_counts(
        self,
        *,
        G: int | float,
        k_fe_nested: int = 0,
        n_fe_fully_nested: int = 0,
    ) -> DegreesOfFreedomCounts:
        "Bundle the model counts that enter get_ssc()."
        return DegreesOfFreedomCounts(
            N=self.sample_info.n_obs,
            k=self._k,
            k_fe=self.fixef_counts.n_levels,
            n_fe=self.fixef_counts.n_fixef,
            k_fe_nested=k_fe_nested,
            n_fe_fully_nested=n_fe_fully_nested,
            G=G,
        )

    def _vcov_crv_cluster(
        self,
        *,
        clustid: np.ndarray,
        cluster_col: np.ndarray,
        vcov_type_detail: str,
    ) -> VcovTerm:
        "Pick CRV1 / CRV3-fast / CRV3-slow for one cluster column."
        if vcov_type_detail == "CRV1":
            return self._vcov_crv1(clustid=clustid, cluster_col=cluster_col)

        use_fast = self._closed_form_ols and not self.model.has_fixef
        crv3 = self._vcov_crv3_fast if use_fast else self._vcov_crv3_slow
        return VcovTerm(vcov=crv3(clustid=clustid, cluster_col=cluster_col), meat=None)

    def _vcov_iid(self) -> VcovTerm:
        vcov = vcov_iid_ols(
            residuals=self._u_hat,
            bread=self.sandwich.bread,
            N=self.sample_info.n_obs,
            weights=self.observation_weights.values,
        )
        return VcovTerm(vcov=vcov, meat=None)

    def _vcov_hetero(self, *, vcov_type_detail: str) -> VcovTerm:
        observation_weights = self.observation_weights.values
        meat = meat_hetero(
            sandwich=self.sandwich,
            X=self.within_data.design,
            frequency_weights=(
                observation_weights.reshape((-1, 1))
                if observation_weights is not None
                and self.options.weights_type == "fweights"
                else None
            ),
            normal_equation_weights=observation_weights,
            vcov_type_detail=cast(HeteroVcovTypeOptions, vcov_type_detail),
        )
        bread = self.sandwich.bread
        return VcovTerm(vcov=bread @ meat @ bread, meat=meat)

    def _vcov_hac(self, spec: VcovSpec) -> VcovTerm:
        _data = self._data
        time_id, panel_id = spec.time_id, spec.panel_id

        # some data checks on input pandas df
        # time needs to be numeric or date else we cannot sort by time
        if not np.issubdtype(_data[time_id], np.number) and not np.issubdtype(
            _data[time_id], np.datetime64
        ):
            raise ValueError(
                "The time variable must be numeric or date, else we cannot sort by time."
            )

        _time_arr = _data[time_id].to_numpy()
        _panel_arr = _data[panel_id].to_numpy() if panel_id is not None else None

        meat = meat_hac(
            scores=self.sandwich.scores,
            time_arr=_time_arr,
            panel_arr=_panel_arr,
            lag=spec.lag,
            vcov_type_detail=cast(HacVcovTypeOptions, spec.vcov_type_detail),
        )
        bread = self.sandwich.bread
        return VcovTerm(vcov=bread @ meat @ bread, meat=meat)

    def _vcov_nid(self) -> VcovTerm:
        raise NotImplementedError(
            "Only models of type Quantreg support a variance-covariance matrix of type 'nid'."
        )

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray) -> VcovTerm:
        meat = meat_crv1(
            scores=self.sandwich.scores,
            clustid=clustid,
            cluster_col=cluster_col,
        )
        bread = self.sandwich.bread
        return VcovTerm(vcov=bread @ meat @ bread, meat=meat)

    def _vcov_crv3_fast(self, clustid, cluster_col) -> np.ndarray:
        return vcov_crv3_fast(
            X=self.within_data.design,
            Y=self.within_data.response,
            weights=self.observation_weights.values,
            beta_hat=self._beta_hat,
            clustid=clustid,
            cluster_col=cluster_col,
        )

    def _vcov_crv3_slow(self, clustid, cluster_col) -> np.ndarray:
        beta_jack = np.zeros((len(clustid), self._k))

        for ixg, g in enumerate(clustid):
            # direct leave one cluster out implementation
            data = self._data[~np.equal(g, cluster_col)]
            fit = refit(
                self,
                data=data,
                # inference not needed, iid fastest to compute
                vcov=VcovSpec(vcov_type="iid", vcov_type_detail="iid"),
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

    def _clear_attributes(self):
        """Apply the configured fitted-model retention policy."""
        for attr in omitted_attributes(self.options.retention):
            if hasattr(self, attr):
                delattr(self, attr)

    def wald_test(
        self,
        R: np.ndarray | None = None,
        q: float | np.ndarray | None = None,
        distribution: WaldDistributionOptions = "F",
    ) -> WaldTest:
        """Conduct Wald test.

        Compute a Wald test for a linear hypothesis of the form `R * beta = q`.
        where R is m x k matrix, beta is a k x 1 vector of coefficients,
        and q is m x 1 vector.
        By default, tests the joint null hypothesis that all coefficients are zero.

        Parameters
        ----------
        R : array-like, optional
            The matrix R of the linear hypothesis.
            If `None`, defaults to an identity matrix.
        q : array-like, optional
            The vector q of the linear hypothesis.
            If `None`, defaults to a vector of zeros.
        distribution : str, optional
            The distribution to use for the p-value. Can be either `F` or `chi2`.
            Defaults to `F`. The F distribution is only used for the joint null
            that all coefficients are zero; any other restriction falls back to
            `chi2` with a warning.

        Returns
        -------
        WaldTest
            The test statistic of the reference distribution used, its p-value,
            both scalings of the Wald quadratic form, and the degrees of
            freedom. See [WaldTest](/reference/estimation.state.WaldTest.qmd).

        Examples
        --------
        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2| f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False))

        R = np.array([[1,-1]] )
        q = np.array([0.0])

        # Wald test
        wald = fit.wald_test(R=R, q=q, distribution = "chi2")

        print(f"Python statistic: {wald.stat}")
        print(f"Python p-value: {wald.pvalue}")
        ```

        Notes
        -----
        Returns the test and stores it as ``fit.wald``. Restrictions must use the
        same coefficient order as ``fit.coef().index``. This operation uses the
        current coefficient covariance and does not refit the model.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        _validate_literal_argument(distribution, WaldDistributionOptions)

        # If R is None, default to the identity matrix
        R = np.eye(self._k) if R is None else np.atleast_2d(np.asarray(R, dtype=float))

        covariance = self.variance_covariance
        if covariance.spec.is_clustered:
            df2: int | float = min(covariance.G) - 1
        else:
            df2 = self.sample_info.n_obs - self._k - self.fixef_counts.n_levels

        # The F distribution is only used for the joint test that all
        # coefficients are zero (R identity, q zero).
        if distribution == "F" and (
            not np.array_equal(R, np.eye(self._k)) or (q is not None and np.any(q))
        ):
            warnings.warn(
                "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector.",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            distribution = "chi2"

        self.wald = wald_test(
            beta_hat=self._beta_hat,
            vcov=covariance.vcov,
            R=R,
            q=q,
            df2=df2,
            distribution=distribution,
            vcov_type=covariance.spec.vcov_type_detail,
        )

        return self.wald

    def wildboottest(
        self,
        reps: int,
        cluster: str | None = None,
        param: str | None = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: int | None = None,
        k_adj: bool = True,
        G_adj: bool = True,
        parallel: bool = False,
        return_bootstrapped_t_stats=False,
    ):
        """Run a wild (cluster) bootstrap test for one coefficient.

        Requires the optional `wildboottest` package. Only unweighted OLS fits
        support the wild bootstrap; IV, GLM, Poisson, quantile regression and
        weighted fits raise a `NotImplementedError`.

        Parameters
        ----------
        reps : int
            The number of bootstrap iterations to run.
        cluster : str or list[str], optional
            The cluster variable. Defaults to `None`. If `None` and the model's
            covariance is clustered, uses the model's cluster variable
            (`fit.variance_covariance.spec.clustervar`). If the model is not
            clustered either, runs a heteroskedasticity-robust bootstrap. Only
            one cluster variable is supported.
        param : str
            The name of the coefficient to test (null: the coefficient is zero).
            Although it defaults to `None`, it is required: with `None`, the
            method returns a `NaN` t-statistic and a meaningless p-value.
        weights_type : str, optional
            The type of bootstrap weights: `rademacher`, `mammen`, `webb` or
            'norm'. The heteroskedastic (unclustered) bootstrap supports only
            `rademacher` and 'norm'. Defaults to `rademacher`.
        impose_null : bool, optional
            Indicates whether to impose the null hypothesis on the bootstrap DGP.
            Defaults to `True`.
        bootstrap_type : str, optional
            The bootstrap type, following MacKinnon, Nielsen and Webb (2023):
            `'11'` or `'31'`; the heteroskedastic bootstrap also accepts '21'.
            Defaults to `'11'`.
        seed : int, optional
            An option to provide a random seed. Defaults to `None`.
        k_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of observations and covariates. Defaults to `True`. Only used by the
            cluster bootstrap.
        G_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of clusters. Defaults to `True`. Only used by the cluster bootstrap.
        parallel : bool, optional
            Indicates whether to run the bootstrap in parallel. Defaults to `False`.
        return_bootstrapped_t_stats : bool, optional
            If `True`, the method returns a tuple of the regular output and the
            bootstrapped t-stats. Defaults to `False`.

        Returns
        -------
        pd.Series
            A `Series` with the entries `param`, `t value` (the original,
            non-bootstrapped t-statistic), `Pr(>|t|)` (the bootstrapped
            p-value), `bootstrap_type`, `inference` (`"HC"` or
            `"CRV(<cluster>)"`), `impose_null` and `ssc`. If
            `return_bootstrapped_t_stats` is `True`, the method returns a tuple
            of this `Series` and the bootstrapped t-statistics.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2 | f1", data)

        fit.wildboottest(param="X1", reps=99, seed=822)
        ```

        A wild cluster bootstrap with the WCR31 bootstrap type:

        ```{python}
        fit.wildboottest(
            param="X1", reps=99, seed=822, cluster="group_id", bootstrap_type="31"
        )
        ```

        Notes
        -----
        Only unweighted OLS fits are supported, with at most one cluster
        variable. Install the optional package with ``pip install wildboottest``.
        The procedure requires retained data for clustered or fixed-effect fits,
        and retained within-scale arrays otherwise. The fitted result is not
        replaced; see [model support](/reference-guide/model-support.qmd).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        if param is not None and param not in self._coefnames:
            raise ValueError(
                f"Parameter {param} not found in the model's coefficients."
            )

        if not self.capabilities.wildboottest:
            if self.options.has_weights:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for WLS estimation."
                )
            raise NotImplementedError(
                "Wild cluster bootstrap is only supported for unweighted OLS models; "
                "it is not available for IV, GLM, quantile regression, or DID2S fits."
            )

        cluster_list = []

        if cluster is not None and isinstance(cluster, str):
            cluster_list = [cluster]
        if cluster is not None and isinstance(cluster, list):
            cluster_list = cluster

        if cluster is None and self.variance_covariance.spec.is_clustered:
            cluster_list = list(self.variance_covariance.spec.clustervar)

        run_heteroskedastic = not cluster_list

        if not run_heteroskedastic and not len(cluster_list) == 1:
            raise NotImplementedError(
                "Multiway clustering is currently not supported with the wild cluster bootstrap."
            )

        if self.model.has_fixef or not run_heteroskedastic:
            require_retained(self, "wildboottest", "_data")
        else:
            require_retained(self, "wildboottest", "within_data")

        if not run_heteroskedastic and cluster_list[0] not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster_list[0]} not found in the data."
            )

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError as exc:
            raise ImportError(
                "wildboottest() requires the optional `wildboottest` package. "
                "Install it with `pip install wildboottest`."
            ) from exc

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
            ssc_value = boot.small_sample_correction

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
                    "2^G < the number of boot iterations, setting full_enumeration to True.",
                    UserWarning,
                    stacklevel=_find_stack_level(),
                )
            ssc_value = boot.ssc

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
            "ssc": ssc_value,
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
        """Compute the Causal Cluster Variance following Abadie et al (QJE 2023).

        Parameters
        ----------
        treatment: str
            The name of the treatment variable.
        cluster : str
            The name of the cluster variable. `None` by default.
            If `None`, uses the cluster variable from the model fit.
        seed : int, optional
            An integer to set the random seed. Defaults to `None`.
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
            A `DataFrame` with inference based on the "Causal Cluster Variance"
            and `regular` CRV1 inference.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        import numpy as np

        data = pf.get_data()
        rng = np.random.default_rng(123)
        data["D"] = rng.choice([0, 1], size=data.shape[0])

        fit = pf.feols("Y ~ D", data=data, vcov={"CRV1": "group_id"})
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=8, seed=123).head()
        ```

        Notes
        -----
        Supported for unweighted OLS without absorbed fixed effects. The
        treatment must be a coefficient in the model and have only values 0 and 1.
        Requires retained estimation data and within-scale arrays. For an
        initially unclustered model, this method stores CRV1 inference on the fit.
        The returned `DataFrame` has rows ``CCV`` and ``CRV1`` with coefficient,
        standard-error, test, and confidence-bound columns.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        if not self.capabilities.cluster_causal_variance:
            raise NotImplementedError(
                "The causal cluster variance estimator is not supported for models "
                f"of type '{self.model.method}'."
            )
        if not isinstance(treatment, str):
            raise TypeError("treatment must be a string.")
        if not (isinstance(cluster, str) or cluster is None):
            raise TypeError("cluster must be a string or None.")
        if not (isinstance(seed, int) or seed is None):
            raise TypeError("seed must be an integer or None.")
        if not isinstance(n_splits, int):
            raise TypeError("n_splits must be an integer.")
        if n_splits < 1:
            raise ValueError(f"n_splits must be a positive integer, got {n_splits}.")
        for name, share in (("pk", pk), ("qk", qk)):
            if not isinstance(share, (int, float)):
                raise TypeError(f"{name} must be a number.")
            if not 0 <= share <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {share}.")

        if self.model.has_fixef:
            raise NotImplementedError(
                "The causal cluster variance estimator is currently not supported for models with fixed effects."
            )
        if self.options.has_weights:
            raise NotImplementedError(
                "The causal cluster variance estimator is currently not supported for models with weights."
            )

        if treatment not in self._coefnames:
            raise ValueError(
                f"Variable {treatment} not found in the model's coefficients."
            )

        if cluster is None:
            clustervar = self.variance_covariance.spec.clustervar
            if not clustervar:
                raise ValueError("No cluster variable found in the model fit.")
            elif len(clustervar) > 1:
                raise ValueError(
                    "Multiway clustering is currently not supported with the causal cluster variance estimator."
                )
            else:
                cluster = clustervar[0]

        # check that cluster is in data
        require_retained(self, "ccv", "_data", "within_data")
        if cluster not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster} not found in the data used for the model fit."
            )

        if not self.variance_covariance.spec.is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object.",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            self.vcov({"CRV1": cluster})

        rng = np.random.default_rng(seed)

        data = self._data
        Y = self.within_data.response.flatten()
        W = data[treatment].to_numpy()
        if not np.all(np.isin(W, [0, 1])):
            raise ValueError(
                f"Treatment variable '{treatment}' must be binary with values 0 and 1."
            )
        X = self.within_data.design
        cluster_vec = data[cluster].to_numpy()
        unique_clusters = np.unique(cluster_vec)

        tau_full = float(self.coef().xs(treatment))  # type: ignore[arg-type]

        N = self.sample_info.n_obs
        G = len(unique_clusters)

        ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
        _compute_CCV = ccv_module._compute_CCV

        vcov_splits = 0.0
        for _ in range(n_splits):
            vcov_ccv = _compute_CCV(
                fit_fn=partial(
                    refit,
                    self,
                    # only the coefficients are read; iid is fastest to compute
                    vcov=VcovSpec(vcov_type="iid", vcov_type_detail="iid"),
                ),
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
        vcov_crv1 = self.variance_covariance.vcov[crv1_idx, crv1_idx]
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

    @overload
    def _model_matrix_one_hot(
        self, output: Literal["numpy"] = ...
    ) -> tuple[np.ndarray, np.ndarray, list[str]]: ...

    @overload
    def _model_matrix_one_hot(
        self, output: Literal["sparse"]
    ) -> tuple[np.ndarray, csc_matrix, list[str]]: ...

    def _model_matrix_one_hot(
        self, output: Literal["numpy", "sparse"] = "numpy"
    ) -> tuple[np.ndarray, np.ndarray | csc_matrix, list[str]]:
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
        if self.model.has_fixef:
            fml_linear, fixef = self.model.formula.split("|")
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
                context=FORMULAIC_TRANSFORMS | {**self.options.context},
            )
            xnames = X.model_spec.column_names
            Y = Y.toarray().flatten() if output == "sparse" else Y.flatten()
            X = csc_matrix(X) if output == "sparse" else X

            # drop the covariates the fit removed as collinear
            collinear = set(self.collinearity.dropped_coef_names)
            if collinear:
                keep = [i for i, name in enumerate(xnames) if name not in collinear]
                X = X[:, keep]
                xnames = [xnames[i] for i in keep]

        else:
            Y = self.within_data.response.flatten()
            X = self.within_data.design
            xnames = self._coefnames

        X = csc_matrix(X) if output == "sparse" else X

        return Y, X, xnames

    def _require_capability(
        self,
        *,
        capability: str,
        method: str,
        exception_type: type[Exception] = NotImplementedError,
    ) -> None:
        """Reject a post-estimation method the model class does not support."""
        if getattr(self.capabilities, capability):
            return
        estimator = f"'{self.model.method}' fits"
        if self.model.is_iv:
            estimator += " with instruments"
        raise exception_type(
            f"{method}() is not supported for {estimator}: "
            f"fit.capabilities.{capability} is False."
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
        """Implement the Gelbach (2016) decomposition method for mediation analysis.

        Compares a short model of the dependent variable on `decomp_var` (and
        `x1_vars`) with the long model specified in the original `feols()` call.

        For details, take a look at
        "When do covariates matter?" by Gelbach (2016, JoLe). You can find
        an ungated version of the paper on SSRN under the following link:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737 .

        Weighted fits are supported only with frequency weights
        (`weights_type="fweights"`) and only with `only_coef=True`; analytic
        weights raise a `NotImplementedError`.

        Parameters
        ----------
        param : str, optional
            Deprecated alias of `decomp_var`; passing it emits a
            `FutureWarning`. Use `decomp_var` instead.
        x1_vars : list[str] or str, optional
            Covariates that are included in both the short and the long
            regression, as a list or as a `"+"`-separated string.
        decomp_var : str
            The name of the focal covariate whose effect is to be decomposed into direct
            and indirect components with respect to the rest of the right-hand side.
            Required unless the deprecated `param` is passed.
        type : str, optional
            The type of decomposition method to use. Defaults to `gelbach`, which
            currently is the only supported option.
        cluster : str, optional
            The name of the cluster variable for the bootstrap. If `None`, uses the
            cluster variable from the model fit. Only one-way clustering is
            supported; a fit with multiway clustering raises ``ValueError`` unless
            a single cluster variable is passed here. Defaults to `None`.
        combine_covariates : dict[str, list[str] | re.Pattern], optional
            A dictionary that maps group names to the covariates combined into
            that group, given as a list of names or as a compiled regular
            expression. See the examples. Defaults to `None`.
        reps : int, optional
            The number of bootstrap iterations to run. Defaults to 1000.
        seed : int, optional
            An integer to set the random seed. Defaults to `None`.
        nthreads : int, optional
            The number of threads to use for the bootstrap. Defaults to `None`,
            which uses all available cores.
        agg_first : bool, optional
            If `True`, use the 'aggregate first' algorithm described in Gelbach (2016).
            `False` by default, unless `combine_covariates` is provided.
            Recommended to set to `True` if `combine_covariates` is argument is provided.
            As a rule of thumb, the more covariates are combined, the larger the performance
            improvement.
        only_coef : bool, optional
            Indicates whether to compute inference for the decomposition. Defaults to `False`.
            If `True`, skips the inference step and only returns the decomposition results.
        digits : int, optional
            Currently unused: the results are not rounded. Defaults to 4.

        Returns
        -------
        GelbachDecomposition
            A `GelbachDecomposition` object with the decomposition results.
            Use `tidy()` and `etable()` to access the estimation results.

        Examples
        --------
        ```{python}
        import re
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        data = gelbach_data(nobs=200)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

        gb = fit.decompose(decomp_var="x1", reps=10, nthreads=1, seed=1)
        gb.tidy()
        ```

        Keep `x21` in both the short and the long model:

        ```{python}
        gb = fit.decompose(decomp_var="x1", x1_vars=["x21"], reps=10, nthreads=1, seed=1)
        gb.tidy()
        ```

        Combine covariates into groups, and skip inference:

        ```{python}
        gb = fit.decompose(
            decomp_var="x1",
            combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
            only_coef=True,
        )
        gb.etable(type="df")
        ```

        Groups can also be given as regular expressions:

        ```{python}
        gb = fit.decompose(
            decomp_var="x1",
            combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")},
            reps=10,
            nthreads=1,
            seed=1,
        )
        gb.tidy()
        ```

        Notes
        -----
        Pass exactly one of ``decomp_var`` and the deprecated ``param``.
        Requires supported OLS and retained within-scale arrays and weight state.
        Original estimation data is also required for models with fixed effects
        or clustering, including a cluster supplied here. Unclustered models
        without fixed effects can use bootstrap inference with
        ``store_data=False``. A partial ``combine_covariates`` mapping reports
        only the named groups, so their sum need not equal the full coefficient
        change. Returns a separate decomposition result without replacing the
        regression fit. Small
        bootstrap examples illustrate syntax; use more repetitions for analysis.

        See Also
        --------
        [GelbachDecomposition](/reference/estimation.post_estimation.decomposition.GelbachDecomposition.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        self._require_capability(capability="decomposition", method="decompose")

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
                FutureWarning,
                stacklevel=_find_stack_level(),
            )
            decomp_var = param

        if x1_vars is not None:
            if isinstance(x1_vars, str):
                x1_vars = [x.strip() for x in x1_vars.split("+")]
            else:
                x1_vars = list(x1_vars)

        _decompose_arg_check(
            type=type,
            has_weights=self.options.has_weights,
            weights_type=self.options.weights_type,
            is_iv=self.model.is_iv,
            method=self.model.method,
            only_coef=only_coef,
        )

        require_retained(self, "decompose", "within_data", "observation_weights")
        if (
            self.model.has_fixef
            or cluster is not None
            or self.variance_covariance.spec.is_clustered
        ):
            require_retained(self, "decompose", "_data")

        nthreads_int = -1 if nthreads is None else nthreads

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        if agg_first is None:
            agg_first = combine_covariates is not None

        cluster_df: pd.Series | None = None
        if cluster is not None:
            cluster_df = self._data[cluster]
        elif self.variance_covariance.spec.is_clustered:
            clustervar = self.variance_covariance.spec.clustervar
            if len(clustervar) > 1:
                raise ValueError(
                    "Multiway clustering is currently not supported with the Gelbach "
                    "decomposition bootstrap. Pass a single cluster variable via "
                    "`cluster` instead."
                )
            cluster_df = self._data[clustervar[0]]

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
            depvarname=self.model.depvar,
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
            weights=(
                np.ones((self.sample_info.n_rows, 1))
                if self.observation_weights.values is None
                else self.observation_weights.values[:, None]
            ),
            store=True,
        )

        if not only_coef:
            med.bootstrap(rng=rng, B=reps)

        self.GelbachDecompositionResults = med

        return med

    def summary(self, **kwargs):
        """Print a model summary.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.summary()`` with this result as ``models``.

        Returns
        -------
        None
            Prints the summary to standard output.

        See Also
        --------
        [summary()](/reference/report.summary.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import summary

        return summary(models=self, **kwargs)

    def etable(self, **kwargs):
        """Create a regression table.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.etable()`` with this result as ``models``.

        Returns
        -------
        pandas.DataFrame, GT, Document, str, or None
            Output selected by ``type``: table, formatted `DataFrame`, document,
            or text. Markdown output is printed and returns `None`. See the
            canonical ``pf.etable()`` page for formats and forwarded options.

        See Also
        --------
        [etable()](/reference/report.etable.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import etable

        return etable(models=self, **kwargs)

    def coefplot(self, **kwargs):
        """Plot model coefficients.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.coefplot()`` with this result as ``models``.

        Returns
        -------
        matplotlib.figure.Figure or lets_plot.plot.core.PlotSpec
            Plot object from the selected backend. See the canonical plotting
            page for arguments and save examples.

        See Also
        --------
        [coefplot()](/reference/report.coefplot.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import coefplot

        return coefplot(models=self, **kwargs)

    def iplot(self, **kwargs):
        """Plot coefficients generated by `i()` terms.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.iplot()`` with this result as ``models``.

        Returns
        -------
        matplotlib.figure.Figure or lets_plot.plot.core.PlotSpec
            Plot object from the selected backend. See the canonical plotting
            page for arguments and save examples.

        See Also
        --------
        [iplot()](/reference/report.iplot.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import iplot

        return iplot(models=self, **kwargs)

    def fixef(self, atol: float = 1e-06, btol: float = 1e-06) -> pd.DataFrame:
        """Compute the coefficients of (swept out) fixed effects for a regression model.

        Publishes the estimates as `fixef_estimates`, a `FixedEffectEstimates`
        value holding the coefficient records, the dummy-coded solution
        `alpha`, and the per-observation contribution `sumFE`.

        For GLM and Poisson fits, the fixed effects are on the link scale.

        Parameters
        ----------
        atol : float, default 1e-6
            Stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : float, default 1e-6
            Another stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html

        Returns
        -------
        pd.DataFrame
            A tidy `DataFrame` with columns `variable`, `code`, `level`, and
            `coefficient` containing the estimated fixed effects.

        Raises
        ------
        ValueError
            If the model has no fixed effects.
        NotImplementedError
            For model classes that cannot recover fixed effects, such as IV
            fits.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fixed_effects = fit.fixef()
        fixed_effects.head()
        ```

        Notes
        -----
        Requires absorbed fixed effects and retained model frames. IV, quantile,
        and DID2S fits do not support recovery. The solution uses treatment
        normalization: the first observed level of the second and each later
        fixed effect is omitted. Individual coefficients depend on normalization;
        their per-observation sum does not. See
        [FixedEffectEstimates](/reference/estimation.state.FixedEffectEstimates.qmd).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        if not self.model.has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        self._require_capability(capability="fixed_effect_recovery", method="fixef")

        require_retained(self, "fixef", "model_matrix")

        fe_spec = self.model_matrix.model_spec[_ModelMatrixKey.fixed_effects]

        # flatten() copies: the weighting below scales uhat in place and must
        # not touch the fitted values the GLM hook may return.
        uhat = self._fixef_dependent().flatten()
        if not self._X_is_empty:
            # model_matrix keeps the columns the collinearity check dropped;
            # _coefnames names the estimated ones.
            X = self.model_matrix.independent[self._coefnames].to_numpy()
            uhat = uhat - X @ self._beta_hat
        # Recover from retained codes, omitting the first observed level of
        # the second and subsequent FEs under treatment normalization.
        # has_fixef guarantees that the retained matrix contains FE codes.
        fixed_effect_codes = cast(pd.DataFrame, self.model_matrix.fixed_effects)
        dummies = build_fixed_effect_dummies(
            codes=fixed_effect_codes, formulaic_columns=fe_spec.column_names
        )
        D = dummies.matrix
        D_w = D
        observation_weights = self.observation_weights.values
        if observation_weights is not None:
            # Weighted least squares: min || sqrt(w) (uhat - D alpha) ||.
            weights_sqrt = np.sqrt(observation_weights)
            uhat *= weights_sqrt
            D_w = diags(weights_sqrt, 0).dot(D)

        alpha = lsqr(D_w, uhat, atol=atol, btol=btol)[0]

        self.fixef_estimates = FixedEffectEstimates(
            fixed_effects=build_fixed_effects(
                alpha=alpha,
                dummies=dummies,
                transform_state=fe_spec.transform_state,
            ),
            alpha=alpha,
            # Fixed-effect contribution per observation, in the units of Y.
            sumFE=D.dot(alpha),
        )

        return fixed_effects_to_frame(self.fixef_estimates.fixed_effects)

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
        """Predict values of the model on new data.

        Without ``newdata``, returns values only for rows retained during
        estimation, in their retained order. With ``newdata``, output follows its
        row order and length. Missing covariate or fixed-effect values and unseen
        categorical or fixed-effect levels produce ``NaN``. For a GLM with an
        offset, an offset expression that evaluates to missing values in
        ``newdata`` raises ``ValueError`` instead.

        Parameters
        ----------
        newdata : DataFrameType, optional
            A narwhals compatible `DataFrame` (polars, pandas, duckdb, etc).
            If `None` (default), the data used for fitting the model is used.
        atol : float, default 1e-6
            Stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : float, default 1e-6
            Another stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        type : str, optional
            The type of prediction: `link` (default) or `response`. Both are
            identical for linear models. For GLM and Poisson fits, `link`
            returns the linear predictor and `response` the predicted mean;
            note that `fixest`'s `predict()` defaults to `response`.
        se_fit : bool, optional
            If `True`, return the standard errors of the predictions instead of
            the predictions. Not available for models with fixed effects, for
            weighted models, or for GLMs. Defaults to `False`.
        interval : str, optional
            The type of interval to compute: `prediction` or `None` (default).
            Prediction intervals are not available for models with fixed
            effects or for weighted models.
        alpha : float, optional
            The significance level of the prediction interval. Defaults to
            0.05. Only used if `interval="prediction"`.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            With ``interval="prediction"``, a `DataFrame` containing ``fit``,
            ``se_fit``, ``ci_low``, and ``ci_high``. Otherwise a flat array
            of predictions, or of standard errors when ``se_fit=True``.

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

        Notes
        -----
        IV and DID2S results reject prediction. Prediction standard errors and
        intervals require an unweighted model without absorbed fixed effects;
        GLM prediction intervals are currently ignored by the GLM override.
        In-sample predictions require retained fitted arrays. Predictions on
        supplied data can use the retained formula specification without the
        original data; fixed-effect prediction additionally needs recovered
        fixed effects or the data to recover them. See
        [model storage](/reference-guide/model-storage.qmd).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        self._require_capability(capability="prediction", method="predict")

        if interval == "prediction" or se_fit:
            if self.model.has_fixef:
                raise NotImplementedError(
                    "Prediction errors are currently not supported for models with fixed effects."
                )

            if self.options.has_weights:
                raise NotImplementedError(
                    "Prediction errors are currently not supported for models with weights."
                )

        _validate_literal_argument(type, PredictionType)
        if interval is not None:
            _validate_literal_argument(interval, PredictionErrorOptions)

        if newdata is None:
            # note: no need to worry about fixed effects, as not supported with
            # prediction errors; will throw error later;
            X = self._prediction_design()
            y_hat = getattr(self.fitted_values, type)
            n_observations = self.sample_info.n_rows
        else:
            newdata = _narwhals_to_pandas(newdata).reset_index(drop=True)
            n_observations = newdata.shape[0]
            context = FORMULAIC_TRANSFORMS | {**self.options.context}
            # Use na_action="drop" on each sub-spec separately because dependent variable
            # may not be available in newdata, then intersect indices so a NaN in *any* variable
            # (covariate or FE) marks the whole row as NaN in the output.
            model_spec = self.model.model_spec
            assert model_spec is not None, (
                "predict() runs after the model matrix is built"
            )
            rhs_spec = model_spec[_ModelMatrixKey.main].rhs
            X_mm, unseen = materialize_model_spec_with_unseen_mask(
                rhs_spec, newdata, context
            )
            valid_idx = X_mm.index.to_numpy()
            # rows with a categorical level unseen during fitting (in C()/i()) would
            # be silently encoded as the reference level -> drop them to NaN instead,
            # matching how unseen fixed-effect levels are handled below.
            valid_idx = valid_idx[~unseen[valid_idx]]
            if self.model.has_fixef:
                fe_spec = model_spec[_ModelMatrixKey.fixed_effects]
                # na_action="ignore" keeps unseen-level rows as NaN codes
                fe_mm = fe_spec.get_model_matrix(
                    newdata,
                    context=context,
                    na_action="ignore",
                )
                valid_fixed_effects = fe_mm.notna().all(axis="columns").to_numpy()
                valid_idx = valid_idx[valid_fixed_effects[valid_idx]]
                if not hasattr(self, "fixef_estimates"):
                    require_retained(self, "predict", "_data")
                    self.fixef(atol, btol)
                fe_hat = sum_fixed_effects(
                    codes=fe_mm.loc[valid_idx],
                    fixed_effects=self.fixef_estimates.fixed_effects,
                )

            X_coef = X_mm.loc[valid_idx, self._coefnames].to_numpy()
            y_hat = np.full(n_observations, np.nan)
            y_hat[valid_idx] = X_coef @ self._beta_hat
            if self.model.has_fixef:
                y_hat[valid_idx] += fe_hat
            # Pad X to full size; NaN rows yield NaN SE/CI via einsum propagation.
            X = np.full((n_observations, X_coef.shape[1]), np.nan)
            X[valid_idx] = X_coef
            if self.options.offset is not None:
                offset_mm = model_spec[_ModelMatrixKey.offset].get_model_matrix(
                    newdata,
                    context=context,
                    na_action="drop",
                    output="pandas",
                )
                if not offset_mm.index.equals(newdata.index):
                    raise ValueError(
                        f"Offset expression '{self.options.offset}' evaluates to missing "
                        "values in `newdata`."
                    )

                y_hat += offset_mm.iloc[:, 0].to_numpy()

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
        """Conduct a randomization inference (RI) test for one coefficient.

        Supported for unweighted `feols()` fits without instruments and for
        `fepois()` fits; weighted fits raise a `NotImplementedError`.

        Parameters
        ----------
        resampvar : str
            The name of the variable to be resampled, optionally with a null
            hypothesis: `"X1"` tests `X1 = 0`, `"X1=0.5"` tests `X1 = 0.5`
            (two-sided), and `"X1>0"` or `"X1<0"` run one-sided tests.
        cluster : str, optional
            The name of the cluster variable in case of cluster random assignment.
            If provided, `resampvar` is held constant within each `cluster`.
            Defaults to `None`.
        reps : int, optional
            The number of randomization iterations. Defaults to 100.
        type : str, optional
            The type of the randomization inference test.
            Can be `randomization-c` (default) or `randomization-t`. Note that
            the `randomization-c` is much faster, while the
            `randomization-t` is recommended by Wu & Ding (JASA, 2021).
        rng : np.random.Generator, optional
            A random number generator. Defaults to `None`.
        choose_algorithm : str, optional
            The algorithm to use for the computation: `auto` (default), `fast`
            or `slow`. `auto` picks the fast algorithm when the optional
            `numba` extra is installed (`pip install pyfixest[numba]`) and the
            slow one otherwise; `fast` without numba raises an `ImportError`.
            `randomization-t` tests and non-OLS fits such as `fepois()` always
            use the slow algorithm.
        store_ritest_statistics : bool, optional
            Whether to store the simulated statistics of the RI procedure.
            Defaults to `False`. If `True`, publishes the draws, the centered
            sample statistic, and the p-value as a `RitestStatistics` value
            in the model's `ritest_statistics` attribute, which
            `plot_ritest()` plots.
        level : float, optional
            The level for the confidence interval of the randomization inference
            p-value. Defaults to 0.95.

        Returns
        -------
        pd.Series
            The null hypothesis (`H0`), the test type, the regression
            coefficient of `resampvar`, the RI p-value, its standard error and
            confidence interval, and the cluster variable if one was given.

        Examples
        --------
        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data)

        fit.ritest("X1", reps=99, rng=np.random.default_rng(1))
        ```

        Use randomization-t instead of randomization-c:

        ```{python}
        fit.ritest("X1", reps=99, type="randomization-t", rng=np.random.default_rng(1))
        ```

        Store the statistics for `plot_ritest()`:

        ```{python}
        fit.ritest(
            "X1", reps=99, store_ritest_statistics=True, rng=np.random.default_rng(1)
        )
        fit.ritest_statistics.pvalue
        ```

        Notes
        -----
        Requires retained estimation data. When ``cluster`` is supplied for an
        initially unclustered fit, CRV1 inference is computed and stored on the
        fitted result. Setting ``store_ritest_statistics=True`` publishes draws
        for ``plot_ritest()``; it does not change the return type.

        See Also
        --------
        [plot_ritest()](/reference/estimation.models.feols_.Feols.plot_ritest.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
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

        self._require_capability(capability="randomization_inference", method="ritest")

        # check that resampvar in _coefnames
        if resampvar_ not in self._coefnames:
            raise ValueError(f"{resampvar_} not found in the model's coefficients.")

        if self.options.has_weights:
            raise NotImplementedError(
                """
                Regression Weights are not supported with Randomization Inference.
                """
            )

        require_retained(self, "ritest", "_data")

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

        if type not in ["randomization-t", "randomization-c"]:
            raise ValueError("type must be 'randomization-t' or 'randomization-c.")

        if isinstance(reps, bool) or not isinstance(reps, int):
            raise TypeError("reps must be an integer.")
        if reps < 1:
            raise ValueError(f"reps must be a positive integer, got {reps}.")

        # update vcov if cluster provided but not in model
        if cluster is not None and not self.variance_covariance.spec.is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object.",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            self.vcov({"CRV1": cluster})

        rng = np.random.default_rng() if rng is None else rng

        sample_coef = np.array(self.coef().xs(resampvar_))
        sample_tstat = np.array(self.tstat().xs(resampvar_))
        sample_stat = sample_tstat if type == "randomization-t" else sample_coef

        # always run slow algorithm for randomization-t
        choose_algorithm = "slow" if type == "randomization-t" else choose_algorithm

        if choose_algorithm == "auto":
            choose_algorithm = "fast" if _HAS_NUMBA else "slow"

        if choose_algorithm == "slow" or not self._closed_form_ols:
            vcov_input: str | dict[str, str]
            if cluster is not None:
                vcov_input = {"CRV1": cluster}
            else:
                # "iid" for models without controls, else HC1
                vcov_input = (
                    "hetero"
                    if (self.model.has_fixef and len(self._coefnames) > 1)
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
                reps=reps,
                type=type,
                rng=rng,
                fit_fn=partial(
                    refit,
                    self,
                    vcov=VcovSpec.from_user_input(vcov_input),
                    # permutations change the treatment, not the rows
                    same_sample=True,
                ),
            )

        else:
            weights = (
                np.ones(self.sample_info.n_rows)
                if self.observation_weights.values is None
                else self.observation_weights.values
            )
            # encoded fixed effects also cover interactions and expressions
            fval_df = self.model_matrix.fixed_effects
            D = self._data[resampvar_].to_numpy()

            ri_stats = _get_ritest_stats_fast(
                Y=self.within_data.response,
                X=self.within_data.design,
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
            self.ritest_statistics = RitestStatistics(
                statistics=ri_stats,
                sample_stat=float(sample_stat - h0_value),
                pvalue=float(ri_pvalue),
            )

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
        """Plot the distribution of the Randomization Inference Statistics.

        Parameters
        ----------
        plot_backend : str, optional
            The plotting backend to use. Defaults to `lets_plot`. Alternatively,
            `matplotlib` is available.

        Returns
        -------
        A `lets_plot` or matplotlib figure with the distribution of the Randomization
        Inference Statistics.

        Examples
        --------
        ```{python}
        import numpy as np
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2", pf.get_data(N=100))
        fit.ritest("X1", reps=99, rng=np.random.default_rng(1),
                   store_ritest_statistics=True)
        fit.plot_ritest(plot_backend="matplotlib")
        ```

        Notes
        -----
        Run ``ritest(store_ritest_statistics=True)`` first; otherwise this method
        raises ``ValueError``. It uses the stored draws without refitting.

        See Also
        --------
        [ritest()](/reference/estimation.models.feols_.Feols.ritest.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        from pyfixest.estimation.post_estimation.ritest import _plot_ritest_pvalue

        if not hasattr(self, "ritest_statistics"):
            raise ValueError(
                """
                            The randomization inference statistics have not been stored
                            in the model object. Please set `store_ritest_statistics=True`
                            when calling `ritest()`
                            """
            )

        stored = self.ritest_statistics

        return _plot_ritest_pvalue(
            ri_stats=stored.statistics,
            sample_stat=stored.sample_stat,
            plot_backend=plot_backend,
        )

    def update(
        self, X_new: np.ndarray, y_new: np.ndarray, inplace: bool = False
    ) -> np.ndarray:
        """Update coefficients for new observations using Sherman-Morrison formula.

        Parameters
        ----------
        X_new : np.ndarray
            Covariates for new data points. Users expected to ensure conformability
            with existing data.
        y_new : np.ndarray
            Outcome values for new data points.
        inplace : bool, optional
            Must be `False`. In-place updates are unsupported because appending
            design rows cannot reconstruct the complete fitted-result state.

        Returns
        -------
        np.ndarray
            Updated coefficients.

        Notes
        -----
        Only unweighted OLS without absorbed fixed effects is supported.
        Requires retained within-scale arrays. The coefficient order follows
        ``fit.coef().index``; use that order for ``X_new``.

        Updates the coefficients in closed form via the Sherman-Morrison
        identity instead of refitting on the full sample. `X_new` has to include
        the intercept column. The returned coefficients do not mutate the fitted
        result. Models with fixed effects are not supported.

        Examples
        --------
        Fit on all but the last observation, then add it:

        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data().dropna()
        fit = pf.feols("Y ~ X1 + X2", data.iloc[:-1])

        last = data.iloc[[-1]]
        X_new = np.column_stack(
            [np.ones(1), last["X1"].to_numpy(), last["X2"].to_numpy()]
        )
        y_new = last["Y"].to_numpy()

        fit.update(X_new, y_new)
        ```

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        if self.model.has_fixef:
            raise NotImplementedError(
                "The update() method is currently not supported for models with fixed effects."
            )
        self._require_capability(capability="sherman_morrison_update", method="update")
        if self.options.has_weights:
            raise NotImplementedError(
                "The update() method is currently not supported for models with weights."
            )
        if inplace:
            raise NotImplementedError(
                "update(..., inplace=True) is not supported because appending design "
                "rows cannot safely update the complete fitted-result state; use the "
                "returned coefficients instead."
            )
        require_retained(self, "update", "within_data")
        if not np.all(X_new[:, 0] == 1):
            X_new = np.column_stack((np.ones(len(X_new)), X_new))
        X_n_plus_1 = np.vstack((self.within_data.design, X_new))
        epsi_n_plus_1 = y_new - X_new @ self._beta_hat
        gamma_n_plus_1 = np.linalg.inv(X_n_plus_1.T @ X_n_plus_1) @ X_new.T
        beta_n_plus_1 = self._beta_hat + gamma_n_plus_1 @ epsi_n_plus_1
        return beta_n_plus_1
