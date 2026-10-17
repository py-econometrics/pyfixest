from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import GlmFamily
from pyfixest.estimation.internals.fit_glm_ import fit_glm_irls
from pyfixest.estimation.internals.fit_statistics import FitStatistics
from pyfixest.estimation.internals.literals import HeteroVcovTypeOptions
from pyfixest.estimation.internals.model_state import (
    Capabilities,
    FittedValues,
    GlmEstimationOptions,
    ModelDescription,
    SampleSplit,
)
from pyfixest.estimation.internals.retention import require_retained
from pyfixest.estimation.internals.separation import check_for_separation
from pyfixest.estimation.internals.vcov_ import meat_hetero, vcov_iid_glm
from pyfixest.estimation.internals.vcov_utils import VcovTerm
from pyfixest.estimation.models.feols_ import (
    Feols,
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.utils.dev_utils import DataFrameType
from pyfixest.utils.utils import DegreesOfFreedomCounts


class Feglm(Feols):
    """Fitted generalized-linear-model result returned by `pf.feglm()`.

    Access as ``fit = pf.feglm(formula, data, family=...)``. The selected
    family determines the concrete subclass: `Felogit`, `Feprobit`,
    `Fegaussian`, or `Fepois`. Construct results with the estimation functions.

    Attributes
    ----------
    options : GlmEstimationOptions
        Access as ``fit.options``: shared estimation settings and the IRLS
        tolerance, iteration limit, separation check, and acceleration option.
    working_state : GlmWorkingState
        Final IRLS response, design, working weights, and residuals.
    fitted_values : FittedValues
        Linear predictor (``link``) and fitted mean (``response``), in retained
        estimation-row order.
    sandwich : SandwichComponents
        Final scores, weighted Hessian, and its inverse.
    convergence : bool
        Whether IRLS converged.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()
    data["Y_bin"] = (data["Y"] > 0).astype(int)
    fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit")
    fit.tidy()
    fit.predict(type="response")[:3]
    ```

    Notes
    -----
    Shared coefficient accessors and reporting methods are described in the
    [fitted-result overview](/reference-guide/fitted-results.qmd). Logit,
    probit, and Gaussian families reject wild bootstrap, randomization
    inference, causal cluster variance, decomposition, coefficient updates,
    and SAVI. They also reject CRV3 covariance. Poisson has its own support
    contract. See [model support](/reference-guide/model-support.qmd).

    Predictions default to the link scale; request ``type="response"`` for
    means or probabilities. Recovered fixed effects are on the link scale.
    Prediction standard errors are unsupported; interval arguments currently
    have no effect. ``lean=True`` removes working arrays; see
    [model storage](/reference-guide/model-storage.qmd).

    Fixed effects are handled by IRLS with demeaning following Stammann
    (2018), [arXiv:1707.01815](https://arxiv.org/abs/1707.01815).

    See Also
    --------
    [feglm()](/reference/estimation.api.feglm.feglm.qmd),
    [Fepois](/reference/estimation.models.fepois_.Fepois.qmd),
    [fitted-result overview](/reference-guide/fitted-results.qmd).
    """

    options: GlmEstimationOptions
    _n_fe_after_separation: int | None = None
    # Iterative IRLS fit: no single least-squares solve to shortcut.
    _closed_form_ols = False
    # Link, variance, deviance, and method name; each subclass declares one.
    _family: ClassVar[GlmFamily]

    # No CRV3: the inherited slow jackknife refits with the linear/Poisson
    # APIs and cannot yet preserve a generic GLM family's estimation contract.
    _declared_capabilities: ClassVar[Capabilities] = Capabilities(
        covariance_update=True,
        crv3_inference=False,
        hac_inference=True,
        multiway_clustering=True,
        wildboottest=False,
        cluster_causal_variance=False,
        decomposition=False,
        prediction=True,
        fixed_effect_recovery=True,
        randomization_inference=False,
        sherman_morrison_update=False,
        anytime_valid_inference=False,
    )

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split: SampleSplit | None = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split=sample_split,
        )

        _glm_input_checks(
            drop_singletons=options.drop_singletons,
            tol=options.tol,
            maxiter=options.maxiter,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Describe a GLM fit by its family's method and inference distribution."""
        return replace(
            super()._describe_model(**kwargs),
            method=self._family.method,
            inference_dist=self._family.inference_dist,
        )

    def prepare_model_matrix(self) -> ModelMatrix:
        "Prepare model inputs for estimation."
        model_matrix = super().prepare_model_matrix()

        # check for separation
        na_separation: list[int] = []
        if (
            model_matrix.fixed_effects is not None
            and self.options.separation_check is not None
            and self.options.separation_check  # not an empty list
        ):
            na_separation = check_for_separation(
                Y=model_matrix.dependent,
                X=model_matrix.independent,
                fe=model_matrix.fixed_effects,
                demeaner=self.options.demeaner,
                collin_tol=self.options.collin_tol,
                methods=self.options.separation_check,
            )

        if na_separation:
            self._data.drop(na_separation, axis=0, inplace=True)
            model_matrix = model_matrix.without_rows(na_separation, stage="separation")
            self._publish_model_matrix(model_matrix)

            # possible to have dropped fixed effects level due to separation
            self._n_fe_after_separation = int(
                np.sum(np.asarray(self.fixef_counts.n_levels_by_fe) > 1)
            )

        return model_matrix

    def _dof_counts(self, **kwargs: Any) -> DegreesOfFreedomCounts:
        "Count only fixed effects with several levels once separation dropped rows."
        counts = super()._dof_counts(**kwargs)
        if self._n_fe_after_separation is None:
            return counts
        return replace(counts, n_fe=self._n_fe_after_separation)

    def get_fit(self) -> None:
        "Fit the GLM via IRLS and write results onto `self.*` attributes."
        model_matrix = self.model_matrix
        response = model_matrix.dependent.to_numpy()
        design = model_matrix.independent.to_numpy()
        fixed_effect_frame = model_matrix.fixed_effects
        offset_frame = model_matrix.offset
        fixed_effects = (
            None if fixed_effect_frame is None else fixed_effect_frame.to_numpy()
        )
        offset = (
            None if offset_frame is None else offset_frame.to_numpy().reshape((-1, 1))
        )

        def _demean(
            v: np.ndarray, X: np.ndarray, weights: np.ndarray, tol: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return self.residualize(
                v=v,
                X=X,
                flist=fixed_effects,
                weights=weights,
                tol=tol,
            )

        fit = fit_glm_irls(
            X=design,
            Y=response,
            family=self._family,
            demean=_demean,
            coefnames=self._coefnames,
            collin_tol=self.options.collin_tol,
            accelerate=self.options.accelerate and fixed_effects is not None,
            offset=offset,
            weights=self.observation_weights.values,
            solver=self.options.solver,
            maxiter=self.options.maxiter,
            tol=self.options.tol,
            fixef_tol=self.options.fixef_tol,
        )

        self.collinearity = fit.collinearity
        self._coefnames = list(fit.collinearity.coefnames)
        working_state = fit.working_state
        self.working_state = working_state
        # The prediction view of the same arrays: eta is the linear predictor
        # (fixed effects and offset included), mu its inverse-link mean.
        self.fitted_values = FittedValues(
            link=working_state.eta, response=working_state.mu
        )
        design_within = working_state.design_within
        self._X_is_empty = design_within.shape[1] == 0
        self._k = design_within.shape[1]

        self._beta_hat = fit.beta
        self.sandwich = fit.sandwich
        self.convergence = fit.converged

    def _fit_statistics(self) -> FitStatistics:
        """Compute the deviance of the final IRLS iteration.

        Families with further measures extend this in their subclass.
        """
        deviance = self._family.deviance(
            self.model_matrix.dependent.to_numpy().flatten(),
            self.working_state.mu,
            self.observation_weights.values,
        )
        return FitStatistics(deviance=deviance)

    def _prediction_design(self) -> np.ndarray:
        """Supply the final IRLS design to the inherited predict() method.

        Unlike linear models, GLMs store this coefficient-ordered design in
        working_state. It is not multiplied by square-root working weights.
        """
        require_retained(self, "predict", "working_state")
        return self.working_state.design_within

    def _fixef_dependent(self) -> np.ndarray:
        """Return the linear predictor net of the offset for `fixef()`.

        The fixed effects are recovered from the estimated linear predictor,
        equation (5.2) in Stammann (2018), http://arxiv.org/abs/1707.01815;
        the observed response is not used. The linear predictor includes
        the offset; subtracting it makes `sumFE` the pure fixed-effect
        contribution, so predict() can add the offset back from newdata
        without double-counting.
        """
        eta = self.fitted_values.link
        if self.options.offset is not None:
            offset = self.model_matrix.offset
            assert offset is not None
            eta = eta - offset.to_numpy().flatten()
        return eta

    def _vcov_iid(self) -> VcovTerm:
        return VcovTerm(vcov=vcov_iid_glm(bread=self.sandwich.bread), meat=None)

    def _vcov_hetero(self, *, vcov_type_detail: str) -> VcovTerm:
        # The IRLS design is unpremultiplied, so the HC2/HC3 leverage takes the
        # final IRLS weights, which already contain the observation weights.
        observation_weights = self.observation_weights.values
        meat = meat_hetero(
            sandwich=self.sandwich,
            X=self.working_state.design_within,
            frequency_weights=(
                observation_weights.reshape((-1, 1))
                if observation_weights is not None
                and self.options.weights_type == "fweights"
                else None
            ),
            normal_equation_weights=self.working_state.working_weights,
            vcov_type_detail=cast(HeteroVcovTypeOptions, vcov_type_detail),
        )
        bread = self.sandwich.bread
        return VcovTerm(vcov=bread @ meat @ bread, meat=meat)

    def resid(self, type: str = "response") -> np.ndarray:
        """Return residuals from a fitted GLM.

        Parameters
        ----------
        type : str, optional
            The type of residuals to return. Either `response` (default) or
            `working`.

        Returns
        -------
        np.ndarray
            A flat array of length ``fit.sample_info.n_rows``, ordered by the
            rows retained during estimation. Response residuals are ``Y - mu``;
            working residuals use the final IRLS working response and design.

        Notes
        -----
        Requires retained ``working_state`` and is unavailable after
        ``lean=True``. It does not pad rows dropped before estimation.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.fepois("Y ~ X1", pf.get_data(model="Fepois"))
        fit.resid(type="response")[:3]
        ```

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        if type not in {"response", "working"}:
            raise ValueError("type must be one of 'response' or 'working'.")
        require_retained(self, "resid", "working_state")
        if type == "response":
            return self.working_state.response_residuals.flatten()
        return self.working_state.working_residuals.flatten()

    def residualize(
        self,
        v: np.ndarray,
        X: np.ndarray,
        flist: np.ndarray | None,
        weights: np.ndarray,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residualize v and X with current working weights and a reused preconditioner."""
        if flist is None:
            return v, X

        effective_demeaner = self.options.demeaner.with_tol(tol)
        vX_tilde = self._demean_cache.demean_array(
            x=np.c_[v, X],
            flist=flist,
            weights=weights.flatten(),
            na_index=self.sample_info.dropped_row_index,
            demeaner=effective_demeaner,
        )
        return vX_tilde[:, 0], vX_tilde[:, 1:]

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
        """Return predicted values from a fitted GLM.

        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations are set to `NaN`.

        Parameters
        ----------
        newdata : DataFrameType, optional
            A Narwhals-compatible data frame with new data, to be used for
            prediction. If `None` (default), uses the data used for fitting the
            model.
        atol : float, default 1e-6
            Stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : float, default 1e-6
            Another stopping tolerance for `scipy.sparse.linalg.lsqr()`.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        type : str, optional
            The type of prediction to be computed: `link` (default) or
            `response`. With `link`, the output is the linear predictor
            `X @ beta` (plus fixed effects and offset). With `response`, the
            output is on the scale of the dependent variable, i.e. the inverse
            link of the linear predictor, E(Y|X). Note that `fixest`'s
            `predict()` defaults to `response`; pass `type="response"`
            explicitly for predicted means or probabilities.
        se_fit : bool, optional
            Prediction standard errors are not implemented for GLMs: `True`
            raises a `NotImplementedError`. Defaults to `False`.
        interval : str, optional
            Prediction intervals are not implemented for GLMs. The argument is
            currently ignored, and the method returns the point predictions.
        alpha : float, optional
            Currently ignored, see `interval`.

        Returns
        -------
        np.ndarray
            A flat array in retained estimation-row order when ``newdata`` is
            absent; otherwise one entry per supplied row, in its order. Missing
            covariate/fixed-effect values and unseen categorical/fixed-effect
            levels yield ``NaN``. An offset expression that evaluates to missing
            values in ``newdata`` raises ``ValueError`` instead.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        data["Y_bin"] = (data["Y"] > 0).astype(int)
        fit = pf.feglm("Y_bin ~ X1 + X2", data, family="logit")
        fit.predict(type="response")[:3]
        ```

        Notes
        -----
        In-sample prediction requires retained working state and fitted values.
        Fixed-effect prediction on supplied data requires retained recovery state
        or a prior call to ``fixef()``. See
        [model storage](/reference-guide/model-storage.qmd).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        if se_fit:
            raise NotImplementedError(
                "Prediction with standard errors is not implemented for GLMs."
            )

        yhat = super().predict(newdata=newdata, type="link", atol=atol, btol=btol)
        if type == "response":
            return self._family.inv_link(
                yhat.to_numpy() if isinstance(yhat, pd.DataFrame) else yhat
            )
        else:
            return yhat

    def _validate_response(self) -> None:
        """Validate the prepared response against the family's constraints."""
        self._family.check_y(self.model_matrix.dependent.to_numpy())

    def _finalize_fit(self) -> None:
        """Skip the OLS Wald test; GLMs run no Wald test at fit time."""


def _glm_input_checks(drop_singletons: bool, tol: float, maxiter: int) -> None:
    if not isinstance(drop_singletons, bool):
        raise TypeError("drop_singletons must be logical.")
    if not isinstance(tol, (int, float)):
        raise TypeError("tol must be numeric.")
    if tol <= 0 or tol >= 1:
        raise AssertionError("tol must be between 0 and 1.")
    if not isinstance(maxiter, int):
        raise TypeError("maxiter must be integer.")
    if maxiter <= 0:
        raise AssertionError("maxiter must be greater than 0.")
