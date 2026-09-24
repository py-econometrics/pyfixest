from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

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
    FittedValues,
    GlmEstimationOptions,
    ModelDescription,
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


class Feglm(Feols):
    """
    Base class for the estimation of a fixed-effects GLM model.

    Returned by [feglm()](/reference/estimation.api.feglm.feglm.qmd). Fixed
    effects are handled via iteratively reweighted least squares with demeaning,
    following Stammann (2018),
    [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815). The family is set with
    the `family` argument and implemented by a subclass. `poisson` dispatches to
    [Fepois](/reference/estimation.models.fepois_.Fepois.qmd).

    Examples
    --------
    ```{python}
    import numpy as np
    import pyfixest as pf

    data = pf.get_data()
    data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)

    fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit")
    fit.tidy()
    ```
    """

    options: GlmEstimationOptions
    # Iterative IRLS fit: no single least-squares solve to shortcut.
    _closed_form_ols = False

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        family: GlmFamily,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
    ) -> None:
        # `_describe_model()`, called by the base constructor, names the
        # family's inference distribution.
        self._family = family
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
        )

        _glm_input_checks(
            drop_singletons=options.drop_singletons,
            tol=options.tol,
            maxiter=options.maxiter,
        )

        # The inherited slow jackknife refits with the linear/Poisson APIs and
        # cannot yet preserve a generic GLM family's estimation contract.
        self.capabilities = replace(
            self.capabilities,
            crv3_inference=False,
            hac_inference=True,
            wildboottest=False,
            cluster_causal_variance=False,
            decomposition=False,
            randomization_inference=False,
            sherman_morrison_update=False,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Describe a GLM fit and the inference distribution of its family."""
        return replace(
            super()._describe_model(**kwargs),
            method="feglm",
            inference_dist=self._family.inference_dist,
        )

    def _refit_estimator(self) -> Callable[..., Any]:
        "Refuse refits: `feglm` refits cannot yet replay the family and options."
        raise NotImplementedError(
            f"Leave-out and resampled refits are not implemented for '{self.model.method}' "
            "models: a refit cannot yet replay their estimation contract."
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
                fml=self.model.formula,
                data=self._data,
                demeaner=self.options.demeaner,
                methods=self.options.separation_check,
            )

        if na_separation:
            self._data.drop(na_separation, axis=0, inplace=True)
            model_matrix = model_matrix.without_rows(na_separation, stage="separation")
            self._publish_model_matrix(model_matrix)

            # possible to have dropped fixed effects level due to separation
            if self.model.has_fixef:
                assert self._k_fe is not None
                self._n_fe = np.sum(self._k_fe > 1)
            else:
                self._n_fe = 0

        return model_matrix

    def get_fit(self) -> None:
        "Fit the GLM via IRLS and write results onto self.* attributes."
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

        self.fitstat = FitStatistics(deviance=fit.deviance)
        self.convergence = fit.converged

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
        """
        Return residuals from a fitted GLM.

        Parameters
        ----------
        type : str, optional
            The type of residuals to return. Either "response" (default) or
            "working".

        Returns
        -------
        np.ndarray
            A flat array with the requested residuals.
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
        "Residualize v and X by flist using weights."
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
        """
        Return predicted values from regression model.

        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations
        will be set to NaN.

        Parameters
        ----------
        newdata : Union[None, pd.DataFrame], optional
            A pd.DataFrame with the new data, to be used for prediction.
            If None (default), uses the data used for fitting the model.
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/
                scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        type : str, optional
            The type of prediction to be computed.
            Can be either "response" (default) or "link".
            If type="response", the output is at the level of the response variable,
            i.e., it is the expected predictor E(Y|X).
            If "link", the output is at the level of the explanatory variables,
            i.e., the linear predictor X @ beta.
        atol : Float, default 1e-6
            Stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        btol : Float, default 1e-6
            Another stopping tolerance for scipy.sparse.linalg.lsqr().
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
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
