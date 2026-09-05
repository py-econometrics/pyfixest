from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.model_matrix import ModelMatrix
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import GlmFamily
from pyfixest.estimation.internals.fit_glm_ import fit_glm_irls
from pyfixest.estimation.internals.separation import check_for_separation
from pyfixest.estimation.internals.vcov_ import vcov_iid_glm
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
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
        ],
        family: GlmFamily,
        demeaner: AnyDemeaner | None = None,
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: str | None = None,
        sample_split_value: str | int | None = None,
        separation_check: list[str] | None = None,
        context: int | Mapping[str, Any] = 0,
        accelerate: bool = True,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            collin_tol=collin_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
            demeaner=demeaner,
            lookup_preconditioner=lookup_preconditioner,
        )

        _glm_input_checks(
            drop_singletons=drop_singletons,
            tol=tol,
            maxiter=maxiter,
        )

        self.maxiter = maxiter
        self.tol = tol
        self.convergence = False
        self.separation_check = separation_check
        self._accelerate = accelerate

        # The inherited slow jackknife refits with the linear/Poisson APIs and
        # cannot yet preserve a generic GLM family's estimation contract.
        self._support_crv3_inference = False
        self._support_iid_inference = True
        self._support_hac_inference = True
        self._supports_wildboottest = False
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

        self._Y_hat_response = np.empty(0)
        self.deviance = None
        self._Xbeta = np.empty((0, 1))

        self._method = "feglm"
        self._family = family
        self._inference_dist = family.inference_dist

    def prepare_model_matrix(self) -> ModelMatrix:
        "Prepare model inputs for estimation."
        model_matrix = super().prepare_model_matrix()

        # check for separation
        na_separation: list[int] = []
        if (
            self._fe is not None
            and self.separation_check is not None
            and self.separation_check  # not an empty list
        ):
            na_separation = check_for_separation(
                Y=model_matrix.dependent,
                X=model_matrix.independent,
                fe=model_matrix.fixed_effects,
                fml=self._fml,
                data=self._data,
                demeaner=self._demeaner,
                methods=self.separation_check,
            )

        if na_separation:
            self._data.drop(na_separation, axis=0, inplace=True)
            model_matrix = model_matrix.without_rows(na_separation)
            self._publish_model_matrix(model_matrix)

            self.n_separation_na = len(na_separation)
            # possible to have dropped fixed effects level due to separation
            self._n_fe = np.sum(self._k_fe > 1) if self._has_fixef else 0

        return model_matrix

    def get_fit(self) -> None:
        "Fit the GLM via IRLS and write results onto self.* attributes."
        model_matrix = self._model_matrix
        response = model_matrix.dependent.to_numpy()
        design = model_matrix.independent.to_numpy()
        fixed_effects = (
            None
            if model_matrix.fixed_effects is None
            else model_matrix.fixed_effects.to_numpy()
        )
        offset = (
            None
            if model_matrix.offset is None
            else model_matrix.offset.to_numpy().reshape((-1, 1))
        )
        self._offset = offset

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
            collin_tol=self._collin_tol,
            accelerate=self._accelerate and fixed_effects is not None,
            offset=offset,
            weights=self._observation_weights.values,
            solver=self._solver,
            maxiter=self.maxiter,
            tol=self.tol,
            fixef_tol=self._fixef_tol,
        )

        self._coefnames = fit.coefnames
        self._collin_vars = fit.collin_vars
        self._collin_index = fit.collin_index
        working_state = fit.working_state
        self._working_state = working_state
        design_within = working_state.design_within
        self._X_is_empty = design_within.shape[1] == 0
        self._k = design_within.shape[1]

        self._beta_hat = fit.beta
        self._Y_hat_response = working_state.mu
        self._Y_hat_link = working_state.eta
        self._u_hat_response = working_state.response_residuals
        self._u_hat_working = working_state.working_residuals
        self._u_hat = self._u_hat_working

        weighted_working_residuals = (
            working_state.working_weights * working_state.working_residuals
        )
        # The IRLS score is W_i x_i e_i. ``working_weights`` already includes
        # any user-supplied observation weight.
        self._scores = design_within * weighted_working_residuals[:, None]
        weighted_design = working_state.working_weights[:, None] * design_within
        self._tZX = design_within.T @ weighted_design
        self._tZXinv = np.linalg.inv(self._tZX)
        self._hessian = self._tZX.copy()

        # Temporary aliases retained until the getter-only alias layer.
        self._Y = working_state.working_response_within
        self._X = design_within
        self._Z = design_within
        self._irls_weights = working_state.working_weights
        self._Xbeta = working_state.eta.reshape(-1, 1)
        self.deviance = fit.deviance
        self.convergence = fit.converged
        if self.convergence:
            self._convergence = True

    def _normal_equation_weights(self) -> np.ndarray:
        """Return the final IRLS weights in the fitted normal equations."""
        return self._working_state.working_weights

    def _fixef_recovery_weights(self) -> np.ndarray | None:
        """Return weights used by fixed-effect coefficient recovery.

        At convergence, ``eta - offset - X @ beta`` is the fixed-effect
        contribution, up to solver tolerance. For weighted fits the recovery
        solve uses the combined IRLS weights, which already contain the
        observation weights; applying observation weights again would double
        count them.
        """
        return self._working_state.working_weights if self._has_weights else None

    def _vcov_iid(self):
        return vcov_iid_glm(bread=self._bread)

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
        if type == "response":
            return self._u_hat_response.flatten()
        if type == "working":
            return self._u_hat_working.flatten()
        raise ValueError("type must be one of 'response' or 'working'.")

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

        effective_demeaner = self._demeaner.with_tol(tol)
        vX_tilde = self._demean_cache.demean_array(
            x=np.c_[v, X],
            flist=flist,
            weights=weights.flatten(),
            na_index=self._na_index,
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
        self._family.check_y(self._model_matrix.dependent)


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
