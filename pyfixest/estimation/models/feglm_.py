from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.families import FAMILY_FROM_METHOD, GlmFamily
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
    "Base class for the estimation of a fixed-effects GLM model."

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
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
        ],
        family: GlmFamily | None = None,
        method: str = "feglm",
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
    ):
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

        self._support_crv3_inference = True
        self._support_iid_inference = True
        self._support_hac_inference = True
        self._supports_cluster_causal_variance = False
        self._support_decomposition = False

        self._Y_hat_response = np.empty(0)
        self.deviance = None
        self._Xbeta = np.empty(0)

        self._method = method
        self._family = family if family is not None else FAMILY_FROM_METHOD[method]
        self._inference_dist = self._family.inference_dist

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        # check for separation
        na_separation: list[int] = []
        if (
            self._fe is not None
            and self.separation_check is not None
            and self.separation_check  # not an empty list
        ):
            na_separation = check_for_separation(
                Y=self._Y,
                X=self._X,
                fe=self._fe,
                fml=self._fml,
                data=self._data,
                methods=self.separation_check,
            )

        if na_separation:
            self._Y.drop(na_separation, axis=0, inplace=True)
            self._X.drop(na_separation, axis=0, inplace=True)
            self._fe.drop(na_separation, axis=0, inplace=True)
            self._data.drop(na_separation, axis=0, inplace=True)
            if self._weights_df is not None:
                self._weights_df.drop(na_separation, axis=0, inplace=True)
            if self._offset_df is not None:
                self._offset_df.drop(na_separation, axis=0, inplace=True)
            self._N = self._Y.shape[0]
            self._N_rows = self._N
            # Re-set weights after dropping rows (handles both weighted and unweighted)
            self._weights = self._set_weights()

            self.na_index = np.concatenate([self.na_index, np.array(na_separation)])
            self.n_separation_na = len(na_separation)
            # possible to have dropped fixed effects level due to separation
            self._k_fe = self._fe.nunique(axis=0) if self._has_fixef else None
            self._n_fe = np.sum(self._k_fe > 1) if self._has_fixef else 0

    def to_array(self):
        "Turn estimation DataFrames to np arrays."
        self._Y, self._X, self._Z = (
            self._Y.to_numpy(),
            self._X.to_numpy(),
            self._X.to_numpy(),
        )
        if self._offset_df is not None:
            self._offset = self._offset_df.to_numpy().reshape((-1, 1))
        if self._fe is not None:
            self._fe = self._fe.to_numpy()
            if self._fe.ndim == 1:
                self._fe = self._fe.reshape((self._N, 1))

    def get_fit(self):
        "Fit the GLM via IRLS and write results onto self.* attributes."
        self.to_array()

        def _demean(
            v: np.ndarray, X: np.ndarray, weights: np.ndarray, tol: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return self.residualize(v=v, X=X, flist=self._fe, weights=weights, tol=tol)

        fit = fit_glm_irls(
            X=self._X,
            Y=self._Y,
            family=self._family,
            demean=_demean,
            coefnames=self._coefnames,
            collin_tol=self._collin_tol,
            accelerate=self._accelerate and self._fe is not None,
            offset=self._offset,
            weights=self._weights if self._has_weights else None,
            solver=self._solver,
            maxiter=self.maxiter,
            tol=self.tol,
            fixef_tol=self._fixef_tol,
        )

        self._coefnames = fit.coefnames
        self._collin_vars = fit.collin_vars
        self._collin_index = fit.collin_index
        self._X = fit.X
        self._X_is_empty = self._X.shape[1] == 0
        self._k = self._X.shape[1]

        self._beta_hat = fit.beta
        self._Y_hat_response = fit.mu.flatten()
        self._Y_hat_link = fit.eta.flatten()

        self._weights = fit.W
        self._irls_weights = fit.W
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat_response = (self._Y.flatten() - fit.mu).flatten()
        e_final = fit.z_tilde - fit.X_tilde @ self._beta_hat
        self._u_hat_working = (
            self._u_hat_response
            if self._method == "feglm-gaussian"
            else e_final.flatten()
        )

        self._scores_response = self._u_hat_response[:, None] * self._X
        self._scores_working = self._u_hat_working[:, None] * self._X

        sqrt_W_vec = fit.sqrt_W.flatten()
        X_wls = sqrt_W_vec[:, None] * fit.X_tilde
        z_wls = sqrt_W_vec * fit.z_tilde

        self._u_hat = (z_wls - X_wls @ self._beta_hat).flatten()
        self._Y = z_wls
        self._X = X_wls
        self._Z = self._X

        self._scores = self._u_hat[:, None] * self._X

        self._tZX = self._Z.T @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = fit.eta.reshape(-1, 1)

        self._hessian = X_wls.T @ X_wls
        self.deviance = fit.deviance
        self.convergence = fit.converged
        if self.convergence:
            self._convergence = True

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
        flist: np.ndarray,
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

    def _check_dependent_variable(self) -> None:
        "Validate the dependent variable according to the family's constraints."
        self._family.check_y(self._Y)


def _glm_input_checks(drop_singletons: bool, tol: float, maxiter: int):
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
