from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.fit_ import fit_iwls_glm
from pyfixest.estimation.models.feols_ import (
    Feols,
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.estimation.models.fepois_ import _check_for_separation
from pyfixest.utils.dev_utils import DataFrameType


class Feglm(Feols, ABC):
    "Abstract base class for the estimation of a fixed-effects GLM model."

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
        demeaner: AnyDemeaner | None = None,
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

        self._method = "feglm"

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
            na_separation = _check_for_separation(
                Y=self._Y_df,
                X=self._X_df,
                fe=self._fe,
                fml=self._fml,
                data=self._data,
                methods=self.separation_check,
            )

        if na_separation:
            self._Y_df.drop(na_separation, axis=0, inplace=True)
            self._X_df.drop(na_separation, axis=0, inplace=True)
            self._fe.drop(na_separation, axis=0, inplace=True)
            self._data.drop(na_separation, axis=0, inplace=True)
            self._N = self._Y_df.shape[0]

            self.na_index = np.concatenate([self.na_index, np.array(na_separation)])
            self.n_separation_na = len(na_separation)

    def to_array(self) -> tuple[np.ndarray, np.ndarray]:
        "Return dependent variable and design matrix as np arrays."
        Y_arr = self._Y_df.to_numpy()
        X_arr = self._X_df.to_numpy()
        if self._fe is not None:
            self._fe = self._fe.to_numpy()
            if self._fe.ndim == 1:
                self._fe = self._fe.reshape((self._N, 1))
        return Y_arr, X_arr

    def get_fit(self):
        """
        Fit the GLM model via iterated weighted least squares.

        The implementation follows ideas developed in
        - Bergé (2018): https://ideas.repec.org/p/luc/wpaper/18-13.html
        - Correia, Guimaraes, Zylkin (2019): https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691
        - Stamann (2018): https://arxiv.org/pdf/1707.01815
        """
        Y_arr, X_arr = self.to_array()

        def _residualize(
            v: np.ndarray, X: np.ndarray, weights: np.ndarray, tol: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return self.residualize(v=v, X=X, flist=self._fe, weights=weights, tol=tol)

        glm_fit = fit_iwls_glm(
            Y=Y_arr,
            X=X_arr,
            family=self,
            method=self._method,
            coefnames=self._coefnames,
            collin_tol=self._collin_tol,
            solver=self._solver,
            tol=self.tol,
            maxiter=self.maxiter,
            accelerate=self._accelerate and self._fe is not None,
            fixef_tol=self._fixef_tol,
            residualize=_residualize,
        )

        # collinearity info from the first IWLS iteration
        self._coefnames = glm_fit.coefnames
        self._collin_vars = glm_fit.collin_vars
        self._collin_index = glm_fit.collin_index
        X_arr = glm_fit.X
        self._X_is_empty = X_arr.shape[1] == 0
        self._k = X_arr.shape[1]
        self.convergence = glm_fit.convergence

        eta = glm_fit.eta
        mu = glm_fit.mu
        W = glm_fit.irls_weights

        self._beta_hat = glm_fit.beta
        self._Y_hat_response = mu.flatten()
        self._Y_hat_link = eta.flatten()

        # Final IRLS weights for inference; user weights stay untouched
        self._weights_irls = W.reshape((self._N, 1)) if W.ndim == 1 else W

        self._u_hat_response = (Y_arr.flatten() - mu).flatten()
        e_final = glm_fit.z_tilde - glm_fit.X_tilde @ self._beta_hat
        self._u_hat_working = (
            self._u_hat_response
            if self._method == "feglm-gaussian"
            else e_final.flatten()
        )

        self._scores_response = self._u_hat_response[:, None] * X_arr
        self._scores_working = self._u_hat_working[:, None] * X_arr

        sqrt_W_vec = glm_fit.sqrt_W.flatten()
        X_wls = sqrt_W_vec[:, None] * glm_fit.X_tilde
        z_wls = sqrt_W_vec * glm_fit.z_tilde

        self._u_hat_wls = (z_wls - X_wls @ self._beta_hat).flatten()
        self._Y_wls = z_wls
        self._X_wls = X_wls

        self._scores = self._u_hat_wls[:, None] * self._X_wls

        self._tZX = self._X_wls.T @ self._X_wls
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._hessian = X_wls.T @ X_wls
        self.deviance = glm_fit.deviance

        if self.convergence:
            self._convergence = True

    def resid(self) -> np.ndarray:
        """
        Return working residuals from the GLM model.

        Returns
        -------
        np.ndarray
            A flat array with the working residuals of the GLM model.
        """
        return self._u_hat_wls.flatten() / np.sqrt(self._weights_irls.flatten())

    def _vcov_iid(self):
        return self._bread

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
            return self._get_mu(
                eta=yhat.to_numpy() if isinstance(yhat, pd.DataFrame) else yhat
            )
        else:
            return yhat

    @abstractmethod
    def _check_dependent_variable(self):
        pass

    @abstractmethod
    def _get_score(
        self, y: np.ndarray, X: np.ndarray, mu: np.ndarray, eta: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        "Compute the deviance for the GLM family."
        pass

    @abstractmethod
    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        "Get the dispersion parameter phi for the GLM family."
        pass

    @abstractmethod
    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        "Get the cumulant function b(theta) for the GLM family."
        pass

    @abstractmethod
    def _get_mu(self, eta: np.ndarray) -> np.ndarray:
        "Apply inverse link function: μ = g⁻¹(η)."
        pass

    @abstractmethod
    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        "Apply link function: η = g(μ)."
        pass

    @abstractmethod
    def _get_gprime(self, mu: np.ndarray) -> np.ndarray:
        "Get the derivative of the link function g'(μ) = dη/dμ."
        pass

    @abstractmethod
    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        "Get the mechanical link theta(mu) for the GLM family."
        pass

    @abstractmethod
    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        "Get the variance function V(mu) for the GLM family."
        pass


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
