from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from pyfixest.errors import (
    NonConvergenceError,
)
from pyfixest.estimation.backends import BACKENDS
from pyfixest.estimation.feols_ import Feols, PredictionErrorOptions, PredictionType
from pyfixest.estimation.fepois_ import _check_for_separation
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.literals import DemeanerBackendOptions
from pyfixest.estimation.solvers import solve_ols
from pyfixest.utils.dev_utils import DataFrameType


class Feglm(Feols, ABC):
    "Abstract base class for the estimation of a fixed-effects GLM model."

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
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: Literal[
            "np.linalg.lstsq",
            "np.linalg.solve",
            "scipy.linalg.solve",
            "scipy.sparse.linalg.lsqr",
            "jax",
        ],
        demeaner_backend: DemeanerBackendOptions = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
        context: Union[int, Mapping[str, Any]] = 0,
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
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            lookup_demeaned_data=lookup_demeaned_data,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
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

        self._demeaner_backend = demeaner_backend
        try:
            impl = BACKENDS[demeaner_backend]
        except KeyError:
            raise ValueError(f"Unknown demeaner backend {demeaner_backend!r}")
        self._demean_func = impl["demean"]

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
            self._N = self._Y.shape[0]

            self.na_index = np.concatenate([self.na_index, np.array(na_separation)])
            self.n_separation_na = len(na_separation)

    def to_array(self):
        "Turn estimation DataFrames to np arrays."
        self._Y, self._X, self._Z = (
            self._Y.to_numpy(),
            self._X.to_numpy(),
            self._X.to_numpy(),
        )
        if self._fe is not None:
            self._fe = self._fe.to_numpy()
            if self._fe.ndim == 1:
                self._fe = self._fe.reshape((self._N, 1))

    def get_fit(self):
        """
        Fit the GLM model via iterated weighted least squares.

        The implementation follows ideas developed in
        - Bergé (2018): https://ideas.repec.org/p/luc/wpaper/18-13.html
        - Correia, Guimaraes, Zylkin (2019): https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691
        - Stamann (2018): https://arxiv.org/pdf/1707.01815
        """
        _mean = np.mean(self._Y)
        if self._method in ("feglm-logit", "feglm-probit"):
            mu = np.full_like(self._Y.flatten(), 0.5, dtype=float)
        else:
            mu = np.full_like(self._Y.flatten(), _mean, dtype=float)

        eta = self._get_link(mu)
        deviance = self._get_deviance(self._Y.flatten(), mu)
        deviance_old = deviance + 1.0

        for r in range(self.maxiter):
            if r > 0:
                converged = self._check_convergence(
                    crit=self._get_relative_deviance_change(deviance, deviance_old),
                    tol=self.tol,
                    r=r,
                    maxiter=self.maxiter,
                    model=self._method,
                )
                if converged:
                    self.convergence = True
                    break

            gprime = self._get_gprime(mu=mu)
            W = self._update_W(mu=mu)
            sqrt_W = np.sqrt(W)

            z = eta + (self._Y.flatten() - mu) * gprime

            z_tilde, X_tilde = self.residualize(
                v=z,
                X=self._X,
                flist=self._fe,
                weights=W.flatten(),
                tol=self._fixef_tol,
                maxiter=self._fixef_maxiter,
            )

            WX = sqrt_W.flatten()[:, None] * X_tilde
            WZ = sqrt_W.flatten() * z_tilde

            tXX = WX.T @ WX
            tXz = WX.T @ WZ
            beta_new = solve_ols(tXX, tXz, self._solver)

            # Residual from demeaned regression (not weighted)
            e_new = z_tilde - X_tilde @ beta_new
            eta_new = z - e_new

            mu_new = self._get_mu(eta=eta_new)
            deviance_new = self._get_deviance(self._Y.flatten(), mu_new)

            # Step-halving if deviance did not decrease
            eta_new, mu_new, deviance_new = self._step_halving(
                eta, eta_new, mu_new, deviance, deviance_new
            )

            deviance_old = deviance
            eta = eta_new
            mu = mu_new
            deviance = deviance_new

            z_tilde_final = z_tilde
            X_tilde_final = X_tilde
            sqrt_W_final = sqrt_W
            beta_final = beta_new

        self._beta_hat = beta_final
        self._Y_hat_response = mu.flatten()
        self._Y_hat_link = eta.flatten()

        # Update weights for inference
        self._weights = W
        self._irls_weights = W
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat_response = (self._Y.flatten() - mu).flatten()
        e_final = z_tilde_final - X_tilde_final @ self._beta_hat
        self._u_hat_working = (
            self._u_hat_response
            if self._method == "feglm-gaussian"
            else e_final.flatten()
        )

        self._scores_response = self._u_hat_response[:, None] * self._X
        self._scores_working = self._u_hat_working[:, None] * self._X

        sqrt_W_vec = sqrt_W_final.flatten()
        X_wls = sqrt_W_vec[:, None] * X_tilde_final
        z_wls = sqrt_W_vec * z_tilde_final

        self._u_hat = (z_wls - X_wls @ self._beta_hat).flatten()
        self._Y = z_wls
        self._X = X_wls
        self._Z = self._X

        self._scores = self._u_hat[:, None] * self._X

        self._tZX = self._Z.T @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._hessian = X_wls.T @ X_wls
        self.deviance = deviance

        if self.convergence:
            self._convergence = True

    def _vcov_iid(self):
        return self._bread

    def _update_v(
        self, y: np.ndarray, mu: np.ndarray, gprime: np.ndarray
    ) -> np.ndarray:
        "Get (running) dependent variable v for the GLM family."
        return (y - mu) * gprime

    def _update_W(self, mu: np.ndarray) -> np.ndarray:
        "Compute IRLS weights: w = 1 / (g'(μ)² · V(μ))."
        return 1 / (self._get_gprime(mu=mu) ** 2 * self._get_V(mu=mu))

    def _update_v_tilde(
        self, y: np.ndarray, mu: np.ndarray, sqrt_W: np.ndarray, gprime: np.ndarray
    ) -> np.ndarray:
        "Get sqrt(W) * v for weighted least squares transformation."
        return sqrt_W * ((y - mu) * gprime)

    def _update_X_tilde(self, sqrt_W: np.ndarray, X: np.ndarray) -> np.ndarray:
        "Get sqrt(W) * X for weighted least squares transformation."
        return sqrt_W.reshape(-1, 1) * X

    def _update_beta_diff(
        self, X_dotdot: np.ndarray, v_dotdot: np.ndarray
    ) -> np.ndarray:
        "Get the beta _update difference (formula 3.5) via WLS fit."
        beta_diff = np.linalg.lstsq(X_dotdot, v_dotdot.reshape(-1, 1), rcond=None)[
            0
        ].flatten()
        return beta_diff

    def _update_eta(
        self,
        sqrt_W: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        X_tilde: np.ndarray,
        beta_diff: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        e = z_tilde - X_tilde @ beta_diff
        e_unweighted = e / sqrt_W
        return z - e_unweighted

    def _get_gradient(self, Z: np.ndarray, W: np.ndarray, v: np.ndarray) -> np.ndarray:
        return Z.T @ W @ v

    def _get_relative_deviance_change(
        self, deviance: np.ndarray, deviance_old: np.ndarray
    ) -> np.ndarray:
        "Compute relative change in deviance for convergence check."
        return np.abs(deviance - deviance_old) / (0.1 + np.abs(deviance_old))

    def _step_halving(
        self,
        eta: np.ndarray,
        eta_new: np.ndarray,
        mu_new: np.ndarray,
        deviance: np.ndarray,
        deviance_new: np.ndarray,
        step_halving_tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply step-halving if deviance did not decrease.

        Returns updated (eta_new, mu_new, deviance_new).
        """
        if deviance_new < deviance:
            return eta_new, mu_new, deviance_new

        alpha = 1.0
        while alpha > step_halving_tol:
            alpha /= 2.0
            eta_try = eta + alpha * (eta_new - eta)
            mu_try = self._get_mu(eta=eta_try)
            deviance_try = self._get_deviance(self._Y.flatten(), mu_try)
            if deviance_try < deviance:
                return eta_try, mu_try, deviance_try

        # Step-halving exhausted - check if change is within tolerance
        if self._get_relative_deviance_change(deviance_new, deviance) < self.tol:
            return eta_new, mu_new, deviance_new

        raise RuntimeError(
            f"Step-halving failed. Deviance: {deviance_new:.6f} vs {deviance:.6f}"
        )

    def residualize(
        self,
        v: np.ndarray,
        X: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray,
        tol: np.ndarray,
        maxiter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        "Residualize v and X by flist using weights."
        if flist is None:
            return v, X
        else:
            vX_tilde, success = self._demean_func(
                x=np.c_[v, X],
                flist=flist.astype(np.uintp),
                weights=weights,
                tol=tol,
                maxiter=maxiter,
            )
            if success is False:
                raise ValueError(f"Demeaning failed after {maxiter} iterations.")
            else:
                return vX_tilde[:, 0], vX_tilde[:, 1:]

    def _check_convergence(
        self,
        crit: float,
        tol: float,
        r: int,
        maxiter: int,
        model: str,
    ) -> bool:
        if model == "feglm-gaussian":
            converged = True
        else:
            converged = crit < tol
            if r == maxiter:
                raise NonConvergenceError(
                    f"""
                    The IRLS algorithm did not converge with {maxiter}
                    iterations. Try to increase the maximum number of iterations.
                    """
                )

        return converged

    def _update_eta_step_halfing(
        self,
        Y: np.ndarray,
        beta: np.ndarray,
        eta: np.ndarray,
        mu: np.ndarray,
        deviance: np.ndarray,
        beta_update_diff: np.ndarray,
        sqrt_W: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        X_tilde: np.ndarray,
        deviance_old: np.ndarray,
        step_halfing_tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        "Update parameters, potentially using step halfing."
        alpha = 1.0
        step_accepted = False

        while alpha > step_halfing_tolerance:
            beta_try = beta + alpha * beta_update_diff
            eta_try = self._update_eta(
                sqrt_W=sqrt_W.flatten(),
                z=z,
                z_tilde=z_tilde,
                X_tilde=X_tilde,
                beta_diff=alpha * beta_update_diff,
                eta=eta,
            )
            mu_try = self._get_mu(eta=eta_try)
            deviance_try = self._get_deviance(Y.flatten(), mu_try)
            if deviance_try < deviance_old:
                beta = beta_try
                eta = eta_try
                mu = mu_try
                deviance = deviance_try
                step_accepted = True
                break
            else:
                alpha /= 2.0

        if not step_accepted:
            raise RuntimeError("Step-halving failed to find improvement.")

        return beta, eta, mu, deviance

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
