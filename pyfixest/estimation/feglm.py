from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyfixest.errors import (
    NonConvergenceError,
)
from pyfixest.estimation.demean_ import demean
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FormulaParser import FixestFormula


class Feglm(Fepois, ABC):
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
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
    ):
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            tol=tol,
            maxiter=maxiter,
            collin_tol=collin_tol,
            fixef_tol=fixef_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
        )

        self._method = "feglm"

    def get_fit(self):
        "Fit the GLM model via iterated weighted least squares."
        _Y = self._Y
        _X = self._X
        _fe = self._fe
        _N = self._N
        _convergence = self.convergence  # False
        _maxiter = self.maxiter
        _tol = self.tol
        _fixef_tol = self._fixef_tol
        _solver = self._solver

        # initialize

        beta = np.zeros(_X.shape[1])
        eta = np.zeros(_N)
        mu = self._get_mu(theta=eta)
        deviance_old = self._get_deviance(_Y.flatten(), mu)

        for r in range(_maxiter):
            # Step 1: _get weights w_tilde(r-1) and v(r-1) (eq. 2.5)
            detadmu = self._update_detadmu(mu=mu)
            # v = self._update_v(y=_Y.flatten(), mu=mu, detadmu=detadmu)
            W = self._update_W(mu=mu)

            # Step 2: _get v_tilde(r-1) and X_tilde(r-1) (eq. 3.2)
            W_tilde = self._update_W_tilde(W=W)
            X_tilde = self._update_X_tilde(W_tilde=W_tilde, X=_X)
            v_tilde = self._update_v_tilde(
                y=_Y.flatten(), mu=mu, W_tilde=W_tilde.flatten(), detadmu=detadmu
            )

            # Step 3 compute v_dotdot(r-1) and X_dotdot(r-1) - demeaning
            v_dotdot, X_dotdot = self.residualize(
                v=v_tilde,
                X=X_tilde,
                flist=_fe,
                weights=W_tilde.flatten(),
                tol=_fixef_tol,
            )

            # Step 4: compute (beta(r) - beta(r-1)) and check for convergence, _update beta(r-1) s(eq. 3.5)
            beta_update_diff = self._update_beta_diff(
                X_dotdot=X_dotdot, v_dotdot=v_dotdot
            )

            # Step 5: _update using step halfing (if required)
            mu_old = mu.copy()
            beta, eta, mu, deviance, step_accepted = self._update_eta_step_halfing(
                Y=_Y,
                beta=beta,
                eta=eta,
                mu=mu,
                deviance=deviance_old,
                beta_update_diff=beta_update_diff,
                W_tilde=W_tilde.flatten(),
                v_tilde=v_tilde,
                v_dotdot=v_dotdot,
                X_dotdot=X_dotdot,
                deviance_old=deviance_old,
                step_halfing_tolerance=1e-12,
            )

            print("beta:", beta)

            deviance_old = deviance.copy()
            converged = self._stop_iterating(
                crit=self._get_diff(deviance, deviance_old),
                tol=_tol,
                r=r,
                maxiter=_maxiter,
                beta_update_diff=beta_update_diff,
            )

            if converged:
                break

        self._beta_hat = beta.flatten()
        self._Y_hat_response = mu.flatten()
        self._Y_hat_link = eta.flatten()
        # (Y - self._Y_hat)
        # needed for the calculation of the vcov

        # _update for inference
        self._weights = mu_old
        self._irls_weights = mu
        # if only one dim
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat = (v_dotdot - X_dotdot @ beta).flatten()
        # self._u_hat_working = resid
        # self._u_hat_response = self._Y - np.exp(eta)

        self._Y = (v_dotdot / np.sqrt(W_tilde)).reshape(-1, 1)
        self._X = X_dotdot / np.sqrt(W_tilde.reshape(-1, 1))
        self._Z = self._Z
        self.deviance = deviance

        self._tZX = np.transpose(self._Z) @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._scores = self._u_hat[:, None] * self._X
        self._hessian = X_dotdot.T @ X_dotdot

        if _convergence:
            self._convergence = True

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

    # @abstractmethod
    # def _get_c(self, y, phi):
    #
    #    "Get the function c(y, phi) for the GLM family."
    #    pass

    @abstractmethod
    def _get_mu(self, theta: np.ndarray) -> np.ndarray:
        "Get the mean mu(theta) for the GLM family."
        pass

    @abstractmethod
    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        "Get the link function theta(mu) for the GLM family."
        pass

    @abstractmethod
    def _update_detadmu(self, mu: np.ndarray) -> np.ndarray:
        "Get the derivative of mu(theta) with respect to theta for the GLM family."
        pass

    @abstractmethod
    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        "Get the mechanical link theta(mu) for the GLM family."
        pass

    @abstractmethod
    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        "Get the variance function V(mu) for the GLM family."
        pass

    def _update_v(
        self, y: np.ndarray, mu: np.ndarray, detadmu: np.ndarray
    ) -> np.ndarray:
        "Get (running) dependent variable v for the GLM family."
        return (y - mu) * detadmu

    def _update_W(self, mu: np.ndarray) -> np.ndarray:
        "Get (running) weights W for the GLM family."
        return 1 / (self._update_detadmu(mu=mu) ** 2 * self._get_V(mu=mu))
        # return (mu * (1 - mu)).reshape(-1, 1)

    def _update_W_tilde(self, W: np.ndarray) -> np.ndarray:
        "Get W_tilde (formula 3.2)."
        return np.sqrt(W)

    def _update_v_tilde(
        self, y: np.ndarray, mu: np.ndarray, W_tilde: np.ndarray, detadmu: np.ndarray
    ) -> np.ndarray:
        "Get v_tilde (formula 3.2)."
        return W_tilde * ((y - mu) * detadmu)

    def _update_X_tilde(self, W_tilde: np.ndarray, X: np.ndarray) -> np.ndarray:
        "Get X_tilde (formula 3.2)."
        # import pdb; pdb.set_trace()
        return W_tilde.reshape(-1, 1) * X

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
        W_tilde: np.ndarray,
        v_tilde: np.ndarray,
        v_dotdot: np.ndarray,
        X_dotdot: np.ndarray,
        beta_diff: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        "Get the eta _update (formula 4.5)."
        # import pdb; pdb.set_trace()
        # return  (v_tilde  - X_dotdot @ beta_diff) / W_tilde +
        return eta + X_dotdot @ beta_diff / W_tilde
        # return  (v_tilde - v_dotdot - X_dotdot @ beta_diff) / W_tilde + eta

    def _get_gradient(self, Z: np.ndarray, W: np.ndarray, v: np.ndarray) -> np.ndarray:
        return Z.T @ W @ v

    def _get_diff(self, deviance: np.ndarray, last: np.ndarray) -> np.ndarray:
        return np.abs(deviance - last) / (0.1 + np.abs(last))

    def residualize(
        self,
        v: np.ndarray,
        X: np.ndarray,
        flist: np.ndarray,
        weights: np.ndarray,
        tol: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        "Residualize v and X by flist using weights."
        if flist is None:
            return v, X
        else:
            vX_resid, success = demean(
                x=np.c_[v, X], flist=flist, weights=weights, tol=tol
            )
            if success is False:
                raise ValueError("Demeaning failed after 100_000 iterations.")
            else:
                return vX_resid[:, 0], vX_resid[:, 1:]

    def _stop_iterating(
        self,
        crit: float,
        tol: float,
        r: int,
        maxiter: int,
        beta_update_diff: np.ndarray,
    ) -> bool:
        converged = crit < tol  # or np.max(beta_update_diff) < crit
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
        W_tilde: np.ndarray,
        v_tilde: np.ndarray,
        v_dotdot: np.ndarray,
        X_dotdot: np.ndarray,
        deviance_old: np.ndarray,
        step_halfing_tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        "Update parameters, potentially using step halfing."
        alpha = 1.0
        step_accepted = False

        while alpha > step_halfing_tolerance:
            beta_try = beta + alpha * beta_update_diff
            eta_try = self._update_eta(
                W_tilde=W_tilde.flatten(),
                v_tilde=v_tilde,
                v_dotdot=v_dotdot,
                X_dotdot=X_dotdot,
                beta_diff=alpha * beta_update_diff,
                eta=eta,
            )
            mu_try = self._get_mu(theta=eta_try)
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

        return beta, eta, mu, deviance, step_accepted


class Felogit(Feglm):
    "Class for the estimation of a fixed-effects logit model."

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
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
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
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
        )

        self._method = "feglm-logit"

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        return 1.0

    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(theta))

    # def _get_c(self, y, phi):
    #    return 0
    def _get_mu(self, theta: np.ndarray) -> np.ndarray:
        return np.exp(theta) / (1 + np.exp(theta))

    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu / (1 - mu))

    def _update_detadmu(self, mu: np.ndarray) -> np.ndarray:
        return 1 / (mu * (1 - mu))

    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu / (1 - mu))

    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

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
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
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
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
        )

        self._method = "feglm-probit"

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return -2 * np.sum(
            y * np.log(norm.cdf(mu)) + (1 - y) * np.log(1 - norm.cdf(mu))
        )

    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        return 1.0

    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        raise ValueError("The function _get_b is not implemented for the probit model.")
        return None

    def _get_mu(self, theta: np.ndarray) -> np.ndarray:
        return norm.cdf(theta)

    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        return norm.ppf(mu)

    def _update_detadmu(self, mu: np.ndarray) -> np.ndarray:
        return 1 / norm.pdf(norm.ppf(mu))

    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        return norm.ppf(mu)

    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)


class Fegaussian(Feglm):
    "Class for the estimation of a fixed-effects GLM with normal errors."

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
        lookup_demeaned_data: dict[str, pd.DataFrame],
        tol: float,
        maxiter: int,
        solver: str = "np.linalg.solve",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        separation_check: Optional[list[str]] = None,
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
            lookup_demeaned_data=lookup_demeaned_data,
            tol=tol,
            maxiter=maxiter,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            separation_check=separation_check,
        )

        self._method = "feglm-gaussian"

    def _get_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.sum((y - mu) ** 2)

    def _get_dispersion_phi(self, theta: np.ndarray) -> float:
        return np.var(theta)

    def _get_b(self, theta: np.ndarray) -> np.ndarray:
        return theta**2 / 2

    def _get_mu(self, theta: np.ndarray) -> np.ndarray:
        return theta

    def _get_link(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def _update_detadmu(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def _get_theta(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def _get_V(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)
