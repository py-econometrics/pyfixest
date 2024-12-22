from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from pyfixest.estimation.demean_ import demean
from pyfixest.estimation.fepois_ import Fepois
from pyfixest.estimation.FormulaParser import FixestFormula


class Feglm(Fepois, ABC):
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
        mu = self.get_mu(theta=eta)
        deviance_old = self.compute_deviance(_Y.flatten(), mu)

        for r in range(_maxiter):
            # Step 1: compute weights w_tilde(r-1) and v(r-1) (eq. 2.5)
            detadmu = self.update_detadmu(mu=mu)
            v = self.update_v(y=_Y.flatten(), mu=mu, detadmu=detadmu)
            W = self.update_W(mu=mu)

            # Step 2: compute v_tilde(r-1) and X_tilde(r-1) (eq. 3.2)
            W_tilde = self.update_W_tilde(W=W)
            X_tilde = self.update_X_tilde(W_tilde=W_tilde, X=_X)
            v_tilde = self.update_v_tilde(
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

            # Step 4: compute (beta(r) - beta(r-1)) and check for convergence, update beta(r-1) s(eq. 3.5)
            beta_update_diff = self.update_beta_diff(
                X_dotdot=X_dotdot, v_dotdot=v_dotdot
            )

            # Step 5: update using step halfing (if required)
            beta, eta, mu, deviance, crit, step_accepted = self.update_eta_step_halfing(
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
            converged = self.stop_iterating(crit=crit, tol=_tol, r=r, maxiter=_maxiter)
            if converged:
                break

        import pdb

        pdb.set_trace()

    @abstractmethod
    def compute_deviance(self, y, mu):
        "Compute the deviance for the GLM family."
        pass

    @abstractmethod
    def get_dispersion_phi(self, theta):
        "Get the dispersion parameter phi for the GLM family."
        pass

    @abstractmethod
    def get_b(self, theta):
        "Get the cumulant function b(theta) for the GLM family."
        pass

    # @abstractmethod
    # def get_c(self, y, phi):
    #
    #    "Get the function c(y, phi) for the GLM family."
    #    pass

    @abstractmethod
    def get_mu(self, theta):
        "Get the mean mu(theta) for the GLM family."
        pass

    @abstractmethod
    def get_link(self, mu):
        "Get the link function theta(mu) for the GLM family."
        pass

    @abstractmethod
    def update_detadmu(self, mu):
        "Get the derivative of mu(theta) with respect to theta for the GLM family."
        pass

    @abstractmethod
    def get_theta(self, mu):
        "Get the mechanical link theta(mu) for the GLM family."
        pass

    @abstractmethod
    def get_V(self, mu):
        "Get the variance function V(mu) for the GLM family."
        pass

    def update_v(self, y, mu, detadmu):
        "Get (running) dependent variable v for the GLM family."
        return (y - mu) * detadmu

    def update_W(self, mu):
        "Get (running) weights W for the GLM family."
        return 1 / (self.update_detadmu(mu=mu) ** 2 * self.get_V(mu=mu))
        # return (mu * (1 - mu)).reshape(-1, 1)

    def update_W_tilde(self, W):
        "Get W_tilde (formula 3.2)."
        return np.sqrt(W)

    def update_v_tilde(self, y, mu, W_tilde, detadmu):
        "Get v_tilde (formula 3.2)."
        return W_tilde * ((y - mu) * detadmu)

    def update_X_tilde(self, W_tilde, X):
        "Get X_tilde (formula 3.2)."
        # import pdb; pdb.set_trace()
        return W_tilde.reshape(-1, 1) * X

    def update_beta_diff(self, X_dotdot, v_dotdot):
        "Get the beta update difference (formula 3.5) via WLS fit."
        beta_diff = np.linalg.lstsq(X_dotdot, v_dotdot.reshape(-1, 1), rcond=None)[
            0
        ].flatten()
        return beta_diff

    def update_eta(self, W_tilde, v_tilde, v_dotdot, X_dotdot, beta_diff, eta):
        "Get the eta update (formula 4.5)."
        # import pdb; pdb.set_trace()
        # return  (v_tilde  - X_dotdot @ beta_diff) / W_tilde +
        return eta + X_dotdot @ beta_diff / W_tilde
        # return  (v_tilde - v_dotdot - X_dotdot @ beta_diff) / W_tilde + eta

    def get_gradient(self, Z, W, v):
        return Z.T @ W @ v

    def compute_diff(self, deviance, last):
        return np.abs(deviance - last) / (0.1 + np.abs(last))

    def residualize(self, v, X, flist, weights, tol):
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

    def stop_iterating(self, crit, tol, r, maxiter):
        stop_iterating = crit < tol
        converged = stop_iterating
        if r == maxiter:
            raise NonConvergenceError(
                f"""
                The IRLS algorithm did not converge with {maxiter}
                iterations. Try to increase the maximum number of iterations.
                """
            )

        return converged

    def update_eta_step_halfing(
        self,
        Y,
        beta,
        eta,
        mu,
        deviance,
        beta_update_diff,
        W_tilde,
        v_tilde,
        v_dotdot,
        X_dotdot,
        deviance_old,
        step_halfing_tolerance,
    ):
        "Update parameters, potentially using step halfing."
        alpha = 1.0
        step_accepted = False

        while alpha > step_halfing_tolerance:
            beta_try = beta + alpha * beta_update_diff
            eta_try = self.update_eta(
                W_tilde=W_tilde.flatten(),
                v_tilde=v_tilde,
                v_dotdot=v_dotdot,
                X_dotdot=X_dotdot,
                beta_diff=alpha * beta_update_diff,
                eta=eta,
            )
            mu_try = self.get_mu(theta=eta_try)
            deviance_try = self.compute_deviance(Y.flatten(), mu_try)
            if deviance_try < deviance_old:
                beta = beta_try
                eta = eta_try
                mu = mu_try
                deviance = deviance_try
                step_accepted = True
                crit = self.compute_diff(deviance, deviance_old)
                break
            else:
                alpha /= 2.0

        if not step_accepted:
            raise RuntimeError("Step-halving failed to find improvement.")

        return beta, eta, mu, deviance, crit, step_accepted


class Felogit(Feglm):
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

    def compute_deviance(self, y, mu):
        eps = 1e-15
        mu_clipped = np.clip(mu, eps, 1 - eps)
        return -2 * np.sum(y * np.log(mu_clipped) + (1 - y) * np.log(1 - mu_clipped))

    def get_dispersion_phi(self, theta):
        return 1

    def get_b(self, theta):
        return np.log(1 + np.exp(theta))

    # def get_c(self, y, phi):
    #    return 0
    def get_mu(self, theta):
        return np.exp(theta) / (1 + np.exp(theta))

    def get_link(self, mu):
        return np.log(mu / (1 - mu))

    def update_detadmu(self, mu):
        return 1 / (mu * (1 - mu))

    def get_theta(self, mu):
        return np.log(mu / (1 - mu))

    def get_V(self, mu):
        return mu * (1 - mu)


class Fepois2(Feglm):
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

    def compute_deviance(self, y, mu):
        return -2 * np.sum(y * np.log(mu) - mu)

    def get_dispersion_phi(self, theta):
        return 1

    def get_b(self, theta):
        return np.exp(theta)

    # def get_c(self, y, phi):
    #    return np.log(y)    # TODO: error!
    def get_mu(self, theta):
        return np.exp(theta)

    def get_link(self, mu):
        return np.log(mu)

    def update_detadmu(self, mu):
        return 1 / mu

    def get_theta(self, mu):
        return np.log(mu)

    def get_V(self, mu):
        return mu
