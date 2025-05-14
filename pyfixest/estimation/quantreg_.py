from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack
from scipy.stats import norm

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula

import warnings
class Quantreg(Feols):
    """
    Quantile regression model.
    """

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
        solver: Literal[
            "np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"
        ] = "np.linalg.solve",
        demeaner_backend: Literal["numba", "jax"] = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        quantile: float = 0.5,
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
            fixef_tol=fixef_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
            demeaner_backend=demeaner_backend,
        )

        self._quantile = quantile
        self._method = "quantreg"

        self._model_name = (
            FixestFormula.fml
            if self._sample_split_var is None
            else f"{FixestFormula.fml} (Sample: {self._sample_split_var} = {self._sample_split_value})"
        )
        # update with quantile name
        self._model_name = f"{self._model_name} (q = {quantile})"
        self._model_name_plot = self._model_name

        self._rng = np.random.default_rng()

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


    def get_fit_multi(self, quantiles: list[float]) -> np.ndarray:

        quantiles = np.sort(np.array(quantiles))

        betas = np.zeros((len(quantiles), self._X.shape[1]))

        for i, q in enumerate(quantiles):
            if i == 0:
                self.get_fit()
                beta_new = self._beta_hat
            else:
                beta_init = beta_new
                u_init = self._u_hat
                self.get_fit(beta_init = beta_init)
                J = jacobian_powell(
                    X = self._X,
                    Y = self._Y,
                    beta = self._beta_hat,
                    tau = q,
                    kernel = "epanechnikov"
                )
                beta_new = beta_init - np.linalg.inv(J) * np.mean(q - u_init < 0)

            betas[i, :] = beta_new

        return betas

    def get_fit(self) -> None:
        """Fit a quantile regression model using the interior point method."""

        self._beta_hat = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile, rng = self._rng, beta_init = None)
        self._u_hat = self._Y.flatten() - self._X @ self._beta_hat
        self._hessian = self._X.T @ self._X
        self._bread = np.linalg.inv(self._hessian)

    def fit_qreg_ip(self, X: np.ndarray, Y: np.ndarray, q: float, rng: np.random.Generator, beta_init: Optional[np.ndarray] = None) -> np.ndarray:

        """Fit a quantile regression model using the interior point method."""

        N, k = self._X.shape
        update_initial_model = True
        has_converged = False
        compute_beta_init = True if beta_init is None else False
        max_bad_fixups = 3
        n_bad_fixups = 0

        # m constant should be set by user
        m = 0.8
        n_init = int(np.ceil((k * N) ** (2 / 3)))
        M = int(np.ceil(m * n_init))


        while not has_converged:

            if compute_beta_init:

                # get initial sample
                idx_init = rng.choice(N, size=n_init, replace=False)
                beta_hat_init = self.fit_qreg(X[idx_init, :], Y[idx_init], q = q)

            else:
                beta_hat_init = beta_init

            r_init = Y.flatten() - X @ beta_hat_init
            # conservative estimate of sigma^2
            z = np.dot(r_init, r_init) / (N - k)
            rz = r_init / np.sqrt(z)

            ql = max(0, q - M / (2 * N))
            qu = min(1, q + M / (2 * N))

            JL = rz < np.quantile(rz, ql)
            JH = rz > np.quantile(rz, qu)

            keep = ~(JL | JH)
            X_sub = X[keep,:]
            Y_sub = Y[keep,:]

            if np.any(JL):
                X_neg = np.sum(X[JL,:], axis = 0)
                Y_neg = np.sum(Y[JL])
                X_sub = np.concatenate([X_sub, X_neg.reshape((1,self._k))], axis = 0)
                Y_sub = np.concatenate([Y_sub, Y_neg.reshape((1,1))], axis = 0)
            if np.any(JH):
                X_pos = np.sum(X[JH,:], axis = 0)
                Y_pos = np.sum(Y[JH])
                X_sub = np.concatenate([X_sub, X_pos.reshape(1,self._k)], axis = 0)
                Y_sub = np.concatenate([Y_sub, Y_pos.reshape((1,1))], axis = 0)


            while not has_converged and n_bad_fixups < max_bad_fixups:

                # solve the modified problem
                beta_hat = self.fit_qreg(X = X_sub, Y = Y_sub, q = q)
                r = Y.flatten() - X @ beta_hat

                # count wrong predictions and get their indices
                mis_L   = JL & (r > 0)
                mis_H   = JH & (r < 0)
                n_bad = np.sum(mis_L) + np.sum(mis_H)

                if n_bad == 0:
                    has_converged = True
                elif n_bad > 0.1 * M:
                    warnings.warn("Too many bad fixups. Doubling m.")
                    n_init = min(N, 2 * n_init)
                    M = int(np.ceil(m * n_init))
                    n_bad_fixups += 1
                    break
                else:
                    JL = JL & ~mis_L
                    JH = JH & ~mis_H

        return beta_hat


    def fit_qreg(self, X: np.ndarray, Y: np.ndarray, q: float) -> np.ndarray:
        """Fit a quantile regression model and return the coefficients."""
        N, k = X.shape
        X_sparse = csc_matrix(X)
        I_N = eye(N, format="csc")
        c1 = np.hstack([np.zeros(k), (1 - q) * np.ones(N), q * np.ones(N)])
        A_eq = hstack([-X_sparse, I_N, -I_N])
        b_eq = -Y
        bounds = [(None, None)] * k + [(0, None)] * (2 * N)

        res = linprog(
            c=c1,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs-ds",
        )

        if not res.success:
            raise ValueError(f"Linear programming failed: {res.message}")

        return res.x[:k].flatten()

    def _vcov_iid(self) -> np.ndarray:

        raise NotImplementedError(
            """vcov = 'iid' for quantile regression is not yet implemented. "
            For iid errors, please select vcov = 'nid'.
            """
        )

    def _vcov_nid(self) -> np.ndarray:

        "Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth."

        h = get_hall_sheather_bandwidth(q = self._quantile, N = self._N)
        beta_hat_plus = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile + h, rng = self._rng)
        yhat_plus = self._X @ beta_hat_plus
        beta_hat_minus = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile - h, rng = self._rng)
        yhat_minus = self._X @ beta_hat_minus

        s = (yhat_plus - yhat_minus) / (2 * h)
        u_hat = self._u_hat
        psi = self._quantile - (u_hat < 0)

        meat = (self._X.T * (psi**2)) @ self._X / self._N
        bread = (self._X.T / s) @ self._X / self._N

        return bread @ meat @ bread


def get_hall_sheather_bandwidth(q: float, N: int) -> float:

    """
    Compute the Hall-Sheater Bandwith.

    Parameters
    ----------
    q : float
        The quantile to compute the bandwidth for.
    N : int
        The number of observations.

    """

    x = norm.cdf(q, loc=0, scale=1)
    f = norm.ppf(x, loc=0, scale=1)

    return N**(-1/3) * norm.ppf(1 - q/2)**(2/3) * ((1.5 * f**2)/(2 * x**2 + 1))**(1/3)
