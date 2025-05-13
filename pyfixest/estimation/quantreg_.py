from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack
from scipy.stats import gaussian_kde

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula

from warnings import warn

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

    def get_fit(self, beta_init: Optional[np.ndarray] = None) -> None:

        rng = np.random.default_rng(2)
        q = self._quantile
        N, k = self._X.shape
        X = self._X
        Y = self._Y
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
                beta_hat_init = self.fit(X[idx_init, :], Y[idx_init])

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
                beta_hat = self.fit(X = X_sub, Y = Y_sub)
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

            self._beta_hat = beta_hat
            self._u_hat = Y - X @ beta_hat
            self._hessian = X.T @ X
            self._bread = np.linalg.inv(self._hessian)

            self._beta_hat

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit a quantile regression model and return the coefficients."""
        q = self._quantile
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

        import pdb; pdb.set_trace()
        u_hat = self._u_hat
        N = self._N
        q = self._quantile

        kde = gaussian_kde(u_hat)
        f0 = kde.evaluate(0)[0]

        k = q * (1 - q) / (N * f0)

        return k * self._bread

    def _vcov_hetero(self) -> np.ndarray:
        u_hat = self._u_hat
        X = self._X
        N = self._N
        q = self._quantile
        bread = self._bread

        psi = q - (u_hat < 0)

        meat = X.T * psi[:, None] ** 2 @ X / N

        return bread @ meat @ bread

    def _vcov_crv1(self) -> np.ndarray:
        raise NotImplementedError(
            "CRV1 is not yet implemented for quantile regression."
        )

    def _vcov_crv3(self) -> np.ndarray:
        raise NotImplementedError(
            "CRV3 is not yet implemented for quantile regression."
        )


def hall_sheather_bandwidth(tau, n, alpha=0.05):
    """
    Compute the Hall and Sheather (1988) bandwidth for quantile tau.

    Parameters:
    -----------
    tau : float
        Quantile level (0 < tau < 1).
    n : int
        Sample size.
    alpha : float, optional
        Significance level for interval (default 0.05).

    Returns:
    --------
    h : float
        Bandwidth for kernel estimation.
    """
    x0 = norm.ppf(tau)
    f0 = norm.pdf(x0)
    z = norm.ppf(1 - alpha/2)
    return n**(-1/3) * z**(2/3) * ((1.5 * f0**2) / (2 * x0**2 + 1))**(1/3)

def jacobian_powell(X, Y, beta, tau, kernel='epanechnikov'):
    """
    Estimate the Jacobian matrix J(tau) using Powell (1991) estimator.

    J(tau) = E[f_{Y|X}(Xβ | X) X X^T] ≈ (1/(n*h)) Σ K(r_i / h) X_i X_i^T

    Parameters:
    -----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
    Y : ndarray, shape (n_samples,)
        Response vector.
    beta : ndarray, shape (n_features,)
        Estimated regression coefficients at quantile tau.
    tau : float
        Quantile level (0 < tau < 1).
    kernel : str, optional
        Kernel to use ('epanechnikov' or 'gaussian').

    Returns:
    --------
    J_hat : ndarray, shape (n_features, n_features)
        Estimated Jacobian matrix.
    h : float
        Bandwidth used for kernel estimation.
    """
    n, k = X.shape
    # Residuals
    r = Y - X.dot(beta)
    # Bandwidth selection
    h = hall_sheather_bandwidth(tau, n)
    u = r / h

    # Kernel weights
    if kernel == 'epanechnikov':
        w = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    elif kernel == 'gaussian':
        w = norm.pdf(u)
    else:
        raise ValueError("Unsupported kernel. Choose 'epanechnikov' or 'gaussian'.")

    # Weighted sum of X_i X_i^T
    WX = X * w[:, np.newaxis]  # shape (n, k)
    J_hat = (1.0 / (n * h)) * (X.T.dot(WX))

    return J_hat, h
