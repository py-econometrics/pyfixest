from tkinter import W
import warnings
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numba as nb
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular
from scipy.stats import norm

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.literals import (
    QuantregMethodOptions,
    SolverOptions,
)
from pyfixest.estimation.quantreg.frisch_newton_ip import (
    frisch_newton_solver,
)
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth
from pyfixest.estimation.vcov_utils import bucket_argsort

class Quantreg(Feols):
    "Quantile regression model."

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
        solver: SolverOptions = "np.linalg.solve",
        demeaner_backend: Literal["numba", "jax"] = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
        quantile: float = 0.5,
        method: QuantregMethodOptions = "fn",
        quantile_tol: float = 1e-06,
        quantile_maxiter: Optional[int] = None,
        seed: Optional[int] = None,
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

        self._supports_wildboottest = False
        self._support_crv3_inference = False
        self._supports_cluster_causal_variance = False

        self._quantile = quantile
        self._method = f"quantreg_{method}"
        self._quantile_tol = quantile_tol
        self._quantile_maxiter = quantile_maxiter

        self._model_name = (
            FixestFormula.fml
            if self._sample_split_var is None
            else f"{FixestFormula.fml} (Sample: {self._sample_split_var} = {self._sample_split_value})"
        )
        # update with quantile name
        self._model_name = f"{self._model_name} (q = {quantile})"
        self._model_name_plot = self._model_name

        self._seed = seed

        # later set in fit method, consant for different quantiles q -> can be reused
        self._chol = None
        self._P = None

        self._method_map: dict[str, Callable[..., tuple[np.ndarray, bool, int]]] = {
            "fn": partial(
                self.fit_qreg_fn,
                q=self._quantile,
                tol=self._quantile_tol,
                maxiter=self._quantile_maxiter,
                beta_init=None,
            ),
            "pfn": partial(
                self.fit_qreg_pfn,
                q=self._quantile,
                rng=np.random.default_rng(seed),
                tol=self._quantile_tol,
                maxiter=self._quantile_maxiter,
                beta_init=None,
            ),
        }

        try:
            self._fit = self._method_map[method]
        except KeyError as exc:
            valid = ", ".join(self._method_map)
            raise ValueError(f"`method` must be one of {{{valid}}}") from exc

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

    def prepare_model_matrix(self):
        "Prepare model inputs for estimation."
        super().prepare_model_matrix()

        if self._fe is not None:
            raise NotImplementedError("Fixed effects are not yet supported for Quantile Regression.")

    def get_fit(self) -> None:
        """Fit a quantile regression model using the interior point method."""
        res = self._fit(X=self._X, Y=self._Y)

        self._beta_hat = res[0]
        self._has_converged = res[1]
        self._it = res[2]
        self._x_final = res[3]
        self._s_final = res[4]
        self._z_final = res[5]
        self._w_final = res[6]
        self._y_final = res[7]

        self._u_hat = self._Y.flatten() - self._X @ self._beta_hat
        self._hessian = self._X.T @ self._X
        self._bread = np.linalg.inv(self._hessian)

    def fit_qreg_fn(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        q: float,
        tol: Optional[float] = None,
        maxiter: Optional[int] = None,
        beta_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit a quantile regression model using the Frisch-Newton Interior Point Solver."""
        N, k = X.shape
        if tol is None:
            tol = 1e-06
        if maxiter is None:
            maxiter = X.shape[0]

        if self._chol is None or self._P is None:
            self._chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
            self._P = solve_triangular(self._chol, X.T, lower=True, check_finite=False)

        fn_res = frisch_newton_solver(
            A=X.T,
            b=(1 - q) * X.T @ np.ones(N),
            c=-Y,
            u=np.ones(N),
            q=q,
            tol=tol,
            max_iter=maxiter,
            backoff=0.9995,
            beta_init=beta_init,
            chol=self._chol,
            P=self._P,
        )

        has_converged = fn_res[1]
        it = fn_res[2]

        if not has_converged:
            warnings.warn(
                f"The Frisch-Newton Interior Point solver has not converged after {it} iterations."
            )

        return fn_res

    def fit_qreg_pfn(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        q: float,
        rng: np.random.Generator,
        tol: Optional[float] = None,
        maxiter: Optional[int] = None,
        beta_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit a quantile regression model using preprocessing and the Frisch-Newton Interior Point Solver."""
        N, k = self._X.shape
        has_converged = False
        compute_beta_init = beta_init is None
        max_bad_fixups = 3
        n_bad_fixups = 0

        if tol is None:
            tol = 1e-06
        if maxiter is None:
            maxiter = X.shape[0]

        # m constant should be set by user
        m = 0.8
        n_init = int(np.ceil((k * N) ** (2 / 3)))
        M = int(np.ceil(m * n_init))

        while not has_converged:
            if compute_beta_init:
                # get initial sample
                idx_init = rng.choice(N, size=n_init, replace=False)
                beta_hat_init, _ = self.fit_qreg_fn(
                    X[idx_init, :], Y[idx_init], q=q, tol=tol, maxiter=maxiter
                )

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
            X_sub = X[keep, :]
            Y_sub = Y[keep, :]

            if np.any(JL):
                X_neg = np.sum(X[JL, :], axis=0)
                Y_neg = np.sum(Y[JL])
                X_sub = np.concatenate([X_sub, X_neg.reshape((1, self._k))], axis=0)
                Y_sub = np.concatenate([Y_sub, Y_neg.reshape((1, 1))], axis=0)
            if np.any(JH):
                X_pos = np.sum(X[JH, :], axis=0)
                Y_pos = np.sum(Y[JH])
                X_sub = np.concatenate([X_sub, X_pos.reshape(1, self._k)], axis=0)
                Y_sub = np.concatenate([Y_sub, Y_pos.reshape((1, 1))], axis=0)

            while not has_converged and n_bad_fixups < max_bad_fixups:
                # solve the modified problem
                beta_hat, _ = self.fit_qreg_fn(
                    X=X_sub, Y=Y_sub, q=q, tol=tol, maxiter=maxiter
                )
                r = Y.flatten() - X @ beta_hat

                # count wrong predictions and get their indices
                mis_L = JL & (r > 0)
                mis_H = JH & (r < 0)
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

        return beta_hat, has_converged

    def _vcov_nid(self) -> np.ndarray:
            """
            Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth
            as developed in Hendricks and Koenker (1991).
            Note: the estimator is actually heteroskedasticity robust, despite its name.
            'nid' stands for 'non-iid'.
            For details, see page 80 in Koenker's "Quantile Regression" (2005) book.
            """

            q = self._quantile
            N = self._N
            X = self._X

            h = get_hall_sheather_bandwidth(q=q, N=N)

            beta_hat_plus = self.fit_qreg_fn(
                X=self._X, Y=self._Y, q=self._quantile + h
            )[0]
            beta_hat_minus = self.fit_qreg_fn(
                X=self._X, Y=self._Y, q=self._quantile - h
            )[0]

            # eps: small tolerance parameter to avoid division by zero
            # when di = 0; set to sqrt of machine epsilon in quantreg
            eps = np.finfo(float).eps ** 0.5
            # equation (2)
            di = X @ (beta_hat_plus - beta_hat_minus)
            # equation (3)
            Fplus = np.maximum(0, (2 * h) / (di - eps))

            # general Huber structure, see page 74 in Koenker.
            J = X.T @ X
            XFplus = X * np.sqrt(Fplus[:, np.newaxis])
            H = XFplus.T @ XFplus
            Hinv = np.linalg.inv(H)

            return q * (1-q) * Hinv @ J @ Hinv

    def _vcov_hetero(self) -> np.ndarray:

        return self._vcov_nid()

    def _vcov_iid(self) -> np.ndarray:
        raise NotImplementedError(
            "The 'iid' vcov is not implemented for quantile regression. "
            "Please use 'nid' instead, which is heteroskedasticity robust."
        )

    def _vcov_crv1(self, clustid: np.ndarray, cluster_col: np.ndarray):

        """
        Implement cluster robust variance estimator for quantile regression following
        Parente and Santos Silva, 2016.
        """

        if len(self._clustervar) > 1:
            raise NotImplementedError(
                "Multiway clustering is not (yet) supported for quantile regression."
            )

        X = self._X
        N, _ = X.shape
        q = self._quantile
        u_hat = self._u_hat

        # kappa: median absolute deviation of the a-th quantile regression residuals
        kappa = np.median(np.abs(u_hat - np.median(u_hat)))
        h_G     = get_hall_sheather_bandwidth(q=q, N=N)
        delta = kappa*( norm.ppf(q + h_G) - norm.ppf(q - h_G) )

        vcov = _crv1_vcov_loop(X = X, clustid = clustid, cluster_col = cluster_col, q = q, u_hat = u_hat, delta = delta)

        return vcov


@nb.njit(parallel = False)
def _crv1_vcov_loop(X: np.ndarray, clustid: np.ndarray, cluster_col: np.ndarray, q: float, u_hat: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:

        N, k = X.shape

        meat = np.zeros((k,k))
        bread = np.zeros((k,k))
        tmp  = np.empty((k,k))
        score_g = np.zeros(k)
        g_indices, g_locs = bucket_argsort(cluster_col)
        G = len(g_locs)

        for i in range(clustid.size):

            g = clustid[i]
            start = g_locs[g]
            end = g_locs[g + 1]
            g_index = g_indices[start:end]
            Xg  = X[g_index]
            WXg = np.zeros_like(Xg)
            ug  = u_hat[g_index]
            Ng = Xg.shape[0]

            psi_g = q - 1.0 * (ug < 0)

            np.dot(Xg.T, psi_g, out=score_g)
            np.outer(score_g, score_g, out=tmp)
            meat += tmp

            mask= (np.abs(ug) <= delta) * 1.0
            for n in range(Ng):
                WXg[n] = Xg[n] * mask[n]

            np.dot(WXg.T, WXg, out=tmp)
            bread  += tmp

        meat  /= G
        bread /= (2 * delta * G)

        vcov     = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread) / N


        return vcov



