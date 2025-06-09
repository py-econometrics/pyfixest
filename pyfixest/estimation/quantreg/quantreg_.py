import warnings
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
import numba as nb
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

        warnings.warn(
            "The Quantile Regression implementation is experimental and may change in future releases.",
            FutureWarning,
        )

        assert isinstance(quantile, float), "quantile must be a float."

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

        self._method_map: dict[
            str,
            Callable[
                ...,
                tuple[
                    np.ndarray,
                    bool,
                    int,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
            ],
        ] = {
            "fn": partial(
                self.fit_qreg_fn,
                q=self._quantile,
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
            raise NotImplementedError(
                "Fixed effects are not yet supported for Quantile Regression."
            )

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
    ) -> tuple[
        np.ndarray,
        bool,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Fit a quantile regression model using the Frisch-Newton Interior Point Solver."""
        N, k = X.shape
        if tol is None:
            tol = 1e-06
        if maxiter is None:
            maxiter = N

        if self._chol is None or self._P is None:
            self._chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
            self._P = solve_triangular(self._chol, X.T, lower=True, check_finite=False)
        if self._chol is None or self._P is None:
            raise ValueError("...")

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
            chol=cast(np.ndarray, self._chol),
            P=cast(np.ndarray, self._P),
        )

        has_converged = fn_res[1]
        it = fn_res[2]

        if not has_converged:
            warnings.warn(
                f"The Frisch-Newton Interior Point solver has not converged after {it} iterations."
            )

        return fn_res

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

        beta_hat_plus = self.fit_qreg_fn(X=self._X, Y=self._Y, q=self._quantile + h)[0]
        beta_hat_minus = self.fit_qreg_fn(X=self._X, Y=self._Y, q=self._quantile - h)[0]

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

        return q * (1 - q) * Hinv @ J @ Hinv

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
        G = len(clustid)

        # kappa: median absolute deviation of the a-th quantile regression residuals
        kappa = np.median(np.abs(u_hat - np.median(u_hat)))
        h_G = get_hall_sheather_bandwidth(q=q, N=N)
        delta = kappa * (norm.ppf(q + h_G) - norm.ppf(q - h_G))

        vcov = _crv1_vcov_loop(
            X=X, clustid=clustid, cluster_col=cluster_col, q=q, u_hat=u_hat, delta=delta
        )

        return vcov


@nb.njit(parallel=False)
def _crv1_vcov_loop(
    X: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
    q: float,
    u_hat: np.ndarray,
    delta: float,
) -> np.ndarray:
    _, k = X.shape

    A = np.zeros((k, k))
    B = np.zeros((k, k))
    g_indices, g_locs = bucket_argsort(cluster_col)
    G = clustid.size

    eps = 1e-7
    mad = np.median(np.abs(u_hat))
    meps = mad * eps

    for g in clustid:
        start = g_locs[g]
        end = g_locs[g + 1]
        g_index = g_indices[start:end]

        Xg = X[g_index, :]
        ug = u_hat[g_index]

        ng = g_index.size
        for i in range(ng):
            Xgi = Xg[i, :]
            psi_i = q - 1.0 * (ug[i] <= meps)
            for j in range(ng):
                Xgj = Xg[j, :]
                psi_j = q - 1.0 * (ug[j] <= meps)
                A += np.outer(Xgi, Xgj) * psi_i * psi_j

            mask_i = (np.abs(ug[i]) < delta) * 1.0
            B += np.outer(Xgi, Xgi) * mask_i

    # A /= G
    B /= 2 * delta

    return np.linalg.inv(B) @ A @ np.linalg.inv(B)
