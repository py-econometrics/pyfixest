from collections.abc import Mapping
from math import e
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack
from scipy.stats import norm

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula
from pyfixest.estimation.frisch_newton import lpfnc

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

        beta_hat_all = np.zeros((len(quantiles), self._X.shape[1]))

        for i, q in enumerate(quantiles):
            if i == 0:
                beta_hat_all[i,:] = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile, rng = self._rng, beta_init = None)
            else:
                beta_hat = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile, rng = self._rng, beta_init = beta_hat)
                u_hat = self._Y.flatten() - self._X @ beta_hat
                h = get_hall_sheather_bandwidth(q = q, N = self._N)
                Ku = norm.pdf(u_hat / h)
                J = (self._X.T * Ku) @ self._X / (self._N * h)
                beta_hat_all[i,:] = beta_hat_all[i,:] - np.linalg.inv(J) * np.mean(q - u_hat < 0)

        return beta_hat_all

    def get_fit(self) -> None:
        """Fit a quantile regression model using the interior point method."""

        self._beta_hat = self.fit_qreg(X = self._X, Y = self._Y, q = self._quantile)
        #self._beta_hat = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile, rng = self._rng, beta_init = None)
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

        beta_hat, has_converged = frisch_newton_solver(
            A = X.T,
            b = (1 - q) * X.T @ np.ones(N),
            c = -Y,
            u = np.ones(N),
            q = q,
            tol = 1e-6,
            max_iter = 50,
            backoff = 0.9995
        )

        return beta_hat.flatten()

    def _vcov_iid(self) -> np.ndarray:

        raise NotImplementedError(
            """vcov = 'iid' for quantile regression is not yet implemented. "
            For iid errors, please select vcov = 'nid'.
            """
        )

    def _vcov_nid(self) -> np.ndarray:

        "Compute nonparametric IID (NID) vcov matrix using the Hall-Sheather bandwidth."

        h = get_hall_sheather_bandwidth(q = self._quantile, N = self._N)
        beta_hat_plus = self.fit_qreg(X = self._X, Y = self._Y, q = self._quantile + h)
        #beta_hat_plus = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile + h, rng = self._rng)
        yhat_plus = self._X @ beta_hat_plus
        beta_hat_minus = self.fit_qreg(X = self._X, Y = self._Y, q = self._quantile - h)
        #beta_hat_minus = self.fit_qreg_ip(X = self._X, Y = self._Y, q = self._quantile - h, rng = self._rng)
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



def frisch_newton_solver(A: np.ndarray, b: np.ndarray, c: np.ndarray, u: np.ndarray, q: float = 0.5,
          tol: float = 1e-6, max_iter: int = 1000, backoff: float = 0.9995) -> tuple[np.ndarray, bool]:
    """
    Solve
        min_x  c^T x
        s.t.   A x = b,
               0 <= x <= u
    via the Frisch–Newton predictor–corrector IPM as described in
    Koenker and Ng ("A FRISCH-NEWTON ALGORITHM FOR SPARSE QUANTILE
    REGRESSION").
    """

    # Initialize
    m, n = A.shape
    c = c.flatten()
    b = b.flatten()
    u = u.flatten()

    m, n = A.shape
    x = (1 - 0.5) * np.ones(n)
    s = u - x
    d = c.copy()
    d_plus  = np.maximum(d,  0)
    d_minus = np.maximum(-d, 0)
    U      = x @ d_plus + s @ d_minus

    mu0   = max(1, U/n)
    alpha = (n*mu0 - U) / (np.sum(1/x) + np.sum(1/s))
    z = d_plus  + alpha / x
    w = d_minus + alpha / s

    y  = np.linalg.solve(A @ A.T, A @ (c - z + w) )

    e = np.ones(n)

    print(f"z: {z[:5]}")
    print(f"w: {w[:5]}")
    print(f"x: {x[:5]}")
    print(f"s: {s[:5]}")
    print(f"d: {d[:5]}")

    # 6) Quick sanity checks (optional)
    if True:
        #import pdb; pdb.set_trace()
        assert np.all(z > 0)
        assert np.all(x > 0)
        assert np.all(s > 0)
        assert np.all(w > 0)
        assert np.all(u > x)

    def duality_gap(x, z, s, w):
        return (x.T @ z + s.T @ w)

    mu_curr = duality_gap(x = x, z = z, s = s, w = w)

    def bound(v, dv):
        """Componentwise bound used by quantreg: returns +∞ where dv≥0."""
        out = np.full_like(v, np.inf)
        mask = dv < 0
        out[mask] = -v[mask] / dv[mask]
        return out

    def step_length(a: tuple, b: tuple, backoff: float = 0.9995):

        a_max = np.min(np.concatenate((bound(a[0], a[1]),
                                    bound(b[0], b[1]))))
        return min(backoff * a_max, 1.0)


    has_converged = False

    # Main loop
    for it in range(max_iter):
        if mu_curr < tol:
            has_converged = True
            break

        # Residuals: equ. (7)
        r1 = (A.T @ y).flatten() + z - w - c
        r2 = A @ x - b

        norm_r1 = np.linalg.norm(r1)
        norm_r2 = np.linalg.norm(r2)
        #print(f"it={it:3d}  mu={mu_curr:.3e}   ||r1||={norm_r1:.3e}  ||r2||={norm_r2:.3e}")

        r1_tilde = c - A.T @ y
        r2_tilde = b - A @ x


        # Affine-Scaling Predictor Direction (eq. (8))
        Q = z / x + w / s
        Qinv = 1.0 / Q  # diag
        M = A @ (Qinv[:, None] * A.T)

        dy_aff = np.linalg.inv(M) @ (r2_tilde + A @ (Qinv * r1_tilde))
        dx_aff = Qinv * ( A.T @ dy_aff - r1_tilde)
        ds_aff = -dx_aff
        dz_aff = -z - (z / x) * dx_aff
        dw_aff = -w - (w / s) * ds_aff

        # Step lengths (eq. (9))
        alpha_p_aff = step_length(a = (x, dx_aff), b = (s, ds_aff))
        alpha_d_aff = step_length(a = (z, dz_aff), b = (w, dw_aff))

        # 6) Compute mu_new  and centering σ  (eq (10))

        x_pred = x      + alpha_p_aff*dx_aff
        s_pred = s      + alpha_p_aff*ds_aff
        y_pred = y      + alpha_d_aff*dy_aff
        z_pred = z      + alpha_d_aff*dz_aff
        w_pred = w      + alpha_d_aff*dw_aff

        mu_aff = duality_gap(
            x = x_pred,
            z = z_pred,
            s = s_pred,
            w = w_pred
        )

        ratio  = mu_aff / mu_curr
        sigma  = ratio ** 2          # 0 ≤ σ ≤ 1

        mu_targ = sigma * mu_curr / n

        # corrector direction
        r1_hat = (mu_targ*(1/s - 1/x)
                + (dx_aff * dz_aff) / x
                - (ds_aff * dw_aff) / s)

        dy_cor = np.linalg.inv(M) @ (A @ (Qinv * r1_hat))
        dx_cor = Qinv * (A.T @ dy_cor - r1_hat)
        ds_cor = -dx_cor
        dz_cor = - (z / x) * dx_cor + (mu_targ - dx_aff * dz_aff) / x
        dw_cor = - (w / s) * ds_cor + (mu_targ - ds_aff * dw_aff) / s

        # 9) Final step lengths (corrector) — eq (12)
        alpha_p_cor = step_length(a = (x, dx_cor), b = (s, ds_cor))
        alpha_d_cor = step_length(a = (z, dz_cor), b = (w, dw_cor))

        # 10) Update all variables / corrector step
        # Update
        # corrector (starting from the predictor point)
        x = x_pred     + alpha_p_cor*dx_cor
        s = s_pred     + alpha_p_cor*ds_cor
        y = y_pred     + alpha_d_cor*dy_cor
        z = z_pred     + alpha_d_cor*dz_cor
        w = w_pred     + alpha_d_cor*dw_cor

        # update
        mu_curr = duality_gap(x = x, z = z, s = s, w = w)
        #print("mu_curr: ", mu_curr)

        # sizes of the Newton directions  (ℓ∞ makes overflow obvious)
        inf_dx   = np.max(np.abs(dx_aff))
        inf_dz   = np.max(np.abs(dz_aff))
        inf_dw   = np.max(np.abs(dw_aff))

        cond_M   = np.linalg.cond(M)                    # κ₂ of normal matrix
        print(f"""
        it {it:3d}
        μ_k        = {mu_curr:16.9e}
        μ_aff      = {mu_aff:16.9e}
        σ          = {sigma:10.4f}
        μ_target   = {mu_targ:16.9e}
        ||r1||₂    = {np.linalg.norm(r1):9.2e}
        ||r2||₂    = {np.linalg.norm(r2):9.2e}
        ||dx_aff||∞= {inf_dx:9.2e}
        ||dz_aff||∞= {inf_dz:9.2e}
        ||dw_aff||∞= {inf_dw:9.2e}
        α_p_aff    = {alpha_p_aff:6.4f}
        α_d_aff    = {alpha_d_aff:6.4f}
        cond(M)    = {cond_M:9.2e}
        min(x)     = {x.min():9.2e}   max(x) = {x.max():9.2e}
        min(s)     = {s.min():9.2e}   max(s) = {s.max():9.2e}
        min(z)     = {z.min():9.2e}   max(z) = {z.max():9.2e}
        min(w)     = {w.min():9.2e}   max(w) = {w.max():9.2e}
        """)
    # Recover β,u,v if this was a quantile‐LP on [β;u;v].
    return y, has_converged