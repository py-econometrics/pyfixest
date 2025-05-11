from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack
from scipy.stats import gaussian_kde

from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormula


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

    def get_fit(self) -> None:

        rng = np.random.default_rng()
        update_initial_model = True
        q = self._quantile
        N, k = self._X.shape
        X = self._X
        Y = self._Y
        update_initial_model = True
        has_converged = False

        # m constant should be set by user
        m = 0.8

        while not has_converged:

            if update_initial_model:

                # get subsample size
                n_init = int(np.ceil((k * N)**(2/3)))
                M = m * n_init

                # get initial sample
                idx_init = rng.choice(N, size=n_init, replace=False)
                X_init = X[idx_init, :]
                Y_init = Y[idx_init]

                # fit initial model
                beta_hat_init = self.fit(X_init, Y_init)
                r_init = Y_init.flatten() - X_init @ beta_hat_init
                # conservative estimate of sigma^2
                z = np.dot(r_init, r_init) / (N - k)
                rz = r_init / z

                q_lower = q - M / (2 * N)
                q_upper = q + M / (2 * N)

                rz_q_lower = np.quantile(rz, q_lower)
                rz_q_upper = np.quantile(rz, q_upper)

                JL_idx = rz < rz_q_lower
                JH_idx = rz > rz_q_upper

            # count wrong predictions and get their indices
            mispredicted_signs_L = rz[JL_idx] > 0
            mispredicted_signs_H = rz[JH_idx] < 0
            mispredicted_signs_L_idx = idx_init[JL_idx][mispredicted_signs_L]
            mispredicted_signs_H_idx = idx_init[JH_idx][mispredicted_signs_H]
            n_mispredicted_signs = np.sum(mispredicted_signs_L) + np.sum(mispredicted_signs_H)

            # get indices of estimation sample
            idx = np.ones(N, dtype=bool)
            idx[idx_init[JL_idx]] = False
            idx[idx_init[JH_idx]] = False

            # solve the modified problem
            beta_hat = self.fit(X[idx, :], Y[idx])

            if n_mispredicted_signs == 0:
                has_converged = True
            elif n_mispredicted_signs > 0.1 * M:
                # drop mispredicted signs from JL, JH
                JL_idx[mispredicted_signs_L_idx] = False
                JH_idx[mispredicted_signs_H_idx] = False
                update_initial_model = False
            else:
                M *= 2
                update_initial_model = True

        self._beta_hat = beta_hat
        self._u_hat = Y - X @ beta_hat
        self._hessian = X.T @ X
        self._bread = np.linalg.inv(self._hessian)


    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit a quantile regression model and return the coefficients."""

        q = self._quantile
        N, k = X.shape
        X_sparse = csc_matrix(X)
        I_N = eye(N, format="csc")
        c1 = np.hstack([np.zeros(k), (1-q) * np.ones(N), q * np.ones(N)])
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
