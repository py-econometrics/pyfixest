import numpy as np
import pandas as pd
from pyfixest.estimation.feols_ import Feols
from scipy.optimize import linprog
from typing import Optional, Union, Literal, Mapping, Any
from pyfixest.estimation.FormulaParser import FixestFormula
from scipy import sparse
from scipy.sparse import csc_matrix, eye, hstack, vstack
from scipy.stats import gaussian_kde

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
        """Fit a quantile regression model."""

        X = self._X
        Y = self._Y
        q = self._quantile

        N, k = X.shape

        # Convert X to sparse matrix
        X_sparse = csc_matrix(X)

        # Create sparse identity matrices
        I_N = eye(N, format='csc')

        # Create sparse coefficient vector
        c = np.hstack([np.zeros(k), q * np.ones(N), (1 - q) * np.ones(N)])

        # Create sparse equality constraint matrix
        A_eq = hstack([-X_sparse, I_N, -I_N])
        b_eq = -Y

        # Bounds: beta free (None), u+ >= 0, u- >= 0
        bounds = [(None, None)] * k + [(0, None)] * (2*N)

        # Solve using sparse matrices
        res = linprog(
            c = c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            #options={"maxiter": 1000, "tol": 1e-10}
        )

        if not res.success:
            raise ValueError(f"Linear programming failed: {res.message}")

        self._beta_hat = res.x[:k]
        self._u_hat = res.x[k:k+N] - res.x[k+N:]
        self._Y_hat = X @ self._beta_hat

        self._hessian = X.T @ X
        self._bread = np.linalg.inv(self._hessian)

    def _vcov_iid(self) -> np.ndarray:

        u_hat = self._u_hat
        N = self._N
        q = self._quantile

        kde = gaussian_kde(u_hat)
        f0 = kde.evaluate(0)[0]

        k = q * (1-q) / (N * f0)

        return k * self._bread

    def _vcov_hetero(self) -> np.ndarray:

        u_hat = self._u_hat
        X = self._X
        N = self._N
        q = self._quantile
        bread = self._bread

        psi = q - (u_hat < 0)

        meat = X.T * psi[:, None] **2 @ X / N

        return bread @ meat @ bread

    def _vcov_crv1(self) -> np.ndarray:

        raise NotImplementedError("CRV1 is not yet implemented for quantile regression.")

    def _vcov_crv3(self) -> np.ndarray:

        raise NotImplementedError("CRV3 is not yet implemented for quantile regression.")