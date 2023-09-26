import numpy as np
from scipy.linalg import solve as scipy_solve
from pyfixest.feols import Feols

class Feiv(Feols):

    """
    # Feiv
    A class to estimate a single model with instrumental variables.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        weights: np.ndarray,
        coefnames: list,
        collin_tol: float,
        solver:str,
    ) -> None:
        """
        Args:
            Y (np.array): dependent variable. two-dimensional np.array
            X (np.array): independent variables. two-dimensional np.array
            Z (np.array): instruments. two-dimensional np.array
            weights (np.array): weights. one-dimensional np.array
            coefnames (list): names of the coefficients
            collin_tol (float): tolerance for collinearity check
            solver (str): solver to use for the least squares problem. Either "np.linalg.solve" or "sp.linalg.solve".
        Returns:
            None
        """

        super().__init__(
            Y=Y, X=X, weights=weights, coefnames=coefnames, collin_tol=collin_tol, solver=solver
        )

        # check if Z is two dimensional array
        if len(Z.shape) != 2:
            raise ValueError("Z must be a two-dimensional array")

        # import pdb; pdb.set_trace()

        if self._collin_index is not None:
            self._Z = Z[:, ~self._collin_index]
        else:
            self._Z = Z

        self._is_iv = True

        self._support_crv3_inference = False
        self._support_iid_inference = True

    def get_fit(self) -> None:
        """
        IV  estimation for a single model, via 2SLS.
        Returns:
            None
        Attributes:
            beta_hat (np.ndarray): The estimated regression coefficients.
            Y_hat (np.ndarray): The predicted values of the regression model.
            u_hat (np.ndarray): The residuals of the regression model.
        """

        _X = self._X
        _Z = self._Z
        _Y = self._Y
        _solver = self._solver

        self._tZX = _Z.T @ _X
        self._tXZ = _X.T @ _Z
        self._tZy = _Z.T @ _Y
        self._tZZinv = np.linalg.inv(_Z.T @ _Z)

        H = self._tXZ @ self._tZZinv
        A = H @ self._tZX
        B = H @ self._tZy

        if self._solver == "np.linalg.solve":
            self._beta_hat = np.linalg.solve(A, B).flatten()
        elif self._solver == "sp.linalg.solve":
            self._beta_hat = scipy_solve(A, B).flatten()
        else:
            raise ValueError("Solver not supported")


        self._Y_hat_link = self._X @ self._beta_hat
        self._u_hat = self._Y.flatten() - self._Y_hat_link.flatten()

        self._scores = self._Z * self._u_hat[:, None]
        self._hessian = self._Z.transpose() @ self._Z

        D = np.linalg.inv(self._tXZ @ self._tZZinv @ self._tZX)
        self._bread = (H.T) @ D @ H
