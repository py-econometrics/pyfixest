import numpy as np
from pyfixest.feols import Feols, _drop_multicollinear_variables


class Feiv(Feols):

    """
    A class to estimate a single model with instrumental variables.

    Inherits from Feols. Overwrites the `get_fit` method.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        weights: np.ndarray,
        coefnames_x: list,
        coefnames_z: list,
        collin_tol: float,
    ) -> None:
        """
        Initialize the Feiv class.

        Parameters
        ----------
        Y : np.ndarray
            Dependent variable. Two-dimensional np.ndarray.
        X : np.ndarray
            Independent variables. Two-dimensional np.ndarray.
        Z : np.ndarray
            Instruments. Two-dimensional np.ndarray.
        weights : np.ndarray
            Weights. One-dimensional np.ndarray.
        coefnames_x : list
            Names of the coefficients of X.
        coefnames_z : list
            Names of the coefficients of Z.
        collin_tol : float
            Tolerance for collinearity check.

        Returns
        -------
        None
        """

        super().__init__(
            Y=Y, X=X, weights=weights, coefnames=coefnames_x, collin_tol=collin_tol
        )

        # import pdb; pdb.set_trace()

        # check if Z is two dimensional array
        if len(Z.shape) != 2:
            raise ValueError("Z must be a two-dimensional array")

        # handle multicollinearity in Z
        (
            self._Z,
            self._coefnames_z,
            self._collin_vars_z,
            self._collin_index_z,
        ) = _drop_multicollinear_variables(Z, coefnames_z, self._collin_tol)

        self._is_iv = True

        self._support_crv3_inference = False
        self._support_iid_inference = True

    def get_fit(self) -> None:
        """
        IV estimation for a single model, via 2SLS.

        Returns
        -------
        None

        Attributes
        ----------
        beta_hat : np.ndarray
            The estimated regression coefficients.
        Y_hat : np.ndarray
            The predicted values of the regression model.
        u_hat : np.ndarray
            The residuals of the regression model.
        """

        # import pdb; pdb.set_trace()

        _X = self._X
        _Z = self._Z
        _Y = self._Y

        self._tZX = _Z.T @ _X
        self._tXZ = _X.T @ _Z
        self._tZy = _Z.T @ _Y
        self._tZZinv = np.linalg.inv(_Z.T @ _Z)

        H = self._tXZ @ self._tZZinv
        A = H @ self._tZX
        B = H @ self._tZy

        self._beta_hat = np.linalg.solve(A, B).flatten()

        self._Y_hat_link = self._X @ self._beta_hat
        self._u_hat = self._Y.flatten() - self._Y_hat_link.flatten()

        self._scores = self._Z * self._u_hat[:, None]
        self._hessian = self._Z.transpose() @ self._Z

        D = np.linalg.inv(self._tXZ @ self._tZZinv @ self._tZX)
        self._bread = (H.T) @ D @ H
