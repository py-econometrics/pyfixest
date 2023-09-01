import numpy as np
from pyfixest.feols import Feols
from typing import Optional

class Feiv(Feols):

    def __init__(
        self, Y: np.ndarray, X: np.ndarray, Z: np.ndarray, weights: np.ndarray) -> None:
        """
        Args:
            Y (np.array): dependent variable. two-dimensional np.array
            X (np.array): independent variables. two-dimensional np.array
            Z (np.array): instruments. two-dimensional np.array
            weights (np.array): weights. one-dimensional np.array
        Returns:
            None
        """

        super().__init__(Y=Y, X=X, weights=weights)

        # check if Z is two dimensional array
        if len(Z.shape) != 2:
            raise ValueError("Z must be a two-dimensional array")

        self.Z = Z
        self._is_iv = True

        self._support_crv3_inference = False
        self._support_iid_inference = True

    def get_fit(self) -> None:
        """
        Regression estimation for a single model, via ordinary least squares (OLS).
        Returns:
            None
        Attributes:
            beta_hat (np.ndarray): The estimated regression coefficients.
            Y_hat (np.ndarray): The predicted values of the regression model.
            u_hat (np.ndarray): The residuals of the regression model.
        """

        #import pdb
        #pdb.set_trace()

        self.tZX = np.transpose(self.Z) @ self.X
        self.tXZ = np.transpose(self.tZX)

        self.tZy = np.transpose(self.Z) @ self.Y

        #self.tXZ = np.transpose(self.X) @ self.Z
        self.tZZinv = np.linalg.inv(np.transpose(self.Z) @ self.Z)

        H = self.tXZ @ self.tZZinv
        A = H @ self.tZX
        B = H @ self.tZy

        self.beta_hat = np.linalg.solve(A, B).flatten()

        self.Y_hat_link = self.X @ self.beta_hat
        self.u_hat = self.Y.flatten() - self.Y_hat_link.flatten()

        self.scores = self.Z * self.u_hat[:, None]
        self.hessian = self.Z.transpose() @ self.Z

        D =  np.linalg.inv(self.tXZ @ self.tZZinv @ self.tZX)
        self.bread = (H.T) @ D @ H












