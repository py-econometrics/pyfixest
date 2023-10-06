import pyhdfe
import numpy as np
import pandas as pd
import warnings


from typing import Union, Optional, List
from formulaic import model_matrix
from pyfixest.feols import Feols
from pyfixest.exceptions import (
    NonConvergenceError,
    NotImplementedError,
)


class Fepois(Feols):

    """
    # Fepois

    Class to estimate Poisson Regressions. Inherits from Feols. The following methods are overwritten: `get_fit()`.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        fe: np.ndarray,
        weights: np.ndarray,
        coefnames: List[str],
        drop_singletons: bool,
        collin_tol: float,
        maxiter: Optional[int] = 25,
        tol: Optional[float] = 1e-08,
    ):
        """
        Args:
            Y (np.array): dependent variable. two-dimensional np.array
            Z (np.array): independent variables. two-dimensional np.array
            fe (np.array): fixed effects. two dimensional np.array or None
            weights (np.array): weights. one dimensional np.array or None
            coefnames (list): names of the coefficients in the design matrix X.
            drop_singletons (bool): whether to drop singleton fixed effects
            collin_tol (float): tolerance level for the detection of collinearity
            maxiter (int): maximum number of iterations for the IRLS algorithm
            tol (float): tolerance level for the convergence of the IRLS algorithm
        """

        super().__init__(
            Y=Y, X=X, weights=weights, coefnames=coefnames, collin_tol=collin_tol
        )

        # input checks
        _fepois_input_checks(fe, drop_singletons, tol, maxiter)

        self.fe = fe
        self.maxiter = maxiter
        self.tol = tol
        self._drop_singletons = drop_singletons
        self._method = "fepois"
        self.convergence = False

        if self.fe is not None:
            self._has_fixef = True
        else:
            self._has_fixef = False

        # check if Y is a weakly positive integer
        self._Y = _to_integer(self._Y)
        # check that self._Y is a weakly positive integer
        if np.any(self._Y < 0):
            raise ValueError(
                "The dependent variable must be a weakly positive integer."
            )

        self.separation_na = None
        self._N_separation_na = None
        self._check_for_separation()

        self._support_crv3_inference = False
        self._support_iid_inference = True

        # attributes that are updated outside of the class (not optimal)
        self._N_separation_na = None
        self._Na_index = None

        self._Y_hat_response = None
        self.deviance = None
        self._Xbeta = None

    def get_fit(self) -> None:
        """
        Fit a Poisson Regression Model via Iterated Weighted Least Squares

        Args:
            tol (float): tolerance level for the convergence of the IRLS algorithm
            maxiter (int): maximum number of iterations for the IRLS algorithm. 25 by default.
        Returns:
            None
        Attributes:
            beta_hat (np.array): estimated coefficients
            Y_hat (np.array): estimated dependent variable
            u_hat (np.array): estimated residuals
            weights (np.array): weights (from the last iteration of the IRLS algorithm)
            X (np.array): demeaned independent variables (from the last iteration of the IRLS algorithm)
            Z (np.array): demeaned independent variables (from the last iteration of the IRLS algorithm)
            Y (np.array): demeaned dependent variable (from the last iteration of the IRLS algorithm)

        """

        _Y = self._Y
        _X = self._X
        _fe = self.fe
        _N = self._N
        _drop_singletons = self._drop_singletons
        _convergence = self.convergence  # False
        _maxiter = self.maxiter
        _iwls_maxiter = 25
        _tol = self.tol

        def compute_deviance(_Y, mu):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deviance = (
                    2 * np.sum(np.where(_Y == 0, 0, _Y * np.log(_Y / mu)) - (_Y - mu))
                ).flatten()
            return deviance

        # initiate demeaning algo (if needed)
        if _fe is not None:
            algorithm = pyhdfe.create(
                ids=_fe, residualize_method="map", drop_singletons=_drop_singletons
            )
            if (
                _drop_singletons == True
                and algorithm.singletons != 0
                and algorithm.singletons is not None
            ):
                print(
                    algorithm.singletons,
                    "columns are dropped due to singleton fixed effects.",
                )
                dropped_singleton_indices = np.where(algorithm._singleton_indices)[
                    0
                ].tolist()
                na_index += dropped_singleton_indices

        accelerate = True
        # inner_tol = 1e-04
        stop_iterating = False
        crit = 1

        for i in range(_maxiter):
            if stop_iterating:
                _convergence = True
                break
            if i == _maxiter:
                raise NonConvergenceError(
                    f"The IRLS algorithm did not converge with {_iwls_maxiter} iterations. Try to increase the maximum number of iterations."
                )

            if i == 0:
                _mean = np.mean(_Y)
                mu = (_Y + _mean) / 2
                eta = np.log(mu)
                Z = eta + _Y / mu - 1
                last_Z = Z.copy()
                reg_Z = Z.copy()
                last = compute_deviance(_Y, mu)

            elif accelerate:
                last_Z = Z.copy()
                Z = eta + _Y / mu - 1
                reg_Z = Z - last_Z + Z_resid
                X = X_resid.copy()

            else:
                # update w and Z
                Z = eta + _Y / mu - 1  # eq (8)
                reg_Z = Z.copy()  # eq (9)

            # tighten HDFE tolerance - currently not possible with PyHDFE
            # if crit < 10 * inner_tol:
            #    inner_tol = inner_tol / 10

            # Step 1: weighted demeaning
            ZX = np.concatenate([reg_Z, _X], axis=1)

            if _fe is not None:
                ZX_resid = algorithm.residualize(ZX, mu)
            else:
                ZX_resid = ZX

            Z_resid = ZX_resid[:, 0].reshape((_N, 1))  # z_resid
            X_resid = ZX_resid[:, 1:]  # x_resid

            # Step 2: estimate WLS
            WX = np.sqrt(mu) * X_resid
            WZ = np.sqrt(mu) * Z_resid

            XWX = WX.transpose() @ WX
            XWZ = WX.transpose() @ WZ

            delta_new = np.linalg.solve(XWX, XWZ)  # eq (10), delta_new -> reg_z
            resid = Z_resid - X_resid @ delta_new

            mu_old = mu.copy()
            # more updating
            eta = Z - resid
            mu = np.exp(eta)

            # same criterion as fixest
            # https://github.com/lrberge/fixest/blob/6b852fa277b947cea0bad8630986225ddb2d6f1b/R/ESTIMATION_FUNS.R#L2746
            deviance = compute_deviance(_Y, mu)
            crit = np.abs(deviance - last) / (0.1 + np.abs(last))
            last = deviance.copy()

            stop_iterating = crit < _tol

        self._beta_hat = delta_new.flatten()
        self._Y_hat_response = mu
        self._Y_hat_link = eta
        # (Y - self._Y_hat)
        # needed for the calculation of the vcov

        # updat for inference
        self._weights = mu_old
        # if only one dim
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape((self._N, 1))

        self._u_hat = resid.flatten()

        self._Y = Z_resid
        self._X = X_resid
        self._Z = self._X
        self.deviance = deviance

        self._tZX = np.transpose(self._Z) @ self._X
        self._tZXinv = np.linalg.inv(self._tZX)
        self._Xbeta = eta

        self._scores = self._u_hat[:, None] * self._weights * X_resid
        self._hessian = XWX
        self._T = self._weights * X_resid

        if _convergence:
            self._convergence = True

    def predict(
        self, newdata: Optional[pd.DataFrame] = None, type="link"
    ) -> np.ndarray:
        """
        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values for such observations
        will be set to NaN.

        Args:
            newdata (Union[None, pd.DataFrame], optional): A pd.DataFrame with the new data, to be used for prediction.
                If None (default), uses the data used for fitting the model.
            type (str, optional): The type of prediction to be computed. Either "response" (default) or "link".
                If type="response", then the output is at the level of the response variable, i.e. it is the expected predictor E(Y|X).
                If "link", then the output is at the level of the explanatory variables, i.e. the linear predictor X @ beta.

        """

        _Xbeta = self._Xbeta
        _has_fixef = self._has_fixef

        if _has_fixef:
            raise NotImplementedError(
                "Prediction with fixed effects is not yet implemented for Poisson regression."
            )
        if newdata is not None:
            raise NotImplementedError(
                "Prediction with function argument `newdata` is not yet implemented for Poisson regression."
            )

        if type not in ["response", "link"]:
            raise ValueError("type must be one of 'response' or 'link'.")

        y_hat = super().predict(data=newdata)
        if type == "link":
            y_hat = np.exp(y_hat)

        return y_hat

    def _check_for_separation(self, check: str = "fe") -> None:
        """
        Check for separation of Poisson Regression. For details, see the pplmhdfe documentation on
        separation checks. Currently, only the "fe" check is implemented.

        Args:
            type: type of separation check. Currently, only "fe" is supported.
        Returns:
            None
        Updates the following attributes (if columns are dropped):
            Y (np.array): dependent variable
            X (np.array): independent variables
            Z (np.array): independent variables
            fe (np.array): fixed effects
            N (int): number of observations
        Creates the following attributes
        separation_na (np.array): indices of dropped observations due to separation
        """

        if check == "fe":
            if not self._has_fixef:
                pass
            elif (self._Y > 0).all():
                pass
            else:
                Y_help = pd.Series(np.where(self._Y.flatten() > 0, 1, 0))
                fe = pd.DataFrame(self.fe)

                separation_na = set()
                # loop over all elements of fe
                for x in fe.columns:
                    ctab = pd.crosstab(Y_help, fe[x])
                    null_column = ctab.xs(0)
                    # fixed effect "nested" in Y == 0. cond 1: fixef combi only in nested in specific value of Y. cond 2: fixef combi only in nested in Y == 0
                    sep_candidate = (np.sum(ctab > 0, axis=0).values == 1) & (
                        null_column > 0
                    ).values.flatten()
                    droplist = ctab.xs(0)[sep_candidate].index.tolist()

                    if len(droplist) > 0:
                        dropset = set(np.where(fe[x].isin(droplist))[0])
                        separation_na = separation_na.union(dropset)

                self.separation_na = list(separation_na)

                self._Y = np.delete(self._Y, self.separation_na, axis=0)
                self._X = np.delete(self._X, self.separation_na, axis=0)
                # self._Z = np.delete( self._Z, self.separation_na, axis = 0)
                self.fe = np.delete(self.fe, self.separation_na, axis=0)

                self._N = self._Y.shape[0]
                if len(self.separation_na) > 0:
                    warnings.warn(
                        f"{str(len(self.separation_na))} observations removed because of separation."
                    )

        else:
            raise NotImplementedError(
                f"Separation check via {check} is not implemented yet."
            )


def _fepois_input_checks(fe, drop_singletons, tol, maxiter):
    # fe must be np.array of dimension 2 or None
    if fe is not None:
        if not isinstance(fe, np.ndarray):
            raise AssertionError("fe must be a numpy array.")
        if fe.ndim != 2:
            raise AssertionError("fe must be a numpy array of dimension 2.")
    # drop singletons must be logical
    if not isinstance(drop_singletons, bool):
        raise AssertionError("drop_singletons must be logical.")
    # tol must be numeric and between 0 and 1
    if not isinstance(tol, (int, float)):
        raise AssertionError("tol must be numeric.")
    if tol <= 0 or tol >= 1:
        raise AssertionError("tol must be between 0 and 1.")
    # maxiter must be integer and greater than 0
    if not isinstance(maxiter, int):
        raise AssertionError("maxiter must be integer.")
    if maxiter <= 0:
        raise AssertionError("maxiter must be greater than 0.")


def _to_integer(x):
    if x.dtype == int:
        return x
    else:
        try:
            x = x.astype(np.int64)
            return x
        except ValueError:
            raise ValueError(
                "Conversion of the dependent variable to integer is not possible. Please do so manually."
            )
