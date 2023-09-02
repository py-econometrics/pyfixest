import pyhdfe
import numpy as np
import pandas as pd
import warnings


from typing import Union, List, Dict
from formulaic import model_matrix
from pyfixest.feols import Feols, _check_vcov_input, _deparse_vcov_input
from pyfixest.ssc_utils import get_ssc
from pyfixest.exceptions import (
    VcovTypeNotSupportedError,
    NanInClusterVarError,
    NonConvergenceError,
    NotImplementedError,
)


class Fepois(Feols):

    """
    Class to estimate Poisson Regressions. Inherits from Feols. The following methods are overwritten: `get_fit()`.
    """

    def __init__(self, Y, X, fe, weights, drop_singletons, maxiter=25, tol=1e-08):
        """
        Args:
            Y (np.array): dependent variable. two-dimensional np.array
            Z (np.array): independent variables. two-dimensional np.array
            fe (np.array): fixed effects. two dimensional np.array or None
            weights (np.array): weights. one dimensional np.array or None
            drop_singletons (bool): whether to drop singleton fixed effects
            maxiter (int): maximum number of iterations for the IRLS algorithm
            tol (float): tolerance level for the convergence of the IRLS algorithm
        """

        super().__init__(Y=Y, X=X, weights=weights)

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
        self.Y = _to_integer(self.Y)
        # check that self.Y is a weakly positive integer
        if np.any(self.Y < 0):
            raise ValueError(
                "The dependent variable must be a weakly positive integer."
            )

        self.separation_na = None
        self.n_separation_na = None
        self._check_for_separation()

        self._support_crv3_inference = False
        self._support_iid_inference = False

        # attributes that are updated outside of the class (not optimal)
        self.n_separation_na = None
        self.na_index = None

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

        _Y = self.Y
        _X = self.X
        _fe = self.fe
        _N = self.N
        _drop_singletons = self._drop_singletons
        _convergence = self.convergence
        _maxiter = self.maxiter
        _iwls_maxiter = 25
        _tol = self.tol

        self.beta_hat = None
        self.Y_hat_response = None
        self.Y_hat_link = None
        self.weights = None
        self.X = None
        self.Z = None
        self.Y = None
        self.u_hat = None
        self.deviance = None
        self.tZX = None
        self.tZXinv = None
        self.Xbeta = None
        self.scores = None
        self.hessian = None
        # only needed for IV
        self.tXZ = None
        self.tZZinv = None


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
            # crit = np.sqrt(((deviance - last)** 2) / (last ** 2))
            # crit = np.sqrt(((deviance - last)** 2))
            last = deviance.copy()

            stop_iterating = crit < _tol

        self.beta_hat = delta_new.flatten()
        self.Y_hat_response = mu
        self.Y_hat_link = eta
        # (Y - self.Y_hat)
        # needed for the calculation of the vcov

        # updat for inference
        self.weights = mu_old
        # if only one dim
        if self.weights.ndim == 1:
            self.weights = self.weights.reshape((self.N, 1))

        self.u_hat = resid.flatten()

        self.Y = Z_resid
        self.X = X_resid
        self.Z = self.X
        self.deviance = deviance

        self.tZX = np.transpose(self.Z) @ self.X
        self.tZXinv = np.linalg.inv(self.tZX)
        self.Xbeta = eta

        self.scores = self.u_hat[:, None] * self.weights * X_resid
        self.hessian = XWX



    def predict(self, data: Union[None, pd.DataFrame] = None, type="link") -> np.ndarray:
        """
        Return a flat np.array with predicted values of the regression model.
        Args:
            data (Union[None, pd.DataFrame], optional): A pd.DataFrame with the data to be used for prediction.
                If None (default), uses the data used for fitting the model.
            type (str, optional): The type of prediction to be computed. Either "response" (default) or "link".
                If type="response", then the output is at the level of the response variable, i.e. it is the expected predictor E(Y|X).
                If "link", then the output is at the level of the explanatory variables, i.e. the linear predictor X @ beta.

        """

        if type not in ["response", "link"]:
            raise ValueError("type must be one of 'response' or 'link'.")

        if data is None:
            y_hat = self.Xbeta

        else:
            fml_linear, _ = self._fml.split("|")
            _, X = model_matrix(fml_linear, data)
            X = X.drop("Intercept", axis=1)

            y_hat = X @ self.beta_hat

        if type == "link":
            if self._method == "fepois":
                y_hat = np.exp(y_hat)

        return y_hat.flatten()

    def _check_for_separation(self, check="fe"):
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

            else:
                Y_help = pd.Series(np.where(self.Y.flatten() > 0, 1, 0))
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

                self.Y = np.delete(self.Y, self.separation_na, axis=0)
                self.X = np.delete(self.X, self.separation_na, axis=0)
                # self.Z = np.delete(self.Z, self.separation_na, axis = 0)
                self.fe = np.delete(self.fe, self.separation_na, axis=0)

                self.N = self.Y.shape[0]
                if len(self.separation_na) > 0:
                    warnings.warn(
                        str(len(self.separation_na))
                        + " observations removed because of only 0 outcomes."
                    )

        else:
            raise NotImplementedError(
                "Separation check via " + check + " is not implemented yet."
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