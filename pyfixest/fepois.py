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

        super().__init__(Y=Y, X=X, Z=X, weights=weights)

        # input checks
        _fepois_input_checks(fe, drop_singletons, tol, maxiter)

        self.fe = fe
        self.maxiter = maxiter
        self.tol = tol
        self._drop_singletons = drop_singletons
        self._method = "fepois"

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
        self._check_for_separation()


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

        Y = self.Y
        X = self.X
        fe = self.fe

        # print(Y.shape, X.shape, fe.shape)

        def _update_w(Xbeta):
            """
            Implements the updating function for the weights in the IRLS algorithm.
            Uses the softmax for numerical stability.
            Args:
                Xbeta (np.array): Xbeta from the last iteration of the IRLS algorithm
            Returns:
                w (np.array): updated weights
            """
            expXbeta = np.exp(Xbeta - np.max(Xbeta))
            return expXbeta / np.sum(expXbeta)

        def _update_Z(Y, Xbeta):
            return (Y - np.exp(Xbeta)) / np.exp(Xbeta) + Xbeta

        # starting values: http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode27.html
        # reference:  McCullagh, P. & Nelder, J. A. ( 1989). Generalized Linear Models,
        #  Vol. 37 of Monographs on Statistics and Applied Probability, 2 edn, Chapman and Hall, London.
        Xbeta = np.log(np.repeat(np.mean(Y, axis=0), self.N).reshape((self.N, 1)))
        w = _update_w(Xbeta)
        Z = _update_Z(Y=Y, Xbeta=Xbeta)

        delta = np.ones((X.shape[1]))

        X2 = X
        Z2 = Z

        algorithm = pyhdfe.create(
            ids=fe, residualize_method="map", drop_singletons=self._drop_singletons
        )

        algorithm = pyhdfe.create(
            ids=fe, residualize_method="map", drop_singletons=self._drop_singletons
        )

        for x in range(self.maxiter):
            # Step 1: weighted demeaning
            ZX = np.concatenate([Z2, X2], axis=1)

            if fe is not None:
                if (
                    self._drop_singletons == True
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
                ZX_d = algorithm.residualize(ZX, w)
            else:
                ZX_d = ZX

            Z_d = ZX_d[:, 0].reshape((self.N, 1))
            X_d = ZX_d[:, 1:]

            WX_d = np.sqrt(w) * X_d
            WZ_d = np.sqrt(w) * Z_d

            XdWXd = WX_d.transpose() @ WX_d
            XdWZd = WX_d.transpose() @ WZ_d

            delta_new = np.linalg.solve(XdWXd, XdWZd)
            e_new = Z_d - X_d @ delta_new

            Xbeta_new = Z - e_new
            w_u = _update_w(Xbeta_new)
            Z_u = _update_Z(Y=Y, Xbeta=Xbeta_new)

            stop_iterating = np.sqrt(np.sum((delta - delta_new) ** 2)) < self.tol

            # update
            delta = delta_new
            Z2 = Z_d + Z_u - Z
            X2 = X_d
            Z = Z_u
            w_old = w.copy()
            w = w_u
            Xbeta = Xbeta_new

            if stop_iterating:
                break
            if x == self.maxiter:
                raise NonConvergenceError(
                    "The IRLS algorithm did not converge. Try to increase the maximum number of iterations."
                )

        self.beta_hat = delta.flatten()
        self.Y_hat_response = np.exp(Xbeta)
        self.Y_hat_link = Xbeta
        self.u_hat = e_new
        # (Y - self.Y_hat)
        # needed for the calculation of the vcov

        # updat for inference
        self.weights = w_old
        # if only one dim
        if self.weights.ndim == 1:
            self.weights = self.weights.reshape((self.N, 1))

        self.X = X_d
        self.Z = X_d
        self.ZX = ZX

        self.tZX = np.transpose(self.Z) @ self.X
        self.tZXinv = np.linalg.inv(self.tZX)
        self.Xbeta = Xbeta

    def get_vcov(self, vcov: Union[str, Dict[str, str], List[str]]) -> None:
        """
        Compute covariance matrices for an estimated regression model.

        Parameters
        ----------
        vcov : Union[str, Dict[str, str], List[str]]
            A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference
            or {"CRV3":"clustervar"} for CRV3 inference.
            Note that CRV3 inference is currently not supported with arbitrary fixed effects and IV estimation.

        Raises
        ------
        AssertionError
            If vcov is not a dict, string, or list.
        AssertionError
            If vcov is a dict and the key is not "CRV1" or "CRV3".
        AssertionError
            If vcov is a dict and the value is not a string.
        AssertionError
            If vcov is a dict and the value is not a column in the data.
        AssertionError
            CRV3 currently not supported with arbitrary fixed effects
        AssertionError
            If vcov is a list and it does not contain strings.
        AssertionError
            If vcov is a list and it does not contain columns in the data.
        AssertionError
            If vcov is a string and it is not one of "iid", "hetero", "HC1", "HC2", or "HC3".


        Returns
        -------
        None

        """

        _check_vcov_input(vcov, self._data)

        (
            self.vcov_type,
            self.vcov_type_detail,
            self.is_clustered,
            self.clustervar,
        ) = _deparse_vcov_input(vcov, self._has_fixef, self._is_iv)

        # compute vcov
        WX = self.weights * self.X
        bread = np.linalg.inv(self.X.transpose() @ WX)

        if self.vcov_type == "iid":
            raise NotImplementedError(
                "iid inference is not supported for non-linear models."
            )

            self.ssc = get_ssc(
                ssc_dict=self._ssc_dict,
                N=self.N,
                k=self.k,
                G=1,
                vcov_sign=1,
                vcov_type="iid",
            )

            # only relevant factor for iid in ssc: fixef.K
            sigma2 = np.sum(self.weights * (self.u_hat**2)) / (self.N - 1)
            self.vcov = self.ssc * bread * sigma2

        elif self.vcov_type == "hetero":
            if self.vcov_type_detail in ["HC2", "HC3"]:
                raise NotImplementedError(
                    "HC2 and HC3 are not implemented for non-linear models."
                )

            self.ssc = get_ssc(
                ssc_dict=self._ssc_dict,
                N=self.N,
                k=self.k,
                G=1,
                vcov_sign=1,
                vcov_type="hetero",
            )

            Sigma = self.u_hat**2
            meat = WX.transpose() @ (Sigma * WX)

            self.vcov = self.ssc * bread @ meat @ bread

        elif self.vcov_type == "CRV":
            cluster_df = self._data[self.clustervar]
            # if there are missings - delete them!

            if cluster_df.dtype != "category":
                cluster_df = pd.Categorical(cluster_df)

            if cluster_df.isna().any():
                raise NanInClusterVarError(
                    "CRV inference not supported with missing values in the cluster variable."
                    "Please drop missing values before running the regression."
                )

            _, clustid = pd.factorize(cluster_df)

            self.G = len(clustid)

            self.ssc = get_ssc(
                ssc_dict=self._ssc_dict,
                N=self.N,
                k=self.k,
                G=self.G,
                vcov_sign=1,
                vcov_type="CRV",
            )

            if self.vcov_type_detail == "CRV1":
                k = self.X.shape[1]
                meat = np.zeros((k, k))

                for g in range(self.G):
                    WX_g = WX[np.where(cluster_df == g)]
                    u_g = self.u_hat[np.where(cluster_df == g)]
                    meat_g = WX_g.transpose() @ u_g @ u_g.transpose() @ WX_g
                    meat += meat_g

                self.vcov = self.ssc * bread @ meat @ bread

            else:
                raise NotImplementedError(
                    "CRV3 inference is not supported for non-linear models."
                )

    def predict(self, data: Union[None, pd.DataFrame] = None, type="link") -> np.array:
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
                fe_combined = fe.apply(lambda row: "-".join(row.astype(str)), axis=1)

                ctab = pd.crosstab(Y_help, fe_combined)
                null_column = ctab.xs(0)
                # fixed effect "nested" in Y == 0. cond 1: fixef combi only in nested in specific value of Y. cond 2: fixef combi only in nested in Y == 0
                sep_candidate = (np.sum(ctab > 0, axis=0).values == 1) & (
                    null_column > 0
                ).values.flatten()
                droplist = ctab.xs(0)[sep_candidate].index.tolist()

                if len(droplist) > 0:
                    self.separation_na = np.where(fe_combined.isin(droplist))[0].tolist()
                    n_separation_na = len(self.separation_na)

                    self.Y = np.delete(self.Y, self.separation_na, axis=0)
                    self.X = np.delete(self.X, self.separation_na, axis=0)
                    # self.Z = np.delete(self.Z, self.separation_na, axis = 0)
                    self.fe = np.delete(self.fe, self.separation_na, axis=0)

                    self.N = self.Y.shape[0]
                    warnings.warn(
                        str(n_separation_na)
                        + " observations removed because of only 0 outcomes."
                    )

                else:
                    self.separation_na = None

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
