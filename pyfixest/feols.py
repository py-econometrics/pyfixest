import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from importlib import import_module
from typing import Union, List, Dict
from scipy.stats import norm, t
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, hstack
from formulaic import model_matrix

from pyfixest.ssc_utils import get_ssc
from pyfixest.exceptions import VcovTypeNotSupportedError, NanInClusterVarError


class Feols:

    """
    A class for estimating regression models with high-dimensional fixed effects via
    ordinary least squares.

    Parameters
    ----------
    Y : Union[np.ndarray, pd.DataFrame]
        Dependent variable of the regression.
    X : Union[np.ndarray, pd.DataFrame]
        Independent variable of the regression.
    weights: np.ndarray
        Weights for the regression.
    Z: Union[np.ndarray, pd.DataFrame]
        Instruments of the regression.

    Attributes
    ----------
    Y : np.ndarray
        The dependent variable of the regression.
    X : np.ndarray
        The independent variable of the regression.
    Z : np.ndarray
        The instruments of the regression. If None, equal to `X`.
    N : int
        The number of observations.
    k : int
        The number of columns in X.


    Methods
    -------
    get_fit()
        Regression estimation for a single model, via ordinary least squares (OLS).
    get_vcov(vcov)
        Compute covariance matrices for an estimated model.

    Raises
    ------
    AssertionError
        If the vcov argument is not a dict, a string, or a list.

    """

    def __init__(
        self, Y: np.ndarray, X: np.ndarray, weights: np.ndarray, Z: np.ndarray = None
    ) -> None:
        self._method = "feols"
        _feols_input_checks(Y, X, Z, weights)

        self.Y = Y
        self.X = X
        if Z is not None:
            self.Z = Z
        else:
            self.Z = X
        self.weights = weights

        self.N, self.k = X.shape
        self.sumFE = None
        self.y_hat_link = None
        self.y_hat_response = None

    def get_fit(self, estimator="ols") -> None:
        """
        Regression estimation for a single model, via ordinary least squares (OLS).
        Args: estimator (str): Estimator to use. Can be one of "ols"", or "2sls".
                If "ols", then the estimator is (X'X)^{-1}X'Y.
                If "iv", then the estimator is (Z'X)^{-1}Z'Y.
                If "2sls", then the estimator is (X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y.
        Returns:
            None
        Attributes:
            beta_hat (np.ndarray): The estimated regression coefficients.
            Y_hat (np.ndarray): The predicted values of the regression model.
            u_hat (np.ndarray): The residuals of the regression model.


        """

        self.tZX = np.transpose(self.Z) @ self.X
        self.tZy = np.transpose(self.Z) @ self.Y

        if estimator == "ols":
            self.tZXinv = np.linalg.inv(self.tZX)
            self.beta_hat = np.linalg.solve(self.tZX, self.tZy).flatten()
            # self.beta_hat = (self.tZXinv @ self.tZy).flatten()

        elif estimator == "2sls":
            self.tXZ = np.transpose(self.X) @ self.Z
            self.tZZinv = np.linalg.inv(np.transpose(self.Z) @ self.Z)
            self.beta_hat = np.linalg.solve(
                self.tXZ @ self.tZZinv @ self.tZX, self.tXZ @ self.tZZinv @ self.tZy
            ).flatten()
            # self.beta_hat = (np.linalg.inv(self.tXZ @ self.tZZinv @ self.tZX) @ self.tXZ @ self.tZZinv @ self.tZy).flatten()

        else:
            raise ValueError("estimator must be one of 'ols' or '2sls'.")

        self.Y_hat_link = self.X @ self.beta_hat
        self.u_hat = self.Y.flatten() - self.Y_hat_link.flatten()

        self.scores = self.X * self.u_hat[:, None]
        self.hessian = self.X.transpose() @ self.X


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

        if self._is_iv:
            if self.vcov_type in ["CRV3"]:
                raise VcovTypeNotSupportedError(
                    "CRV3 inference is not supported for IV regressions."
                )


        if self._is_iv:
            bread = np.linalg.inv(self.tXZ @ self.tZZinv @ self.tZX)
        else:
            bread = np.linalg.inv(self.hessian)
            #bread = self.tZXinv


        # compute vcov
        if self.vcov_type == "iid":
            self.ssc = get_ssc(
                ssc_dict=self._ssc_dict,
                N=self.N,
                k=self.k,
                G=1,
                vcov_sign=1,
                vcov_type="iid",
            )

            sigma2 = np.sum((self.weights.flatten() * self.u_hat.flatten()) **2) / (self.N - 1)
            # only relevant factor for iid in ssc: fixef.K
            if self._is_iv == False:
                self.vcov = self.ssc * bread * sigma2 # NOTE:   only valid for diagonal values
            else:
                self.vcov = (
                    self.ssc *  np.sum((self.weights * self.u_hat) ** 2) / (self.N - 1)
                )

        elif self.vcov_type == "hetero":
            self.ssc = get_ssc(
                ssc_dict=self._ssc_dict,
                N=self.N,
                k=self.k,
                G=1,
                vcov_sign=1,
                vcov_type="hetero",
            )

            if self.vcov_type_detail in ["hetero", "HC1"]:
                #u = self.u_hat
                transformed_scores = self.scores
            elif self.vcov_type_detail in ["HC2", "HC3"]:
                leverage = np.sum(self.X * (self.X @ self.tZXinv), axis=1)
                if self.vcov_type_detail == "HC2":
                    u = self.u_hat / np.sqrt(1 - leverage)
                    transformed_scores = self.scores / np.sqrt(1 - leverage)[:, None]
                else:
                    transformed_scores = self.scores / (1 - leverage)[:, None]


            if self._is_iv == False:
                meat = transformed_scores.transpose() @ transformed_scores
                self.vcov = self.ssc * bread @ meat @ bread
            else:
                if u.ndim == 1:
                    u = u.reshape((self.N, 1))
                Omega = transformed_scores.transpose() @ transformed_scores     #np.transpose(self.Z) @ (self.Z * (u**2))  # k x k
                meat = self.tXZ @ self.tZZinv @ Omega @ self.tZZinv @ self.tZX  # k x k
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

                k_instruments = self.Z.shape[1]
                meat = np.zeros((k_instruments, k_instruments))

                if self.weights is not None:
                    weighted_uhat = (self.weights.flatten() * self.u_hat.flatten()).reshape((self.N, 1))
                else:
                    weighted_uhat = self.u_hat

                for (
                    _,
                    g,
                ) in enumerate(clustid):
                    Zg = self.Z[np.where(cluster_df == g)]
                    ug = weighted_uhat[np.where(cluster_df == g)]
                    score_g = (np.transpose(Zg) @ ug).reshape((k_instruments, 1))
                    meat += np.dot(score_g, score_g.transpose())

                if self._is_iv == False:
                    self.vcov = self.ssc * bread @ meat @ bread
                #if self._is_iv == False:
                #    self.vcov = self.ssc * bread @ meat @ bread
                else:
                    meat = self.tXZ @ self.tZZinv @ meat @ self.tZZinv @ self.tZX
                    self.vcov = self.ssc * bread @ meat @ bread

            elif self.vcov_type_detail == "CRV3":
                # check: is fixed effect cluster fixed effect?
                # if not, either error or turn fixefs into dummies
                # for now: don't allow for use with fixed effects

                # if self._has_fixef:
                #    raise ValueError("CRV3 inference is currently not supported with fixed effects.")

                if self._is_iv:
                    raise VcovTypeNotSupportedError(
                        "CRV3 inference is not supported with IV estimation."
                    )

                k_params = self.k

                beta_hat = self.beta_hat

                clusters = clustid
                n_groups = self.G
                group = cluster_df

                beta_jack = np.zeros((n_groups, k_params))

                if self._has_fixef == False:
                    # inverse hessian precomputed?
                    tXX = np.transpose(self.X) @ self.X
                    tXy = np.transpose(self.X) @ self.Y

                    # compute leave-one-out regression coefficients (aka clusterjacks')
                    for ixg, g in enumerate(clusters):
                        Xg = self.X[np.equal(ixg, group)]
                        Yg = self.Y[np.equal(ixg, group)]
                        tXgXg = np.transpose(Xg) @ Xg
                        # jackknife regression coefficient
                        beta_jack[ixg, :] = (
                            np.linalg.pinv(tXX - tXgXg) @ (tXy - np.transpose(Xg) @ Yg)
                        ).flatten()

                else:
                    # lazy loading to avoid circular import
                    fixest_module = import_module("pyfixest.fixest")
                    Fixest_ = getattr(fixest_module, "Fixest")

                    for ixg, g in enumerate(clusters):
                        # direct leave one cluster out implementation
                        data = self._data[~np.equal(ixg, group)]
                        model = Fixest_(data)
                        model.feols(self._fml, vcov="iid")
                        beta_jack[ixg, :] = model.coef().to_numpy()

                # optional: beta_bar in MNW (2022)
                # center = "estimate"
                # if center == 'estimate':
                #    beta_center = beta_hat
                # else:
                #    beta_center = np.mean(beta_jack, axis = 0)
                beta_center = beta_hat

                vcov = np.zeros((k_params, k_params))
                for ixg, g in enumerate(clusters):
                    beta_centered = beta_jack[ixg, :] - beta_center
                    vcov += np.outer(beta_centered, beta_centered)

                self.vcov = self.ssc * vcov

    def get_inference(self, alpha=0.95):
        """
        Compute standard errors, t-statistics and p-values for the regression model.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals, by default 0.95

        Returns
        -------
        None

        """

        self._se = np.sqrt(np.diagonal(self.vcov))

        self._tstat = self.beta_hat / self._se

        if self.vcov_type in ["iid", "hetero"]:
            df = self.N - self.k

        else:
            df = self.G - 1

        # use t-dist for linear models, but normal for non-linear models
        if self._method == "feols":
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), df))
            z = np.abs(t.ppf((1 - alpha) / 2, df))

        else:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf((1 - alpha) / 2))

        z_se = z * self._se
        self.conf_int = np.array([self.beta_hat - z_se, self.beta_hat + z_se])

    def get_Ftest(self, vcov, is_iv=False):
        """
        compute an F-test statistic of the form H0: R*beta = q
        Args: is_iv (bool): If True, the F-test is computed for the first stage regression of an IV model. Default is False.
        Returns: None
        """

        R = np.ones(self.k).reshape((1, self.k))
        q = 0
        beta = self.beta_hat
        Rbetaq = R @ beta - q
        # Rbetaq = self.beta_hat

        if self._is_iv:
            first_stage = Feols(self.Y, self.Z, self.Z)
            first_stage.get_fit()
            first_stage.get_vcov(vcov=vcov)
            vcov = first_stage.vcov
        else:
            vcov = self.vcov

        self.F_stat = Rbetaq @ np.linalg.inv(R @ self.vcov @ np.transpose(R)) @ Rbetaq

    def get_wildboottest(
        self,
        B: int,
        cluster: Union[np.ndarray, pd.Series, pd.DataFrame, None],
        param: Union[str, None],
        weights_type: str,
        impose_null: bool,
        bootstrap_type: str,
        seed: Union[str, None],
        adj: bool,
        cluster_adj: bool,
    ):
        """
        Run a wild cluster bootstrap based on an object of type "Feols"

        Args:

        B (int): The number of bootstrap iterations to run
        cluster (Union[None, np.ndarray, pd.Series, pd.DataFrame], optional): If None (default), a 'heteroskedastic' wild boostrap
            is run. For a wild cluster bootstrap, requires a numpy array of dimension one,a  pandas Series or DataFrame, containing the clustering variable.
        param (Union[str, None], optional): A string of length one, containing the test parameter of interest. Defaults to None.
        weights_type (str, optional): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb' or 'normal'.
                            'rademacher' by default. Defaults to 'rademacher'.
        impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                            Defaults to True.
        bootstrap_type (str, optional):A string of length one. Allows to choose the bootstrap type
                            to be run. Either '11', '31', '13' or '33'. '11' by default. Defaults to '11'.
        seed (Union[str, None], optional): Option to provide a random seed. Defaults to None.

        Returns: a pd.DataFrame with bootstrapped t-statistic and p-value
        """

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest'. Note that it 'wildboottest 'requires Python < 3.11 due to its dependency on 'numba'."
            )

        if self._is_iv:
            raise VcovTypeNotSupportedError(
                "Wild cluster bootstrap is not supported with IV estimation."
            )
        if self._has_fixef:
            raise VcovTypeNotSupportedError(
                "Wild cluster bootstrap is not supported with fixed effects."
            )

        xnames = self._coefnames
        Y = self.Y.flatten()
        X = self.X

        # later: allow r <> 0 and custom R
        R = np.zeros(len(xnames))
        R[xnames.index(param)] = 1
        r = 0

        if cluster is None:
            boot = WildboottestHC(X=X, Y=Y, R=R, r=r, B=B, seed=seed)
            boot.get_adjustments(bootstrap_type=bootstrap_type)
            boot.get_uhat(impose_null=impose_null)
            boot.get_tboot(weights_type=weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            full_enumeration_warn = False

        else:
            cluster = self._data[self.clustervar]

            boot = WildboottestCL(X=X, Y=Y, cluster=cluster, R=R, B=B, seed=seed)
            boot.get_scores(
                bootstrap_type=bootstrap_type,
                impose_null=impose_null,
                adj=adj,
                cluster_adj=cluster_adj,
            )
            _, _, full_enumeration_warn = boot.get_weights(weights_type=weights_type)
            boot.get_numer()
            boot.get_denom()
            boot.get_tboot()
            boot.get_vcov()
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")

            if full_enumeration_warn:
                warnings.warn(
                    "2^G < the number of boot iterations, setting full_enumeration to True."
                )

        res = {
            "param": param,
            "statistic": boot.t_stat,
            "pvalue": boot.pvalue,
            "bootstrap_type": bootstrap_type,
            "impose_null": impose_null,
        }

        res_df = pd.Series(res)

        return res_df

    def fixef(self) -> np.array:
        """
        Return a np.array with estimated fixed effects of a fixed effects regression model.
        Additionally, computes the sum of fixed effects for each observation (this is required for the predict() method)
        If the model does not have a fixed effect, raises an error.
        Args:
            None
        Returns:
            alphaDF (pd.DataFrame): A pd.DataFrame with the estimated fixed effects. For only one fixed effects,
                                    no level of the fixed effects is dropped. For multiple fixed effects, one
                                    level of each fixed effect is dropped to avoid perfect multicollinearity.
            sumDF (np.array): A np.array with the sum of fixed effects for each of the i = 1, ..., N observations.
        Creates the following attributes:
            alphaDF, sumDF
        """

        if not self._has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        if self._is_iv:
            raise NotImplementedError(
                "The fixef() method is currently not supported for IV models."
            )

        if self._method == "fepois":
            raise NotImplementedError(
                "The fixef() method is currently not supported for Poisson models."
            )

        # fixef_vars = self._fixef.split("+")[0]

        fml = self._fml
        depvars, res = fml.split("~")
        covars, fixef_vars = res.split("|")

        df = self._data.copy()
        # all fixef vars to pd.Categorical
        for x in fixef_vars.split("+"):
            df[x] = pd.Categorical(df[x])

        fml_linear = depvars + "~" + covars
        Y, X = model_matrix(fml_linear, df)
        X = X.drop("Intercept", axis=1)
        Y = Y.to_numpy().flatten().astype(np.float64)
        X = X.to_numpy()
        uhat = csr_matrix(Y - X @ self.beta_hat).transpose()

        D2 = model_matrix("-1+" + fixef_vars, df).astype(np.float64)
        cols = D2.columns

        D2 = csr_matrix(D2.values)

        alpha = spsolve(D2.transpose() @ D2, D2.transpose() @ uhat)
        k_fe = len(alpha)

        var, level = [], []

        for _, x in enumerate(cols):
            res = x.replace("[", "").replace("]", "").split("T.")
            var.append(res[0])
            level.append(res[1])

        self.fixef_dict = dict()
        ki_start = 0
        for x in np.unique(var):
            ki = len(list(filter(lambda x: x == "group", var)))
            alphai = alpha[ki_start : (ki + ki_start)]
            levi = level[ki_start : (ki + ki_start)]
            fe_dict = (
                pd.DataFrame({"level": levi, "value": alphai}).set_index("level").T
            )

            self.fixef_dict[x] = fe_dict
            ki_start = ki

        for key, df in self.fixef_dict.items():
            print(f"{key}:\n{df.to_string(index=True)}\n")

        self.sumFE = D2 @ alpha

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
            depvar = self._fml.split("~")[0]
            y_hat = self._data[depvar].to_numpy() - self.u_hat.flatten()

        else:
            fml_linear, _ = self._fml.split("|")
            _, X = model_matrix(fml_linear, data)
            X = X.drop("Intercept", axis=1)
            X = X.to_numpy()
            y_hat = X @ self.beta_hat

        return y_hat.flatten()

    def get_nobs(self):
        """
        Fetch the number of observations used in fitting the regression model.

        Returns
        -------
        None
        """
        self.N = len(self.Y)

    def get_performance(self) -> None:
        """
        Compute multiple additional measures commonly reported with linear regression output,
        including R-squared and adjusted R-squared. Not that variables with suffix _within
        use demeand dependent variables Y, while variables without do not or are invariat to
        demeaning.

        Returns:
            None

        Creates the following instances:
            r2 (float): R-squared of the regression model.
            adj_r2 (float): Adjusted R-squared of the regression model.
            r2_within (float): R-squared of the regression model, computed on demeaned dependent variable.
            adj_r2_within (float): Adjusted R-squared of the regression model, computed on demeaned dependent variable.
        """

        Y_no_demean = self.Y

        ssu = np.sum(self.u_hat**2)
        ssy_within = np.sum((self.Y - np.mean(self.Y)) ** 2)
        ssy = np.sum((Y_no_demean - np.mean(Y_no_demean)) ** 2)

        self.rmse = np.sqrt(ssu / self.N)

        self.r2_within = 1 - (ssu / ssy_within)
        self.r2 = 1 - (ssu / ssy)

        self.adj_r2_within = 1 - (1 - self.r2_within) * (self.N - 1) / (
            self.N - self.k - 1
        )
        self.adj_r2 = 1 - (1 - self.r2) * (self.N - 1) / (self.N - self.k - 1)

    def tidy(self) -> pd.DataFrame:
        """
        Return a tidy pd.DataFrame with the point estimates, standard errors, t statistics and p-values.
        Returns:
            tidy_df (pd.DataFrame): A tidy pd.DataFrame with the regression results.
        """

        tidy_df = pd.DataFrame(
            {
                "Coefficient": self._coefnames,
                "Estimate": self.beta_hat,
                "Std. Error": self._se,
                "t value": self._tstat,
                "Pr(>|t|)": self._pvalue,
                "2.5 %": self.conf_int[0],
                "97.5 %": self.conf_int[1],
            }
        )

        return tidy_df.set_index("Coefficient")

    def coef(self) -> pd.Series:
        """
        Return a pd.Series with estimated regression coefficients.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Return a pd.Series with standard errors of the estimated regression model.
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Return a pd.Series with t-statistics of the estimated regression model.
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Return a pd.Series with p-values of the estimated regression model.
        """
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        """
        Return a pd.DataFrame with confidence intervals for the estimated regression model.
        """
        return self.tidy()[["2.5 %", "97.5 %"]]

    def resid(self) -> np.ndarray:
        """
        Returns a one dimensional np.array with the residuals of the estimated regression model.
        """
        return self.u_hat


def _check_vcov_input(vcov, data):
    """
    Check the input for the vcov argument in the Feols class.
    Args:
        vcov (dict, str, list): The vcov argument passed to the Feols class.
        data (pd.DataFrame): The data passed to the Feols class.
    Returns:
        None
    """

    assert isinstance(vcov, (dict, str, list)), "vcov must be a dict, string or list"
    if isinstance(vcov, dict):
        assert list(vcov.keys())[0] in [
            "CRV1",
            "CRV3",
        ], "vcov dict key must be CRV1 or CRV3"
        assert isinstance(
            list(vcov.values())[0], str
        ), "vcov dict value must be a string"
        assert (
            list(vcov.values())[0] in data.columns
        ), "vcov dict value must be a column in the data"
    if isinstance(vcov, list):
        assert all(isinstance(v, str) for v in vcov), "vcov list must contain strings"
        assert all(
            v in data.columns for v in vcov
        ), "vcov list must contain columns in the data"
    if isinstance(vcov, str):
        assert vcov in [
            "iid",
            "hetero",
            "HC1",
            "HC2",
            "HC3",
        ], "vcov string must be iid, hetero, HC1, HC2, or HC3"


def _deparse_vcov_input(vcov, has_fixef, is_iv):
    """
    Deparse the vcov argument passed to the Feols class.

    Args:
        vcov (dict, str, list): The vcov argument passed to the Feols class.
        has_fixef (bool): Whether the regression has fixed effects.
        is_iv (bool): Whether the regression is an IV regression.
    Returns:
        vcov_type (str): The type of vcov to be used. Either "iid", "hetero", or "CRV"
        vcov_type_detail (str, list): The type of vcov to be used, with more detail. Either "iid", "hetero", "HC1", "HC2", "HC3", "CRV1", or "CRV3"
        is_clustered (bool): Whether the vcov is clustered.
        clustervar (str): The name of the cluster variable.
    """

    if isinstance(vcov, dict):
        vcov_type_detail = list(vcov.keys())[0]
        clustervar = list(vcov.values())[0]
    elif isinstance(vcov, list):
        vcov_type_detail = vcov
    elif isinstance(vcov, str):
        vcov_type_detail = vcov
    else:
        assert False, "arg vcov needs to be a dict, string or list"

    if vcov_type_detail == "iid":
        vcov_type = "iid"
        is_clustered = False
    elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
        vcov_type = "hetero"
        is_clustered = False
        if vcov_type_detail in ["HC2", "HC3"]:
            if has_fixef:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for regressions with fixed effects."
                )
            if is_iv:
                raise VcovTypeNotSupportedError(
                    "HC2 and HC3 inference types are not supported for IV regressions."
                )
    elif vcov_type_detail in ["CRV1", "CRV3"]:
        vcov_type = "CRV"
        is_clustered = True

    if is_clustered:
        clustervar = list(vcov.values())[0]
    else:
        clustervar = None

    return vcov_type, vcov_type_detail, is_clustered, clustervar


def _feols_input_checks(Y, X, Z, weights):
    """
    Some basic checks on the input matrices Y, X, and Z.
    Args:
        Y (np.ndarray): FEOLS input matrix Y
        X (np.ndarray): FEOLS input matrix X
        Z (np.ndarray): FEOLS input matrix Z
    Returns:
        None
    """

    if not isinstance(Y, (np.ndarray)):
        raise TypeError("Y must be a numpy array.")
    if not isinstance(X, (np.ndarray)):
        raise TypeError("X must be a numpy array.")
    if not isinstance(Z, (np.ndarray)):
        if Z is not None:
            raise TypeError("Z must be a numpy array or None.")
    if not isinstance(weights, (np.ndarray)):
        raise TypeError("weights must be a numpy array.")

    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Z is not None:
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array")
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")
