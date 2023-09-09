import numpy as np
import pandas as pd
import warnings

from importlib import import_module
from typing import Optional, Union, List, Dict
from scipy.stats import norm, t
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from formulaic import model_matrix

from pyfixest.utils import get_ssc
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
    vcov(vcov)
        Compute covariance matrices for an estimated model.

    Raises
    ------
    AssertionError
        If the vcov argument is not a dict, a string, or a list.

    """

    def __init__(self, Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> None:
        self._method = "feols"

        self._Y = Y
        self._X = X
        self._Z = X
        self.get_nobs()

        _feols_input_checks(Y, X, weights)

        self._weights = weights
        self._is_iv = False

        self._N, self._k = X.shape

        self._support_crv3_inference = True
        self._support_iid_inference = True

        # attributes that have to be enriched outside of the class - not really optimal code
        # change later
        self._data = None
        self._fml = None
        self._has_fixef = False
        self._fixef = None
        self._coefnames = None
        self._icovars = None
        self._ssc_dict = None

        # set in get_fit()
        self._tZX = None
        self._tZXinv = None
        self._tXZ = None
        self._tZy = None
        self._tZZinv = None
        self._beta_hat = None
        self._Y_hat_link = None
        self._Y_hat_response = None
        self._u_hat = None
        self._scores = None
        self._hessian = None
        self._bread = None

        # set in vcov()
        self._vcov_type = None
        self._vcov_type_detail = None
        self._is_clustered = None
        self._clustervar = None
        self._G = None
        self._ssc = None
        self._vcov = None

        # set in get_inference()
        self._se = None
        self._tstat = None
        self._pvalue = None
        self._conf_int = None

        # set in get_Ftest()
        self._F_stat = None

        # set in fixef()
        self._fixef_dict = None
        self._sumFE = None

        # set in get_performance()
        self._rmse = None
        self._r2 = None
        self._r2_within = None
        self._adj_r2 = None
        self._adj_r2_within = None

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

        _X = self._X
        _Y = self._Y
        _Z = self._Z

        self._tZX = _Z.T @ _X
        self._tZy = _Z.T @ _Y

        self._tZXinv = np.linalg.inv(self._tZX)
        self._beta_hat = np.linalg.solve(self._tZX, self._tZy).flatten()
        # self._beta_hat = (self._tZXinv @ self._tZy).flatten()

        self._Y_hat_link = self._X @ self._beta_hat
        self._u_hat = self._Y.flatten() - self._Y_hat_link.flatten()

        self._scores = self._Z * self._u_hat[:, None]
        self._hessian = self._Z.transpose() @ self._Z

        # IV attributes, set to None for OLS, Poisson
        self._tXZ = None
        self._tZZinv = None

    def vcov(self, vcov: Union[str, Dict[str, str], List[str]]) -> None:
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

        Returns
        -------
        None

        """

        _data = self._data
        _fml = self._fml
        _has_fixef = self._has_fixef
        _is_iv = self._is_iv
        _method = self._method
        _support_iid_inference = self._support_iid_inference
        _support_crv3_inference = self._support_crv3_inference

        _beta_hat = self._beta_hat

        _X = self._X
        _Z = self._Z
        _tXZ = self._tXZ
        _tZZinv = self._tZZinv
        _tZX = self._tZX
        _tZXinv = self._tZXinv
        _hessian = self._hessian
        _scores = self._scores

        _weights = self._weights
        _ssc_dict = self._ssc_dict
        _N = self._N
        _k = self._k

        _u_hat = self._u_hat

        _check_vcov_input(vcov, _data)

        (
            self._vcov_type,
            self._vcov_type_detail,
            self._is_clustered,
            self._clustervar,
        ) = _deparse_vcov_input(vcov, _has_fixef, _is_iv)

        if _is_iv:
            if self._vcov_type in ["CRV3"]:
                raise VcovTypeNotSupportedError(
                    "CRV3 inference is not supported for IV regressions."
                )

        if _is_iv:
            bread = np.linalg.inv(_tXZ @ _tZZinv @ _tZX)
        else:
            bread = np.linalg.inv(_hessian)

        # compute vcov
        if self._vcov_type == "iid":
            if not _support_iid_inference:
                raise NotImplementedError(
                    f"'iid' inference is not supported for {_method} regressions."
                )

            self._ssc = get_ssc(
                ssc_dict=_ssc_dict,
                N=_N,
                k=_k,
                G=1,
                vcov_sign=1,
                vcov_type="iid",
            )

            sigma2 = np.sum((_u_hat.flatten()) ** 2) / (_N - 1)
            self._vcov = self._ssc * bread * sigma2

        elif self._vcov_type == "hetero":
            self._ssc = get_ssc(
                ssc_dict=_ssc_dict,
                N=_N,
                k=_k,
                G=1,
                vcov_sign=1,
                vcov_type="hetero",
            )

            if self._vcov_type_detail in ["hetero", "HC1"]:
                u = _u_hat
                transformed_scores = _scores
            elif self._vcov_type_detail in ["HC2", "HC3"]:
                if _is_iv:
                    raise VcovTypeNotSupportedError(
                        "HC2 and HC3 inference is not supported for IV regressions."
                    )

                leverage = np.sum(_X * (_X @ _tZXinv), axis=1)
                if self._vcov_type_detail == "HC2":
                    u = _u_hat / np.sqrt(1 - leverage)
                    transformed_scores = _scores / np.sqrt(1 - leverage)[:, None]
                else:
                    transformed_scores = _scores / (1 - leverage)[:, None]

            if _is_iv == False:
                meat = transformed_scores.transpose() @ transformed_scores
                self._vcov = self._ssc * bread @ meat @ bread
            else:
                if u.ndim == 1:
                    u = u.reshape((_N, 1))
                Omega = (
                    transformed_scores.transpose() @ transformed_scores
                )  # np.transpose( self._Z) @ ( self._Z * (u**2))  # k x k
                meat = _tXZ @ _tZZinv @ Omega @ _tZZinv @ _tZX  # k x k
                self._vcov = self._ssc * bread @ meat @ bread

        elif self._vcov_type == "CRV":
            cluster_df = _data[self._clustervar]
            # if there are missings - delete them!

            if cluster_df.dtype != "category":
                cluster_df = pd.Categorical(cluster_df)

            if cluster_df.isna().any():
                raise NanInClusterVarError(
                    "CRV inference not supported with missing values in the cluster variable."
                    "Please drop missing values before running the regression."
                )

            _, clustid = pd.factorize(cluster_df)

            self._G = len(clustid)

            self._ssc = get_ssc(
                ssc_dict=_ssc_dict,
                N=_N,
                k=_k,
                G=self._G,
                vcov_sign=1,
                vcov_type="CRV",
            )

            if self._vcov_type_detail == "CRV1":
                k_instruments = _Z.shape[1]
                meat = np.zeros((k_instruments, k_instruments))

                if _weights is not None:
                    weighted_uhat = (_weights.flatten() * _u_hat.flatten()).reshape(
                        (_N, 1)
                    )
                else:
                    weighted_uhat = _u_hat

                for (
                    _,
                    g,
                ) in enumerate(clustid):
                    Zg = _Z[np.where(cluster_df == g)]
                    ug = weighted_uhat[np.where(cluster_df == g)]
                    score_g = (np.transpose(Zg) @ ug).reshape((k_instruments, 1))
                    meat += np.dot(score_g, score_g.transpose())

                if _is_iv == False:
                    self._vcov = self._ssc * bread @ meat @ bread
                # if self._is_iv == False:
                #    self._vcov = self._ssc * bread @ meat @ bread
                else:
                    meat = _tXZ @ _tZZinv @ meat @ _tZZinv @ self._tZX
                    self._vcov = self._ssc * bread @ meat @ bread

            elif self._vcov_type_detail == "CRV3":
                # check: is fixed effect cluster fixed effect?
                # if not, either error or turn fixefs into dummies
                # for now: don't allow for use with fixed effects

                # if self._has_fixef:
                #    raise ValueError("CRV3 inference is currently not supported with fixed effects.")

                if _is_iv:
                    raise VcovTypeNotSupportedError(
                        "CRV3 inference is not supported with IV estimation."
                    )

                if not _support_crv3_inference:
                    raise NotImplementedError(
                        f"'CRV3' inference is not supported for {_method} regressions."
                    )

                clusters = clustid
                n_groups = self._G
                group = cluster_df

                beta_jack = np.zeros((n_groups, _k))

                if self._has_fixef == False:
                    # inverse hessian precomputed?
                    tXX = np.transpose(self._X) @ self._X
                    tXy = np.transpose(self._X) @ self._Y

                    # compute leave-one-out regression coefficients (aka clusterjacks')
                    for ixg, g in enumerate(clusters):
                        Xg = self._X[np.equal(ixg, group)]
                        Yg = self._Y[np.equal(ixg, group)]
                        tXgXg = np.transpose(Xg) @ Xg
                        # jackknife regression coefficient
                        beta_jack[ixg, :] = (
                            np.linalg.pinv(tXX - tXgXg) @ (tXy - np.transpose(Xg) @ Yg)
                        ).flatten()

                else:
                    # lazy loading to avoid circular import
                    fixest_module = import_module("pyfixest.estimation")
                    feols_ = getattr(fixest_module, "feols")

                    for ixg, g in enumerate(clusters):
                        # direct leave one cluster out implementation
                        data = _data[~np.equal(ixg, group)]
                        fit = feols_(fml=self._fml, data=data, vcov="iid")
                        beta_jack[ixg, :] = fit.coef().to_numpy()

                # optional: beta_bar in MNW (2022)
                # center = "estimate"
                # if center == 'estimate':
                #    beta_center = beta_hat
                # else:
                #    beta_center = np.mean(beta_jack, axis = 0)
                beta_center = _beta_hat

                vcov = np.zeros((_k, _k))
                for ixg, g in enumerate(clusters):
                    beta_centered = beta_jack[ixg, :] - beta_center
                    vcov += np.outer(beta_centered, beta_centered)

                self._vcov = self._ssc * vcov

        self.get_inference()

        return self

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

        _vcov = self._vcov
        _beta_hat = self._beta_hat
        _vcov_type = self._vcov_type
        _N = self._N
        _k = self._k
        _G = self._G
        _method = self._method

        self._se = np.sqrt(np.diagonal(_vcov))
        self._tstat = _beta_hat / self._se

        if _vcov_type in ["iid", "hetero"]:
            df = _N - _k
        else:
            df = _G - 1

        # use t-dist for linear models, but normal for non-linear models
        if _method == "feols":
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), df))
            z = np.abs(t.ppf((1 - alpha) / 2, df))
        else:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf((1 - alpha) / 2))

        z_se = z * self._se
        self._conf_int = np.array([_beta_hat - z_se, _beta_hat + z_se])

    def get_Ftest(self, vcov, is_iv=False):
        """
        compute an F-test statistic of the form H0: R*beta = q
        Args: is_iv (bool): If True, the F-test is computed for the first stage regression of an IV model. Default is False.
        Returns: None
        """

        R = np.ones(self._k).reshape((1, self._k))
        q = 0
        beta = self._beta_hat
        Rbetaq = R @ beta - q
        # Rbetaq = self._beta_hat

        if self._is_iv:
            first_stage = Feols(self._Y, self._Z, self._Z)
            first_stage.get_fit()
            first_stage.vcov(vcov=vcov)
            vcov = first_stage.vcov
        else:
            vcov = self._vcov

        self._F_stat = Rbetaq @ np.linalg.inv(R @ self._vcov @ np.transpose(R)) @ Rbetaq

    def coefplot(
        self,
        alpha=0.05,
        figsize=(10, 10),
        yintercept=None,
        xintercept=None,
        rotate_xticks=0,
        coefficients: Optional[List[str]] = None,
    ):
        # lazy loading to avoid circular import
        visualize_module = import_module("pyfixest.visualize")
        _coefplot = getattr(visualize_module, "coefplot")

        plot = _coefplot(
            models=self,
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            coefficients=coefficients,
        )

        return plot

    def iplot(
        self,
        alpha=0.05,
        figsize=(10, 10),
        yintercept=None,
        xintercept=None,
        rotate_xticks=0,
    ):
        visualize_module = import_module("pyfixest.visualize")
        _iplot = getattr(visualize_module, "iplot")

        plot = _iplot(
            models=self,
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
        )

        return plot

    def wildboottest(
        self,
        B: int,
        cluster: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        param: Optional[str] = None,
        weights_type: Optional[str] = "rademacher",
        impose_null: Optional[bool] = True,
        bootstrap_type: Optional[str] = "11",
        seed: Optional[int] = None,
        adj: Optional[bool] = True,
        cluster_adj: Optional[bool] = True,
        digits: Optional[int] = 3,
    ):
        """
        Run a wild cluster bootstrap based on an object of type "Feols"

        Args:

        B (int): The number of bootstrap iterations to run
        cluster (Union[None, np.ndarray, pd.Series, pd.DataFrame], optional): If None (default), checks if the model's vcov type was CRV. If yes, uses
                            `self._clustervar` as cluster. If None and no clustering was employed in the initial model, runs a heteroskedastic wild bootstrap.
                            If an argument is supplied, uses the argument as cluster variable for the wild cluster bootstrap.
                            Requires a numpy array of dimension one,a  pandas Series or DataFrame, containing the clustering variable.
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

        _is_iv = self._is_iv
        _has_fixef = self._has_fixef
        _Y = self._Y.flatten()
        _X = self._X
        _xnames = self._coefnames
        _data = self._data
        _clustervar = self._clustervar

        if cluster is None:
            if hasattr(self, "clustervar"):
                cluster = _clustervar

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest'. Note that it 'wildboottest 'requires Python < 3.11 due to its dependency on 'numba'."
            )

        if _is_iv:
            raise VcovTypeNotSupportedError(
                "Wild cluster bootstrap is not supported with IV estimation."
            )
        if _has_fixef:
            raise VcovTypeNotSupportedError(
                "Wild cluster bootstrap is not supported with fixed effects."
            )

        # later: allow r <> 0 and custom R
        R = np.zeros(len(_xnames))
        R[_xnames.index(param)] = 1
        r = 0

        if cluster is None:
            boot = WildboottestHC(X=_X, Y=_Y, R=R, r=r, B=B, seed=seed)
            boot.get_adjustments(bootstrap_type=bootstrap_type)
            boot.get_uhat(impose_null=impose_null)
            boot.get_tboot(weights_type=weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            full_enumeration_warn = False

        else:
            cluster = _data[cluster]

            boot = WildboottestCL(X=_X, Y=_Y, cluster=cluster, R=R, B=B, seed=seed)
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
            boot.vcov()
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")

            if full_enumeration_warn:
                warnings.warn(
                    "2^G < the number of boot iterations, setting full_enumeration to True."
                )

        res = {
            "param": param,
            "statistic": np.round(boot.t_stat, digits),
            "pvalue": np.round(boot.pvalue, digits),
            "bootstrap_type": bootstrap_type,
            "impose_null": impose_null,
        }

        res_df = pd.Series(res)

        return res_df

    def fixef(self) -> np.ndarray:
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

        _has_fixef = self._has_fixef
        _is_iv = self._is_iv
        _method = self._method
        _fml = self._fml
        _data = self._data

        if not _has_fixef:
            raise ValueError("The regression model does not have fixed effects.")

        if _is_iv:
            raise NotImplementedError(
                "The fixef() method is currently not supported for IV models."
            )

        if _method == "fepois":
            raise NotImplementedError(
                "The fixef() method is currently not supported for Poisson models."
            )

        # fixef_vars = self._fixef.split("+")[0]

        depvars, res = _fml.split("~")
        covars, fixef_vars = res.split("|")

        df = _data.copy()
        # all fixef vars to pd.Categorical
        for x in fixef_vars.split("+"):
            df[x] = pd.Categorical(df[x])

        fml_linear = depvars + "~" + covars
        Y, X = model_matrix(fml_linear, df)
        X = X.drop("Intercept", axis=1)
        Y = Y.to_numpy().flatten().astype(np.float64)
        X = X.to_numpy()
        uhat = csr_matrix(Y - X @ self._beta_hat).transpose()

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

        self._fixef_dict = dict()
        ki_start = 0
        for x in np.unique(var):
            ki = len(list(filter(lambda x: x == "group", var)))
            alphai = alpha[ki_start : (ki + ki_start)]
            levi = level[ki_start : (ki + ki_start)]
            fe_dict = (
                pd.DataFrame({"level": levi, "value": alphai}).set_index("level").T
            )

            self._fixef_dict[x] = fe_dict
            ki_start = ki

        for key, df in self._fixef_dict.items():
            print(f"{key}:\n{df.to_string(index=True)}\n")

        self._sumFE = D2 @ alpha

    def predict(self, data: Optional[pd.DataFrame] = None, type="link") -> np.ndarray:
        """
        Return a flat np.array with predicted values of the regression model.
        Args:
            data (Optional[pd.DataFrame], optional): A pd.DataFrame with the data to be used for prediction.
                If None (default), uses the data used for fitting the model.
            type (str, optional): The type of prediction to be computed. Either "response" (default) or "link".
                If type="response", then the output is at the level of the response variable, i.e. it is the expected predictor E(Y|X).
                If "link", then the output is at the level of the explanatory variables, i.e. the linear predictor X @ beta.

        """

        _fml = self._fml
        _data = self._data
        _u_hat = self._u_hat
        _beta_hat = self._beta_hat

        if type not in ["response", "link"]:
            raise ValueError("type must be one of 'response' or 'link'.")

        if data is None:
            depvar = _fml.split("~")[0]
            y_hat = _data[depvar].to_numpy() - _u_hat.flatten()

        else:
            fml_linear, _ = _fml.split("|")
            _, X = model_matrix(fml_linear, data)
            X = X.drop("Intercept", axis=1)
            X = X.to_numpy()
            y_hat = X @ _beta_hat

        return y_hat.flatten()

    def get_nobs(self):
        """
        Fetch the number of observations used in fitting the regression model.

        Returns
        -------
        None
        """
        self._N = len(self._Y)

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

        _Y = self._Y
        _u_hat = self._u_hat
        _N = self._N
        _k = self._k

        Y_no_demean = _Y

        ssu = np.sum(_u_hat**2)
        ssy_within = np.sum((_Y - np.mean(_Y)) ** 2)
        ssy = np.sum((Y_no_demean - np.mean(Y_no_demean)) ** 2)

        self._rmse = np.sqrt(ssu / _N)

        self._r2_within = 1 - (ssu / ssy_within)
        self._r2 = 1 - (ssu / ssy)

        self._adj_r2_within = 1 - (1 - self._r2_within) * (_N - 1) / (_N - _k - 1)
        self._adj_r2 = 1 - (1 - self._r2) * (_N - 1) / (_N - _k - 1)

    def tidy(self) -> pd.DataFrame:
        """
        Return a tidy pd.DataFrame with the point estimates, standard errors, t statistics and p-values.
        Returns:
            tidy_df (pd.DataFrame): A tidy pd.DataFrame with the regression results.
        """

        _coefnames = self._coefnames
        _se = self._se
        _tstat = self._tstat
        _pvalue = self._pvalue
        _beta_hat = self._beta_hat
        _conf_int = self._conf_int

        tidy_df = pd.DataFrame(
            {
                "Coefficient": _coefnames,
                "Estimate": _beta_hat,
                "Std. Error": _se,
                "t value": _tstat,
                "Pr(>|t|)": _pvalue,
                "2.5 %": _conf_int[0],
                "97.5 %": _conf_int[1],
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
        return self._u_hat

    def summary(self, digits=3) -> None:
        """
        Print a summary of the estimated regression model.
        Args:
            digits (int, optional): Number of digits to be printed. Defaults to 3.
        Returns:
            None
        """

        # lazy loading to avoid circular import
        summarize_module = import_module("pyfixest.summarize")
        _summary = getattr(summarize_module, "summary")

        return _summary(models=self, digits=digits)


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


def _feols_input_checks(Y, X, weights):
    """
    Some basic checks on the input matrices Y, X, and Z.
    Args:
        Y (np.ndarray): FEOLS input matrix Y
        X (np.ndarray): FEOLS input matrix X
    Returns:
        None
    """

    if not isinstance(Y, (np.ndarray)):
        raise TypeError("Y must be a numpy array.")
    if not isinstance(X, (np.ndarray)):
        raise TypeError("X must be a numpy array.")
    if not isinstance(weights, (np.ndarray)):
        raise TypeError("weights must be a numpy array.")

    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")


def _get_vcov_type(vcov, fval):
    """
    Passes the specified vcov type. If no vcov type specified, sets the default vcov type as iid if no fixed effect
    is included in the model, and CRV1 clustered by the first fixed effect if a fixed effect is included in the model.
    Args:
        vcov (str): The specified vcov type.
        fval (str): The specified fixed effects. (i.e. "X1+X2")
    Returns:
        vcov_type (str): The specified vcov type.
    """

    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}
    else:
        vcov_type = vcov

    return vcov_type
