import re
import numpy as np
import pandas as pd
import warnings

from importlib import import_module
from typing import Optional, Union, List, Dict, Tuple

from scipy.stats import norm, t
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from formulaic import model_matrix

from pyfixest.utils import get_ssc
from pyfixest.exceptions import (
    MatrixNotFullRankError,
    VcovTypeNotSupportedError,
    NanInClusterVarError,
)


class Feols:

    """

    # Feols

    A class for estimating regression models with high-dimensional fixed effects via
    ordinary least squares.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        weights: np.ndarray,
        collin_tol: float,
        coefnames: List[str],
    ) -> None:
        """
        Initiate an instance of class `Feols`.

        Args:
            Y (np.array): dependent variable. two-dimensional np.array
            X (np.array): independent variables. two-dimensional np.array
            weights (np.array): weights. one-dimensional np.array
            collin_tol (float): tolerance level for collinearity checks
            coefnames (List[str]): names of the coefficients (of the design matrix X)
        Returns:
            None

        """

        self._method = "feols"

        self._Y = Y
        self._X = X
        self.get_nobs()

        _feols_input_checks(Y, X, weights)

        self._collin_tol = collin_tol
        (
            self._X,
            self._coefnames,
            self._collin_vars,
            self._collin_index,
        ) = _drop_multicollinear_variables(self._X, coefnames, self._collin_tol)
        self._Z = self._X

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
        # self._coefnames = None
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
        Fit a single regression model, via ordinary least squares (OLS).

        Args:
            None
        Returns:
            None
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

    def vcov(self, vcov: Union[str, Dict[str, str]]):
        """
        Compute covariance matrices for an estimated regression model.

        Args:
            vcov : Union[str, Dict[str, str]
                A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference
                or {"CRV3":"clustervar"} for CRV3 inference.
                Note that CRV3 inference is currently not supported with arbitrary fixed effects and IV estimation.

        Returns:
            An instance of class `Feols` with updated inference.
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

    def get_inference(self, alpha: float = 0.95) -> None:
        """
        Compute standard errors, t-statistics and p-values for the regression model.

        Args:
            alpha (float): The significance level for confidence intervals. Defaults to 0.95.

        Returns:
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

        Args:
            is_iv (bool): If True, the F-test is computed for the first stage regression of an IV model. Default is False.
        Returns:
            None
        """

        raise NotImplementedError("The F-test is currently not supported.")

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
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (500, 300),
        yintercept: Optional[float] = 0,
        xintercept: Optional[float] = None,
        rotate_xticks: int = 0,
        coefficients: Optional[List[str]] = None,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Create a coefficient plot to visualize model coefficients.

        Args:
            alpha (float, optional): Significance level for highlighting significant coefficients.
            figsize (Tuple[int, int], optional): Size of the plot (width, height) in inches.
            yintercept (float, optional): Value to set as the y-axis intercept (vertical line).
            xintercept (float, optional): Value to set as the x-axis intercept (horizontal line).
            rotate_xticks (int, optional): Rotation angle for x-axis tick labels.
            coefficients (List[str], optional): List of coefficients to include in the plot.
                If None, all coefficients are included.
            title (str, optional): Title of the plot.
            coord_flip (bool, optional): Whether to flip the coordinates of the plot.

        Returns:
            A lets-plot figure with coefficient estimates and confidence intervals.

        """

        # lazy loading to avoid circular import
        visualize_module = import_module("pyfixest.visualize")
        _coefplot = getattr(visualize_module, "coefplot")

        plot = _coefplot(
            models=[self],
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            coefficients=coefficients,
            title=title,
            coord_flip=coord_flip,
        )

        return plot

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (500, 300),
        yintercept: Optional[float] = None,
        xintercept: Optional[float] = None,
        rotate_xticks: int = 0,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Create a coefficient plots for variables interaceted via `i()` syntax.

        Args:
            alpha (float, optional): Significance level for visualization options.
            figsize (Tuple[int, int], optional): Size of the plot (width, height) in inches.
            yintercept (float, optional): Value to set as the y-axis intercept (vertical line).
            xintercept (float, optional): Value to set as the x-axis intercept (horizontal line).
            rotate_xticks (int, optional): Rotation angle for x-axis tick labels.
            title (str, optional): Title of the plot.
            coord_flip (bool, optional): Whether to flip the coordinates of the plot.

        Returns:
            A lets-plot figure with coefficient estimates and confidence intervals.

        """

        visualize_module = import_module("pyfixest.visualize")
        _iplot = getattr(visualize_module, "iplot")

        plot = _iplot(
            models=[self],
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
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
        parallel: Optional[bool] = False,
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
        impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not? Defaults to True.
        bootstrap_type (str, optional):A string of length one. Allows to choose the bootstrap type
                            to be run. Either '11', '31', '13' or '33'. '11' by default. Defaults to '11'.
        seed (Union[int, None], optional): Option to provide a random seed. Defaults to None.
        adj (bool, optional): Should a small sample adjustment be applied for number of observations and covariates? Defaults to True.
                              Note that the small sample adjustment in the bootstrap might differ from the one in the original model.
                              This will only affect the returned non-bootstrapped t-statistic, but not the bootstrapped p-value.
                              For exact matches, set `adj = False` and `cluster_adj = False` in `wildboottest()` and via the
                              `ssc(adj = False, cluster_adj = False)` option in `feols()`.
        cluster_adj (bool, optional): Should a small sample adjustment be applied for the number of clusters? Defaults to True.
                                Note that the small sample adjustment in the bootstrap might differ from the one in the original model.
                                This will only affect the returned non-bootstrapped t-statistic, but not the bootstrapped p-value.
                                For exact matches, set `adj = False` and `cluster_adj = False` in `wildboottest()` and via the
                                `ssc(adj = False, cluster_adj = False)` option in `feols()`.
        parallel (bool, optional): Should the bootstrap be run in parallel? Defaults to False.
        seed (Union[str, None], optional): Option to provide a random seed. Defaults to None.

        Returns: a pd.DataFrame with the original, non-bootstrapped t-statistic and bootstrapped p-value as well as
                the bootstrap type, inference type (HC vs CRV) and whether the null hypothesis was imposed on the bootstrap dgp.
        """

        _is_iv = self._is_iv
        _has_fixef = self._has_fixef
        _Y = self._Y.flatten()
        _X = self._X
        _xnames = self._coefnames
        _data = self._data
        _clustervar = self._clustervar

        _ssc = self._ssc

        if cluster is None:
            if self._clustervar is not None:
                cluster = self._clustervar

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
            )

        if _is_iv:
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported with IV estimation."
            )

        if self._method == "fepois":
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported for Poisson regression."
            )

        if _has_fixef:
            # update _X, _xnames
            fml_linear, fixef = self._fml.split("|")
            fixef_vars = fixef.split("+")
            # wrap all fixef vars in "C()"
            fixef_vars_C = [f"C({x})" for x in fixef_vars]
            fixef_fml = "+".join(fixef_vars_C)

            fml_dummies = f"{fml_linear} + {fixef_fml}"

            # make this sparse once wildboottest allows it
            _, _X = model_matrix(fml_dummies, _data, output="numpy")
            _xnames = _X.model_spec.column_names

        # later: allow r <> 0 and custom R
        R = np.zeros(len(_xnames))
        R[_xnames.index(param)] = 1
        r = 0

        if cluster is None:
            inference = "HC"

            boot = WildboottestHC(X=_X, Y=_Y, R=R, r=r, B=B, seed=seed)
            boot.get_adjustments(bootstrap_type=bootstrap_type)
            boot.get_uhat(impose_null=impose_null)
            boot.get_tboot(weights_type=weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            full_enumeration_warn = False

        else:
            inference = f"CRV({self._clustervar})"
            cluster = _data[cluster]

            boot = WildboottestCL(
                X=_X, Y=_Y, cluster=cluster, R=R, B=B, seed=seed, parallel=parallel
            )
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

        if np.isscalar(boot.t_stat):
            boot.t_stat = boot.t_stat
        else:
            boot.t_stat = boot.t_stat[0]

        res = {
            "param": param,
            "t value": boot.t_stat.astype(np.float64),
            "Pr(>|t|)": boot.pvalue.astype(np.float64),
            "bootstrap_type": bootstrap_type,
            "inference": inference,
            "impose_null": impose_null,
        }

        res_df = pd.Series(res)

        return res_df

    def fixef(self) -> None:
        """
        Compute the coefficients of (sweeped out) fixed effects for a regression model.

        This method creates the following attributes:

        - `alphaDF` (pd.DataFrame): A DataFrame with the estimated fixed effects.
        - `sumFE` (np.array): An array with the sum of fixed effects for each observation (i = 1, ..., N).

        Args:
            None

        Returns:
            None
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

        # fixef_vars = self._fixef.split("+")[0]

        depvars, res = _fml.split("~")
        covars, fixef_vars = res.split("|")

        fixef_vars = fixef_vars.split("+")
        fixef_vars_C = [f"C({x})" for x in fixef_vars]
        fixef_fml = "+".join(fixef_vars_C)

        fml_linear = f"{depvars} ~ {covars}"
        Y, X = model_matrix(fml_linear, _data)
        X = X[self._coefnames]  # drop intercept, potentially multicollinear vars
        Y = Y.to_numpy().flatten().astype(np.float64)
        X = X.to_numpy()
        uhat = csr_matrix(Y - X @ self._beta_hat).transpose()

        D2 = model_matrix("-1+" + fixef_fml, _data, output="sparse")
        cols = D2.model_spec.column_names

        alpha = spsolve(D2.transpose() @ D2, D2.transpose() @ uhat)

        res = dict()
        for i, col in enumerate(cols):
            matches = re.match(r"(.+?)\[T\.(.+?)\]", col)
            if matches:
                variable = matches.group(1)
                level = matches.group(2)
            else:
                raise ValueError(
                    "Something went wrong with the regex. Please open a PR in the github repo!"
                )

            # check if res already has a key variable
            if variable not in res:
                res[variable] = dict()
                res[variable][level] = alpha[i]
                continue
            else:
                if level not in res[variable]:
                    res[variable][level] = alpha[i]

        self._fixef_dict = res
        self._alpha = alpha
        self._sumFE = D2.dot(alpha)

        return self._fixef_dict

    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values for such observations
        will be set to NaN.

        Args:
            newdata (Optional[pd.DataFrame], optional): A pd.DataFrame with the data to be used for prediction.
                If None (default), uses the data used for fitting the model.

        Returns:
            y_hat (np.ndarray): A flat np.array with predicted values of the regression model.

        """

        _fml = self._fml
        _data = self._data
        _u_hat = self._u_hat
        _beta_hat = self._beta_hat
        _is_iv = self._is_iv
        # _fixef = "+".split(self._fixef) # name of the fixef effects variables

        if _is_iv:
            raise NotImplementedError(
                "The predict() method is currently not supported for IV models."
            )

        if newdata is None:
            depvar = _fml.split("~")[0]
            y_hat = _data[depvar].to_numpy() - _u_hat.flatten()

        else:
            if self._has_fixef:
                fml_linear, _ = _fml.split("|")

                if self._sumFE is None:
                    self.fixef()

                fvals = self._fixef.split("+")
                df_fe = newdata[fvals].astype(str)

                # populate matrix with fixed effects estimates
                fixef_mat = np.zeros((newdata.shape[0], len(fvals)))
                # fixef_mat = np.full((newdata.shape[0], len(fvals)), np.nan)

                for i, fixef in enumerate(df_fe.columns):
                    new_levels = df_fe[fixef].unique()
                    old_levels = _data[fixef].unique().astype(str)
                    subdict = self._fixef_dict[f"C({fixef})"] # as variables are called C(var) in the fixef_dict

                    for level in new_levels:
                        # if level estimated: either estimated value (or 0 for reference level)
                        if level in old_levels:
                            if level in subdict:
                                fixef_mat[df_fe[fixef] == level, i] = subdict[level]
                            else:
                                fixef_mat[df_fe[fixef] == level, i] = 0
                        # if new level not estimated: set to NaN
                        else:
                            fixef_mat[df_fe[fixef] == level, i] = np.nan

            else:
                fml_linear = _fml
                fml_fe = None

            # deal with the linear part
            _, X = model_matrix(fml_linear, newdata)
            X = X[self._coefnames]
            X = X.to_numpy()
            y_hat = X @ _beta_hat
            if self._has_fixef:
                y_hat += np.sum(fixef_mat, axis=1)

        return y_hat.flatten()

    def get_nobs(self):
        """
        Fetch the number of observations used in fitting the regression model.

        Params:
            None
        Returns:
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


def _drop_multicollinear_variables(
    X: np.ndarray, names: List[str], collin_tol: float
) -> None:
    """
    Checks for multicollinearity in the design matrices X and Z.

    Args:
        X (numpy.ndarray): The design matrix X.
        names (List[str]): The names of the coefficients.
        collin_tol (float): The tolerance level for the multicollinearity check.

    Returns:
        Xd (numpy.ndarray): The design matrix X.
        names (List[str]): The names of the coefficients.
        collin_vars (List[str]): The collinear variables.
        collin_index (numpy.ndarray): Logical array, True if the variable is collinear.
    """

    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = X.T @ X
    res = _find_collinear_variables(tXX, collin_tol)

    collin_vars = None
    collin_index = None

    if res["n_excl"] > 0:
        names = np.array(names)
        collin_vars = names[res["id_excl"]]
        warnings.warn(
            f"""
            The following variables are collinear: {collin_vars}.
            The variables are dropped from the model.
            """
        )

        X = np.delete(X, res["id_excl"], axis=1)
        names = np.delete(names, res["id_excl"])
        names = names.tolist()
        collin_index = res["id_excl"]

    return X, names, collin_vars, collin_index


def _find_collinear_variables(X, tol=1e-10):
    """
    Detect multicollinear variables.
    Brute force copy of Laurent's c++ implementation.
    See the fixest repo here: https://github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130
    Args:
        X (numpy.ndarray): A symmetrix matrix X.
        tol (float): The tolerance level for the multicollinearity check.
    Returns:
        res (dict): A dictionary with the following keys:
            id_excl (numpy.ndarray): A boolean array, True if the variable is collinear.
            n_excl (int): The number of collinear variables.
            all_removed (bool): True if all variables are collinear.
    """

    # import pdb; pdb.set_trace()

    res = dict()
    K = X.shape[1]
    R = np.zeros((K, K))
    id_excl = np.zeros(K, dtype=bool)
    n_excl = 0
    min_norm = X[0, 0]

    for j in range(K):
        R_jj = X[j, j]
        for k in range(j):
            if id_excl[k]:
                continue
            R_jj -= R[k, j] * R[k, j]

        if R_jj < tol:
            n_excl += 1
            id_excl[j] = True

            if n_excl == K:
                res["all_removed"] = True
                return res

            continue

        if min_norm > R_jj:
            min_norm = R_jj

        R_jj = np.sqrt(R_jj)
        R[j, j] = R_jj

        for i in range(j + 1, K):
            value = X[i, j]
            for k in range(j):
                if id_excl[k]:
                    continue
                value -= R[k, i] * R[k, j]
            R[j, i] = value / R_jj

    res["id_excl"] = id_excl
    res["n_excl"] = n_excl
    res["all_removed"] = False

    return res
