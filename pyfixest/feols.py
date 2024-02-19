import re
import warnings
from importlib import import_module
from typing import Optional, Union

import numba as nb
import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import f, norm, t

from pyfixest.dev_utils import DataFrameType, _polars_to_pandas
from pyfixest.exceptions import (
    NanInClusterVarError,
    VcovTypeNotSupportedError,
)
from pyfixest.utils import get_ssc


class Feols:
    """
    Non user-facing class to estimate an IV model using a 2SLS estimator.

    Inherits from the Feols class. Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.feols.qmd) function. Note that
    no demeaning is performed in this class: demeaning is performed in the
    [FixestMulti](/reference/estimation.fixest_multi.qmd) class (to allow for caching
    of demeaned variables for multiple estimation).

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable, a two-dimensional numpy array.
    X : np.ndarray
        Independent variables, a two-dimensional numpy array.
    weights : np.ndarray
        Weights, a one-dimensional numpy array.
    collin_tol : float
        Tolerance level for collinearity checks.
    coefnames : list[str]
        Names of the coefficients (of the design matrix X).

    Attributes
    ----------
    _method : str
        Specifies the method used for regression, set to "feols".
    _is_iv : bool
        Indicates whether instrumental variables are used, initialized as False.

    _Y : np.ndarray
        The dependent variable array.
    _X : np.ndarray
        The independent variables array.
    _X_is_empty : bool
        Indicates whether the X array is empty.
    _collin_tol : float
        Tolerance level for collinearity checks.
    _coefnames : list
        Names of the coefficients (of the design matrix X).
    _collin_vars : list
        Variables identified as collinear.
    _collin_index : list
        Indices of collinear variables.
    _Z : np.ndarray
        Alias for the _X array, used for calculations.
    _weights : np.ndarray
        Array of weights for each observation.
    _N : int
        Number of observations.
    _k : int
        Number of independent variables (or features).
    _support_crv3_inference : bool
        Indicates support for CRV3 inference.
    _support_iid_inference : bool
        Indicates support for IID inference.
    _data : Any
        Data used in the regression, to be enriched outside of the class.
    _fml : Any
        Formula used in the regression, to be enriched outside of the class.
    _has_fixef : bool
        Indicates whether fixed effects are used.
    _fixef : Any
        Fixed effects used in the regression.
    _icovars : Any
        Internal covariates, to be enriched outside of the class.
    _ssc_dict : dict
        Dictionary for sum of squares and cross products matrices.
    _tZX : np.ndarray
        Transpose of Z multiplied by X, set in get_fit().
    _tXZ : np.ndarray
        Transpose of X multiplied by Z, set in get_fit().
    _tZy : np.ndarray
        Transpose of Z multiplied by Y, set in get_fit().
    _tZZinv : np.ndarray
        Inverse of the transpose of Z multiplied by Z, set in get_fit().
    _beta_hat : np.ndarray
        Estimated regression coefficients.
    _Y_hat_link : np.ndarray
        Predicted values of the dependent variable.
    _Y_hat_response : np.ndarray
        Response predictions of the model.
    _u_hat : np.ndarray
        Residuals of the regression model.
    _scores : np.ndarray
        Scores used in the regression analysis.
    _hessian : np.ndarray
        Hessian matrix used in the regression.
    _bread : np.ndarray
        Bread matrix, used in calculating the variance-covariance matrix.
    _vcov_type : Any
        Type of variance-covariance matrix used.
    _vcov_type_detail : Any
        Detailed specification of the variance-covariance matrix type.
    _is_clustered : bool
        Indicates if clustering is used in the variance-covariance calculation.
    _clustervar : Any
        Variable used for clustering in the variance-covariance calculation.
    _G : Any
        Group information used in clustering.
    _ssc : Any
        Sum of squares and cross products matrix.
    _vcov : np.ndarray
        Variance-covariance matrix of the estimated coefficients.
    _se : np.ndarray
        Standard errors of the estimated coefficients.
    _tstat : np.ndarray
        T-statistics of the estimated coefficients.
    _pvalue : np.ndarray
        P-values associated with the t-statistics.
    _conf_int : np.ndarray
        Confidence intervals for the estimated coefficients.
    _F_stat : Any
        F-statistic for the model, set in get_Ftest().
    _fixef_dict : dict
        Dictionary containing fixed effects estimates.
    _sumFE : np.ndarray
        Sum of all fixed effects for each observation.
    _rmse : float
        Root mean squared error of the model.
    _r2 : float
        R-squared value of the model.
    _r2_within : float
        R-squared value computed on demeaned dependent variable.
    _adj_r2 : float
        Adjusted R-squared value of the model.
    _adj_r2_within : float
        Adjusted R-squared value computed on demeaned dependent variable.

    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        weights: np.ndarray,
        collin_tol: float,
        coefnames: list[str],
        weights_name: Optional[str],
    ) -> None:
        self._method = "feols"
        self._is_iv = False

        self._weights = weights
        self._weights_name = weights_name
        self._has_weights = False
        if weights_name is not None:
            self._has_weights = True

        if self._has_weights:
            w = np.sqrt(weights)
            self._Y = Y * w
            self._X = X * w
        else:
            self._Y = Y
            self._X = X

        self.get_nobs()

        _feols_input_checks(Y, X, weights)

        if self._X.shape[1] == 0:
            self._X_is_empty = True
        else:
            self._X_is_empty = False
            self._collin_tol = collin_tol
            (
                self._X,
                self._coefnames,
                self._collin_vars,
                self._collin_index,
            ) = _drop_multicollinear_variables(self._X, coefnames, self._collin_tol)

        self._Z = self._X

        self._N, self._k = self._X.shape

        self._support_crv3_inference = True
        if self._weights_name is not None:
            self._support_crv3_inference = False
        self._support_iid_inference = True
        self._supports_wildboottest = True
        if self._has_weights or self._is_iv:
            self._supports_wildboottest = False

        # attributes that have to be enriched outside of the class -
        # not really optimal code change later
        self._data = None
        self._fml = None
        self._has_fixef = False
        self._fixef = None
        # self._coefnames = None
        self._icovars = None
        self._ssc_dict = None

        # set in get_fit()
        self._tZX = None
        # self._tZXinv = None
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
        Fit an OLS model.

        Returns
        -------
        None
        """
        _X = self._X
        _Y = self._Y
        _Z = self._Z

        self._tZX = _Z.T @ _X
        self._tZy = _Z.T @ _Y

        # self._tZXinv = np.linalg.inv(self._tZX)
        self._beta_hat = np.linalg.solve(self._tZX, self._tZy).flatten()
        # self._beta_hat, _, _, _ = lstsq(self._tZX, self._tZy, lapack_driver='gelsy')

        # self._beta_hat = (self._tZXinv @ self._tZy).flatten()

        self._Y_hat_link = self._X @ self._beta_hat
        self._u_hat = self._Y.flatten() - self._Y_hat_link.flatten()

        self._scores = self._u_hat[:, None] * _X
        self._hessian = self._tZX.copy()

        # IV attributes, set to None for OLS, Poisson
        self._tXZ = None
        self._tZZinv = None

    def vcov(self, vcov: Union[str, dict[str, str]]):
        """
        Compute covariance matrices for an estimated regression model.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]]
            A string or dictionary specifying the type of variance-covariance matrix
            to use for inference.
            If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format {"CRV1": "clustervar"} for
            CRV1 inference or {"CRV3": "clustervar"}
            for CRV3 inference. Note that CRV3 inference is currently not supported
            for IV estimation.

        Returns
        -------
        Feols
            An instance of class [Feols(/reference/Feols.qmd) with updated inference.
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
        # _tZXinv = self._tZXinv
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

            if self._method == "feols":
                sigma2 = np.sum(_u_hat.flatten() ** 2) / (_N - 1)
            elif self._method == "fepois":
                sigma2 = 1
            else:
                raise NotImplementedError(
                    f"'iid' inference is not supported for {_method} regressions."
                )

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
                _tZXinv = np.linalg.inv(_tZX)
                leverage = np.sum(_X * (_X @ _tZXinv), axis=1)
                if self._vcov_type_detail == "HC2":
                    u = _u_hat / np.sqrt(1 - leverage)
                    transformed_scores = _scores / np.sqrt(1 - leverage)[:, None]
                else:
                    transformed_scores = _scores / (1 - leverage)[:, None]

            if _is_iv is False:
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
            if cluster_df.isna().any().any():
                raise NanInClusterVarError(
                    "CRV inference not supported with missing values in the cluster variable."
                    "Please drop missing values before running the regression."
                )

            if cluster_df.shape[1] > 1:
                # paste both columns together
                # set cluster_df to string
                cluster_df = cluster_df.astype(str)
                cluster_df["cluster_intersection"] = cluster_df.iloc[:, 0].str.cat(
                    cluster_df.iloc[:, 1], sep="-"
                )

            G = []
            for col in cluster_df.columns:
                G.append(cluster_df[col].nunique())

            if _ssc_dict["cluster_df"] == "min":
                G = [min(G)] * 3

            # loop over columns of cluster_df
            vcov_sign_list = [1, 1, -1]
            self._ssc = []
            self._G = G

            self._vcov = np.zeros((self._k, self._k))

            for x, col in enumerate(cluster_df.columns):
                cluster_col_pd = cluster_df[col]
                cluster_col, _ = pd.factorize(cluster_col_pd)
                clustid = np.unique(cluster_col)

                ssc = get_ssc(
                    ssc_dict=_ssc_dict,
                    N=_N,
                    k=_k,
                    G=self._G[x],
                    vcov_sign=vcov_sign_list[x],
                    vcov_type="CRV",
                )

                self._ssc.append(ssc)

                if self._vcov_type_detail == "CRV1":
                    k_instruments = _Z.shape[1]
                    meat = np.zeros((k_instruments, k_instruments))

                    # import pdb; pdb.set_trace()
                    # deviance uniquely for Poisson
                    if hasattr(self, "deviance"):
                        weighted_uhat = _weights.flatten() * _u_hat.flatten()
                    else:
                        weighted_uhat = _u_hat

                    meat = _crv1_meat_loop(
                        _Z=_Z.astype(np.float64),
                        weighted_uhat=weighted_uhat.astype(np.float64).reshape((_N, 1)),
                        clustid=clustid,
                        cluster_col=cluster_col,
                    )

                    if _is_iv is False:
                        self._vcov += self._ssc[x] * bread @ meat @ bread
                    else:
                        meat = _tXZ @ _tZZinv @ meat @ _tZZinv @ self._tZX
                        self._vcov += self._ssc[x] * bread @ meat @ bread

                elif self._vcov_type_detail == "CRV3":
                    # check: is fixed effect cluster fixed effect?
                    # if not, either error or turn fixefs into dummies
                    # for now: don't allow for use with fixed effects

                    if not _support_crv3_inference:
                        raise VcovTypeNotSupportedError(
                            "CRV3 inference is not supported with IV regression or WLS."
                        )

                    beta_jack = np.zeros((len(clustid), _k))

                    if (
                        (self._has_fixef is False)
                        and (self._method == "feols")
                        and (_is_iv is False)
                    ):
                        # inverse hessian precomputed?
                        tXX = np.transpose(self._X) @ self._X
                        tXy = np.transpose(self._X) @ self._Y

                        # compute leave-one-out regression coefficients (aka clusterjacks')  # noqa: W505
                        for ixg, g in enumerate(clustid):
                            Xg = self._X[np.equal(g, cluster_col)]
                            Yg = self._Y[np.equal(g, cluster_col)]
                            tXgXg = np.transpose(Xg) @ Xg
                            # jackknife regression coefficient
                            beta_jack[ixg, :] = (
                                np.linalg.pinv(tXX - tXgXg)
                                @ (tXy - np.transpose(Xg) @ Yg)
                            ).flatten()

                    else:
                        # lazy loading to avoid circular import
                        fixest_module = import_module("pyfixest.estimation")
                        if self._method == "feols":
                            fit_ = getattr(fixest_module, "feols")
                        else:
                            fit_ = getattr(fixest_module, "fepois")

                        for ixg, g in enumerate(clustid):
                            # direct leave one cluster out implementation
                            data = _data[~np.equal(g, cluster_col)]
                            fit = fit_(fml=self._fml, data=data, vcov="iid")
                            beta_jack[ixg, :] = fit.coef().to_numpy()

                    # optional: beta_bar in MNW (2022)
                    # center = "estimate"
                    # if center == 'estimate':
                    #    beta_center = beta_hat
                    # else:
                    #    beta_center = np.mean(beta_jack, axis = 0)
                    beta_center = _beta_hat

                    vcov = np.zeros((_k, _k))
                    for ixg, g in enumerate(clustid):
                        beta_centered = beta_jack[ixg, :] - beta_center
                        vcov += np.outer(beta_centered, beta_centered)

                    self._vcov += self._ssc[x] * vcov

        self.get_inference()

        return self

    def get_inference(self, alpha: float = 0.95) -> None:
        """
        Compute standard errors, t-statistics, and p-values for the regression model.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.95.

        Returns
        -------
        None
        """
        _vcov = self._vcov
        _beta_hat = self._beta_hat
        _vcov_type = self._vcov_type
        _N = self._N
        _k = self._k
        _G = np.min(np.array(self._G))  # fixest default
        _method = self._method

        self._se = np.sqrt(np.diagonal(_vcov))
        self._tstat = _beta_hat / self._se

        df = _N - _k if _vcov_type in ["iid", "hetero"] else _G - 1

        # use t-dist for linear models, but normal for non-linear models
        if _method == "feols":
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), df))
            z = np.abs(t.ppf((1 - alpha) / 2, df))
        else:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf((1 - alpha) / 2))

        z_se = z * self._se
        self._conf_int = np.array([_beta_hat - z_se, _beta_hat + z_se])

    def add_fixest_multi_context(
        self,
        fml: str,
        depvar: str,
        Y: pd.Series,
        _data: pd.DataFrame,
        _ssc_dict: dict,
        _k_fe: int,
        fval: str,
        na_index: np.ndarray,
    ) -> None:
        """
        Enrich Feols object.

        Enrich an instance of `Feols` Class with additional
        attributes set in the `FixestMulti` class.

        Parameters
        ----------
        fml : str
            The formula used for estimation.
        depvar : str
            The dependent variable of the regression model.
        Y : pd.Series
            The dependent variable of the regression model.
        _data : pd.DataFrame
            The data used for estimation.
        _ssc_dict : dict
            A dictionary with the sum of squares and cross products matrices.
        _k_fe : int
            The number of fixed effects.
        fval : str
            The fixed effects formula.
        na_index : np.ndarray
            An array with the indices of missing values.

        Returns
        -------
        None
        """
        # some bookkeeping
        self._fml = fml
        self._depvar = depvar
        self._Y_untransformed = Y
        self._data = _data.iloc[~_data.index.isin(na_index)]
        self._ssc_dict = _ssc_dict
        self._k_fe = _k_fe
        if fval != "0":
            self._has_fixef = True
            self._fixef = fval
        else:
            self._has_fixef = False
            self._fixef = None

    def wald_test(self, R=None, q=None, distribution="F") -> None:
        """
        Conduct Wald test.

        Compute a Wald test for a linear hypothesis of the form Rb = q.
        By default, tests the joint null hypothesis that all coefficients are zero.

        Parameters
        ----------
        R : array-like, optional
            The matrix R of the linear hypothesis.
            If None, defaults to an identity matrix.
        q : array-like, optional
            The vector q of the linear hypothesis.
            If None, defaults to a vector of zeros.
        distribution : str, optional
            The distribution to use for the p-value. Can be either "F" or "chi2".
            Defaults to "F".

        Returns
        -------
        pd.Series
            A pd.Series with the Wald statistic and p-value.
        """
        raise ValueError("wald_tests will be released as a feature with pyfixest 0.14.")

        _vcov = self._vcov
        _N = self._N
        _k = self._k
        _beta_hat = self._beta_hat
        _k_fe = np.sum(self._k_fe.values) if self._has_fixef else 0

        dfn = _N - _k_fe - _k
        dfd = _k

        # if R is not two dimensional, make it two dimensional
        if R is not None:
            if R.ndim == 1:
                R = R.reshape((1, len(R)))
            assert (
                R.shape[1] == _k
            ), "R must have the same number of columns as the number of coefficients."
        else:
            R = np.eye(_k)

        if q is not None:
            assert isinstance(
                q, (int, float, np.ndarray)
            ), "q must be a numeric scalar."
            if isinstance(q, np.ndarray):
                assert q.ndim == 1, "q must be a one-dimensional array or a scalar."
                assert (
                    q.shape[0] == R.shape[0]
                ), "q must have the same number of rows as R."
            warnings.warn(
                "Note that the argument q is experimental and no unit tests are implemented. Please use with caution / take a look at the source code."
            )
        else:
            q = np.zeros(R.shape[0])

        assert distribution in [
            "F",
            "chi2",
        ], "distribution must be either 'F' or 'chi2'."

        bread = R @ _beta_hat - q
        meat = np.linalg.inv(R @ _vcov @ R.T)
        W = bread.T @ meat @ bread

        # this is chi-squared(k) distributed, with k = number of coefficients
        self._wald_statistic = W
        self._f_statistic = W / dfd

        if distribution == "F":
            self._f_statistic_pvalue = f.sf(self._f_statistic, dfn=dfn, dfd=dfd)
            # self._f_statistic_pvalue = 1 - chi2(df = _k).cdf(self._f_statistic)
            res = pd.Series(
                {"statistic": self._f_statistic, "pvalue": self._f_statistic_pvalue}
            )
        else:
            raise NotImplementedError("chi2 distribution not yet implemented.")
            # self._wald_pvalue = 1 - chi2(df = _k).cdf(self._wald_statistic)

        return res

    def coefplot(
        self,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (500, 300),
        yintercept: Optional[float] = 0,
        xintercept: Optional[float] = None,
        rotate_xticks: int = 0,
        coefficients: Optional[list[str]] = None,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Create a coefficient plot to visualize model coefficients.

        Parameters
        ----------
        alpha : float, optional
            Significance level for highlighting significant coefficients.
            Defaults to None.
        figsize : tuple[int, int], optional
            Size of the plot (width, height) in inches. Defaults to None.
        yintercept : float, optional
            Value to set as the y-axis intercept (vertical line). Defaults to None.
        xintercept : float, optional
            Value to set as the x-axis intercept (horizontal line). Defaults to None.
        rotate_xticks : int, optional
            Rotation angle for x-axis tick labels. Defaults to None.
        coefficients : list[str], optional
            List of coefficients to include in the plot.
            If None, all coefficients are included.
        title : str, optional
            Title of the plot. Defaults to None.
        coord_flip : bool, optional
            Whether to flip the coordinates of the plot. Defaults to None.

        Returns
        -------
        lets-plot figure
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
        figsize: tuple[int, int] = (500, 300),
        yintercept: Optional[float] = None,
        xintercept: Optional[float] = None,
        rotate_xticks: int = 0,
        title: Optional[str] = None,
        coord_flip: Optional[bool] = True,
    ):
        """
        Create coefficient plots for variables interacted via `i()` syntax.

        Parameters
        ----------
        alpha : float, optional
            Significance level for visualization options. Defaults to 0.05.
        figsize : tuple[int, int], optional
            Size of the plot (width, height) in inches. Defaults to (500, 300).
        yintercept : float, optional
            Value to set as the y-axis intercept (vertical line). Defaults to None.
        xintercept : float, optional
            Value to set as the x-axis intercept (horizontal line). Defaults to None.
        rotate_xticks : int, optional
            Rotation angle for x-axis tick labels. Defaults to 0.
        title : str, optional
            Title of the plot. Defaults to None.
        coord_flip : bool, optional
            Whether to flip the coordinates of the plot. Defaults to True.

        Returns
        -------
        lets-plot figure
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
        return_bootstrapped_t_stats=False,
    ):
        """
        Run a wild cluster bootstrap based on an object of type "Feols".

        Parameters
        ----------
        B : int
            The number of bootstrap iterations to run.
        cluster : Union[None, np.ndarray, pd.Series, pd.DataFrame], optional
            If None (default), checks if the model's vcov type was CRV.
            If yes, uses `self._clustervar` as cluster. If None and no clustering
            was employed in the initial model, runs a heteroskedastic wild bootstrap.
            If an argument is supplied, it is used as the cluster variable for the
            wild cluster bootstrap. Requires a numpy array of dimension one, a
            pandas Series, or DataFrame, containing the clustering variable.
        param : Union[str, None], optional
            A string of length one, containing the test parameter of interest.
            Defaults to None.
        weights_type : str, optional
            The type of bootstrap weights. Options are 'rademacher', 'mammen',
            'webb', or 'normal'. Defaults to 'rademacher'.
        impose_null : bool, optional
            Indicates whether to impose the null hypothesis on the bootstrap DGP.
            Defaults to True.
        bootstrap_type : str, optional
            A string of length one to choose the bootstrap type.
            Options are '11', '31', '13', or '33'. Defaults to '11'.
        seed : Union[int, None], optional
            An option to provide a random seed. Defaults to None.
        adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of observations and covariates. Defaults to True.
        cluster_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of clusters. Defaults to True.
        parallel : bool, optional
            Indicates whether to run the bootstrap in parallel. Defaults to False.
        seed : Union[str, None], optional
            An option to provide a random seed. Defaults to None.
        return_bootstrapped_t_stats : bool, optional:
            If True, the method returns a tuple of the regular output and the
            bootstrapped t-stats. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the original, non-bootstrapped t-statistic and
            bootstrapped p-value, along with the bootstrap type, inference type
            (HC vs CRV), and whether the null hypothesis was imposed on the
            bootstrap DGP. If `return_bootstrapped_t_stats` is True, the method
            returns a tuple of the regular output and the bootstrapped t-stats.
        """
        _is_iv = self._is_iv
        _has_fixef = self._has_fixef
        _Y = self._Y.flatten()
        _X = self._X
        _xnames = self._coefnames
        _data = self._data
        _clustervar = self._clustervar
        _supports_wildboottest = self._supports_wildboottest

        if param is not None and param not in _xnames:
            raise ValueError(
                f"Parameter {param} not found in the model's coefficients."
            )

        if not _supports_wildboottest:
            if self._is_iv:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for IV estimation."
                )
            if self._has_weights:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for WLS estimation."
                )

        if cluster is None and _clustervar is not None:
            cluster = _clustervar

        if isinstance(cluster, str):
            cluster = [cluster]

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
            inference = f"CRV({cluster})"

            if len(cluster) > 1:
                raise ValueError(
                    "Multiway clustering is currently not supported with the wild cluster bootstrap."
                )

            cluster = _data[cluster[0]]

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

        if return_bootstrapped_t_stats:
            return res_df, boot.t_boot
        else:
            return res_df

    def fixef(self) -> None:
        """
        Compute the coefficients of (swept out) fixed effects for a regression model.

        This method creates the following attributes:
        - `alphaDF` (pd.DataFrame): A DataFrame with the estimated fixed effects.
        - `sumFE` (np.array): An array with the sum of fixed effects for each
        observation (i = 1, ..., N).

        Returns
        -------
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
        if self._X_is_empty:
            Y = Y.to_numpy()
            uhat = Y

        else:
            X = X[self._coefnames]  # drop intercept, potentially multicollinear vars
            Y = Y.to_numpy().flatten().astype(np.float64)
            X = X.to_numpy()
            uhat = csr_matrix(Y - X @ self._beta_hat).transpose()

        D2 = model_matrix("-1+" + fixef_fml, _data, output="sparse")
        cols = D2.model_spec.column_names

        alpha = spsolve(D2.transpose() @ D2, D2.transpose() @ uhat)

        res = {}
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

    def predict(self, newdata: Optional[DataFrameType] = None) -> np.ndarray:  # type: ignore
        """
        Predict values of the model on new data.

        Return a flat np.array with predicted values of the regression model.
        If new fixed effect levels are introduced in `newdata`, predicted values
        for such observations will be set to NaN.

        Parameters
        ----------
        newdata : Optional[DataFrameType], optional
            A pd.DataFrame or pl.DataFrame with the data to be used for prediction.
            If None (default), the data used for fitting the model is used.

        Returns
        -------
        y_hat : np.ndarray
            A flat np.array with predicted values of the regression model.
        """
        _fml = self._fml
        _data = self._data
        _u_hat = self._u_hat
        _beta_hat = self._beta_hat
        _is_iv = self._is_iv

        _Y_untransformed = self._Y_untransformed.values.flatten()

        if _is_iv:
            raise NotImplementedError(
                "The predict() method is currently not supported for IV models."
            )

        if newdata is None:
            y_hat = _Y_untransformed - _u_hat.flatten()

        else:
            newdata = _polars_to_pandas(newdata).reset_index(drop=False)

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
                    subdict = self._fixef_dict[
                        f"C({fixef})"
                    ]  # as variables are called C(var) in the fixef_dict

                    for level in new_levels:
                        # if level estimated: either estimated value (or 0 for reference level)  # noqa: W505
                        if level in old_levels:
                            fixef_mat[df_fe[fixef] == level, i] = subdict.get(level, 0)
                        # if new level not estimated: set to NaN
                        else:
                            fixef_mat[df_fe[fixef] == level, i] = np.nan

            else:
                fml_linear = _fml  # noqa: F841
                fml_fe = None  # noqa: F841

            if not self._X_is_empty:
                # deal with linear part
                xfml = _fml.split("|")[0].split("~")[1]
                X = model_matrix(xfml, newdata)
                X_index = X.index
                coef_idx = np.isin(self._coefnames, X.columns)
                X = X[np.array(self._coefnames)[coef_idx]]
                X = X.to_numpy()
                # fill y_hat with np.nans
                y_hat = np.full(newdata.shape[0], np.nan)
                y_hat[X_index] = X @ _beta_hat[coef_idx]

            else:
                y_hat = np.zeros(newdata.shape[0])

            if self._has_fixef:
                y_hat += np.sum(fixef_mat, axis=1)

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
        Get Goodness-of-Fit measures.

        Compute multiple additional measures commonly reported with linear
        regression output, including R-squared and adjusted R-squared. Note that
        variables with the suffix _within use demeaned dependent variables Y,
        while variables without do not or are invariant to demeaning.

        Returns
        -------
        None

        Creates the following instances:
        - r2 (float): R-squared of the regression model.
        - adj_r2 (float): Adjusted R-squared of the regression model.
        - r2_within (float): R-squared of the regression model, computed on
        demeaned dependent variable.
        - adj_r2_within (float): Adjusted R-squared of the regression model,
        computed on demeaned dependent variable.
        """
        _Y_within = self._Y
        _Y = self._Y_untransformed.values
        _u_hat = self._u_hat
        _N = self._N
        _k = self._k
        _has_fixef = self._has_fixef
        _weights = self._weights
        _has_weights = self._has_weights

        if _has_fixef:
            _k_fe = np.sum(self._k_fe.values - 1) + 1
            _adj_factor = (_N - _k_fe) / (_N - _k - _k_fe)
        else:
            _adj_factor = (_N) / (_N - 1)

        ssu = np.sum(_u_hat**2)
        ssy = np.sum((_Y - np.mean(_Y)) ** 2)

        if _has_weights:
            self._rmse = None
            self._r2 = None
            self._adj_r2 = None
        else:
            self._rmse = np.sqrt(ssu / _N)
            self._r2 = 1 - (ssu / ssy)
            self._adj_r2 = 1 - (ssu / ssy) * _adj_factor

        if _has_fixef:
            ssy_within = np.sum((_Y_within - np.mean(_Y_within)) ** 2)
            self._r2_within = 1 - (ssu / ssy_within)
            self._r2_adj_within = 1 - (ssu / ssy_within) * _adj_factor
        else:
            self._r2_within = np.nan
            self._adj_r2_within = np.nan

        # overwrite self._adj_r2 and self._adj_r2_within
        # reason: currently I cannot match fixest dof correction, so
        # better not to report it
        self._adj_r2 = None
        self._adj_r2_within = None

    def tidy(self) -> pd.DataFrame:
        """
        Tidy model outputs.

        Return a tidy pd.DataFrame with the point estimates, standard errors,
        t-statistics, and p-values.

        Returns
        -------
        tidy_df : pd.DataFrame
            A tidy pd.DataFrame containing the regression results, including point
            estimates, standard errors, t-statistics, and p-values.
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
        Fitted model coefficents.

        Returns
        -------
        pd.Series
            A pd.Series with the estimated coefficients of the regression model.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Fitted model standard errors.

        Returns
        -------
        pd.Series
            A pd.Series with the standard errors of the estimated regression model.
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Fitted model t-statistics.

        Returns
        -------
        pd.Series
            A pd.Series with t-statistics of the estimated regression model.
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Fitted model p-values.

        Returns
        -------
        pd.Series
            A pd.Series with p-values of the estimated regression model.
        """
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        """
        Fitted model confidence intervals.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame with confidence intervals of the estimated regression model.
        """
        return self.tidy()[["2.5 %", "97.5 %"]]

    def resid(self) -> np.ndarray:
        """
        Fitted model residuals.

        Returns
        -------
        np.ndarray
            A np.ndarray with the residuals of the estimated regression model.
        """
        return self._u_hat

    def summary(self, digits=3) -> None:
        """
        Summary of estimated model.

        Parameters
        ----------
        digits : int, optional
            The number of digits to be displayed. Defaults to 3.

        Returns
        -------
        None
        """
        # lazy loading to avoid circular import
        summarize_module = import_module("pyfixest.summarize")
        _summary = getattr(summarize_module, "summary")

        return _summary(models=self, digits=digits)


def _check_vcov_input(vcov, data):
    """
    Check the input for the vcov argument in the Feols class.

    Parameters
    ----------
    vcov : dict, str, list
        The vcov argument passed to the Feols class.
    data : pd.DataFrame
        The data passed to the Feols class.

    Returns
    -------
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
        deparse_vcov = list(vcov.values())[0].split("+")
        assert all(
            col.replace(" ", "") in data.columns for col in deparse_vcov
        ), "vcov dict value must be a column in the data"

        assert len(deparse_vcov) <= 2, "not more than twoway clustering is supported"

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

    Parameters
    ----------
    vcov : dict, str, list
        The vcov argument passed to the Feols class.
    has_fixef : bool
        Whether the regression has fixed effects.
    is_iv : bool
        Whether the regression is an IV regression.

    Returns
    -------
    vcov_type : str
        The type of vcov to be used. Either "iid", "hetero", or "CRV".
    vcov_type_detail : str or list
        The type of vcov to be used, with more detail. Options include "iid",
        "hetero", "HC1", "HC2", "HC3", "CRV1", or "CRV3".
    is_clustered : bool
        Indicates whether the vcov is clustered.
    clustervar : str
        The name of the cluster variable.
    """
    if isinstance(vcov, dict):
        vcov_type_detail = list(vcov.keys())[0]
        deparse_vcov = list(vcov.values())[0].split("+")
        if isinstance(deparse_vcov, str):
            deparse_vcov = [deparse_vcov]
        deparse_vcov = [x.replace(" ", "") for x in deparse_vcov]
    elif isinstance(vcov, (list, str)):
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

    clustervar = deparse_vcov if is_clustered else None

    return vcov_type, vcov_type_detail, is_clustered, clustervar


def _feols_input_checks(Y, X, weights):
    """
    Perform basic checks on the input matrices Y and X for the FEOLS.

    Parameters
    ----------
    Y : np.ndarray
        FEOLS input matrix Y.
    X : np.ndarray
        FEOLS input matrix X.

    Returns
    -------
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
    Get variance-covariance matrix type.

    Passes the specified vcov type and sets the default vcov type based on the
    inclusion of fixed effects in the model.

    Parameters
    ----------
    vcov : str
        The specified vcov type.
    fval : str
        The specified fixed effects, formatted as a string (e.g., "X1+X2").

    Returns
    -------
    vcov_type : str
        The specified or default vcov type. Defaults to 'iid' if no fixed effect
        is included in the model, and 'CRV1' clustered by the first fixed effect
        if a fixed effect is included.
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
    X: np.ndarray, names: list[str], collin_tol: float
) -> None:
    """
    Check for multicollinearity in the design matrices X and Z.

    Parameters
    ----------
    X : numpy.ndarray
        The design matrix X.
    names : list[str]
        The names of the coefficients.
    collin_tol : float
        The tolerance level for the multicollinearity check.

    Returns
    -------
    Xd : numpy.ndarray
        The design matrix X after checking for multicollinearity.
    names : list[str]
        The names of the coefficients, excluding those identified as collinear.
    collin_vars : list[str]
        The collinear variables identified during the check.
    collin_index : numpy.ndarray
        Logical array, where True indicates that the variable is collinear.
    """
    # TODO: avoid doing this computation twice, e.g. compute tXXinv here as fixest does

    tXX = X.T @ X
    res = _find_collinear_variables(tXX, collin_tol)

    collin_vars = None
    collin_index = None

    if res["all_removed"]:
        raise ValueError(
            """
            All variables are collinear. Maybe your model specification introduces multicollinearity? If not, please reach out to the package authors!.
            """
        )

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

    Detect multicollinear variables, replicating Laurent Berge's C++ implementation
    from the fixest package. See the fixest repo [here](https://github.com/lrberge/fixest/blob/a4d1a9bea20aa7ab7ab0e0f1d2047d8097971ad7/src/lm_related.cpp#L130)

    Parameters
    ----------
    X : numpy.ndarray
        A symmetric matrix X used to check for multicollinearity.
    tol : float
        The tolerance level for the multicollinearity check.

    Returns
    -------
    res : dict
        A dictionary containing:
        - id_excl (numpy.ndarray): A boolean array, where True indicates a collinear
        variable.
        - n_excl (int): The number of collinear variables.
        - all_removed (bool): True if all variables are identified as collinear.
    """
    res = {}
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


# CODE from Styfen Schaer (@styfenschaer)
@nb.njit(parallel=False)
def bucket_argsort(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sorts the input array using the bucket sort algorithm.

    Parameters
    ----------
    arr : array_like
        An array_like object that needs to be sorted.

    Returns
    -------
    array_like
        A sorted copy of the input array.

    Raises
    ------
    ValueError
        If the input is not an array_like object.

    Notes
    -----
    The bucket sort algorithm works by distributing the elements of an array
    into a number of buckets. Each bucket is then sorted individually, either
    using a different sorting algorithm, or by recursively applying the bucket
    sorting algorithm.
    """
    counts = np.zeros(arr.max() + 1, dtype=np.uint32)
    for i in range(arr.size):
        counts[arr[i]] += 1

    locs = np.empty(counts.size + 1, dtype=np.uint32)
    locs[0] = 0
    pos = np.empty(counts.size, dtype=np.uint32)
    for i in range(counts.size):
        locs[i + 1] = locs[i] + counts[i]
        pos[i] = locs[i]

    args = np.empty(arr.size, dtype=np.uint32)
    for i in range(arr.size):
        e = arr[i]
        args[pos[e]] = i
        pos[e] += 1

    return args, locs


# CODE from Styfen Schaer (@styfenschaer)
@nb.njit(parallel=False)
def _crv1_meat_loop(
    _Z: np.ndarray,
    weighted_uhat: np.ndarray,
    clustid: np.ndarray,
    cluster_col: np.ndarray,
) -> np.ndarray:
    k = _Z.shape[1]
    dtype = _Z.dtype
    meat = np.zeros((k, k), dtype=dtype)

    g_indices, g_locs = bucket_argsort(cluster_col)

    score_g = np.empty((k, 1), dtype=dtype)
    meat_i = np.empty((k, k), dtype=dtype)

    for i in range(clustid.size):
        g = clustid[i]
        start = g_locs[g]
        end = g_locs[g + 1]
        g_index = g_indices[start:end]

        Zg = _Z[g_index]
        ug = weighted_uhat[g_index]

        np.dot(Zg.T, ug, out=score_g)
        np.outer(score_g, score_g, out=meat_i)
        meat += meat_i

    return meat
