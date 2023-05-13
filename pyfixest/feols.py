import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from wildboottest.wildboottest import WildboottestCL, WildboottestHC
from importlib import import_module
from typing import Union, List, Dict
from scipy.stats import norm, t
from pyfixest.ssc_utils import get_ssc


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
    Z: Union[np.ndarray, pd.DataFrame]
        Instruments of the regression.

    Attributes
    ----------
    Y : np.ndarray
        The dependent variable of the regression.
    X : np.ndarray
        The independent variable of the regression.
    Z : np.ndarray
        The instruments of the regression.
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

    def __init__(self, Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> None:

        if not isinstance(Y, (np.ndarray)):
            raise TypeError("Y must be a numpy array.")
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be a numpy array.")

        self.Y = Y
        self.X = X
        self.Z = Z

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.Z.ndim != 2:
            raise ValueError("Z must be a 2D array")

        self.N, self.k = X.shape

    def get_fit(self) -> None:
        '''
        Regression estimation for a single model, via ordinary least squares (OLS).
        '''

        self.tZX = np.transpose(self.Z) @ self.X
        self.tZXinv = np.linalg.inv(self.tZX)

        self.tZy = (np.transpose(self.Z) @ self.Y)
        beta_hat = self.tZXinv @ self.tZy
        self.beta_hat = beta_hat.flatten()
        self.Y_hat = (self.X @ self.beta_hat)
        self.u_hat = (self.Y.flatten() - self.Y_hat)

    def get_vcov(self, vcov: Union[str, Dict[str, str], List[str]]) -> None:
        '''
        Compute covariance matrices for an estimated regression model.

        Parameters
        ----------
        vcov : Union[str, Dict[str, str], List[str]]
            A string or dictionary specifying the type of variance-covariance matrix to use for inference.
            If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference
            or {"CRV3":"clustervar"} for CRV3 inference.

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

        '''

        assert isinstance(vcov, (dict, str, list)), "vcov must be a dict, string or list"
        if isinstance(vcov, dict):
            assert list(vcov.keys())[0] in ["CRV1", "CRV3"], "vcov dict key must be CRV1 or CRV3"
            assert isinstance(list(vcov.values())[0], str), "vcov dict value must be a string"
            assert list(vcov.values())[0] in self.data.columns, "vcov dict value must be a column in the data"
        if isinstance(vcov, list):
            assert all(isinstance(v, str) for v in vcov), "vcov list must contain strings"
            assert all(v in self.data.columns for v in vcov), "vcov list must contain columns in the data"
        if isinstance(vcov, str):
            assert vcov in ["iid", "hetero", "HC1", "HC2", "HC3"], "vcov string must be iid, hetero, HC1, HC2, or HC3"

        if isinstance(vcov, dict):
            vcov_type_detail = list(vcov.keys())[0]
            self.clustervar = list(vcov.values())[0]
        elif isinstance(vcov, list):
            vcov_type_detail = vcov
        elif isinstance(vcov, str):
            vcov_type_detail = vcov
        else:
            assert False, "arg vcov needs to be a dict, string or list"

        if vcov_type_detail == "iid":
            self.vcov_type = "iid"
            self.is_clustered = False
        elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
            self.vcov_type = "hetero"
            self.is_clustered = False
            if vcov_type_detail in ["HC2", "HC3"]:
                if self.has_fixef:
                    raise ValueError("HC2 and HC3 inference types are not supported for regressions with fixed effects.")
        elif vcov_type_detail in ["CRV1", "CRV3"]:
            self.vcov_type = "CRV"
            self.is_clustered = True


        if self.is_iv:
            if self.vcov_type in ["CRV3"]:
                raise ValueError("CRV3 inference is not supported for IV regressions.")

        # compute vcov
        if self.vcov_type == 'iid':

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = 1,
                vcov_sign = 1,
                vcov_type='iid'
            )

            # only relevant factor for iid in ssc: fixef.K
            if self.is_iv == False:
                self.vcov =  self.ssc * self.tZXinv * (np.sum(self.u_hat ** 2) / (self.N - 1))
            else:
                sigma2 = (np.sum(self.u_hat ** 2) / (self.N - 1))
                tZZinv = np.linalg.inv(np.transpose(self.Z) @ self.Z) # k x k
                tXZ = np.transpose(self.X) @ self.Z
                self.vcov = self.ssc * np.linalg.inv(tXZ @ tZZinv @ self.tZX ) * sigma2 #

        elif self.vcov_type == 'hetero':

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = 1,
                vcov_sign = 1,
                vcov_type = "hetero"
            )

            if vcov_type_detail in ["hetero", "HC1"]:
                u = self.u_hat
            elif vcov_type_detail in ["HC2", "HC3"]:
                leverage = np.sum(self.X * (self.X @ self.tZXinv), axis=1)
                if vcov_type_detail == "HC2":
                    u = self.u_hat / np.sqrt(1 - leverage)
                else:
                    u = self.u_hat / (1-leverage)

            if self.is_iv == False:
                meat = np.transpose(self.Z) * (u ** 2) @ self.Z
                # set off diagonal elements to zero
                self.vcov =  self.ssc * self.tZXinv @ meat @  self.tZXinv
            else:
                tZZinv = np.linalg.inv(np.transpose(self.Z) @ self.Z)  # k x k
                tXZ = np.transpose(self.X) @ self.Z # k x k
                if u.ndim == 1:
                    u = u.reshape((self.N,1))
                Omega = np.transpose(self.Z) @ (self.Z * (u ** 2))  # k x k
                meat = tXZ @ tZZinv  @ Omega  @ tZZinv @ self.tZX # k x k
                bread = np.linalg.inv(tXZ @ tZZinv @ self.tZX)
                self.vcov = self.ssc * bread @ meat @ bread



        elif self.vcov_type == "CRV":

            cluster_df = self.data[self.clustervar]
            # if there are missings - delete them!

            if cluster_df.dtype != "category":
                cluster_df = pd.Categorical(cluster_df)

            if cluster_df.isna().any():
                raise ValueError("CRV inference not supported with missing values in the cluster variable. Please drop missing values before running the regression.")

            cluster_mat, clustid = pd.factorize(cluster_df)
            #cluster_mat = cluster_df.to_numpy()

            #clustid = np.unique(cluster_mat)
            self.G = len(clustid)

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = self.G,
                vcov_sign = 1,
                vcov_type = "CRV"
            )

            if vcov_type_detail == "CRV1":


                meat = np.zeros((self.k, self.k))

                for _, g, in enumerate(clustid):

                    Zg = self.Z[np.where(cluster_df == g)]
                    ug = self.u_hat[np.where(cluster_df == g)]
                    score_g = (np.transpose(Zg) @ ug).reshape((self.k, 1))
                    meat += np.dot(score_g, score_g.transpose())

                if self.is_iv == False:
                    self.vcov = self.ssc * self.tZXinv @ meat @ self.tZXinv
                else:
                    tZZinv = np.linalg.inv(np.transpose(self.Z) @ self.Z)  # k x k
                    tXZ = np.transpose(self.X) @ self.Z # k x k
                    meat = tXZ @ tZZinv @ meat @ tZZinv @ self.tZX
                    bread = np.linalg.inv(tXZ @ tZZinv @ self.tZX)
                    self.vcov = self.ssc * bread @ meat @ bread

            elif vcov_type_detail == "CRV3":

                # check: is fixed effect cluster fixed effect?
                # if not, either error or turn fixefs into dummies
                # for now: don't allow for use with fixed effects

                #if self.has_fixef:
                #    raise ValueError("CRV3 inference is currently not supported with fixed effects.")

                if self.is_iv:
                    raise ValueError("CRV3 inference is not supported with IV estimation.")

                k_params = self.k

                beta_hat = self.beta_hat

                clusters = clustid
                n_groups = self.G
                group = cluster_df

                beta_jack = np.zeros((n_groups, k_params))


                if self.has_fixef == False:
                    # inverse hessian precomputed?
                    tXX = np.transpose(self.X) @ self.X
                    tXy = np.transpose(self.X) @ self.Y

                    # compute leave-one-out regression coefficients (aka clusterjacks')
                    for ixg, g in enumerate(clusters):

                        Xg = self.X[np.equal(ixg, group)]
                        Yg = self.Y[np.equal(ixg, group)]
                        tXgXg = np.transpose(Xg) @ Xg
                        # jackknife regression coefficient
                        beta_jack[ixg,:] = (
                            np.linalg.pinv(tXX - tXgXg) @ (tXy - np.transpose(Xg) @ Yg)
                        ).flatten()

                else:

                    # lazy loading to avoid circular import
                    fixest_module = import_module('pyfixest.fixest')
                    Fixest_ = getattr(fixest_module, 'Fixest')

                    for ixg, g in enumerate(clusters):
                        # direct leave one cluster out implementation
                        data = self.data[~np.equal(ixg, group)]
                        model = Fixest_(data)
                        model.feols(self.fml, vcov = "iid")
                        beta_jack[ixg,:] = model.coef()["Estimate"].to_numpy()


                # optional: beta_bar in MNW (2022)
                #center = "estimate"
                #if center == 'estimate':
                #    beta_center = beta_hat
                #else:
                #    beta_center = np.mean(beta_jack, axis = 0)
                beta_center = beta_hat

                vcov = np.zeros((k_params, k_params))
                for ixg, g in enumerate(clusters):
                    beta_centered = beta_jack[ixg,:] - beta_center
                    vcov += np.outer(beta_centered, beta_centered)

                self.vcov = self.ssc * vcov

    def get_inference(self, alpha = 0.95):
        '''
        Compute standard errors, t-statistics and p-values for the regression model.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals, by default 0.95

        Returns
        -------
        None

        '''

        self.se = (
            np.sqrt(np.diagonal(self.vcov))
        )

        self.tstat = (
            self.beta_hat / self.se
        )

        #if self.vcov_type in ['iid', 'CRV']:
            # t(G-1) distribution for clustered errors
        if self.vcov_type in ["iid", "hetero"]:
            df = self.N - self.k
        else:
            df = self.G - 1
        self.pvalue = (
            2*(1-t.cdf(np.abs(self.tstat), df))
        )
        #else:
        #    # normal distribution for non-clustered errors
        #    self.pvalue = (
        #            2*(1-norm.cdf(np.abs(self.tstat)))
        #    )

        z = norm.ppf(1 - (alpha / 2))
        self.conf_int = (
            np.array([z * self.se - self.beta_hat, z * self.se + self.beta_hat])
        )


    def get_wildboottest(self, B:int, cluster : Union[np.ndarray, pd.Series, pd.DataFrame, None], param : Union[str, None], weights_type: str, impose_null: bool , bootstrap_type: str, seed: Union[str, None] , adj: bool , cluster_adj: bool):

        '''
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
        '''

        if self.is_iv:
            raise ValueError("Wild cluster bootstrap is not supported with IV estimation.")
        if self.has_fixef:
            raise ValueError("Wild cluster bootstrap is not supported with fixed effects.")

        xnames = self.coefnames.to_list()
        Y = self.Y.flatten()
        X = self.X

        # later: allow r <> 0 and custom R
        R = np.zeros(len(xnames))
        R[xnames.index(param)] = 1
        r = 0

        if cluster is None:

            boot = WildboottestHC(X = X, Y = Y, R = R, r = r, B = B, seed = seed)
            boot.get_adjustments(bootstrap_type = bootstrap_type)
            boot.get_uhat(impose_null = impose_null)
            boot.get_tboot(weights_type = weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type = "two-tailed")
            full_enumeration_warn = False

        else:

            cluster = self.data[self.clustervar]

            boot = WildboottestCL(X = X, Y = Y, cluster = cluster,
                                R = R, B = B, seed = seed)
            boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null, adj=adj, cluster_adj=cluster_adj)
            _, _, full_enumeration_warn = boot.get_weights(weights_type = weights_type)
            boot.get_numer()
            boot.get_denom()
            boot.get_tboot()
            boot.get_vcov()
            boot.get_tstat()
            boot.get_pvalue(pval_type = "two-tailed")

            if full_enumeration_warn:
                warnings.warn("2^G < the number of boot iterations, setting full_enumeration to True.")

        res = {
            'param':param,
            'statistic': boot.t_stat,
            'pvalue': boot.pvalue,
            'bootstrap_type': bootstrap_type,
            'impose_null' : impose_null
        }

        res_df = pd.Series(res)

        return res_df




    def get_nobs(self):

        '''
        Fetch the number of observations used in fitting the regression model.

        Returns
        -------
        None
        '''
        self.N = len(self.Y)


    def get_performance(self):
        '''
        Compute multiple additional measures commonly reported with linear regression output.
        '''

        self.r_squared = 1 - np.sum(self.u_hat ** 2) / \
            np.sum((self.Y - np.mean(self.Y))**2)
        self.adj_r_squared = (self.N - 1) / (self.N - self.k) * self.r_squared
