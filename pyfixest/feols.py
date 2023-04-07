import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    Attributes
    ----------
    Y : np.ndarray
        The dependent variable of the regression.
    X : np.ndarray
        The independent variable of the regression.
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

    def __init__(self, Y: np.ndarray, X: np.ndarray) -> None:

        if not isinstance(Y, (np.ndarray)):
            raise TypeError("Y must be a numpy array.")
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be a numpy array.")

        self.Y = Y
        self.X = X

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.N, self.k = X.shape

    def get_fit(self) -> None:
        '''
        Regression estimation for a single model, via ordinary least squares (OLS).
        '''

        self.tXX = np.transpose(self.X) @ self.X
        self.tXXinv = np.linalg.inv(self.tXX)

        self.tXy = (np.transpose(self.X) @ self.Y)
        beta_hat = self.tXXinv @ self.tXy
        self.beta_hat = beta_hat.flatten()
        self.Y_hat = (self.X @ self.beta_hat)
        self.u_hat = (self.Y - self.Y_hat)

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
            assert list(vcov.keys())[0] != "CRV3", "CRV3 currently not supported with arbitrary fixed effects"
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
            vcov_type = "iid"
            self.is_clustered = False
        elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
            vcov_type = "hetero"
            self.is_clustered = False
        elif vcov_type_detail in ["CRV1", "CRV3"]:
            vcov_type = "CRV"
            self.is_clustered = True

        # compute vcov
        if vcov_type == 'iid':

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = 1,
                vcov_sign = 1,
                is_clustered=False
            )

            print("ssc: ", self.ssc)

            self.vcov =  self.ssc * (self.tXXinv * np.mean(self.u_hat ** 2))

        elif vcov_type == 'hetero':

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = 1,
                vcov_sign = 1,
                is_clustered=False
            )

            print("ssc: ", self.ssc)

            if vcov_type_detail in ["hetero", "HC1"]:
                u = self.u_hat
            elif vcov_type_detail in ["HC2", "HC3"]:
                leverage = np.mean(self.X * (self.X @ self.tXXinv), axis=1)
                if vcov_type_detail == "HC2":
                    u = (1 - leverage) * self.u_hat
                else:
                    u = np.sqrt(1 - leverage) * self.u_hat

            meat = np.transpose(self.X) * (u ** 2) @ self.X
            self.vcov = self.ssc * self.tXXinv @ meat @  self.tXXinv

        elif vcov_type == "CRV":

            cluster_df = self.data[self.clustervar]
            # if there are missings - delete them!
            # cluster_df = pd.Categorical(cluster_df)

            if np.any(pd.isna(cluster_df)):
                raise ValueError("CRV inference not supported with missing values in the cluster variable. Please drop missing values before running the regression.")

            cluster_mat = cluster_df.to_numpy()

            clustid = np.unique(cluster_mat)
            self.G = len(clustid)

            self.ssc = get_ssc(
                ssc_dict = self.ssc_dict,
                N = self.N,
                k = self.k,
                G = self.G,
                vcov_sign = 1,
                is_clustered=True
            )

            if vcov_type_detail == "CRV1":

                meat = np.zeros((self.k, self.k))

                for igx, g, in enumerate(clustid):

                    Xg = self.X[np.where(cluster_df == g)]
                    ug = self.u_hat[np.where(cluster_df == g)]
                    score_g = (np.transpose(Xg) @ ug).reshape((self.k, 1))
                    meat += np.dot(score_g, score_g.transpose())

                self.vcov = self.ssc * self.tXXinv @ meat @ self.tXXinv

            elif vcov_type_detail == "CRV3":

                # check: is fixed effect cluster fixed effect?
                # if not, either error or turn fixefs into dummies
                # for now: don't allow for use with fixed effects
                assert self.has_fixef == False, "CRV3 currently not supported with arbitrary fixed effects"

                beta_jack = np.zeros((self.G, self.k))
                tXX = np.transpose(self.X) @ self.X

                for ixg, g in enumerate(clustid):

                    Xg = self.X[np.where(cluster_df == g)]
                    Yg = self.Y[:, x][np.where(cluster_df == g)]

                    tXgXg = np.transpose(Xg) @ Xg

                    # jackknife regression coefficient
                    beta_jack[ixg, :] = (
                        np.linalg.pinv(
                            tXX - tXgXg) @ (self.tXy - np.transpose(Xg) @ Yg)
                    ).flatten()

                beta_center = self.beta_hat

                vcov = np.zeros((self.k, self.k))
                for ixg, g in enumerate(clustid):
                    beta_centered = beta_jack[ixg, :] - beta_center
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

        if self.is_clustered:
            # t(G-1) distribution for clustered errors
            self.pvalue = (
                2*(1-t.cdf(np.abs(self.tstat), self.G - 1))
            )
        else:
            # normal distribution for non-clustered errors
            self.pvalue = (
                    2*(1-norm.cdf(np.abs(self.tstat)))
            )

        self.pvalue = (
            2*(1-norm.cdf(np.abs(self.tstat)))
        )

        z = norm.ppf(1 - (alpha / 2))
        self.conf_int = (
            np.array([z * self.se - self.beta_hat, z * self.se + self.beta_hat])
        )

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



