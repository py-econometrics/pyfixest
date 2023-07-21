import pyhdfe
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

from pyfixest.feols import Feols, _check_vcov_input, _deparse_vcov_input
from pyfixest.ssc_utils import get_ssc
from pyfixest.exceptions import VcovTypeNotSupportedError, NanInClusterVarError, NonConvergenceError

class Fepois(Feols):

    '''
    Class to estimate Poisson Regressions. Inherits from Feols. The following methods are overwritten: `get_fit()`.
    '''

    def get_fit(self, tol = 1e-08, maxiter = 25) -> None:

        '''
        Fit a Poisson Regression Model via Iterated Weighted Least Squares

        Args:
            tol (float): tolerance level for the convergence of the IRLS algorithm
            maxiter (int): maximum number of iterations for the IRLS algorithm. 25 by default.
        Returns:
            None
        Attributes:
            beta_hat (np.array): estimated coefficients
            Y_hat (np.array): predicted values of the dependent variable
            u_hat (np.array): estimated residuals
            tZX (np.array): transpose of the product of the demeaned Z and X matrices (used for vcov calculation)
            tZy (np.array): transpose of the product of the demeaned Z and Y matrices (used for vcov calculation)
            tZXinv (np.array): inverse of tZX (used for vcov calculation)
        Updates the following attributes:
            X (np.array): demeaned X matrix from the last iteration of the IRLS algorithm (X_d) x weights
            Z (np.array): demeaned X matrix from the last iteration of the IRLS algorithm (X_d) x weights
            Y (np.array): demeaned Y matrix from the last iteration of the IRLS algorithm (Z_d) x weights
        '''


        X = self.X
        Y = self.Y

        fe = self.fe_df.to_numpy()

        # check if fe is a one-dimensional array
        if self.fe_df.ndim == 1:
            fe = fe.reshape((self.N, 1))

        def _update_w(Xbeta):
            return np.exp(Xbeta)

        def _update_Z(Y, Xbeta):
            return (Y - np.exp(Xbeta)) / np.exp(Xbeta) + Xbeta

        # starting values: http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode27.html
        # reference:  McCullagh, P. & Nelder, J. A. ( 1989). Generalized Linear Models,
        #  Vol. 37 of Monographs on Statistics and Applied Probability, 2 edn, Chapman and Hall, London.
        Xbeta = np.log(np.repeat(np.mean(Y), self.N).reshape((self.N, 1)))
        w = _update_w(Xbeta)
        Z = _update_Z(Y = Y, Xbeta = Xbeta)

        delta = np.ones((X.shape[1], 1))

        X2 = X.copy()
        Z2 = Z

        for x in range(maxiter):

            # Step 1: weighted demeaning
            ZX = np.concatenate([Z2, X2], axis = 1)

            algorithm = pyhdfe.create(
                    ids=fe,
                    residualize_method='map',
                    drop_singletons=True,
                    weights = np.exp(Xbeta)
            )

            #if self.drop_singletons == True and algorithm.singletons != 0 and algorithm.singletons is not None:
            #    print(algorithm.singletons, "columns are dropped due to singleton fixed effects.")
            #    dropped_singleton_indices = np.where(algorithm._singleton_indices)[0].tolist()
            #    na_index += dropped_singleton_indices

            ZX_d = algorithm.residualize(ZX)
            Z_d = ZX_d[:,0].reshape((self.N, 1))
            X_d = ZX_d[:,1:]

            WX_d = np.sqrt(w) * X_d
            WZ_d = np.sqrt(w) * Z_d

            XdWXd = WX_d.transpose() @ WX_d
            XdWZd = WX_d.transpose() @ WZ_d

            delta_new = np.linalg.solve(XdWXd, XdWZd)
            e_new = Z_d - X_d.transpose().reshape((self.N, self.k)) @ delta_new

            Xbeta_new = Z - e_new
            w_u = _update_w(Xbeta_new)
            Z_u = _update_Z(Y = Y, Xbeta = Xbeta_new)

            stop_iterating = np.sqrt(np.sum((delta - delta_new) ** 2) / np.sum(delta ** 2)) < tol

            # update
            delta = delta_new
            Z2 = Z_d + Z_u - Z
            #Z2 = Z_u
            X2 = X
            #X2 = X_d
            Z = Z_u
            #w_old = w.copy()
            w = w_u
            #Xbeta_old = Xbeta.copy()
            Xbeta = Xbeta_new

            if stop_iterating:
                break
            if x == maxiter:
                raise NonConvergenceError("The IRLS algorithm did not converge. Try to increase the maximum number of iterations.")

        self.beta_hat = delta
        self.Y_hat = Xbeta
        self.u_hat = (Y - np.exp(Xbeta))
        # needed for the calculation of the vcov

        # updat for inference
        #self.weights = w_old
        # if only one dim
        #if self.weights.ndim == 1:
        #    self.weights = self.weights.reshape((self.N, 1))

        self.X = WX_d
        self.Z = WZ_d
        #self.Y = np.sqrt(weights) * Z_d
        #self.weights = weights

        self.tZX = np.transpose(self.Z) @ self.X
        #self.tZy = (np.transpose(self.Z) @ self.Y)
        self.tZXinv = np.linalg.inv(self.tZX)
        self.Xbeta = Xbeta

    def get_vcov(self, vcov):

        _check_vcov_input(vcov, self.data)

        self.vcov_type, self.vcov_type_detail, self.is_clustered, self.clustervar = _deparse_vcov_input(vcov, self.has_fixef, self.is_iv)

        if self.is_iv:
            if self.vcov_type in ["CRV3"]:
                raise VcovTypeNotSupportedError(
                    "CRV3 inference is not supported for IV regressions."
                )

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
                #self.vcov = self.ssc * (self.u_hat ** 2) @ np.linalg.inv(self.X.transpose() @ self.X)
                self.vcov =  self.ssc * self.tZXinv * np.sum((self.u_hat ** 2) / (self.N - 1))
            else:
                sigma2 = (np.sum(self.u_hat ** 2) / (self.N - 1))
                self.vcov = self.ssc * np.linalg.inv(self.tXZ @ self.tZZinv @ self.tZX ) * sigma2


        else:

            raise NotImplementedError("Only iid is supported for Poisson regressions.")
