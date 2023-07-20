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

from pyfixest.feols import Feols
from pyfixest.ssc_utils import get_ssc
from pyfixest.exceptions import VcovTypeNotSupportedError, NanInClusterVarError

class Fepois(Feols):

    '''
    Class to estimate Poisson Regressions. Inherits from Feols. The following methods are overwritten: `get_fit()`.
    '''

    def get_fit(self, tol = 1e-06) -> None:

        '''
        Fit a Poisson Regression Model via Iterated Weighted Least Squares

        Args:
            tol (float): tolerance level for the convergence of the IRLS algorithm
        Returns:
            None
        '''


        X = self.X
        Y = self.Y
        fe = self.fe_df.to_numpy().reshape((self.N, 1))

        def _update_W(Xbeta):
            return np.diag(np.exp(Xbeta).flatten())

        def _update_Z(Y, Xbeta):
            return (Y - np.exp(Xbeta)) / np.exp(Xbeta) + Xbeta

        # starting values: http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode27.html
        # reference:  McCullagh, P. & Nelder, J. A. ( 1989). Generalized Linear Models,
        #  Vol. 37 of Monographs on Statistics and Applied Probability, 2 edn, Chapman and Hall, London.
        Xbeta = np.log(np.repeat(np.mean(Y), self.N).reshape((self.N, 1)))
        W = _update_W(Xbeta)
        Z = _update_Z(Y = Y, Xbeta = Xbeta)

        keep_iterating = True
        delta = np.zeros((X.shape[1], 1))

        while keep_iterating:

            # Step 1: weighted demeaning
            ZX = np.concatenate([Z, X], axis = 1)

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

            XdWXd = X_d.transpose() @ W @ X_d
            XdWZd = X_d.transpose() @ W @ Z_d

            delta_new = np.linalg.solve(XdWXd, XdWZd)
            e_new = Z_d - X_d.transpose().reshape((self.N, 1)) @ delta_new

            Xbeta_new = Z - e_new
            W_u = _update_W(Xbeta_new)
            Z_u = _update_Z(Y = Y, Xbeta = Xbeta_new)

            keep_iterating = np.mean((delta - delta_new) ** 2) > tol

            # update
            delta = delta_new
            Z = Z_u
            W = W_u
            Xbeta = Xbeta_new



        self.beta_hat = delta
        self.yhat = Xbeta






