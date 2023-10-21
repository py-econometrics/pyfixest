from pyfixest.estimation import feols
from pyfixest.demean import demean
from abc import ABC, abstractmethod
from formulaic import model_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import numpy as np
import warnings
import time

def event_study(data, yname, idname, tname, gname, xfml, estimator):

    if estimator == "did2s":

        did2s = DID2S(data = data, yname = yname, idname=idname, tname = tname, gname = gname, xfml = xfml)
        fit = did2s.estimate()
        vcov = did2s.vcov()
        fit._vcov = vcov
        fit._vcov_type = "CRV1"
        fit._vcov_type_detail = "CRV (GMM)"
        fit._G = did2s._G
        fit._method = "did2s"

    elif estimator == "twfe":

        twfe = TWFE(data = data, yname = yname, idname=idname, tname = tname, gname = gname, xfml = xfml)
        fit = twfe.estimate()
        vcov = fit.vcov(vcov = {"CRV1": twfe._idname})
        fit._method = "twfe"

    else:
        raise Exception("Estimator not supported")

    # update inference with vcov matrix
    fit.get_inference()

    return fit


class DID(ABC):

    @abstractmethod
    def __init__(self, data, yname, idname, tname, gname, xfml):

        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period.
            gname: unit-specific time of initial treatment.
            xfml: The formula for the covariates.
            estimator: The estimator to use. Options are "did2s".
        Returns:
            None
        """

        # do some checks here

        self._data = data.copy()
        self._yname = yname
        self._idname = idname
        self._tname = tname
        self._gname = gname
        self._xfml = xfml

        # create additional columns


    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def vcov(self):
        pass

    #@abstractmethod
    #def aggregate(self):
    #    pass


class TWFE(DID):

    def __init__(self, data, yname, idname, tname, gname, xfml):
        super().__init__(data, yname, idname, tname, gname, xfml)

        self._estimator = "twfe"

        if self._xfml is not None:
            self._fml = f"{yname} ~ treat + {xfml} | {idname} + {tname}"
        else:
            self._fml = f"{yname} ~ treat | {idname} + {tname}"

    def estimate(self):

            _fml = self._fml
            _data = self._data

            fit = feols(fml = _fml, data = _data)
            self._fit = fit

            return fit

    def vcov(self, cluster = None):

        pass



class DID2S(DID):

    def __init__(self, data, yname, idname, tname, gname, xfml):

        super().__init__(data, yname, idname, tname, gname, xfml)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f"{yname} ~ {xfml} | {idname} + {tname}"
            self._fml2 = f"{yname} ~ {xfml} + treat"
        else:
            self._fml1 = f"{yname} ~ 0 | {idname} + {tname}"
            self._fml2 = f"{yname} ~ - 1 + treat"

        #self._not_yet_treated_idx = data[data["treat"] == False] #data[gname] <= data[tname]


    def estimate(self):

        _fml1 = self._fml1
        _fml2 = self._fml2
        _data = self._data
        _not_yet_treated_data = _data[_data["treat"] == False]
        _yname = self._yname


        tic = time.time()
        fit1 = feols(fml = _fml1, data = _not_yet_treated_data)
        toc = time.time() - tic
        print(f"Time for first fit: {toc} seconds")

        tic = time.time()
        fit1.fixef()
        toc = time.time() - tic
        print(f"Time for fixef: {toc} seconds")

        tic = time.time()
        Y_hat = fit1.predict(newdata = _data)
        toc = time.time() - tic
        print(f"Time for predict: {toc} seconds")

        self._first_u = _data[f"{_yname}"].to_numpy().flatten() - Y_hat
        _data[f"{_yname}"] = self._first_u

        tic = time.time()
        fit2 = feols(_fml2, data = _data)
        toc = time.time() - tic
        print(f"Time for second fit: {toc} seconds")
        self._second_u = fit2.resid()

        return fit2

    def vcov(self, cluster = None):


        _data = self._data
        _first_u = self._first_u
        _second_u = self._second_u


        if cluster is None:
            cluster = self._idname

        cluster_col =  _data[cluster]
        _, clustid = pd.factorize(cluster_col)

        self._G = clustid.nunique()

        fml_group_time = f"~C({self._idname}) + C({self._tname})"   # add covariates
        fml_treatment_vars = "~0+treat"                               # add covariates

        X1 = model_matrix(fml_group_time, _data, output = "sparse")
        #X10 = X1[self._not_yet_treated_idx, :]
        X2 = model_matrix(fml_treatment_vars, _data, output = "sparse")

        X10 = X1.copy().tocsr()
        treated_rows = np.where(_data["treat"], 0, 1)
        X10 = X10.multiply(treated_rows[:, None])

        X10X10 = X10.T.dot(X10)
        X2X1 = X2.T.dot(X1)
        X2X2 = X2.T.dot(X2)

        V = spsolve(X10X10, X2X1.T).T

        k = X2.shape[1]
        vcov = np.zeros((k, k))

        X10 = X10.tocsr()
        X2 = X2.tocsr()

        for (_,g,) in enumerate(clustid):

            X10g = X10[cluster_col == g, :]
            X2g = X2[cluster_col == g, :]
            first_u_g = _first_u[cluster_col == g]
            second_u_g = _second_u[cluster_col == g]

            W_g = X2g.T.dot(second_u_g) - V @ X10g.T.dot(first_u_g)

            score = spsolve(X2X2, W_g)
            cov_g = score.dot(score.T)

            vcov += cov_g

        self._vcov = vcov

        return self._vcov










