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
        #vcov = did2s.vcov()
        #fit._vcov = vcov
        # update inference with new vcov matrix
        #fit._get_inference()

    else:
        raise Exception("Estimator not supported")

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


class DID2S(DID):

    def __init__(self, data, yname, idname, tname, gname, xfml):

        super().__init__(data, yname, idname, tname, gname, xfml)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f"{yname} ~ {xfml} | {idname} + {tname}"
            self._fml2 = f"Y_hat ~ {xfml} + treat"
        else:
            self._fml1 = f"{yname} ~ 0 | {idname} + {tname}"
            self._fml2 = f"Y_hat ~ treat"

        #self._not_yet_treated_idx = data[data["treat"] == False] #data[gname] <= data[tname]


    def estimate(self):

        _fml1 = self._fml1
        _fml2 = self._fml2
        _data = self._data
        _not_yet_treated_data = _data[_data["treat"] == False]


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

        _data["Y_hat"] = Y_hat

        tic = time.time()
        fit2 = feols(_fml2, data = _data)
        toc = time.time() - tic
        print(f"Time for second fit: {toc} seconds")

        return fit2

    def vcov(self, cluster = None):

        _data = self._data
        if cluster is None:
            cluster = self._idname

        cluster_col =  _data[cluster]
        _, clustid = pd.factorize(cluster_col)

        fml_group_time = f"~C({self._idname}) + C({self._tname})"
        fml_treatment_vars = "~treat"

        X1 = model_matrix(fml_group_time, _data, output = "sparse")
        #X10 = X1[self._not_yet_treated_idx, :]
        X2 = model_matrix(fml_treatment_vars, _data, output = "sparse")

        X2X2 = (X2.T.dot(X2)).toarray()
        X2X2inv = np.linalg.inv(X2X2)

        # clustering

        sum_WgWg = np.zeros((X1.shape[1], X1.shape[1]))
        for (_,g,) in enumerate(clustid):

            X1g = X1[cluster_col == g, :]
            X2g = X2[cluster_col == g, :]
            e1g = None
            e2g = None

            A = X2g.T @ e2g
            B = e1g.T @ X1g
            C = spsolve(X1g.T.dot(X1g))
            D = X1g.T.dot(X2g)

            sum_WgWg += A - B @ C @ D

        self._vcov = X2X2inv @ sum_WgWg @ X2X2inv

        return self._vcov










