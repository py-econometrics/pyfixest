from multiprocessing import Value
from pyfixest.estimation import feols
from pyfixest.demean import demean
from abc import ABC, abstractmethod
from formulaic import model_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import numpy as np
import warnings
import time

def event_study(data, yname, idname, tname, gname, xfml = None, estimator = "twfe", att = True):

    """
    Estimate a treatment effect using an event study design. If estimator is "twfe", then
    the treatment effect is estimated using the two-way fixed effects estimator. If estimator
    is "did2s", then the treatment effect is estimated using Gardner's two-step DID2S estimator.
    Other estimators are work in progress, please contact the package author if you are interested
    / need other estimators (i.e. Mundlak, Sunab, Imputation DID or Projections).

    Args:
        data: The DataFrame containing all variables.
        yname: The name of the dependent variable.
        idname: The name of the id variable.
        tname: Variable name for calendar period.
        gname: unit-specific time of initial treatment.
        xfml: The formula for the covariates.
        estimator: The estimator to use. Options are "did2s" and "twfe".
        att: Whether to estimate the average treatment effect on the treated (ATT) or the
            canonical event study design with all leads and lags. Default is True.
    Returns:
        A fitted model object of class feols.
    """

    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(yname, str), "yname must be a string"
    assert isinstance(idname, str), "idname must be a string"
    assert isinstance(tname, str), "tname must be a string"
    assert isinstance(gname, str), "gname must be a string"
    assert isinstance(xfml, str) or xfml is None, "xfml must be a string or None"
    assert isinstance(estimator, str), "estimator must be a string"
    assert isinstance(att, bool), "att must be a boolean"

    if estimator == "did2s":

        did2s = DID2S(data = data, yname = yname, idname=idname, tname = tname, gname = gname, xfml = xfml, att = att)
        fit = did2s.estimate()
        vcov = did2s.vcov()
        fit._vcov = vcov
        fit._vcov_type = "CRV1"
        fit._vcov_type_detail = "CRV (GMM)"
        fit._G = did2s._G
        fit._method = "did2s"

    elif estimator == "twfe":

        twfe = TWFE(data = data, yname = yname, idname=idname, tname = tname, gname = gname, xfml = xfml, att = att)
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
    def __init__(self, data, yname, idname, tname, gname, xfml, att):

        """
        Args:
            data: The DataFrame containing all variables.
            yname: The name of the dependent variable.
            idname: The name of the id variable.
            tname: Variable name for calendar period. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted.
            gname: unit-specific time of initial treatment. Must be an integer in the format YYYYMMDDHHMMSS, i.e. it must be
                   possible to compare two dates via '>'. Date time variables are currently not accepted. Never treated units
                   must have a value of 0.
            xfml: The formula for the covariates.
            estimator: The estimator to use. Options are "did2s".
            att: Whether to estimate the average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags. Default is True.
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

        # check if idname, tname and gname are in data
        if self._idname not in self._data.columns:
            raise ValueError(f"The variable {self._idname} is not in the data.")
        if self._tname not in self._data.columns:
            raise ValueError(f"The variable {self._tname} is not in the data.")
        if self._gname not in self._data.columns:
            raise ValueError(f"The variable {self._gname} is not in the data.")

        # check if tname and gname are of type int (either int 64, 32, 8)
        if self._data[self._tname].dtype not in ["int64", "int32", "int8"]:
            raise ValueError(f"The variable {self._tname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS.")
        if self._data[self._gname].dtype not in ["int64", "int32", "int8"]:
            raise ValueError(f"The variable {self._gname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS.")

        # check if there is a never treated unit
        if 0 not in self._data[self._gname].unique():
            raise ValueError(f"There must be at least one unit that is never treated.")

        # create a treatment variable
        self._data["zz00_treat"] = (self._data[self._tname] >= self._data[self._gname]) * (self._data[self._gname] > 0)

        # create treat variable
        if att:
            if "treat" not in self._data.columns:
                # Step 1: check if tname and gname are of type int
                if self._data[self._tname].dtype != "int64":
                    raise ValueError(f"The variable {self._tname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS.")
                if self._data[self._gname].dtype != "int64":
                    raise ValueError(f"The variable {self._gname} must be of type int64, and more specifically, in the format YYYYMMDDHHMMSS.")
                self._data["treat"] = (self._data[self._tname] >= self._data[self._gname])
        #else:
        #    # get rel_year variable
        #    if "rel_year" not in self._data.columns:
        #        self._data["rel_year"] = self._data[self._tname] - self._data[self._gname]

        # get never treated units
        #ct = pd.crosstab(self._data[self._idname], self._data["treat"])




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

    def __init__(self, data, yname, idname, tname, gname, xfml, att):
        super().__init__(data, yname, idname, tname, gname, xfml, att)

        self._estimator = "twfe"

        if self._xfml is not None:
            self._fml = f"{yname} ~ zz00_treat + {xfml} | {idname} + {tname}"
        else:
            self._fml = f"{yname} ~ zz00_treat | {idname} + {tname}"

    def estimate(self):

            _fml = self._fml
            _data = self._data

            fit = feols(fml = _fml, data = _data)
            self._fit = fit

            return fit

    def vcov(self, cluster = None):

        pass



class DID2S(DID):

    def __init__(self, data, yname, idname, tname, gname, xfml, att):

        super().__init__(data, yname, idname, tname, gname, xfml, att)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f"{yname} ~ {xfml} | {idname} + {tname}"
            self._fml2 = f"{yname} ~ 0 + zz00_treat + {xfml}"
        else:
            self._fml1 = f"{yname} ~ 0 | {idname} + {tname}"
            self._fml2 = f"{yname} ~ 0 + zz00_treat"

        #self._not_yet_treated_idx = data[data["treat"] == False] #data[gname] <= data[tname]


    def estimate(self):

        _fml1 = self._fml1
        _fml2 = self._fml2
        _data = self._data
        _not_yet_treated_data = _data[_data["zz00_treat"] == False]
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
        fml_treatment_vars = "~0+zz00_treat"                               # add covariates

        X1 = model_matrix(fml_group_time, _data, output = "sparse")
        #X10 = X1[self._not_yet_treated_idx, :]
        X2 = model_matrix(fml_treatment_vars, _data, output = "sparse")

        X10 = X1.copy().tocsr()
        treated_rows = np.where(_data["zz00_treat"], 0, 1)
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










