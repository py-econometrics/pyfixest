import warnings
from typing import Optional, Union, list

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyfixest.did.did import DID
from pyfixest.estimation import feols
from pyfixest.model_matrix_fixest import model_matrix_fixest


class DID2S(DID):
    """
    Difference-in-Differences estimation using Gardner's two-step DID2S estimator (2021).

    Multiple parts of this code are direct translations of Kyle Butt's R code `did2s`, published
    under MIT license: https://github.com/kylebutts/did2s/tree/main.
    """

    def __init__(self, data, yname, idname, tname, gname, xfml, att, cluster):
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
            att: Whether to estimate the pooled average treatment effect on the treated (ATT) or the
                canonical event study design with all leads and lags / the ATT for each period. Default is True.
            cluster (str): The name of the cluster variable.
        Returns:
            None
        """

        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f" ~ {xfml} | {idname} + {tname}"
            self._fml2 = f" ~ 0 + ATT + {xfml}"
        else:
            self._fml1 = f" ~ 0 | {idname} + {tname}"
            self._fml2 = " ~ 0 + ATT"

    def estimate(self):
        """
        Args:
            data (pd.DataFrame): The DataFrame containing all variables.
            yname (str): The name of the dependent variable.
            _first_stage (str): The formula for the first stage.
            _second_stage (str): The formula for the second stage.
            treatment (str): The name of the treatment variable.
        Returns:
            tba
        """

        return _did2s_estimate(
            data=self._data,
            yname=self._yname,
            _first_stage=self._fml1,
            _second_stage=self._fml2,
            treatment="ATT",
        )  # returns triple Feols, first_u, second_u

    def vcov(self):
        return _did2s_vcov(
            data=self._data,
            yname=self._yname,
            first_stage=self._fml1,
            second_stage=self._fml2,
            treatment="ATT",
            first_u=self._first_u,
            second_u=self._second_u,
            cluster=self._cluster,
        )

    def iplot(
        self,
        alpha=0.05,
        figsize=(500, 300),
        yintercept=None,
        xintercept=None,
        rotate_xticks=0,
        title="DID2S Event Study Estimate",
        coord_flip=False,
    ):
        self.iplot(
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

    def tidy(self):
        return self.tidy()

    def summary(self):
        return self.summary()


def _did2s_estimate(
    data: pd.DataFrame,
    yname: str,
    _first_stage: str,
    _second_stage: str,
    treatment: str,
    i_ref1: Optional[Union[int, str, list]] = None,
    i_ref2: Optional[Union[int, str, list]] = None,
):
    """
    Args:
        data (pd.DataFrame): The DataFrame containing all variables.
        yname (str): The name of the dependent variable.
        _first_stage (str): The formula for the first stage.
        _second_stage (str): The formula for the second stage.
        treatment (str): The name of the treatment variable. Must be boolean.
        i_ref1 (int, str or list): The reference value(s) for the first variable used with "i()" syntax. Only applicable for the second stage formula.
        i_ref2 (int, str or list): The reference value(s) for the second variable used with "i()" syntax. Only applicable for the second stage formula.
    Returns:
        A fitted model object of class feols and the first and second stage residuals.
    """

    _first_stage_full = f"{yname} {_first_stage}"
    _second_stage_full = f"{yname}_hat {_second_stage}"

    if treatment is not None:
        if treatment not in data.columns:
            raise ValueError(f"The variable {treatment} is not in the data.")
        # check that treatment is boolean
        if data[treatment].dtype != "bool":
            if data[treatment].dtype in [
                "int64",
                "int32",
                "int8",
                "float64",
                "float32",
            ]:
                if data[treatment].nunique() == 2:
                    data[treatment] = data[treatment].astype(bool)
                    warnings.warn(
                        f"The treatment variable {treatment} was converted to boolean."
                    )
                else:
                    raise ValueError(
                        f"The treatment variable {treatment} must be boolean."
                    )
        _not_yet_treated_data = data[~data[treatment]]
    else:
        _not_yet_treated_data = data[~data["ATT"]]

    # check if first stage formulas has fixed effects
    if "|" not in _first_stage:
        raise ValueError("First stage formula must contain fixed effects.")
    # check if second stage formulas has fixed effects
    if "|" in _second_stage:
        raise ValueError("Second stage formula must not contain fixed effects.")

    # estimate first stage
    fit1 = feols(
        fml=_first_stage_full,
        data=_not_yet_treated_data,
        vcov="iid",
        i_ref1=None,
        i_ref2=None,
    )  # iid as it might be faster than CRV

    # obtain estimated fixed effects
    fit1.fixef()

    # demean data
    Y_hat = fit1.predict(newdata=data)
    _first_u = data[f"{yname}"].to_numpy().flatten() - Y_hat
    data[f"{yname}_hat"] = _first_u

    # intercept needs to be dropped by hand due to the presence of fixed effects in the first stage
    fit2 = feols(
        _second_stage_full,
        data=data,
        vcov="iid",
        drop_intercept=True,
        i_ref1=i_ref1,
        i_ref2=i_ref2,
    )
    _second_u = fit2.resid()

    return fit2, _first_u, _second_u


def _did2s_vcov(
    data: pd.DataFrame,
    yname: str,
    first_stage: str,
    second_stage: str,
    treatment: str,
    first_u: np.ndarray,
    second_u: np.ndarray,
    cluster: str,
    i_ref1: Optional[Union[int, str, list]] = None,
    i_ref2: Optional[Union[int, str, list]] = None,
):
    """
    Compute a variance covariance matrix for Gardner's 2-stage Difference-in-Differences Estimator.
    Args:
        data (pd.DataFrame): The DataFrame containing all variables.
        yname (str): The name of the dependent variable.
        first_stage (str): The formula for the first stage.
        second_stage (str): The formula for the second stage.
        treatment (str): The name of the treatment variable.
        first_u (np.ndarray): The first stage residuals.
        second_u (np.ndarray): The second stage residuals.
        cluster (str): The name of the cluster variable.
        i_ref1 (int, str or list): The reference value(s) for the first variable used with "i()" syntax. Only applicable for the second stage formula.
        i_ref2 (int, str or list): The reference value(s) for the second variable used with "i()" syntax. Only applicable for the second stage formula.
    Returns:
        A variance covariance matrix.
    """

    cluster_col = data[cluster]
    _, clustid = pd.factorize(cluster_col)

    _G = clustid.nunique()  # actually not used here, neither in did2s

    # some formula parsing to get the correct formula for the first and second stage model matrix
    first_stage_x, first_stage_fe = first_stage.split("|")
    first_stage_fe = [f"C({i})" for i in first_stage_fe.split("+")]
    first_stage_fe = "+".join(first_stage_fe)
    first_stage = f"{first_stage_x}+{first_stage_fe}"

    second_stage = f"{second_stage}"

    # note for future Alex: intercept needs to be dropped! it is not as fixed effects are converted to
    # dummies, hence has_fixed checks are False
    _, X1, _, _, _, _, _, _, _, _ = model_matrix_fixest(
        fml=f"{yname} {first_stage}",
        data=data,
        weights=None,
        drop_singletons=False,
        drop_intercept=False,
        i_ref1=i_ref1,
        i_ref2=i_ref2,
    )
    _, X2, _, _, _, _, _, _, _, _ = model_matrix_fixest(
        fml=f"{yname} {second_stage}",
        data=data,
        weights=None,
        drop_singletons=False,
        drop_intercept=True,
        i_ref1=i_ref1,
        i_ref2=i_ref2,
    )  # reference values not dropped, multicollinearity error

    X1 = csr_matrix(X1.values)
    X2 = csr_matrix(X2.values)

    X10 = X1.copy().tocsr()
    treated_rows = np.where(data[treatment], 0, 1)
    X10 = X10.multiply(treated_rows[:, None])

    X10X10 = X10.T.dot(X10)
    X2X1 = X2.T.dot(X1)
    X2X2 = X2.T.dot(X2)  # tocsc() to fix spsolve efficiency warning

    V = spsolve(X10X10.tocsc(), X2X1.T.tocsc()).T

    k = X2.shape[1]
    vcov = np.zeros((k, k))

    X10 = X10.tocsr()
    X2 = X2.tocsr()

    for (
        _,
        g,
    ) in enumerate(clustid):
        X10g = X10[cluster_col == g, :]
        X2g = X2[cluster_col == g, :]
        first_u_g = first_u[cluster_col == g]
        second_u_g = second_u[cluster_col == g]

        W_g = X2g.T.dot(second_u_g) - V @ X10g.T.dot(first_u_g)
        score = spsolve(X2X2, W_g)
        if score.ndim == 1:
            score = score.reshape(-1, 1)
        cov_g = score.dot(score.T)

        vcov += cov_g

    return vcov, _G
