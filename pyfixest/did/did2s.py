import warnings
from typing import Optional, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyfixest.did.did import DID
from pyfixest.estimation.estimation import feols
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.FormulaParser import FixestFormulaParser
from pyfixest.estimation.model_matrix_fixest_ import model_matrix_fixest


class DID2S(DID):
    """
    Difference-in-Differences estimation using Gardner(2021) two-step DID2S estimator.

    Multiple parts of this code are direct translations of Kyle Butt's R code
    `did2s`, published under MIT license: https://github.com/kylebutts/did2s/tree/main.

    Attributes
    ----------
    data : pandas.DataFrame
        The DataFrame containing all variables.
    yname : str
        The name of the dependent variable.
    idname : str
        The name of the identifier variable.
    tname : str
        Variable name for calendar period. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted.
    gname : str
        unit-specific time of initial treatment. Must be an integer in the format
        YYYYMMDDHHMMSS, i.e. it must be possible to compare two dates via '>'.
        Datetime variables are currently not accepted. Never treated units
        must have a value of 0.
    xfml : str
        The formula for the covariates.
    att : str
        Whether to estimate the pooled average treatment effect on the treated
        (ATT) or the canonical event study design with all leads and lags / the
        ATT for each period. Default is True.
    cluster : str
        The name of the cluster variable.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        xfml: str,
        att: bool,
        cluster: str,
    ):
        super().__init__(data, yname, idname, tname, gname, xfml, att, cluster)

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f" ~ {xfml} | {idname} + {tname}"
            self._fml2 = f" ~ 0 + ATT + {xfml}"
        else:
            self._fml1 = f" ~ 0 | {idname} + {tname}"
            self._fml2 = " ~ 0 + ATT"

    def estimate(self):
        """Estimate the two-step DID2S model."""
        return _did2s_estimate(
            data=self._data,
            yname=self._yname,
            _first_stage=self._fml1,
            _second_stage=self._fml2,
            treatment="ATT",
        )  # returns triple Feols, first_u, second_u

    def vcov(self):
        """
        Variance-covariance matrix.

        Calculates the variance-covariance matrix of the coefficient estimates
        for the Difference-in-Differences (DiD) estimator.

        Returns
        -------
        array_like
            The variance-covariance matrix of the coefficient estimates.
        """
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
        alpha: float = 0.05,
        figsize: tuple[int, int] = (500, 300),
        yintercept: Optional[int] = None,
        xintercept: Optional[int] = None,
        rotate_xticks: int = 0,
        title: str = "DID2S Event Study Estimate",
        coord_flip: bool = False,
    ):
        """Plot DID estimates."""
        self.iplot(
            alpha=alpha,
            figsize=figsize,
            yintercept=yintercept,
            xintercept=xintercept,
            rotate_xticks=rotate_xticks,
            title=title,
            coord_flip=coord_flip,
        )

    def tidy(self):  # noqa: D102
        return self.tidy()

    def summary(self):  # noqa: D102
        return self.summary()


def _did2s_estimate(
    data: pd.DataFrame,
    yname: str,
    _first_stage: str,
    _second_stage: str,
    treatment: str,
):
    """
    Estimate the two-step DID2S model.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame containing all variables.
    yname:str
        The name of the dependent variable.
    _first_stage: str
        The formula for the first stage.
    _second_stage: str
        The formula for the second stage.
    treatment: str
        The name of the treatment variable. Must be boolean.

    Returns
    -------
    object
        A fitted model object of class feols and the first and second stage residuals.
    """
    _first_stage_full = f"{yname} {_first_stage}"
    _second_stage_full = f"{yname}_hat {_second_stage}"

    if treatment is not None:
        if treatment not in data.columns:
            raise ValueError(f"The variable {treatment} is not in the data.")
        # check that treatment is boolean
        if data[treatment].dtype != "bool" and data[treatment].dtype in [
            "int64",
            "int32",
            "int8",
            "float64",
            "float32",
        ]:
            if data[treatment].nunique() != 2:
                raise ValueError(f"The treatment variable {treatment} must be boolean.")
            data[treatment] = data[treatment].astype(bool)
            warnings.warn(
                f"The treatment variable {treatment} was converted to boolean."
            )
        _not_yet_treated_data = data[data[treatment] == False]  # noqa: E712
    else:
        _not_yet_treated_data = data[data["ATT"] == False]  # noqa: E712

    # check if first stage formulas has fixed effects
    if "|" not in _first_stage:
        raise ValueError("First stage formula must contain fixed effects.")
    # check if second stage formulas has fixed effects
    if "|" in _second_stage:
        raise ValueError("Second stage formula must not contain fixed effects.")

    # estimate first stage
    fit1 = cast(
        Feols,
        feols(
            fml=_first_stage_full,
            data=_not_yet_treated_data,
            vcov="iid",
        ),
    )  # iid as it might be faster than CRV

    # obtain estimated fixed effects
    fit1.fixef()

    # demean data
    Y_hat = fit1.predict(newdata=data)
    _first_u = data[f"{yname}"].to_numpy().flatten() - Y_hat
    data[f"{yname}_hat"] = _first_u

    # intercept needs to be dropped by hand due to the presence of fixed effects in the first stage  # noqa: W505
    fit2 = cast(
        Feols,
        feols(
            _second_stage_full,
            data=data,
            vcov="iid",
            drop_intercept=True,
        ),
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
):
    """
    Variance-Covariance matrix for DID2S.

    Compute a variance covariance matrix for Gardner's 2-stage Difference-in-Differences
    Estimator.

    Parameters
    ----------
    data: pandas.DataFrame
        The DataFrame containing all variables.
    yname: str
        The name of the dependent variable.
    first_stage: str
        The formula for the first stage.
    second_stage: str
        The formula for the second stage.
    treatment: str
        The name of the treatment variable.
    first_u: np.ndarray
        The first stage residuals.
    second_u: np.ndarray
        The second stage residuals.
    cluster: str
        The name of the cluster variable.

    Returns
    -------
        A variance covariance matrix.
    """
    cluster_col = data[cluster]
    _, clustid = pd.factorize(cluster_col)

    _G = clustid.nunique()  # actually not used here, neither in did2s

    # some formula parsing to get the correct formula for the first and second stage model matrix  # noqa: W505
    first_stage_x, first_stage_fe = first_stage.split("|")
    first_stage_fe_list = [f"C({i})" for i in first_stage_fe.split("+")]
    first_stage_fe_fml = "+".join(first_stage_fe_list)
    first_stage = f"{first_stage_x}+{first_stage_fe_fml}"

    second_stage = f"{second_stage}"

    # note for future Alex: intercept needs to be dropped! it is not as fixed
    # effects are converted to dummies, hence has_fixed checks are False

    FML1 = FixestFormulaParser(f"{yname} {first_stage}")
    FML2 = FixestFormulaParser(f"{yname} {second_stage}")
    FixestFormulaDict1 = FML1.FixestFormulaDict
    FixestFormulaDict2 = FML2.FixestFormulaDict

    mm_dict_first_stage = model_matrix_fixest(
        FixestFormula=next(iter(FixestFormulaDict1.values()))[0],
        data=data,
        weights=None,
        drop_singletons=False,
        drop_intercept=False,
    )
    X1 = cast(pd.DataFrame, mm_dict_first_stage.get("X"))

    mm_second_stage = model_matrix_fixest(
        FixestFormula=next(iter(FixestFormulaDict2.values()))[0],
        data=data,
        weights=None,
        drop_singletons=False,
        drop_intercept=True,
    )  # reference values not dropped, multicollinearity error
    X2 = cast(pd.DataFrame, mm_second_stage.get("X"))

    X1 = csr_matrix(X1.to_numpy())
    X2 = csr_matrix(X2.to_numpy())

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
