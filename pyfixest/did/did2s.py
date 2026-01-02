from typing import Optional, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyfixest.did.did import DID
from pyfixest.estimation.estimation import feols
from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.formula import model_matrix
from pyfixest.estimation.formula.parse import Formula


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
    weights : Optional[str].
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        yname: str,
        idname: str,
        tname: str,
        gname: str,
        cluster: str,
        weights: Optional[str] = None,
        att: bool = True,
        xfml: Optional[str] = None,
    ):
        super().__init__(
            data=data,
            yname=yname,
            idname=idname,
            tname=tname,
            gname=gname,
            xfml=xfml,
            att=att,
            cluster=cluster,
        )

        self._estimator = "did2s"

        if self._xfml is not None:
            self._fml1 = f" ~ {xfml} | {idname} + {tname}"
            self._fml2 = f" ~ 0 + is_treated + {xfml}"
        else:
            self._fml1 = f" ~ 0 | {idname} + {tname}"
            self._fml2 = " ~ 0 + is_treated"

        # first and second stage residuals
        self._first_u = np.array([])
        self._second_u = np.array([])

        # column name with weights. None by default
        self._weights_name = weights

    def estimate(self):
        """Estimate the two-step DID2S model."""
        return _did2s_estimate(
            data=self._data,
            yname=self._yname,
            _first_stage=self._fml1,
            _second_stage=self._fml2,
            weights=self._weights_name,
            treatment="is_treated",
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
            treatment="is_treated",
            first_u=self._first_u,
            second_u=self._second_u,
            cluster=self._cluster,
            weights=self._weights_name,
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
    weights: Optional[str] = None,
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
    weights : Optional[str].
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

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
        treat_u = data[treatment].unique()
        if len(treat_u) != 2:
            raise ValueError(
                f"The treatment variable {treatment} must have 2 unique values but it has unique values {treat_u}."
            )
        if data[treatment].dtype in ["bool", "int64", "int32", "float64", "float32"]:
            data[treatment] = data[treatment].astype(bool)
        else:
            raise ValueError(
                f"The treatment variable {treatment} must be boolean or numeric but it is of type {data[treatment].dtype}."
            )
        _not_yet_treated_data = data[data[treatment] == False]  # noqa: E712
    else:
        _not_yet_treated_data = data[data["is_treated"] == False]  # noqa: E712

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
            weights=weights,
        ),
    )  # iid as it might be faster than CRV

    # obtain estimated fixed effects
    fit1.fixef()

    # demean data
    Y_hat = fit1.predict(newdata=data)
    _first_u = data[f"{yname}"].to_numpy().flatten() - Y_hat
    data[f"{yname}_hat"] = _first_u

    # intercept needs to be dropped by hand due to the presence of fixed effects in the first stage
    fit2 = cast(
        Feols,
        feols(
            _second_stage_full,
            data=data,
            vcov="iid",
            drop_intercept=True,
            weights=weights,
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
    weights: Optional[str] = None,
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
    weights : Optional[str].
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    Returns
    -------
        A variance covariance matrix.
    """
    cluster_col = data[cluster]
    _, clustid = pd.factorize(cluster_col)

    _G = clustid.nunique()  # actually not used here, neither in did2s

    if weights is None:
        weights_array = np.repeat(1.0, data.shape[0])
    else:
        weights_array = np.sqrt(data[weights].to_numpy())

    # some formula parsing to get the correct formula for the first and second stage model matrix
    first_stage_x, first_stage_fe = first_stage.split("|")
    first_stage_fe_list = [f"C({i.strip()})" for i in first_stage_fe.split("+")]
    first_stage_fe_fml = "+".join(first_stage_fe_list)
    first_stage_fml = f"{first_stage_x}+{first_stage_fe_fml}"

    # note for future Alex: intercept needs to be dropped! it is not as fixed
    # effects are converted to dummies, hence has_fixed checks are False

    # Create Formula objects for the new model_matrix system
    # First stage: convert fixed effects to dummy variables (C() syntax)
    FML1 = Formula(
        dependent=yname,
        independent=first_stage_fml.replace("~", "").strip(),
        intercept=False,  # first_stage typically has ~0
    )

    # Second stage: use the formula as-is (new system handles i() syntax natively)
    FML2 = Formula(
        dependent=yname,
        independent=second_stage.replace("~", "").strip(),
        intercept=False,  # intercept dropped due to fixed effects in first stage
    )

    mm_first_stage = model_matrix.get(
        formula=FML1,
        data=data,
        weights=None,
        drop_singletons=False,
        ensure_full_rank=True,
    )
    X1 = mm_first_stage.independent

    mm_second_stage = model_matrix.get(
        formula=FML2,
        data=data,
        weights=None,
        drop_singletons=False,
        ensure_full_rank=True,
    )
    X2 = mm_second_stage.independent

    X1 = csr_matrix(X1.to_numpy() * weights_array[:, None])
    X2 = csr_matrix(X2.to_numpy() * weights_array[:, None])

    # Weight first and second stage residuals
    first_u *= weights_array
    second_u *= weights_array

    X10 = X1.copy().tocsr()  # type: ignore
    treated_rows = np.where(data[treatment], 0, 1)
    X10 = X10.multiply(treated_rows[:, None])

    X10X10 = X10.T.dot(X10)
    X2X1 = X2.T.dot(X1)
    X2X2 = X2.T.dot(X2)  # tocsc() to fix spsolve efficiency warning

    V = spsolve(X10X10.tocsc(), X2X1.T.tocsc()).T  # type: ignore

    k = X2.shape[1]
    vcov = np.zeros((k, k))

    X10 = X10.tocsr()
    X2 = X2.tocsr()  # type: ignore

    for (
        _,
        g,
    ) in enumerate(clustid):
        idx_g: np.ndarray = cluster_col.values == g
        X10g = X10[idx_g, :]
        X2g = X2[idx_g, :]
        first_u_g = first_u[idx_g]
        second_u_g = second_u[idx_g]

        W_g = X2g.T.dot(second_u_g) - V @ X10g.T.dot(first_u_g)
        score = spsolve(X2X2, W_g)
        if score.ndim == 1:
            score = score.reshape(-1, 1)
        cov_g = score.dot(score.T)

        vcov += cov_g

    return vcov, _G
