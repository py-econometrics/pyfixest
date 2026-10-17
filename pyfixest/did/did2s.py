from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import spsolve

from pyfixest.did.did import DID
from pyfixest.estimation import feols
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.model_state import VarianceCovariance, VcovSpec
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.post_estimation.fixed_effects import build_fixed_effect_dummies


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
        weights: str | None = None,
        att: bool = True,
        xfml: str | None = None,
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
            self._fml2 = f" ~ is_treated + {xfml}"
        else:
            self._fml1 = f" ~ 0 | {idname} + {tname}"
            self._fml2 = " ~ is_treated"

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
            cluster=cast(str, self._cluster),
        )

    def vcov(self):
        """Estimate both stages and return their corrected covariance."""
        return self.estimate().variance_covariance

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (500, 300),
        yintercept: int | None = None,
        xintercept: int | None = None,
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
    cluster: str,
    weights: str | None = None,
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
    cluster: str
        The name of the cluster variable.
    weights : Optional[str].
        Default is None. Weights for WLS estimation. If None, all observations
        are weighted equally. If a string, the name of the column in `data` that
        contains the weights.

    Returns
    -------
    object
        A fitted model object of class feols with the two-stage covariance.
    """
    _first_stage_full = f"{yname} {_first_stage}"
    _second_stage_full = f"{yname}_hat {_second_stage}"

    for name, stage in (("first_stage", _first_stage), ("second_stage", _second_stage)):
        if not stage.lstrip().startswith("~"):
            raise ValueError(f"{name} must start with '~', got '{stage}'.")
    first_formulas = Formula.parse(_first_stage_full)
    second_formulas = Formula.parse(_second_stage_full)
    for name, formulas in (
        ("first_stage", first_formulas),
        ("second_stage", second_formulas),
    ):
        if len(formulas) != 1 or len(formulas[0].dependent) != 1:
            raise ValueError(f"{name} must specify a single model.")
        if formulas[0].is_instrumental_variable:
            raise ValueError(f"{name} must not contain instrumental variables.")
    if not first_formulas[0].is_fixed_effects:
        raise ValueError("First stage formula must contain fixed effects.")
    if second_formulas[0].is_fixed_effects:
        raise ValueError("Second stage formula must not contain fixed effects.")
    if not any(term == "1" for term in second_formulas[0].exogenous):
        raise ValueError(
            "The second stage formula must not remove the intercept with '0' or '-1'. "
            "The intercept is dropped automatically after categorical encoding."
        )

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
    if np.isnan(Y_hat).any():
        # levels present in the full data but absent from the first stage fit
        # (fit on not-yet-treated observations only) predict to NaN
        raise ValueError(
            "First-stage predictions contain NaN values because some fixed effect "
            "levels were not observed in the not-yet-treated first-stage sample. "
            "Drop those observations before calling did2s()."
        )
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

    vcov, G = _did2s_vcov(
        data=data,
        first_stage=fit1,
        second_stage=fit2,
        treatment=treatment,
        first_u=_first_u,
        second_u=_second_u,
        cluster=cluster,
        weights=weights,
    )
    fit2.variance_covariance = _did2s_covariance(
        fit=fit2, vcov=vcov, G=G, cluster=cluster
    )
    return fit2


def _did2s_vcov(
    data: pd.DataFrame,
    first_stage: Feols,
    second_stage: Feols,
    treatment: str,
    first_u: np.ndarray,
    second_u: np.ndarray,
    cluster: str,
    weights: str | None = None,
):
    """
    Variance-Covariance matrix for DID2S.

    Compute a variance covariance matrix for Gardner's 2-stage Difference-in-Differences
    Estimator.

    Parameters
    ----------
    data: pandas.DataFrame
        The DataFrame containing all variables.
    first_stage: Feols
        The fitted first-stage model on not-yet-treated observations.
    second_stage: Feols
        The fitted second-stage model on the full sample.
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

    # `clustid` (factorize's `uniques`) is an `Index` for a Series input; the
    # pandas stubs also allow `ndarray`, which lacks `.nunique()`.
    _G = cast(pd.Index, clustid).nunique()

    if weights is None:
        weights_array = np.repeat(1.0, data.shape[0])
    else:
        weights_array = np.sqrt(data[weights].to_numpy())

    # Replay fitted encodings on the full sample: raw FE expressions must never
    # be evaluated as Python (in particular, `a^b` is not a bitwise XOR).
    model_spec = first_stage.model.model_spec
    assert model_spec is not None
    context = FORMULAIC_TRANSFORMS | dict(first_stage.options.context)
    fe_codes = model_spec["fe"].get_model_matrix(
        data, context=context, na_action="ignore"
    )
    dummies = build_fixed_effect_dummies(
        codes=fe_codes, formulaic_columns=fe_codes.columns
    )
    covariates = model_spec["second_stage"].rhs.get_model_matrix(data, context=context)
    covariates = covariates[first_stage._coefnames]
    X1 = hstack([csr_matrix(covariates.to_numpy()), dummies.matrix], format="csr")
    X1 = X1.multiply(weights_array[:, None]).tocsr()
    X2 = second_stage.model_matrix.independent[second_stage._coefnames]
    X2 = csr_matrix(X2.to_numpy() * weights_array[:, None])

    # Weight first and second stage residuals
    first_u *= weights_array
    second_u *= weights_array

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

    for _, g in enumerate(clustid):
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


def _did2s_covariance(
    *, fit: Feols, vcov: np.ndarray, G: int, cluster: str
) -> VarianceCovariance:
    """Wrap the GMM cluster covariance of did2s in the fitted-model contract.

    The second-stage fit was estimated with ``vcov="iid"``; its degrees of
    freedom and adjustment factor are kept. The spec records the cluster
    variable, so `wald_test`, `wildboottest`, and `ccv` treat the fit as
    clustered.
    """
    inner = fit.variance_covariance
    return VarianceCovariance(
        vcov=vcov,
        meat=None,
        ssc=np.ones(1),
        df_k=inner.df_k,
        df_t=inner.df_t,
        spec=VcovSpec(
            vcov_type="CRV", vcov_type_detail="CRV1 (GMM)", clustervar=(cluster,)
        ),
        G=(int(G),),
    )
