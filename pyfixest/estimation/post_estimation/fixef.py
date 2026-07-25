from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import lsqr

from pyfixest.utils.dev_utils import _extract_variable_level


@dataclass(frozen=True, slots=True)
class FixedEffects:
    """Fixed-effect coefficients recovered after estimation.

    Attributes
    ----------
    fixef_dict : dict[str, dict[str, float]]
        Estimated fixed effects, keyed by fixed-effect term and then by level.
    alpha : np.ndarray
        The stacked fixed-effect coefficients, shape (sum of levels,).
    sumFE : np.ndarray
        Sum of all fixed effects for each observation, shape (N,).
    """

    fixef_dict: dict[str, dict[str, float]]
    alpha: np.ndarray
    sumFE: np.ndarray


def _residuals_for_fixef(
    *,
    fml: str,
    data: pd.DataFrame,
    context: Mapping[str, Any],
    coefnames: list[str],
    beta_hat: np.ndarray,
    X_is_empty: bool,
    is_glm: bool,
    Y_hat_link: np.ndarray,
    offset: np.ndarray | None,
) -> np.ndarray:
    """Residualize the dependent variable on the non-absorbed covariates.

    For GLMs the residual is taken from the estimated linear predictor rather
    than from the outcome, following equation (5.2) in Stammann (2018),
    http://arxiv.org/abs/1707.01815 .
    """
    depvars, rhs = fml.split("~")
    covars, _ = rhs.split("|")

    Y, X = Formula(f"{depvars} ~ {covars}").get_model_matrix(
        data, output="pandas", context=context
    )
    Y = Y.to_numpy().flatten().astype(np.float64)

    if X_is_empty:
        return Y.flatten()

    # drop intercept, potentially multicollinear vars
    X = X[coefnames].to_numpy()
    if is_glm:
        Y = Y_hat_link
        # _Y_hat_link contains the offset as part of eta; subtract it so that
        # sumFE represents the pure FE contribution and predict() can add the
        # offset back from newdata without double-counting.
        if offset is not None:
            Y = Y - offset.flatten()

    return (Y - X @ beta_hat).flatten()


def _fixef_design_matrix(*, fml: str, data: pd.DataFrame) -> tuple[spmatrix, list[str]]:
    "Build the one-hot design matrix of the absorbed fixed effects."
    _, rhs = fml.split("~")
    _, fixef_vars = rhs.split("|")
    fixef_fml = "+".join(f"C({x})" for x in fixef_vars.split("+"))

    D2 = Formula("-1+" + fixef_fml).get_model_matrix(data, output="sparse")
    return D2, D2.model_spec.column_names


def solve_fixef(
    *,
    fml: str,
    data: pd.DataFrame,
    context: Mapping[str, Any],
    coefnames: list[str],
    beta_hat: np.ndarray,
    X_is_empty: bool,
    is_glm: bool,
    Y_hat_link: np.ndarray,
    offset: np.ndarray | None,
    weights: np.ndarray,
    has_weights: bool,
    atol: float = 1e-06,
    btol: float = 1e-06,
) -> FixedEffects:
    """Recover the swept-out fixed effects by least squares on the residuals.

    Regresses the residuals of the non-absorbed part of the model on the
    one-hot encoded fixed effects via `scipy.sparse.linalg.lsqr`.

    Parameters
    ----------
    fml : str
        The model formula, with a fixed-effects part after `|`.
    data : pandas.DataFrame
        The estimation data.
    context : Mapping[str, Any]
        Evaluation scope handed to formulaic.
    coefnames : list[str]
        Names of the retained covariates.
    beta_hat : np.ndarray
        Coefficient estimates, shape (k,).
    X_is_empty : bool
        Whether the model has no non-absorbed covariates.
    is_glm : bool
        Whether residuals come from the linear predictor rather than the outcome.
    Y_hat_link : np.ndarray
        The linear predictor, shape (N,). Only read when `is_glm`.
    offset : np.ndarray | None
        Offset included in the linear predictor, shape (N, 1) or None.
    weights : np.ndarray
        Regression weights, shape (N, 1).
    has_weights : bool
        Whether the model is weighted.
    atol : float, optional
        Stopping tolerance for `scipy.sparse.linalg.lsqr`. Defaults to 1e-06.
    btol : float, optional
        Second stopping tolerance for `scipy.sparse.linalg.lsqr`. Defaults to 1e-06.

    Returns
    -------
    FixedEffects
        The estimated fixed effects, the stacked coefficients and their
        per-observation sum.
    """
    uhat = _residuals_for_fixef(
        fml=fml,
        data=data,
        context=context,
        coefnames=coefnames,
        beta_hat=beta_hat,
        X_is_empty=X_is_empty,
        is_glm=is_glm,
        Y_hat_link=Y_hat_link,
        offset=offset,
    )

    D2, cols = _fixef_design_matrix(fml=fml, data=data)

    if has_weights:
        weights_sqrt = np.sqrt(weights).flatten()
        uhat = uhat * weights_sqrt
        D2 = diags(weights_sqrt, 0).dot(D2)

    alpha = lsqr(D2, uhat, atol=atol, btol=btol)[0]

    fixef_dict: dict[str, dict[str, float]] = {}
    for i, col in enumerate(cols):
        variable, level = _extract_variable_level(col)
        if variable not in fixef_dict:
            fixef_dict[variable] = {level: alpha[i]}
        elif level not in fixef_dict[variable]:
            fixef_dict[variable][level] = alpha[i]

    return FixedEffects(fixef_dict=fixef_dict, alpha=alpha, sumFE=D2.dot(alpha))
