from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import t

from pyfixest.estimation.formula.formulaic_compat import (
    bin_mapping_state_key,
    iter_model_spec_categorical_levels,
)

if TYPE_CHECKING:
    import formulaic


def _rows_with_unseen_categories(
    rhs_spec: "formulaic.ModelSpec", newdata: pd.DataFrame
) -> np.ndarray:
    """
    Flag `newdata` rows that carry a categorical level not seen during fitting.

    When a model matrix is rebuilt from a stored `ModelSpec`, both formulaic's
    native `C()` and pyfixest's `i()` silently cast unseen categorical levels to
    the reference level (an all-zero dummy row), which yields a finite-but-wrong
    prediction. We mark those rows so the caller can return NaN for them - the
    same outcome already used for unseen fixed-effect levels.

    Returns
    -------
    np.ndarray
        Boolean mask of length `len(newdata)`; True where a row must be dropped.
    """
    mask = np.zeros(newdata.shape[0], dtype=bool)
    for variable, levels, state in iter_model_spec_categorical_levels(
        rhs_spec, newdata
    ):
        column = newdata[variable]
        # For binned i() terms, apply the stored bin mapping before checking
        # so that valid raw levels (e.g. "a" -> "low") are not flagged unseen.
        bin_key = bin_mapping_state_key(variable)
        if bin_key in state:
            column = column.replace(state[bin_key])
        unseen = ~column.isin(levels) & column.notna()
        mask |= unseen.to_numpy()
    return mask


def _get_prediction_se(model, X: np.ndarray) -> np.ndarray:
    """
    Compute prediction standard error for each row in X.

    Parameters
    ----------
    model : Feols
        The fitted Feols model.
    X : np.ndarray
        The design matrix for newdata.

    Returns
    -------
    se : np.ndarray
        The prediction standard error for each observation.
    """
    return np.sqrt(np.einsum("ij,jk,ik->i", X, model._vcov, X))


def _compute_prediction_error(
    model, nobs: int, yhat: np.ndarray, X: np.ndarray, alpha: float
) -> pd.DataFrame:
    """
    Fill a DataFrame with predictions and confidence intervals.

    X must have shape (nobs, k). Rows containing NaN in X produce NaN se_fit and
    confidence intervals via natural NaN propagation in the einsum.

    Parameters
    ----------
    model : Feols
        The fitted Feols model.
    nobs : int
        The number of rows in the prediction DataFrame.
    yhat : np.ndarray
        The predicted values.
    X : np.ndarray
        The design matrix, shape (nobs, k).
    alpha : float
        The confidence level.

    Returns
    -------
    prediction_df : pd.DataFrame
        The DataFrame with predictions, prediction SEs and confidence intervals.
    """
    columns = ["fit", "se_fit", "ci_low", "ci_high"]

    prediction_df = pd.DataFrame(np.nan, index=range(nobs), columns=columns)

    z_crit = t.ppf(1 - alpha / 2, model._N - model._k)
    sigma2 = np.sum(model._u_hat**2) / (model._N - model._k)

    prediction_df["fit"] = yhat
    prediction_df["se_fit"] = _get_prediction_se(model=model, X=X)
    prediction_df["ci_low"] = prediction_df["fit"] - z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )
    prediction_df["ci_high"] = prediction_df["fit"] + z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )

    return prediction_df
