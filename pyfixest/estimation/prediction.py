import warnings
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.stats import t

from pyfixest.utils.dev_utils import (
    DataFrameType,
    _narwhals_to_pandas,
)


def get_design_matrix_and_yhat(
    model,
    newdata: Optional[DataFrameType] = None,
    context: Optional[Union[int, Mapping[str, Any]]] = None,
):
    """
    Build the design matrix X and initializes y_hat for predictions.

    Parameters
    ----------
    model : Feols
        The fitted Feols model (self inside Feols).
    newdata : DataFrameType, optional
        The new data on which predictions are made, or None for original data.

    Returns
    -------
    X : np.ndarray or None
        The design matrix. None if X is empty.
    y_hat : np.ndarray
        The (initial) prediction for each observation, ignoring fixed effects.
    X_index : pd.Index or None
        The index of rows used in X (if needed).
    """
    if newdata is None:
        y_hat = model._Y_untransformed.to_numpy().flatten() - model.resid()
        xfml = model._fml.split("|")[0].split("~")[1]
        X = Formula(xfml).get_model_matrix(model._data)
        X_index = X.index

        coef_idx = np.isin(model._coefnames, X.columns)
        X = X[np.array(model._coefnames)[coef_idx]]
        X = X.to_numpy()
        return y_hat, X, X_index

    else:
        # Convert newdata to a Pandas DataFrame if needed
        newdata = _narwhals_to_pandas(newdata).reset_index(drop=False)

        if not model._X_is_empty:
            if model._icovars is not None:
                raise NotImplementedError(
                    "predict() with argument newdata is not supported with i() syntax."
                )

            if hasattr(model, "_model_spec") and model._model_spec is not None:
                rhs_spec = model._model_spec.fml_second_stage.rhs
                X = rhs_spec.get_model_matrix(newdata, context=context)
            else:
                xfml = model._fml.split("|")[0].split("~")[1]
                X = Formula(xfml).get_model_matrix(newdata, context=context)

            X_index = X.index

            coef_idx = np.isin(model._coefnames, X.columns)
            X = X[np.array(model._coefnames)[coef_idx]]
            X = X.to_numpy()

            # Initialize y_hat with NaNs, fill in only for valid (X_index) rows
            y_hat = np.full(newdata.shape[0], np.nan)
            y_hat[X_index] = X @ model._beta_hat[coef_idx]
        else:
            # If no X in the model, predictions start at zero
            X = None
            X_index = None
            y_hat = np.zeros(newdata.shape[0])

        return y_hat, X, X_index


def _get_fixed_effects_prediction_component(
    model, newdata: DataFrameType, atol: float = 1e-6, btol: float = 1e-6
):
    """
    Compute the fixed effect contribution to the prediction.

    Parameters
    ----------
    model : Feols
        The fitted Feols model.
    newdata : DataFrameType
        Data for predictions.
    atol : float
    btol : float

    Returns
    -------
    np.ndarray
        fe_hat: the sum of fixed effects contributions for each observation.
    """
    # Convert newdata to a Pandas DataFrame if needed
    newdata = _narwhals_to_pandas(newdata).reset_index(drop=False)

    fe_hat = np.zeros(newdata.shape[0])

    if model._has_fixef:
        if model._sumFE is None:
            model.fixef(atol, btol)

        fvals = model._fixef.split("+")

        # warn if newdata types do not match
        mismatched_fixef_types = [
            x for x in fvals if newdata[x].dtypes != model._data[x].dtypes
        ]
        if mismatched_fixef_types:
            warnings.warn(
                f"Data types of fixed effects {mismatched_fixef_types} do not match "
                "the model data. This leads to possible mismatched keys in the fixed "
                "effect dictionary -> NaN predictions for new levels."
            )

        df_fe = newdata[fvals].astype(str)

        # populate fixed effect dicts with omitted categories handling
        fixef_dicts = {}
        for f in fvals:
            fdict = model._fixef_dict[f"C({f})"]
            omitted_cat = set(model._data[f].unique().astype(str).tolist()) - set(
                fdict.keys()
            )
            if omitted_cat:
                fdict.update({x: 0 for x in omitted_cat})
            fixef_dicts[f"C({f})"] = fdict

        _fixef_mat = _apply_fixef_numpy(df_fe.values, fixef_dicts)
        fe_hat += np.sum(_fixef_mat, axis=1)

    return fe_hat


def _apply_fixef_numpy(df_fe_values, fixef_dicts):
    fixef_mat = np.zeros_like(df_fe_values, dtype=float)
    for i, (_, subdict) in enumerate(fixef_dicts.items()):
        unique_levels, inverse = np.unique(df_fe_values[:, i], return_inverse=True)
        mapping = np.array([subdict.get(level, np.nan) for level in unique_levels])
        fixef_mat[:, i] = mapping[inverse]

    return fixef_mat


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
    model, nobs: int, yhat: np.ndarray, X: np.ndarray, X_index: np.ndarray, alpha: float
) -> pd.DataFrame:
    """
    Fill a DataFrame with predictions and confidence intervals.

    Parameters
    ----------
    model : Feols
        The fitted Feols model.
    nobs : int
        The number of rows in the prediction DataFrame.
    yhat : np.ndarray
        The predicted values.
    X : np.ndarray
        The design matrix.
    X_index : np.ndarray
        The index of rows used in X.
    alpha : float
        The confidence level.

    Returns
    -------
    prediction_df : pd.DataFrame
        The DataFrame with predictions, prediction SEs and confidence intervals.
    """
    columns = ["fit", "se_fit", "ci_low", "ci_high"]

    prediction_df = pd.DataFrame(np.nan, index=range(nobs), columns=columns)

    prediction_df["fit"] = yhat
    prediction_df.loc[X_index, "se_fit"] = _get_prediction_se(model=model, X=X)
    z_crit = t.ppf(1 - alpha / 2, model._N - model._k)
    sigma2 = np.sum(model._u_hat**2) / (model._N - model._k)
    prediction_df.loc[X_index, "ci_low"] = prediction_df["fit"] - z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )
    prediction_df.loc[X_index, "ci_high"] = prediction_df["fit"] + z_crit * np.sqrt(
        prediction_df["se_fit"] ** 2 + sigma2
    )

    return prediction_df
