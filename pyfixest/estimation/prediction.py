import warnings
from typing import Optional

import numpy as np
import pandas as pd
from formulaic import Formula

from pyfixest.utils.dev_utils import (
    _narwhals_to_pandas,
)


def get_design_matrix_and_yhat(
    model,
    newdata: Optional[pd.DataFrame] = None,
    atol: float = 1e-6,
    btol: float = 1e-6,
):
    """
    Build the design matrix X and initializes y_hat for predictions.

    Parameters
    ----------
    model : Feols
        The fitted Feols model (self inside Feols).
    newdata : Optional DataFrame
        The new data on which predictions are made, or None for original data.
    atol : float
        ...
    btol : float
        ...

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
            xfml = model._fml.split("|")[0].split("~")[1]
            if model._icovars is not None:
                raise NotImplementedError(
                    "predict() with argument newdata is not supported with i() syntax."
                )
            X = Formula(xfml).get_model_matrix(newdata)
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
    model, newdata: pd.DataFrame, atol: float = 1e-6, btol: float = 1e-6
):
    """
    Compute the fixed effect contribution to the prediction.

    Parameters
    ----------
    model : Feols
        The fitted Feols model.
    newdata : pd.DataFrame
        Data for predictions.
    atol : float
    btol : float

    Returns
    -------
    np.ndarray
        fe_hat: the sum of fixed effects contributions for each observation.
    """
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


def _get_prediction_se(model, X: np.ndarray):
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
    if X is None or X.shape[0] == 0:
        # If there's no X, or an empty design matrix, just return None arrays
        N = X.shape[0] if X is not None else 0
        return (np.full(N, np.nan),)

    return np.array(_get_newdata_stdp(model, X))


def _get_newdata_stdp(model, X):
    """
    Get standard error of predictions for new data.

    Parameters
    ----------
    X : np.ndarray
        Covariates for new data points.

    Returns
    -------
    list
        Standard errors for each prediction
    """
    # for now only compute prediction error if model has no fixed effects
    # TODO: implement for fixed effects
    if not model._has_fixef:
        if not model._X_is_empty:
            return [_get_single_row_stdp(model=model, row=row) for row in X]
        else:
            warnings.warn(
                """
                Standard error of the prediction cannot be computed if X is empty.
                Prediction dataframe stdp column will be None.
                 """
            )
    else:
        warnings.warn(
            """
            Standard error of the prediction is not implemented for fixed effects models.
            Prediction dataframe stdp column will be None.
            """
        )


def _get_single_row_stdp(model, row):
    """
    Get standard error of predictions for a single row.

    Parameters
    ----------
    row : np.ndarray
        Single row of new covariate data

    Returns
    -------
    np.ndarray
        Standard error of prediction for single row
    """
    return np.sqrt(np.linalg.multi_dot([row, model._vcov, np.transpose(row)]))
