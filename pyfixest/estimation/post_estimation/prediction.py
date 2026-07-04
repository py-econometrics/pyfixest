from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from formulaic.parser.types import Factor
from formulaic.utils.variables import get_expression_variables
from scipy.stats import t

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
    for factor_expr, value in rhs_spec.encoder_state.items():
        # formulaic internal: encoder_state values are (Factor.Kind, state_dict)
        # 2-tuples set by formulaic's materializer; this shape is undocumented.
        kind, state = value
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        for variable, levels in _categorical_levels(factor_expr, state, newdata):
            column = newdata[variable]
            # For binned i() terms, apply the stored bin mapping before checking
            # so that valid raw levels (e.g. "a" -> "low") are not flagged unseen.
            bin_key = f"__bin_mapping_{variable}__"
            if bin_key in state:
                column = column.replace(state[bin_key])
            unseen = ~column.isin(levels) & column.notna()
            mask |= unseen.to_numpy()
    return mask


def _categorical_levels(
    factor_expr: str, state: dict, newdata: pd.DataFrame
) -> Iterator[tuple[str, set]]:
    """
    Yield `(variable_name, seen_levels)` pairs for one categorical factor.

    Handles the two state layouts used in pyfixest: formulaic-native `C()` /
    bare categoricals store `categories` at the top level (the source variable
    is a data column referenced in the factor expression), while pyfixest's
    `i()` stores per-variable contrast sub-states keyed `__contrasts_<var>__`.

    Variable names for ``C(...)`` factors are extracted via
    ``get_expression_variables`` rather than regex, so keyword-argument names
    (e.g. ``base`` in ``C(f, Treatment(base='a'))``) are not mistaken for data
    columns.
    """
    if "categories" in state:
        # formulaic-native C() / bare categoricals: use structured variable
        # extraction instead of regex to avoid false positives from keyword
        # argument names that happen to match column names (e.g. `base`).
        for variable in (str(v) for v in get_expression_variables(factor_expr)):
            if variable in newdata.columns:
                yield variable, set(state["categories"])
    else:
        # formulaic internal: pyfixest's i() stores per-variable contrast state under
        # "__contrasts_<var>__" keys inside formulaic's encoder_state dict (see
        # transforms/factor_interaction.py); we read those keys back here.
        for key, substate in state.items():
            if key.startswith("__contrasts_") and key.endswith("__"):
                variable = key[len("__contrasts_") : -len("__")]
                if variable in newdata.columns and "categories" in substate:
                    yield variable, set(substate["categories"])


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
