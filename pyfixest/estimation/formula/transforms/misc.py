from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def encode_groups(data: pd.DataFrame) -> pd.Series:
    """Assign one integer ID to each distinct row of grouping variables."""
    return data.groupby(data.columns.tolist()).ngroup()


def log(array: np.ndarray) -> np.ndarray:
    """
    Compute the natural logarithm of an array, replacing non-finite values with NaN.

    Parameters
    ----------
    array : np.ndarray
        Input array for which to compute the logarithm.

    Returns
    -------
    np.ndarray
        Array with natural logarithm values, where non-finite results (such as
        -inf from log(0) or NaN from log(negative)) are replaced with NaN.
    """
    result = np.full_like(array, np.nan, dtype="float64")
    valid = (array > 0.0) & np.isfinite(array)
    if not valid.all():
        warnings.warn(
            f"{np.sum(~valid)} rows with infinite values detected. These rows are dropped from the model.",
        )
    np.log(array, out=result, where=valid)
    return result
