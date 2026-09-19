from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pyfixest.utils.dev_utils import _find_stack_level


def encode_groups(data: pd.DataFrame) -> pd.Series:
    """Assign group IDs, with NaN for missing keys across pandas versions."""
    codes = (
        data.groupby(data.columns.tolist(), sort=True, dropna=True, observed=True)
        .ngroup()
        .astype("float64")
    )
    # ngroup numbers each group from 0 to the number of groups - 1
    # but pandas < 1.5 returned -1 for missing keys
    # https://github.com/pandas-dev/pandas/issues/50100
    codes.loc[codes < 0] = np.nan
    return codes


def log(array: ArrayLike) -> np.ndarray:
    """
    Compute the natural logarithm of an array, replacing non-finite values with NaN.

    Parameters
    ----------
    array : ArrayLike
        Input array for which to compute the logarithm. Missing values of
        nullable pandas dtypes (`pd.NA`) are treated as NaN.

    Returns
    -------
    np.ndarray
        Array with natural logarithm values, where non-finite results (such as
        -inf from log(0) or NaN from log(negative)) are replaced with NaN.
    """
    raw = np.asarray(array)
    values = np.where(pd.isna(raw), np.nan, raw).astype("float64")
    result = np.full_like(values, np.nan, dtype="float64")
    valid = (values > 0.0) & np.isfinite(values)
    if not valid.all():
        warnings.warn(
            f"{np.sum(~valid)} rows with infinite values detected. These rows are dropped from the model.",
            UserWarning,
            stacklevel=_find_stack_level(),
        )
    np.log(values, out=result, where=valid)
    return result
