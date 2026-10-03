import warnings

import numpy as np
import pandas as pd

from pyfixest.utils.dev_utils import _find_stack_level


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
