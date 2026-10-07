from __future__ import annotations

import numpy as np
import pandas as pd


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
