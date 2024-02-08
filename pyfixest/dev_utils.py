import importlib.util
from typing import Union

import pandas as pd

try:
    import polars as pl

    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame


def _polars_to_pandas(data: DataFrameType) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        polars_spec = importlib.util.find_spec("polars")
        if polars_spec is not None:
            return data.to_pandas()
        else:
            raise ImportError(
                "Polars is not installed. Please install Polars to use it as an alternative."
            )
    return data

