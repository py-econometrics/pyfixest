import pandas as pd
from typing import Union

try:
    import polars as pl
    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame