import pandas as pd
from typing import Union

try:
    import polars as pl

    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame


def _polars_to_pandas(data: DataFrameType) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        try:
            # Try to import Polars
            import polars as pl

            # Convert user_input to a Polars DataFrame
            data = data.to_pandas()

        except ImportError:
            # Polars is not available
            raise ImportError(
                "Polars is not installed. Please install Polars to use it as an alternative."
            )

    return data
