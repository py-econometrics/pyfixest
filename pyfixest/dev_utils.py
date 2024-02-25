from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import polars as pl

    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame


def _polars_to_pandas(data: DataFrameType) -> pd.DataFrame:  # type: ignore
    if not isinstance(data, pd.DataFrame):
        try:
            import polars as pl  # noqa: F401

            data = data.to_pandas()
        except ImportError:
            raise ImportError(
                """Polars is not installed. Please install Polars to use it as
                an alternative."""
            )

    return data


def _create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a random number generator.

    Parameters
    ----------
    seed : int, optional
        The seed of the random number generator. If None, a random seed is chosen.

    Returns
    -------
    numpy.random.Generator
        A random number generator.
    """
    if seed is None:
        seed = np.random.randint(100_000_000)
    return np.random.default_rng(seed)
