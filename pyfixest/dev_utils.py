import re
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

def _select_order_coefs(
    coefs: list,
    keep: Optional[Union[list, str]] = [],
    drop: Optional[Union[list, str]] = [],
    exact_match: Optional[bool] = False,
):
    r"""
    Select and order the coefficients based on the pattern.

    Parameters
    ----------
    coefs: list
        Coefficient names to be selected and ordered.
    keep: str or list of str, optional
        The pattern for retaining coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Default is keeping all coefficients.
        You should use regular expressions to select coefficients.
            "age",            # would keep all coefficients containing age
            r"^tr",           # would keep all coefficients starting with tr
            r"\\d$",          # would keep all coefficients ending with number
        Output will be in the order of the patterns.
    drop: str or list of str, optional
        The pattern for excluding coefficient names. You can pass a string (one
        pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
        Default is keeping all coefficients. Parameter `keep` and `drop` can be
        used simultaneously.
    exact_match: bool, optional
        Whether to use exact match for `keep` and `drop`. Default is False.
        If True, the pattern will be matched exactly to the coefficient name
        instead of using regular expressions.

    Returns
    -------
    res: list
        The filtered and ordered coefficient names.
    """
    if isinstance(keep, str):
        keep = [keep]
    if isinstance(drop, str):
        drop = [drop]

    coefs = list(coefs)
    res = [] if keep else coefs[:]  # Store matched coefs
    for pattern in keep:
        _coefs = []  # Store remaining coefs
        for coef in coefs:
            if (
                exact_match
                and pattern == coef
                or exact_match is False
                and re.findall(pattern, coef)
            ):
                res.append(coef)
            else:
                _coefs.append(coef)
        coefs = _coefs

    for pattern in drop:
        _coefs = []
        for coef in res:  # Remove previously matched coefs that match the drop pattern
            if (
                exact_match
                and pattern == coef
                or exact_match is False
                and re.findall(pattern, coef)
            ):
                continue
            else:
                _coefs.append(coef)
        res = _coefs

    return res
