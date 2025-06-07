import re
from typing import Optional, Union

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoDataFrame

DataFrameType = IntoDataFrame

def _narwhals_to_pandas(data: IntoDataFrame) -> pd.DataFrame:  # type: ignore
    return nw.from_native(data, eager_or_interchange_only=True).to_pandas()


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
    keep: Optional[Union[list, str]] = None,
    drop: Optional[Union[list, str]] = None,
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
    if keep is None:
        keep = []
    if drop is None:
        drop = []

    if isinstance(keep, str):
        keep = [keep]
    if isinstance(drop, str):
        drop = [drop]

    coefs = list(coefs)
    res = [] if keep else coefs[:]  # Store matched coefs
    for pattern in keep:
        _coefs = []  # Store remaining coefs
        for coef in coefs:
            if (exact_match and pattern == coef) or (
                exact_match is False and re.findall(pattern, coef)
            ):
                res.append(coef)
            else:
                _coefs.append(coef)
        coefs = _coefs

    for pattern in drop:
        _coefs = []
        for coef in res:  # Remove previously matched coefs that match the drop pattern
            if (exact_match and pattern == coef) or (
                exact_match is False and re.findall(pattern, coef)
            ):
                continue
            else:
                _coefs.append(coef)
        res = _coefs

    return res


def docstring_from(func, custom_doc=""):
    """Copy the docstring of another function."""

    def decorator(target_func):
        target_func.__doc__ = custom_doc + "\n\n" + func.__doc__
        return target_func

    return decorator


def _check_series_or_dataframe(x: Union[pd.Series, pd.DataFrame]):
    if not isinstance(x, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame")
    else:
        return x


def _to_list(x):
    if x is not None and not isinstance(x, list):
        return [x]
    return x


def _drop_cols(_data: pd.DataFrame, na_index: np.ndarray):
    """
    Drop columns from data based on the indices in na_index.

    Parameters
    ----------
    _data : pd.DataFrame
        The input DataFrame.
    na_index : np.ndarray
        An array of indices to drop.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with NAs dropped.
    """
    if na_index.size > 0:
        all_indices = np.arange(_data.shape[0])
        max_index = all_indices.max() + 1
        keep = np.ones(max_index, dtype=bool)
        keep[na_index] = False
        return _data[keep]
    else:
        return _data


def _extract_variable_level(fe_string: str):
    """
    Extract the variable and level from a given string.

    Parameters
    ----------
    fe_string: str
        The string encapsulating the fixed effect factor variable and level.

    Returns
    -------
    tuple
        A tuple containing the extracted variable and level for the fixed
        effect.
    """
    pattern = r"C\(([^)]*)\)\[(?:T\.)?(.*)\]$"
    match = re.search(pattern, fe_string)
    if not match:
        raise ValueError(f"Cannot parse: {fe_string}")

    variable = match.group(1)
    level = match.group(2)

    return f"C({variable})", level
