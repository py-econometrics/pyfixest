import inspect
import os
import re
from collections.abc import Callable, Iterable

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoDataFrame

DataFrameType = IntoDataFrame

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _find_stack_level() -> int:
    """
    Return the `stacklevel` that attributes a warning to the caller of pyfixest.

    Follows the approach of pandas' `find_stack_level`
    (https://github.com/pandas-dev/pandas/blob/v3.0.1/pandas/util/_exceptions.py#L37-L63),
    but attributes the warning to the frame past the outermost pyfixest frame.

    Returns
    -------
    int
        The `stacklevel` argument for `warnings.warn`.
    """
    frame = inspect.currentframe()
    outermost_level = 1
    try:
        # Level 1 is the function that calls `warnings.warn`.
        frame = frame.f_back if frame is not None else None
        level = 1
        while frame is not None:
            filename = frame.f_code.co_filename
            # `warnings.warn` skips import-machinery frames when it counts
            # `stacklevel`, so they must not count here either.
            if "importlib" in filename and "_bootstrap" in filename:
                frame = frame.f_back
                continue
            if filename.startswith(_PACKAGE_DIR):
                outermost_level = level
            frame = frame.f_back
            level += 1
    finally:
        del frame
    return outermost_level + 1


def _narwhals_to_pandas(data: IntoDataFrame, *, argument: str = "data") -> pd.DataFrame:
    try:
        frame = nw.from_native(data, eager_or_interchange_only=True)
    except TypeError as exc:
        raise TypeError(
            f"{argument} must be a supported DataFrame, such as a pandas or Polars "
            f"DataFrame; got {type(data).__name__}. Convert {argument} before passing it."
        ) from exc
    return frame.to_pandas()


def _missing_columns(data: pd.DataFrame, columns: Iterable[str | None]) -> list[str]:
    """
    Return the entries of ``columns`` that do not name a column of ``data``.

    ``None`` entries are skipped, so optional arguments can be passed through
    unchanged.

    Parameters
    ----------
    data : pd.DataFrame
        The frame whose columns are checked.
    columns : Iterable[str | None]
        The column names to look up; ``None`` entries are ignored.

    Returns
    -------
    list[str]
        The entries of ``columns`` absent from ``data.columns``, in order.
    """
    return [
        column
        for column in columns
        if column is not None and column not in data.columns
    ]


def _require_columns(
    data: pd.DataFrame,
    columns: Iterable[tuple[str, str | None]],
    *,
    error: type[Exception] = ValueError,
    message: Callable[[str, str], str],
) -> None:
    """
    Raise for the first named column that is absent from ``data``.

    Parameters
    ----------
    data : pd.DataFrame
        The frame whose columns are checked.
    columns : Iterable[tuple[str, str | None]]
        Pairs of an argument label and the column name to validate. ``None``
        names are skipped, so optional arguments need no guard.
    error : type[Exception], optional
        The exception type to raise. Default is `ValueError`.
    message : Callable[[str, str], str]
        Builds the exception message from the argument label and the missing
        column name.

    Raises
    ------
    Exception
        ``error(message(label, column))`` for the first missing column.
    """
    for label, column in columns:
        if column is not None and _missing_columns(data, [column]):
            raise error(message(label, column))


def _create_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create a random number generator.

    Parameters
    ----------
    seed : int, optional
        The seed of the random number generator. If None, the generator draws
        fresh entropy from the operating system; the global NumPy RNG is
        neither read nor modified.

    Returns
    -------
    numpy.random.Generator
        A random number generator.
    """
    return np.random.default_rng(seed)


def _select_order_coefs(
    coefs: list,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
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


def _select_coefnames_and_indices(
    coefnames_all: list,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
) -> tuple[list[str], list[int]]:
    if keep is None:
        keep = []
    if drop is None:
        drop = []

    if keep or drop:
        if isinstance(keep, str):
            keep = [keep]
        if isinstance(drop, str):
            drop = [drop]
        selected = _select_order_coefs(coefnames_all, keep, drop, bool(exact_match))
    else:
        selected = coefnames_all

    indices = [coefnames_all.index(name) for name in selected]
    if not indices:
        raise ValueError(
            f"No coefficients match the keep/drop patterns (keep={keep!r}, drop={drop!r}). "
            "Compare the patterns with fit.coef().index."
        )
    return selected, indices
