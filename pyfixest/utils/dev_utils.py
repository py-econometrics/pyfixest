import inspect
import os
import re

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoDataFrame

DataFrameType = IntoDataFrame

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _find_stack_level() -> int:
    """
    Return the `stacklevel` that attributes a warning to the caller of pyfixest.

    Use as `warnings.warn(message, category, stacklevel=_find_stack_level())`.
    The call stack is walked outward from the frame that issues the warning,
    and the returned level points just past the outermost pyfixest frame. A
    warning therefore names the user's call site however deep inside pyfixest
    it is raised, including pyfixest code called back by a third-party library
    such as a formulaic transform. Adapted from pandas' `find_stack_level`.

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
            if frame.f_code.co_filename.startswith(_PACKAGE_DIR):
                outermost_level = level
            frame = frame.f_back
            level += 1
    finally:
        del frame
    return outermost_level + 1


def _narwhals_to_pandas(data: IntoDataFrame) -> pd.DataFrame:
    return nw.from_native(data, eager_or_interchange_only=True).to_pandas()


def _create_rng(seed: int | None = None) -> np.random.Generator:
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
        raise ValueError("No coefficients match the keep/drop patterns.")
    return selected, indices
