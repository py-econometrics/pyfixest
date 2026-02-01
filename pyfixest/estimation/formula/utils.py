import re
import warnings

import numpy as np
import pandas as pd


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
    result = np.full_like(array, np.nan, dtype="float64")
    valid = (array > 0.0) & np.isfinite(array)
    if not valid.all():
        warnings.warn(
            f"{np.sum(~valid)} rows with infinite values detected. These rows are dropped from the model.",
        )
    np.log(array, out=result, where=valid)
    return result


def _split_paranthesis_preserving(string: str, separator: str) -> list[str]:
    """Split on top-level separator, respecting nested parentheses."""
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for c in string:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == separator and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(c)
    args.append("".join(current).strip())
    return args


def _get_position_of_first_parenthesis_pair(string: str) -> tuple[int, int]:
    position_open = string.find("(")
    if position_open == -1:
        raise ValueError(f"No parenthesis in `{string}`")
    else:
        position_open += 1
    position: int = position_open
    depth: int = 1
    while position < len(string) and depth:
        position += 1
        if string[position] == "(":
            depth += 1
        elif string[position] == ")":
            depth -= 1
    if depth != 0:
        raise ValueError(f"Unmatched '(' in `{string}`")
    return position_open, position


def _interact_fixed_effects(fixed_effects: str, data: pd.DataFrame) -> pd.DataFrame:
    fes = re.split(r"\s*\+\s*", fixed_effects)
    for fixed_effect in fes:
        if "^" not in fixed_effect:
            continue
        # Encode interacted fixed effects
        vars = fixed_effect.split("^")
        data[fixed_effect.replace("^", "_")] = (
            data[vars[0]]
            .astype(pd.StringDtype())
            .str.cat(
                data[vars[1:]].astype(pd.StringDtype()),
                sep="^",
                na_rep=None,  # a row containing a missing value in any of the columns (before concatenation) will have a missing value in the result
            )
        )
    return data.loc[:, [fe.replace("^", "_") for fe in fes]]


def _factorize(series: pd.Series) -> np.ndarray:
    factorized, _ = pd.factorize(series, use_na_sentinel=True)
    # use_sentinel=True replaces np.nan with -1, so we revert to np.nan
    factorized = np.where(factorized == -1, np.nan, factorized)
    return factorized


def _get_weights(data: pd.DataFrame, weights: str) -> pd.Series:
    w = data[weights]
    try:
        w = pd.to_numeric(w, errors="raise")
    except ValueError:
        raise ValueError(f"The weights column '{weights}' must be numeric.")
    if not (w.dropna() > 0.0).all():
        raise ValueError(
            f"The weights column '{weights}' must have only non-negative values."
        )
    return w
