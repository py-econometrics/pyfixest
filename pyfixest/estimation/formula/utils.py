import re
from enum import StrEnum

import pandas as pd


def _str_split_by_sep(string: str, separator: str = "+") -> list[str]:
    """
    Split on top-level *separator*, skipping any occurrences nested inside
    parentheses.  The main use-case is splitting formula terms on ``+``
    without breaking apart multi-estimation operators like ``sw(a, b + c)``.
    """
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
    """
    Return ``(start, end)`` indices of the content inside the first matched
    parenthesis pair, so that ``string[start:end]`` gives the inner content.

    Example: ``"sw(X1, X2)"`` → ``(3, 9)`` and ``string[3:9] == "X1, X2"``.
    """
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


class _MultipleEstimationType(StrEnum):
    # See https://lrberge.github.io/fixest/reference/stepwise.html
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"
    mvsw = "multiverse stepwise"


_MULTIPLE_ESTIMATION_PATTERN = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\b\(.+\)"
)
