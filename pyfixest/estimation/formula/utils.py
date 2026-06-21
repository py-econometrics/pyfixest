import re
import warnings
from enum import Enum

import pandas as pd

from pyfixest.errors import FormulaSyntaxError


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


class _MultipleEstimationType(Enum):
    # See https://lrberge.github.io/fixest/reference/stepwise.html
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"
    mvsw = "multiverse stepwise"


_MULTIPLE_ESTIMATION_PATTERN = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\b\(.+\)"
)


def _preprocess(formula: str) -> str:
    formula = _preprocess_fixest_instrumental_variable(formula)
    formula = _preprocess_fixest_multiple_dependents(formula)
    return formula


def _preprocess_fixest_instrumental_variable(formula: str) -> str:
    """Convert fixest-style instrumental variable syntax to formulaic.
    Y ~ X1 | X2 ~ Z2 will be converted to Y ~ X1 + [X2 ~ Z2].
    """
    parts = re.split(r"\s*\|\s*", formula)
    main = parts.pop(0)
    instrumental_variables = [part for part in parts if "~" in part]
    if len(instrumental_variables) > 1:
        raise FormulaSyntaxError()
    elif instrumental_variables:
        parts = [part for part in parts if part not in instrumental_variables]
        formula_old = formula
        formula = f"{main} + {' + '.join(f'[{iv}]' for iv in instrumental_variables)}"
        if parts:
            formula = f"{formula} | {' | '.join(parts)}"
        warnings.warn(
            "The fixest-style syntax for instrumental variable regressions is deprecated and will throw an error in a future version."
            f"Instead of `{formula_old}` use `{formula}`",
            DeprecationWarning,
            stacklevel=2,
        )
    return formula


def _preprocess_fixest_multiple_dependents(formula: str) -> str:
    """Convert multiple dependent variables to multiple estimation syntax.
    Y + Y2 ~ X1 + X2 will be converted to sw(Y, Y2) ~ X1 + X2.
    """
    if "~" not in formula:
        raise FormulaSyntaxError()
    dependent, rest = re.split(r"\s*~\s*", formula, maxsplit=1)
    if "+" in dependent:
        # Multiple dependent variables
        formula_old = formula
        formula = f"{_MultipleEstimationType.sw.name}({', '.join(_str_split_by_sep(dependent, separator='+'))}) ~ {rest}"
        warnings.warn(
            "Specifiying multiple dependent variables with `+` is deprecated and will throw an error in a future version."
            f"Instead of `{formula_old}` use `{formula}`",
            DeprecationWarning,
            stacklevel=2,
        )
    return formula
