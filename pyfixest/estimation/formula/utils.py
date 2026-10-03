from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from enum import Enum

import pandas as pd

from pyfixest.errors import FormulaSyntaxError
from pyfixest.utils.dev_utils import _find_stack_level


def _iter_formula_characters(string: str) -> Iterator[tuple[int, str, tuple[int, ...]]]:
    """Yield unquoted, unescaped characters and their enclosing bracket positions."""
    brackets: list[int] = []
    quote: str | None = None
    escaped = False
    for position, char in enumerate(string):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "\"'`":
            quote = char
            continue
        yield position, char, tuple(brackets)
        if char in "([{":
            brackets.append(position)
        elif char in ")]}" and brackets:
            brackets.pop()


def _str_split_by_sep(string: str, separator: str = "+") -> list[str]:
    """
    Split on top-level *separator*, skipping any occurrences nested inside
    brackets, quotes, or escapes. The main use-case is splitting terms on ``+`` without
    breaking apart multi-estimation operators like ``sw(a, b + c)`` or
    Formulaic multistage expressions like ``[x ~ z1 + z2]``.
    """
    args: list[str] = []
    start = 0
    for position, char, brackets in _iter_formula_characters(string):
        if char == separator and not brackets:
            args.append(string[start:position].strip())
            start = position + 1
    args.append(string[start:].strip())
    return args


def _get_position_of_first_parenthesis_pair(string: str) -> tuple[int, int]:
    """
    Return ``(start, end)`` indices of the content inside the first matched
    parenthesis pair, so that ``string[start:end]`` gives the inner content.

    Example: ``"sw(X1, X2)"`` → ``(3, 9)`` and ``string[3:9] == "X1, X2"``.
    """
    position_open = None
    for position, char, brackets in _iter_formula_characters(string):
        if char == "(" and position_open is None:
            position_open = position
        elif char == ")" and brackets and brackets[-1] == position_open:
            return position_open + 1, position
    if position_open is None:
        raise ValueError(f"No parenthesis in `{string}`")
    raise ValueError(f"Unmatched '(' in `{string}`")


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
    formula = _preprocess_fixed_effect_interactions(formula)
    formula = _preprocess_fixest_multiple_dependents(formula)
    return formula


def _preprocess_fixed_effect_interactions(formula: str) -> str:
    """Translate legacy FE interactions, including inside stepwise calls."""
    parts = _str_split_by_sep(formula, separator="|")
    if len(parts) < 2:
        return formula

    fixed_effects = parts[1]
    stepwise_openings = {
        match.end() - 1
        for match in re.finditer(
            rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\(",
            fixed_effects,
        )
    }
    characters = list(fixed_effects)
    for position, char, brackets in _iter_formula_characters(fixed_effects):
        if char == "^" and all(opening in stepwise_openings for opening in brackets):
            characters[position] = ":"
    normalized = "".join(characters)
    if normalized == fixed_effects:
        return formula

    formula_old = formula
    parts[1] = normalized
    formula = " | ".join(parts)
    warnings.warn(
        "The `^` operator for fixed-effect interactions is deprecated and will "
        "throw an error in a future version. "
        f"Instead of `{formula_old}` use `{formula}`",
        DeprecationWarning,
        stacklevel=_find_stack_level(),
    )
    return formula


def _preprocess_fixest_instrumental_variable(formula: str) -> str:
    """Convert fixest-style instrumental variable syntax to formulaic.
    Y ~ X1 | X2 ~ Z2 will be converted to Y ~ X1 + [X2 ~ Z2].
    """
    parts = _str_split_by_sep(formula, separator="|")
    supported_syntax = "Use `Y ~ X1 + [X2 ~ Z1] | f1` or `Y ~ X1 | f1 | X2 ~ Z1`."
    # Validate before any deprecation rewrite can move FE terms into covariates.
    for part in parts[1:]:
        iv_openings: set[int] = set()
        for position, char, brackets in _iter_formula_characters(part):
            if char == "[":
                prefix = part[:position].rstrip()
                # A standalone bracket starts an IV block; subscripts and
                # varying slopes attach to the preceding factor instead.
                if not prefix or prefix[-1] in "+-*/:(,[~|":
                    iv_openings.add(position)
            if char == "~" and any(opening in iv_openings for opening in brackets):
                raise FormulaSyntaxError(
                    "A bracketed IV block cannot appear in the fixed-effects part. "
                    + supported_syntax
                )
    instrumental_variables = [
        index
        for index, part in enumerate(parts[1:], start=1)
        if len(_str_split_by_sep(part, separator="~")) > 1
    ]
    if len(instrumental_variables) > 1:
        raise FormulaSyntaxError(
            "Only one instrumental variable block is supported. "
            "Use a single `[endogenous ~ instruments]` block."
        )
    if len(parts) > 3 or (len(parts) == 3 and not instrumental_variables):
        raise FormulaSyntaxError("Invalid formula parts. " + supported_syntax)
    if instrumental_variables:
        if instrumental_variables[0] != len(parts) - 1:
            raise FormulaSyntaxError("The IV part must come last. " + supported_syntax)
        iv = parts[-1]
        iv_sides = _str_split_by_sep(iv, separator="~")
        if len(iv_sides) != 2 or not all(iv_sides):
            raise FormulaSyntaxError("Invalid IV part. " + supported_syntax)
        formula_old = formula
        formula = f"{parts[0]} + [{iv}]"
        if len(parts) == 3:
            formula = f"{formula} | {parts[1]}"
        warnings.warn(
            "The fixest-style syntax for instrumental variable regressions is deprecated and will throw an error in a future version. "
            f"Instead of `{formula_old}` use `{formula}`",
            DeprecationWarning,
            stacklevel=_find_stack_level(),
        )
    return formula


def _preprocess_fixest_multiple_dependents(formula: str) -> str:
    """Convert multiple dependent variables to multiple estimation syntax.
    Y + Y2 ~ X1 + X2 will be converted to sw(Y, Y2) ~ X1 + X2.
    """
    parts = _str_split_by_sep(formula, separator="~")
    if len(parts) < 2:
        raise FormulaSyntaxError("Formula must contain '~'.")
    dependent, rest = parts[0], " ~ ".join(parts[1:])
    # Only a top-level `+` separates dependents: `I(Y + Y2)` is a single
    # transformed dependent, not two.
    dependents = _str_split_by_sep(dependent, separator="+")
    if len(dependents) > 1:
        formula_old = formula
        formula = f"{_MultipleEstimationType.sw.name}({', '.join(dependents)}) ~ {rest}"
        warnings.warn(
            "Specifiying multiple dependent variables with `+` is deprecated and will throw an error in a future version. "
            f"Instead of `{formula_old}` use `{formula}`",
            DeprecationWarning,
            stacklevel=_find_stack_level(),
        )
    return formula
