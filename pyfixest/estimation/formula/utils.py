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
_MULTIPLE_ESTIMATION_CALL = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})$"
)
_SUPPORTED_IV_SYNTAX = "Use `Y ~ X1 + [X2 ~ Z1] | f1` or `Y ~ X1 | f1 | X2 ~ Z1`."


def _encloses_formula_syntax(string: str, position: int) -> bool:
    """
    Whether the bracket opening at *position* encloses formula syntax.

    Grouping parentheses, multiple-estimation calls such as ``sw(...)``, and
    standalone ``[...]`` blocks contain formula syntax. Braces, function calls
    such as ``I(...)``, subscripts, and varying slopes such as ``f1[x]``
    contain Python expressions, whose operators must stay untouched.
    """
    bracket = string[position]
    if bracket == "{":
        return False
    prefix = string[:position].rstrip()
    attached = bool(prefix) and (prefix[-1].isalnum() or prefix[-1] in "_.)]}`'\"")
    if bracket == "(" and attached:
        return _MULTIPLE_ESTIMATION_CALL.search(string[:position]) is not None
    return not attached


def _in_formula_syntax(string: str, brackets: tuple[int, ...]) -> bool:
    """Whether every enclosing bracket keeps a character in formula syntax."""
    return all(_encloses_formula_syntax(string, opening) for opening in brackets)


def _preprocess(formula: str) -> str:
    formula = _preprocess_fixest_instrumental_variable(formula)
    formula = _preprocess_fixed_effect_interactions(formula)
    formula = _preprocess_fixest_multiple_dependents(formula)
    return formula


def _preprocess_fixed_effect_interactions(formula: str) -> str:
    """Translate legacy FE interactions, including inside stepwise calls.

    ``^`` becomes ``:`` at the top level of the fixed-effects part and inside
    grouping parentheses and multiple-estimation calls, which the expansion
    turns into grouping parentheses. Python expressions, quoted names, and
    varying-slope expressions keep their ``^``.
    """
    parts = _str_split_by_sep(formula, separator="|")
    if len(parts) < 2:
        return formula

    fixed_effects = parts[1]
    characters = list(fixed_effects)
    for position, char, brackets in _iter_formula_characters(fixed_effects):
        if char == "^" and _in_formula_syntax(fixed_effects, brackets):
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


def _validate_formula_parts(formula: str) -> list[str]:
    """
    Split *formula* on top-level ``|`` and validate its multipart structure.

    Supported spellings are ``Y ~ X1 + [X2 ~ Z1] | f1`` and the deprecated
    ``Y ~ X1 | f1 | X2 ~ Z1``. Validation runs before any deprecation rewrite
    so that misplaced IV blocks cannot turn fixed effects into covariates.
    """
    for _position, char, brackets in _iter_formula_characters(formula):
        if char == "|" and brackets and _in_formula_syntax(formula, brackets):
            raise FormulaSyntaxError(
                "`|` separates formula parts and cannot appear inside "
                f"parentheses, brackets, or multiple-estimation calls in `{formula}`. "
                + _SUPPORTED_IV_SYNTAX
            )
    parts = _str_split_by_sep(formula, separator="|")
    for part in parts[1:]:
        for _position, char, brackets in _iter_formula_characters(part):
            if char == "~" and brackets and _in_formula_syntax(part, brackets):
                raise FormulaSyntaxError(
                    "An instrumental-variable block can only appear among the "
                    "covariates or as the last part after the fixed effects, "
                    f"not in `{part}`. " + _SUPPORTED_IV_SYNTAX
                )
    iv_parts = [
        index
        for index, part in enumerate(parts[1:], start=1)
        if len(_str_split_by_sep(part, separator="~")) > 1
    ]
    if len(iv_parts) > 1:
        raise FormulaSyntaxError(
            "Only one instrumental variable block is supported. "
            "Use a single `[endogenous ~ instruments]` block."
        )
    if len(parts) > 3 or (len(parts) == 3 and not iv_parts):
        raise FormulaSyntaxError(
            f"Invalid formula parts in `{formula}`. " + _SUPPORTED_IV_SYNTAX
        )
    if iv_parts:
        if iv_parts[0] != len(parts) - 1:
            raise FormulaSyntaxError(
                "The IV part must come last. " + _SUPPORTED_IV_SYNTAX
            )
        iv_sides = _str_split_by_sep(parts[-1], separator="~")
        if len(iv_sides) != 2 or not all(iv_sides):
            raise FormulaSyntaxError(
                f"Invalid IV part `{parts[-1]}`. " + _SUPPORTED_IV_SYNTAX
            )
    return parts


def _preprocess_fixest_instrumental_variable(formula: str) -> str:
    """Convert fixest-style instrumental variable syntax to formulaic.
    Y ~ X1 | X2 ~ Z2 will be converted to Y ~ X1 + [X2 ~ Z2].
    """
    parts = _validate_formula_parts(formula)
    if len(parts) == 1 or len(_str_split_by_sep(parts[-1], separator="~")) == 1:
        return formula
    formula_old = formula
    formula = f"{parts[0]} + [{parts[-1]}]"
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
