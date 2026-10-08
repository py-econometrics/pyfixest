from __future__ import annotations

import ast
import re
import warnings
from enum import Enum

import pandas as pd
from formulaic.parser.algos import tokenize
from formulaic.parser.types import Token

from pyfixest.errors import FormulaSyntaxError
from pyfixest.utils.dev_utils import _find_stack_level


def _str_split_by_sep(string: str, separator: str = "+") -> list[str]:
    """
    Split on top-level *separator*, skipping any occurrences nested inside
    brackets. The main use-case is splitting formula terms on ``+`` without
    breaking apart multi-estimation operators like ``sw(a, b + c)`` or
    Formulaic multistage expressions like ``[x ~ z1 + z2]``.
    """
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for c in string:
        if c in "([{":
            depth += 1
        elif c in ")]}":
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
    depth: int = 1
    for position in range(position_open, len(string)):
        if string[position] == "(":
            depth += 1
        elif string[position] == ")":
            depth -= 1
            if depth == 0:
                return position_open, position
    raise ValueError(f"Unmatched '(' in `{string}`")


def _get_weights(data: pd.DataFrame, weights: str) -> pd.Series:
    w = data[weights]
    try:
        w = pd.to_numeric(w, errors="raise")
    except ValueError:
        raise ValueError(f"The weights column '{weights}' must be numeric.")
    if not (w.dropna() >= 0.0).all():
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


# Matches an operator call such as `csw(X1, X2)`, also when a space precedes the
# parenthesis (`csw (X1, X2)`) or the arguments span lines, as fixest allows.
_MULTIPLE_ESTIMATION_PATTERN = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\b\s*\(.+\)",
    flags=re.DOTALL,
)


def _preprocess(formula: str) -> str:
    formula = _preprocess_fixest_instrumental_variable(formula)
    formula = _preprocess_fixed_effect_interactions(formula)
    formula = _preprocess_fixest_multiple_dependents(formula)
    return formula


def _preprocess_fixed_effect_interactions(formula: str) -> str:
    """Translate legacy top-level fixed-effect interactions to Formulaic syntax."""
    parts = _str_split_by_sep(formula, separator="|")
    if len(parts) < 2:
        return formula

    fixed_effect_parts = _str_split_by_sep(parts[1], separator="^")
    if len(fixed_effect_parts) == 1:
        return formula

    formula_old = formula
    parts[1] = ":".join(fixed_effect_parts)
    formula = " | ".join(parts)
    warnings.warn(
        "The `^` operator for fixed-effect interactions is deprecated and will "
        "throw an error in a future version. "
        f"Instead of `{formula_old}` use `{formula}`",
        DeprecationWarning,
        stacklevel=_find_stack_level(),
    )
    return formula


def _count_formula_tildes(part: str) -> int:
    """Count formula ``~`` operators, excluding Python expressions and names."""
    count = 0
    for token in tokenize(part):
        if token.kind is Token.Kind.OPERATOR:
            count += token.token.count("~")
        elif token.kind is Token.Kind.PYTHON and _MULTIPLE_ESTIMATION_PATTERN.fullmatch(
            token.token
        ):
            start, end = _get_position_of_first_parenthesis_pair(token.token)
            count += _count_formula_tildes(token.token[start:end])
    return count


def _preprocess_fixest_instrumental_variable(formula: str) -> str:
    """Convert legacy IV syntax after rejecting misplaced IV blocks.

    ``Y ~ X1 | f1 | X2 ~ Z2`` becomes ``Y ~ X1 + [X2 ~ Z2] | f1``.
    Bracketed IV blocks belong only in the first formula part.
    """
    main_part, *fixef_iv_parts = _str_split_by_sep(formula, separator="|")
    supported = "Use `Y ~ X1 + [X2 ~ Z1] | f1` or `Y ~ X1 | f1 | X2 ~ Z1`."
    # Bracketed IV in the first part is already in Formulaic syntax.
    # Only later parts need legacy-IV conversion or misplaced-IV validation.
    iv_tilde_counts = [_count_formula_tildes(part) for part in fixef_iv_parts]
    if not any(iv_tilde_counts):
        return formula

    iv_part = fixef_iv_parts[-1]
    iv_sides = _str_split_by_sep(iv_part, separator="~")
    too_many_formula_parts = len(fixef_iv_parts) > 2
    iv_before_final_part = any(iv_tilde_counts[:-1])
    iv_lacks_single_tilde = iv_tilde_counts[-1] != 1
    iv_lacks_two_top_level_sides = len(iv_sides) != 2
    iv_has_empty_side = not all(iv_sides)

    if too_many_formula_parts:
        reason = (
            "Legacy IV syntax allows only the main formula, optional fixed effects, "
            "and one IV part."
        )
    elif iv_before_final_part:
        reason = "The legacy IV part must come last, after any fixed effects."
    elif iv_lacks_single_tilde:
        reason = "The IV part must contain exactly one formula-level `~`."
    elif iv_lacks_two_top_level_sides:
        reason = (
            "The legacy IV separator `~` must be outside brackets. "
            "Bracketed IV syntax belongs before `|`, alongside the covariates."
        )
    elif iv_has_empty_side:
        reason = "Specify endogenous variables before `~` and instruments after it."
    else:
        reason = None

    if reason is not None:
        raise FormulaSyntaxError(reason + " " + supported)

    formula_old = formula
    formula = f"{main_part} + [{iv_part}]"
    if len(fixef_iv_parts) == 2:
        fixed_effects_part = fixef_iv_parts[0]
        formula = f"{formula} | {fixed_effects_part}"
    warnings.warn(
        "The fixest-style syntax for instrumental variable regressions is deprecated and will throw an error in a future version. "
        f"Instead of `{formula_old}` use `{formula}`",
        DeprecationWarning,
        stacklevel=_find_stack_level(),
    )
    return formula


def _validate_dependent_expression(dependent: str, *, allow_sw: bool) -> None:
    """Reject formula algebra on outcomes before stepwise expansion can alter it."""
    expression = dependent.strip()
    while expression.startswith("("):
        start, end = _get_position_of_first_parenthesis_pair(string=expression)
        if end != len(expression) - 1:
            break
        expression = expression[start:end].strip()

    supported = (
        "Use `sw(Y, Y2)` for multiple dependent variables and "
        "`I(...)` for outcome arithmetic, for example `I(Y - 1)`."
    )
    match = re.match(r"(sw|sw0|csw|csw0|mvsw|c)\s*\(", expression)
    if match:
        operator = match.group(1)
        if operator != "sw":
            raise FormulaSyntaxError(
                f"`{operator}()` is not supported on the left-hand side. " + supported
            )
        start, end = _get_position_of_first_parenthesis_pair(string=expression)
        if allow_sw and end == len(expression) - 1:
            for argument in _str_split_by_sep(
                string=expression[start:end], separator=","
            ):
                _validate_dependent_expression(dependent=argument, allow_sw=False)
            return
        raise FormulaSyntaxError(
            "`sw()` must be the entire left-hand side, with one outcome expression "
            "per argument. " + supported
        )

    tokens = list(tokenize(expression))
    if not tokens or any(
        token.kind in (Token.Kind.OPERATOR, Token.Kind.VALUE) for token in tokens
    ):
        raise FormulaSyntaxError(
            f"Invalid dependent expression `{dependent}`: bare formula operators "
            "and constant outcomes are not supported on the left-hand side. "
            + supported
        )
    # Transforms and explicit Python expressions are allowed, but nested
    # stepwise calls would still be expanded by the formula-wide expander.
    for token in tokens:
        if token.kind is Token.Kind.PYTHON:
            try:
                expression_ast = ast.parse(token.token, mode="eval")
            except SyntaxError:
                # Leave malformed Python expressions to Formulaic's parser.
                continue
            for node in ast.walk(expression_ast):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {*_MultipleEstimationType.__members__, "c"}
                ):
                    raise FormulaSyntaxError(
                        "Multiple-dependent syntax cannot be nested inside an "
                        "outcome transform. " + supported
                    )


def _preprocess_fixest_multiple_dependents(formula: str) -> str:
    """Convert multiple dependent variables to multiple estimation syntax.
    Y + Y2 ~ X1 + X2 will be converted to sw(Y, Y2) ~ X1 + X2.
    """
    if "~" not in formula:
        raise FormulaSyntaxError("Formula must contain '~'.")
    dependent, rest = re.split(r"\s*~\s*", formula, maxsplit=1)
    # Only a top-level `+` separates dependents: `I(Y + Y2)` is a single
    # transformed dependent, not two.
    dependents = _str_split_by_sep(dependent, separator="+")
    for expression in dependents:
        _validate_dependent_expression(
            dependent=expression, allow_sw=len(dependents) == 1
        )
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
