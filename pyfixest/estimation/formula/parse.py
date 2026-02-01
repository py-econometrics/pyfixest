import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pyfixest.errors import FormulaSyntaxError
from pyfixest.estimation.formula.utils import (
    _get_position_of_first_parenthesis_pair,
    _split_paranthesis_preserving,
)


@dataclass(kw_only=True, frozen=True, slots=True)
class Formula:
    """A formulaic-compliant formula."""

    second_stage: str
    fixed_effects: str | None
    first_stage: str | None

    @property
    def formula(self) -> str:
        """Full fixest-style formula."""
        formula = self.second_stage
        if self.fixed_effects is not None:
            formula = f"{formula} | {self.fixed_effects}"
        if self.first_stage is not None:
            formula = f"{formula} | {self.first_stage}"
        return formula

    @classmethod
    def parse(cls, formula: str) -> list["Formula"]:
        """Parse fixest-style formula."""
        _validate(formula)
        formula = _preprocess(formula)
        return [
            _split_formula_into_parts(formula)
            for formula in _expand_all_multiple_estimation(formula)
        ]

    @classmethod
    def parse_to_dict(cls, formula: str) -> dict[str | None, list["Formula"]]:
        """Group parsed formulas into dictionary keyed by fixed effects."""
        formulas = cls.parse(formula)
        result: dict[str | None, list[Formula]] = {}
        for parsed_formula in formulas:
            result.setdefault(parsed_formula.fixed_effects, []).append(parsed_formula)
        return result


class _MultipleEstimationType(StrEnum):
    # See https://lrberge.github.io/fixest/reference/stepwise.html
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"


_MULTIPLE_ESTIMATION_PATTERN = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\b\(.+\)"
)


def _validate(formula: str) -> None:
    max_parts: Final[int] = 3
    parts = _split_paranthesis_preserving(string=formula, separator="|")

    # Check: at most 3 parts
    if len(parts) > max_parts:
        raise FormulaSyntaxError(
            f"Formula can have at most {max_parts} parts separated by '|'. "
            f"Received {len(parts)}: '{formula}'"
        )

    # Check: no part has more than one tilde
    parts_with_multiple_tildes = [p for p in parts if p.count("~") > 1]
    if parts_with_multiple_tildes:
        raise FormulaSyntaxError(
            f"Each formula part can contain at most one '~'. "
            f"Invalid parts: {parts_with_multiple_tildes}"
        )

    # Check structure based on number of parts
    if len(parts) == 1 and "~" not in parts[0]:
        # Format: Y ~ X
        raise FormulaSyntaxError(f"Formula must contain '~': '{formula}'")
    elif len(parts) == 2 and "~" not in parts[0]:
        # Format: Y ~ X | fe  OR  Y ~ X | endog ~ instr
        # Part 0 must have a tilde
        raise FormulaSyntaxError(
            f"First part must contain '~' (dependent ~ independent): '{parts[0]}'"
        )
    elif len(parts) == 3:
        # Format: Y ~ X | fe | endog ~ instr
        # Parts 0 and 2 must have tildes, part 1 must NOT
        if "~" not in parts[0]:
            raise FormulaSyntaxError(
                f"First part must contain '~' (dependent ~ independent): '{parts[0]}'"
            )
        if "~" in parts[1]:
            raise FormulaSyntaxError(
                f"Second part (fixed effects) cannot contain '~': '{parts[1]}'. "
                "Fixed effects should be specified as 'f1 + f2', not as a formula."
            )
        if "~" not in parts[2]:
            raise FormulaSyntaxError(
                "Three-part formula requires IV specification in third part: "
                "'dependent ~ independent | fixed_effects | endogenous ~ instruments'. "
            )


def _preprocess(formula: str) -> str:
    """Convert multiple dependent variables to multiple estimation syntax.
    Y + Y2 ~ X1 + X2 will be converted to sw(Y, Y2) ~ X1 + X2.
    """
    dependents, rhs = re.split(r"\s*~\s*", formula, maxsplit=1)
    dependents = _split_paranthesis_preserving(dependents.strip(), separator="+")
    if len(dependents) > 1:
        # Multiple dependent variables
        formula = f"sw({', '.join(dependents)}) ~ {rhs}"
    return formula


def _expand_first_multiple_estimation(formula: str) -> list[str] | None:
    """Expand the first multiple estimation syntax in formula."""
    match = _MULTIPLE_ESTIMATION_PATTERN.search(formula)
    if not match:
        return None
    kind = _MultipleEstimationType[match.group(1)]
    paranthesis_open, paranthesis_closed = _get_position_of_first_parenthesis_pair(
        string=formula[match.start() :]
    )
    paranthesis_open += match.start()
    paranthesis_closed += match.start()
    arguments = _split_paranthesis_preserving(
        string=formula[paranthesis_open:paranthesis_closed],
        separator=",",
    )
    if kind.name.startswith("csw"):
        # Cumulative stepwise
        arguments = [" + ".join(arguments[: i + 1]) for i, _ in enumerate(arguments)]
    if kind.name.endswith("0"):
        # Add zero step
        arguments = ["1", *arguments]
    multiple_estimation_call = formula[match.start() : paranthesis_closed + 1]
    return [
        formula.replace(multiple_estimation_call, argument) for argument in arguments
    ]


def _expand_all_multiple_estimation(formula: str) -> list[str]:
    """Recursively expand all multiple estimation calls."""
    expansion = _expand_first_multiple_estimation(formula)
    if expansion is None:
        # No multiple estimation syntax present
        return [formula]
    else:
        return [
            parsed
            for formula_expanded in expansion
            for parsed in _expand_all_multiple_estimation(formula_expanded)
        ]


def _split_formula_into_parts(formula: str) -> Formula:
    parts = re.split(r"\s*\|\s*", formula)
    second_stage = parts.pop(0).strip()
    first_stage = next((part.strip() for part in parts if "~" in part), None)
    fixed_effects = next((part.strip() for part in parts if "~" not in part), None)
    if fixed_effects == "1":
        fixed_effects = None
    return Formula(
        second_stage=second_stage, fixed_effects=fixed_effects, first_stage=first_stage
    )
