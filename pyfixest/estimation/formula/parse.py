import itertools
import re
from dataclasses import dataclass
from typing import Final

import formulaic

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula.utils import (
    _MULTIPLE_ESTIMATION_PATTERN,
    _get_position_of_first_parenthesis_pair,
    _MultipleEstimationType,
    _str_split_by_sep,
)


@dataclass(kw_only=True, frozen=True, slots=True)
class Formula:
    """A formulaic-compliant formula."""

    # second and first stage are formulas **excluding** fixed effects
    _second_stage: str
    _fixed_effects: str | None = None
    _first_stage: str | None = None

    def __post_init__(self) -> None:
        if self._first_stage is not None:
            second_stage = formulaic.Formula(self._second_stage)
            first_stage = formulaic.Formula(self._first_stage)
            exogenous = second_stage.rhs.required_variables
            endogenous = first_stage.lhs.required_variables
            instruments = first_stage.rhs.required_variables
            if len(endogenous) > 1:
                raise FormulaSyntaxError(
                    "Multiple endogenous variables are currently not supported."
                )
            if len(endogenous) > len(instruments):
                raise UnderDeterminedIVError(
                    "The IV system is underdetermined. "
                    "Please provide at least as many instruments as endogenous variables."
                )
            endogenous_are_covariates = endogenous.intersection(exogenous)
            if endogenous_are_covariates:
                raise EndogVarsAsCovarsError(
                    f"Endogeneous variables specified as covariates: {endogenous_are_covariates}"
                )
            instruments_are_covariates = instruments.intersection(exogenous)
            if instruments_are_covariates:
                raise InstrumentsAsCovarsError(
                    f"Instruments specified as covariates: {instruments_are_covariates}"
                )

    @property
    def formula(self) -> str:
        """Full fixest-style formula."""
        formula = self._second_stage
        if self._fixed_effects is not None:
            formula = f"{formula} | {self._fixed_effects}"
        if self._first_stage is not None:
            formula = f"{formula} | {self._first_stage}"
        return formula

    @property
    def endogenous(self) -> str | None:
        """Endogenous variables of an instrumental variable specification."""
        if self._first_stage is None:
            return None
        else:
            endogenous, _ = re.split(r"\s*~\s*", self._first_stage, maxsplit=1)
            return endogenous

    @property
    def exogenous(self) -> str:
        """Exogenous aka covariates aka independent variables."""
        _, exogenous = re.split(r"\s*~\s*", self._second_stage, maxsplit=1)
        return exogenous

    @property
    def second_stage(self) -> str:
        """The second stage formula."""
        second_stage = self._second_stage
        if self._first_stage is not None:
            # Add endogenous variables as covariates in second stage
            second_stage = f"{second_stage} + {self.endogenous}"
        return second_stage

    @property
    def first_stage(self) -> str | None:
        """The first stage formula of an instrumental variable specification."""
        if self._first_stage is None:
            return None
        else:
            # Add exogenous variables as covariates in first stage
            return f"{self._first_stage} + {self.exogenous}"

    @property
    def fixed_effects(self) -> str | None:
        """The fixed effects of a formula."""
        return self._fixed_effects

    @classmethod
    def parse(cls, formula: str) -> list["Formula"]:
        """
        Parse fixest-style formula. In case of multiple estimation syntax,
        returns a list of multiple regression formulas.
        """
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
            result.setdefault(parsed_formula._fixed_effects, []).append(parsed_formula)
        return result


def _validate(formula: str) -> None:
    max_parts: Final[int] = 3
    parts = _str_split_by_sep(string=formula, separator="|")

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
    dependents = _str_split_by_sep(dependents.strip(), separator="+")
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
    parenthesis_open, parenthesis_closed = _get_position_of_first_parenthesis_pair(
        string=formula[match.start() :]
    )
    parenthesis_open += match.start()
    parenthesis_closed += match.start()
    arguments = _str_split_by_sep(
        string=formula[parenthesis_open:parenthesis_closed],
        separator=",",
    )
    if len(arguments) < 2 and kind is not _MultipleEstimationType.mvsw:
        raise FormulaSyntaxError(
            f"'{kind.name}(...)' requires at least 2 arguments, got {len(arguments)}. "
            f"Check for extra parentheses, e.g. sw((a, b)) should be sw(a, b)."
        )
    if kind is _MultipleEstimationType.mvsw:
        # Multiverse stepwise: all combinations of arguments
        arguments = [
            " + ".join(combination)
            for combination in itertools.chain.from_iterable(
                itertools.combinations(arguments, r=length)
                for length in range(1, len(arguments) + 1)
            )
        ]
    elif kind is _MultipleEstimationType.csw or kind is _MultipleEstimationType.csw0:
        # Cumulative stepwise
        arguments = [" + ".join(arguments[: i + 1]) for i, _ in enumerate(arguments)]
    if (
        kind is _MultipleEstimationType.sw0
        or kind is _MultipleEstimationType.csw0
        or kind is _MultipleEstimationType.mvsw  # Following fixest there's no mvsw0
    ):
        # Add zero step
        arguments = ["1", *arguments]
    multiple_estimation_call = formula[match.start() : parenthesis_closed + 1]
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
    if fixed_effects in ("0", "1"):
        fixed_effects = None
    return Formula(
        _second_stage=second_stage,
        _fixed_effects=fixed_effects,
        _first_stage=first_stage,
    )
