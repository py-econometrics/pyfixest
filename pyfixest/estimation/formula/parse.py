import functools
import itertools
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import formulaic
import formulaic.formula
from formulaic.parser import DefaultFormulaParser, DefaultOperatorResolver
from formulaic.parser.types import Operator, OrderedSet

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.deprecated import FormulaParser
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.utils import (
    _MULTIPLE_ESTIMATION_PATTERN,
    _get_position_of_first_parenthesis_pair,
    _MultipleEstimationType,
    _str_split_by_sep,
)


class _FixedEffectsOperatorResolver(DefaultOperatorResolver):
    def __init__(self):
        super().__init__()

    @property
    def operators(self) -> list[Operator]:
        operators = [
            operator for operator in super().operators if operator.symbol != "^"
        ]

        operators.append(
            Operator(
                symbol="^",
                arity=2,
                precedence=500,
                associativity="left",
                to_terms=lambda *term_sets: OrderedSet(
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*term_sets)
                ),
            )
        )
        return operators


_PARSER: Final[FormulaParser] = DefaultFormulaParser(
    feature_flags=FORMULAIC_FEATURE_FLAG,
    operator_resolver=_FixedEffectsOperatorResolver(),
    include_intercept=True,
)


@dataclass(kw_only=True, frozen=True, slots=True, repr=False)
class Formula:
    """A formulaic-compliant formula."""

    _formula: formulaic.Formula

    def __post_init__(self) -> None:
        if not hasattr(self._formula, "lhs") or not hasattr(self._formula, "rhs"):
            raise FormulaSyntaxError(
                f"Formula must specify a left-hand and right-hand side separated by '~':\n"
                f"{self._formula}"
            )
        elif (
            isinstance(self._formula.rhs, tuple)
            and len(self._formula.rhs) > self._max_parts
        ):
            raise FormulaSyntaxError(
                f"Formula can have at most {self._max_parts} parts separated by '|'. "
                f"Received {len(self._right_hand_side)}:\n"
                f"{self._formula}"
            )
        if self.is_instrumental_variable:
            self._validate_instrumental_variable_specification()

    def _validate_instrumental_variable_specification(self) -> None:
        if len(self.endogenous.required_variables) > 1:
            raise FormulaSyntaxError(
                "Multiple endogenous variables are currently not supported. "
                "See https://github.com/py-econometrics/pyfixest/issues/791"
            )
        underdetermined = len(self.endogenous.required_variables) > len(
            self.instruments.required_variables
        )
        if underdetermined:
            raise UnderDeterminedIVError(
                "The IV system is underdetermined. "
                "Please provide at least as many instruments as endogenous variables."
            )
        endogenous_are_covariates = self.endogenous.required_variables.intersection(
            self.exogenous.required_variables
        )
        if endogenous_are_covariates:
            raise EndogVarsAsCovarsError(
                f"Endogeneous variables specified as covariates: {endogenous_are_covariates}"
            )
        instruments_are_covariates = self.instruments.required_variables.intersection(
            self.exogenous.required_variables
        )
        if instruments_are_covariates:
            raise InstrumentsAsCovarsError(
                f"Instruments specified as covariates: {instruments_are_covariates}"
            )

    def __repr__(self) -> str:
        return self.formula

    @property
    def _max_parts(self) -> int:
        return 2

    @property
    def formula(self) -> str:
        """The string representation of the formula."""
        formula = f"{self.dependent} ~ {self.exogenous}"
        if self.is_instrumental_variable:
            formula = f"{formula} + [{self.endogenous} ~ {self.instruments}]"
        if self.is_fixed_effects:
            formula = f"{formula} | {self.fixed_effects}"
        return formula

    @property
    def _left_hand_side(self) -> formulaic.formula.SimpleFormula:
        """The left hand side of the formula."""
        return self._formula.lhs

    @property
    def _right_hand_side(self) -> formulaic.formula.SimpleFormula:
        """The right hand side of the formula excluding fixed effects."""
        return (
            self._formula.rhs[0]
            if isinstance(self._formula.rhs, tuple)
            else self._formula.rhs
        )

    @property
    def is_instrumental_variable(self) -> bool:
        """Boolean indicating whether the formula is an instrumental variable specification."""
        # A MULTISTAGE formula is a formulaic.formula.StructuredFormula on the right hand side
        return isinstance(self._right_hand_side, formulaic.formula.StructuredFormula)

    @property
    def is_fixed_effects(self) -> bool:
        """Boolean indicating whether the formula is a fixed effects specification."""
        # A MULTIPART formula is a tuple of formulas on the right hand side
        return (
            isinstance(self._formula.rhs, tuple)
            and str(self._formula.rhs[-1]) not in ["", "0", "1"]  # ignore intercept
        )

    @property
    def dependent(self) -> formulaic.formula.Formula:
        """The dependent variable."""
        return self._left_hand_side

    @property
    def exogenous(self) -> formulaic.formula.Formula:
        """Exogenous aka covariates aka independent variables."""
        if self.is_instrumental_variable:
            # formulaic deterministically renames endogenous variables in the second stage
            # https://github.com/matthewwardrop/formulaic/blob/1f04a0b6d1d55ec4e43bf9f81898f6738c1f839a/formulaic/parser/parser.py#L360
            endogenous = {f"{c}_hat" for c in self.endogenous.required_variables}
            exogenous = (
                term for term in self._right_hand_side.root if term not in endogenous
            )
        else:
            exogenous = self._right_hand_side

        return formulaic.formula.SimpleFormula(exogenous)

    @property
    def endogenous(self) -> formulaic.formula.Formula:
        """Endogenous variables of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Endogenous variables are available only in instrumental variables specifications."
            )
        return self._right_hand_side.deps[0].lhs

    @property
    def instruments(self) -> formulaic.formula.Formula:
        """Instruments of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Endogenous variables are available only in instrumental variables specifications"
            )
        return self._right_hand_side.deps[0].rhs

    @property
    def fixed_effects(self) -> formulaic.formula.Formula:
        """The fixed effects of a formula."""
        if not self.is_fixed_effects:
            raise AttributeError("Not a fixed effects specification")
        return formulaic.formula.SimpleFormula(
            [term for term in self._formula.rhs[1] if term != "1"]
        )

    @property
    def wrap_fixed_effects(self) -> set[str]:
        return {f"__fixed_effect__{term.factors}" for term in self.fixed_effects}

    @property
    def second_stage(self) -> str:
        """The second stage formula."""
        right_hand_side = [t for t in self.exogenous]
        if self.is_instrumental_variable:
            right_hand_side += [term for term in self.endogenous]
        if not right_hand_side:
            right_hand_side = ["1"]
        formula = f"{self.dependent} ~ {formulaic.formula.Formula(right_hand_side)}"
        return formula

    @property
    def first_stage(self) -> str:
        """The first stage formula of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise TypeError("Not an instrumental variable specification.")
        return f"{self.endogenous} ~ {self.instruments} + {self.exogenous}"

    @classmethod
    def parse(cls, formula: str) -> list["Formula"]:
        """
        Parse fixest-style formula. In case of multiple estimation syntax,
        returns a list of multiple regression formulas.
        """
        formula = _preprocess(formula)
        return [
            Formula(_formula=formulaic.Formula(formulaic_compliant, _parser=_PARSER))
            for formulaic_compliant in _expand_all_multiple_estimation(formula)
        ]

    @classmethod
    def parse_to_dict(cls, formula: str) -> dict[str | None, list["Formula"]]:
        """Group parsed formulas into dictionary keyed by fixed effects."""
        formulas = cls.parse(formula)
        result: dict[str | None, list[Formula]] = {}
        for parsed_formula in formulas:
            fixed_effects = (
                str(parsed_formula.fixed_effects)
                if parsed_formula.is_fixed_effects
                else None
            )
            result.setdefault(fixed_effects, []).append(parsed_formula)
        return result


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


def _expand_all_multiple_estimation(formula: str) -> Iterator[str]:
    """Recursively expand all multiple estimation calls."""
    expansion = _expand_first_multiple_estimation(formula)
    if expansion is None:
        # No multiple estimation syntax present
        yield formula
    else:
        for formula_expanded in expansion:
            yield from _expand_all_multiple_estimation(formula_expanded)
