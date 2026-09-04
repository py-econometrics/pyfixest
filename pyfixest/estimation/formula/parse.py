import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, cast

import formulaic
import formulaic.formula
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.types import FormulaParser

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.formulaic_compat import (
    FormulaSide,
    count_multistage_blocks,
    filter_multistage_endogenous_terms,
    formula_lhs,
    formula_rhs,
    get_first_multistage_lhs,
    get_first_multistage_rhs,
    parsed_simple_formula,
    simple_formula,
    terms_without_intercept,
)
from pyfixest.estimation.formula.utils import (
    _MULTIPLE_ESTIMATION_PATTERN,
    _get_position_of_first_parenthesis_pair,
    _MultipleEstimationType,
    _preprocess,
    _str_split_by_sep,
)

_PARSER: Final[FormulaParser] = DefaultFormulaParser(
    feature_flags=FORMULAIC_FEATURE_FLAG,
    include_intercept=True,
)


@dataclass(kw_only=True, frozen=True, slots=True, repr=False)
class Formula:
    """
    A formulaic-compliant formula.

    Splits a fixest-style formula into second stage, fixed effects and, for IV
    models, first stage. Use `parse()` instead of calling the class directly.
    `parse()` also expands the multiple estimation operators (`sw`, `sw0`,
    `csw`, `csw0`, `mvsw`) into one `Formula` per model. This is an internal
    API. Formulas are written as strings and passed to `feols()`. See the
    [formula syntax tutorial](/tutorials/formula-syntax.qmd).

    Examples
    --------
    ```{python}
    from pyfixest.estimation.formula.parse import Formula

    fml = Formula.parse("Y ~ X1 + X2 | f1 + f2")[0]
    fml.second_stage, fml.fixed_effects
    ```

    Stepwise syntax expands into one formula per estimated model.

    ```{python}
    Formula.parse("Y ~ X1 + csw(X2, X3)")
    ```
    """

    _formula: formulaic.Formula

    def __post_init__(self) -> None:
        if not hasattr(self._formula, "lhs") or not hasattr(self._formula, "rhs"):
            raise FormulaSyntaxError(
                f"Formula must specify a left-hand and right-hand side separated by '~':\n"
                f"{self._formula}"
            )
        if (
            isinstance(self._formula.rhs, tuple)
            and len(self._formula.rhs) > self._max_parts
        ):
            raise FormulaSyntaxError(
                f"Formula can have at most {self._max_parts} parts separated by '|'. "
                f"Received {len(self._formula.rhs)}:\n"
                f"{self._formula}"
            )
        # Count terms, not source variables: `I(Y + Y2)` is one dependent
        # expression evaluated from two columns.
        if len(self.dependent) > 1:
            raise FormulaSyntaxError(
                f"Formula must have exactly one term on the left hand side. "
                f"Received: {[str(term) for term in self.dependent]}"
            )
        if self.is_instrumental_variable:
            self._validate_instrumental_variable_specification()

    def _validate_instrumental_variable_specification(self) -> None:
        n_multistage_blocks = count_multistage_blocks(self._right_hand_side)
        if n_multistage_blocks > 1:
            raise FormulaSyntaxError(
                "Multiple instrumental variable specifications are not supported. "
                "Use a single `[endogenous ~ instruments]` block. "
                f"Received {n_multistage_blocks} multistage blocks:\n "
                f"{self._formula}"
            )
        if len(self.endogenous) > 1:
            raise FormulaSyntaxError(
                "Multiple endogenous variables are currently not supported. "
                "See https://github.com/py-econometrics/pyfixest/issues/791"
            )
        underdetermined = len(self.endogenous) > len(
            tuple(terms_without_intercept(self.instruments))
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
        return formula_lhs(self._formula)

    @property
    def _right_hand_side(self) -> FormulaSide:
        """The right hand side of the formula excluding fixed effects."""
        right_hand_side = formula_rhs(self._formula)
        return (
            right_hand_side[0]
            if isinstance(right_hand_side, tuple)
            else right_hand_side
        )

    @property
    def is_instrumental_variable(self) -> bool:
        """Boolean indicating whether the formula is an instrumental variable specification."""
        # formulaic parses an IV right hand side into a nested StructuredFormula
        return isinstance(self._right_hand_side, formulaic.formula.StructuredFormula)

    @property
    def is_fixed_effects(self) -> bool:
        """Boolean indicating whether the formula is a fixed effects specification."""
        # A MULTIPART formula is a tuple of formulas on the right hand side
        right_hand_side = formula_rhs(self._formula)
        return (
            isinstance(right_hand_side, tuple)
            and str(right_hand_side[-1]) not in ["", "0", "1"]  # ignore intercept
        )

    @property
    def dependent(self) -> formulaic.formula.SimpleFormula:
        """The dependent variable."""
        return self._left_hand_side

    @property
    def exogenous(self) -> formulaic.formula.SimpleFormula:
        """Exogenous aka covariates aka independent variables."""
        right_hand_side = self._right_hand_side
        exogenous = (
            filter_multistage_endogenous_terms(right_hand_side, self.endogenous)
            if isinstance(right_hand_side, formulaic.formula.StructuredFormula)
            else right_hand_side
        )

        exogenous_terms = tuple(terms_without_intercept(exogenous))
        if self.is_fixed_effects and exogenous_terms:
            # Drop the intercept for fixed effects regressions, except for
            # intercept-only specifications such as `Y ~ 1 | f1`; these can be
            # used to demean dependent variables.
            exogenous = simple_formula(exogenous_terms)

        return exogenous

    @property
    def endogenous(self) -> formulaic.formula.SimpleFormula:
        """Endogenous variables of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Endogenous variables are available only in instrumental variables specifications."
            )
        return get_first_multistage_lhs(self._right_hand_side)

    @property
    def instruments(self) -> formulaic.formula.SimpleFormula:
        """Instruments of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Instruments are available only in instrumental variables specifications."
            )
        return get_first_multistage_rhs(self._right_hand_side)

    @property
    def fixed_effects(self) -> formulaic.formula.SimpleFormula:
        """The fixed effects of a formula."""
        if not self.is_fixed_effects:
            raise AttributeError("Not a fixed effects specification")
        # `is_fixed_effects` guarantees a MULTIPART right hand side, and
        # formulaic nests an IV block only in its first part, so the
        # fixed-effect part is always a flat list of terms.
        parts = cast(
            "tuple[FormulaSide, formulaic.formula.SimpleFormula]",
            formula_rhs(self._formula),
        )
        return simple_formula(terms_without_intercept(parts[1]))

    @property
    def fixed_effects_wrapped(self) -> formulaic.formula.SimpleFormula:
        """Wrapped fixed effects for proper encoding."""
        return parsed_simple_formula(
            f"__fixed_effect__{term.factors}" for term in self.fixed_effects
        )

    @property
    def second_stage(self) -> str:
        """The second stage formula."""
        right_hand_side = list(self.exogenous)
        if self.is_instrumental_variable:
            right_hand_side += list(self.endogenous)
        return f"{self.dependent} ~ {simple_formula(right_hand_side)}"

    @property
    def first_stage(self) -> str:
        """The first stage formula of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise TypeError("Not an instrumental variable specification.")
        return f"{self.endogenous} ~ {simple_formula(itertools.chain(self.instruments, self.exogenous))}"

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
