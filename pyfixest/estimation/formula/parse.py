import ast
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import formulaic
import formulaic.formula
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.types import Factor, FormulaParser, Term

from pyfixest.errors import (
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.formulaic_compat import (
    count_multistage_blocks,
    filter_multistage_endogenous_terms,
    get_first_multistage_lhs,
    get_first_multistage_rhs,
    is_python_expression,
    is_structured_formula,
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
_PARSER_NO_INTERCEPT: Final[FormulaParser] = DefaultFormulaParser(
    include_intercept=False
)


@dataclass(frozen=True, slots=True)
class FixedEffectSpecification:
    """Specification for materialization of fixed effect term.

    Attributes
    ----------
    levels : Term
        Formulaic term that identifies the fixed-effect levels.
    intercept : bool
        Whether the effect includes a constant loading for every level.
    slopes : tuple[Term, ...]
        Formulaic terms representing varying slopes by fixed-effect level.
    """

    levels: Term
    intercept: bool
    slopes: tuple[Term, ...] = ()

    @classmethod
    def from_term(cls, term: Term) -> "FixedEffectSpecification":
        """Convert one Formulaic fixed-effect term to a symbolic effect spec."""
        # A fixed-effect term can have multiple factors if it represents interactions
        # For example f1:f2[z] has two factors: `(f1, f2[z])`
        varying_slope_expressions: list[tuple[ast.Subscript, int]] = []
        for position, factor in enumerate(term.factors):
            factor_expression = _factor_ast(factor)
            if factor_expression is None or not isinstance(
                factor_expression, ast.Subscript
            ):
                continue
            varying_slope_expressions.append((factor_expression, position))

        if not varying_slope_expressions:
            return cls(levels=term, intercept=True)
        elif len(varying_slope_expressions) != 1:
            # Reject expressions of the form `f1[z1]:f2[z2]`
            raise FormulaSyntaxError(
                "Cannot specify more than one varying-slope expression in a single fixed-effect term."
            )
        elif varying_slope_expressions[0][1] != len(term.factors) - 1:
            # Varying slope syntax must be attached to last factor in term
            # For example, accept `f1:f2[z]` but reject `f1[z]:f2`
            raise FormulaSyntaxError(
                "Varying-slope syntax is only supported on the final factor "
                "of a fixed-effect interaction."
            )
        # Decompose expression f1[z] into its "value" (f1) and its "slice" (z)
        # Note: we exploit that `f1[z]` is valid Python syntax to construct an AST
        # First, get the factor with varying slopes (e.g., `f2[z]` in `f1:f2[z]`)
        expression = varying_slope_expressions[0][0]
        fixed_effect_level = expression.value  # `f2`
        fixed_effect_slopes = expression.slice  # `z`
        if isinstance(fixed_effect_level, ast.Subscript):
            raise FormulaSyntaxError(
                "Nested varying-slope subscripts are not supported."
            )
        if isinstance(fixed_effect_slopes, ast.List):
            # Varying slopes without fixed effect: f1[[z1, z2]]
            intercept = False
            slope_nodes = tuple(fixed_effect_slopes.elts)
        else:
            # Varying slopes fixed effect: f1[z1] or f1[z1, z2]
            intercept = True
            slope_nodes = (
                tuple(fixed_effect_slopes.elts)
                if isinstance(fixed_effect_slopes, ast.Tuple)  # f1[z1, z2]
                else (fixed_effect_slopes,)  # f1[z1]
            )

        if not slope_nodes:
            # Guard against `f1[]` or `f1[[]]`
            raise FormulaSyntaxError(
                "A varying-slope term must specify at least one slope."
            )
        return cls(
            levels=Term(term.factors[:-1] + _term_from_ast(fixed_effect_level).factors),
            intercept=intercept,
            slopes=tuple(_term_from_ast(node) for node in slope_nodes),
        )


def _term_from_ast(node: ast.expr) -> Term:
    """Parse one extracted Python expression as one Formulaic term."""
    expression = ast.unparse(node)
    formula = formulaic.Formula(expression, _parser=_PARSER_NO_INTERCEPT)
    if not isinstance(formula, formulaic.formula.SimpleFormula) or len(formula) != 1:
        raise FormulaSyntaxError(
            f"`{expression}` is not valid here. Each fixed-effect level and slope "
            "must resolve to exactly one formula term. To specify multiple slopes, "
            "separate them with commas, for example `f1[z1, z2]`."
        )
    return formula[0]


def _factor_ast(factor: Factor) -> ast.expr | None:
    """Return the AST for a Python-evaluated Formulaic factor."""
    # Factor must encoded as Python expression by formulaic
    # (because `f1[z]` is Python syntax)
    if not is_python_expression(factor):
        return None
    try:
        return ast.parse(factor.expr, mode="eval").body
    except SyntaxError as exception:
        raise FormulaSyntaxError(
            f"Could not parse fixed-effect expression: {factor.expr}"
        ) from exception


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
        return is_structured_formula(self._right_hand_side)

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
        exogenous = self._right_hand_side
        if self.is_instrumental_variable:
            exogenous = filter_multistage_endogenous_terms(exogenous, self.endogenous)

        exogenous_terms = tuple(terms_without_intercept(exogenous))
        if self.is_fixed_effects and exogenous_terms:
            # Drop the intercept for fixed effects regressions, except for
            # intercept-only specifications such as `Y ~ 1 | f1`; these can be
            # used to demean dependent variables.
            exogenous = formulaic.formula.SimpleFormula(exogenous_terms)

        return exogenous

    @property
    def endogenous(self) -> formulaic.formula.Formula:
        """Endogenous variables of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Endogenous variables are available only in instrumental variables specifications."
            )
        return get_first_multistage_lhs(self._right_hand_side)

    @property
    def instruments(self) -> formulaic.formula.Formula:
        """Instruments of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Instruments are available only in instrumental variables specifications."
            )
        return get_first_multistage_rhs(self._right_hand_side)

    @property
    def fixed_effects(self) -> formulaic.formula.Formula:
        """The fixed effects of a formula."""
        if not self.is_fixed_effects:
            raise AttributeError("Not a fixed effects specification")
        return formulaic.formula.SimpleFormula(
            terms_without_intercept(self._formula.rhs[1])
        )

    @property
    def fixed_effect_specifications(self) -> tuple[FixedEffectSpecification, ...]:
        """Fixed effects represented as symbolic `within::Effect` terms."""
        if not self.is_fixed_effects:
            return ()
        return tuple(
            FixedEffectSpecification.from_term(term) for term in self.fixed_effects
        )

    @property
    def fixed_effects_wrapped(self) -> formulaic.formula.Formula:
        """Wrapped fixed effects for proper encoding."""
        return formulaic.formula.Formula(
            [f"__fixed_effect__{term.factors}" for term in self.fixed_effects],
            _parser=_PARSER_NO_INTERCEPT,
        )

    @property
    def second_stage(self) -> str:
        """The second stage formula."""
        right_hand_side = list(self.exogenous)
        if self.is_instrumental_variable:
            right_hand_side += list(self.endogenous)
        return f"{self.dependent} ~ {formulaic.formula.SimpleFormula(right_hand_side)}"

    @property
    def first_stage(self) -> str:
        """The first stage formula of an instrumental variable specification."""
        if not self.is_instrumental_variable:
            raise TypeError("Not an instrumental variable specification.")
        return f"{self.endogenous} ~ {formulaic.formula.SimpleFormula([term for term in itertools.chain(self.instruments, self.exogenous)])}"

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
