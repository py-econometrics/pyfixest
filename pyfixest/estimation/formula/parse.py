from __future__ import annotations

import ast
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final

import formulaic
import formulaic.formula
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.algos import tokenize
from formulaic.parser.types import Factor, FormulaParser, Term, Token

from pyfixest.errors import (
    FormulaSyntaxError,
    IVIdentificationError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG
from pyfixest.estimation.formula.formulaic_compat import (
    PyfixestOperatorResolver,
    count_multistage_blocks,
    get_first_multistage_lhs,
    get_first_multistage_rhs,
    is_python_expression,
    is_structured_formula,
    terms_without_intercept,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    as_formulaic_fixed_effect_term,
)
from pyfixest.estimation.formula.utils import (
    _MULTIPLE_ESTIMATION_PATTERN,
    _get_position_of_first_parenthesis_pair,
    _MultipleEstimationType,
    _preprocess,
    _str_split_by_sep,
)

_PARSER: Final[FormulaParser] = DefaultFormulaParser(
    operator_resolver=PyfixestOperatorResolver(),
    feature_flags=FORMULAIC_FEATURE_FLAG,
    include_intercept=True,
)
_PARSER_NO_INTERCEPT: Final[FormulaParser] = DefaultFormulaParser(
    include_intercept=False
)


def _render_factor(factor: Factor) -> str:
    """Render one factor without confusing lookups with Python expressions."""
    expression = factor.expr
    if factor.eval_method is Factor.EvalMethod.LOOKUP:
        safe_unquoted = expression.isascii() and all(
            part.isidentifier() for part in expression.split(".")
        )
        return expression if safe_unquoted else f"`{expression}`"

    if factor.eval_method is Factor.EvalMethod.PYTHON:
        tokens = list(tokenize(expression))
        if (
            len(tokens) == 1
            and tokens[0].kind is Token.Kind.PYTHON
            and tokens[0].token == expression
        ):
            return expression
        return "{" + expression + "}"

    return expression


def render_term(term: Term) -> str:
    """Render one term; distinct parsed terms render to distinct strings."""
    return ":".join(_render_factor(factor) for factor in term.factors)


def _render_formula_part(
    terms: Iterable[Term], *, explicit_no_intercept: bool = False
) -> str:
    terms = tuple(terms)
    result = " + ".join(render_term(term) for term in terms) or "0"
    if (
        explicit_no_intercept
        and result != "0"
        and not any(str(term) == "1" for term in terms)
    ):
        result = "0 + " + result
    return result


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

    @property
    def name(self) -> str:
        """Label the effect with fixest syntax, e.g. `f1`, `f1[z]`, or `f1[[z]]`."""
        levels = ":".join(str(factor) for factor in self.levels.factors)
        if not self.slopes:
            return levels
        slopes = ", ".join(str(slope) for slope in self.slopes)
        return f"{levels}[{slopes}]" if self.intercept else f"{levels}[[{slopes}]]"

    @classmethod
    def from_term(cls, term: Term) -> FixedEffectSpecification:
        """Convert one Formulaic fixed-effect term to a symbolic effect spec."""
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
        if len(varying_slope_expressions) != 1:
            raise FormulaSyntaxError(
                f"Invalid fixed-effect term `{term}`: cannot specify more than one "
                "varying-slope expression in a single term. For slopes on interacted "
                "levels, use one bracket pair, for example `f1:f2[z1, z2]`."
            )

        expression, slope_factor_position = varying_slope_expressions[0]
        last_factor_position = len(term.factors) - 1
        if slope_factor_position != last_factor_position:
            raise FormulaSyntaxError(
                f"Invalid fixed-effect term `{term}`: varying-slope syntax is only "
                "supported on the final factor of a fixed-effect interaction, "
                "for example `f1:f2[z]`."
            )

        fixed_effect_level = expression.value
        fixed_effect_slopes = expression.slice
        if isinstance(fixed_effect_level, ast.Subscript):
            raise FormulaSyntaxError(
                f"Invalid fixed-effect term `{term}`: nested varying-slope "
                "subscripts are not supported. Separate slopes with commas, "
                "for example `f1[z1, z2]` (with fixed-effect intercepts) or "
                "`f1[[z1, z2]]` (without fixed-effect intercepts)."
            )
        if isinstance(fixed_effect_level, (ast.BinOp, ast.UnaryOp)):
            raise FormulaSyntaxError(
                f"Invalid fixed-effect term `{term}`: unsupported fixed-effect "
                f"level expression `{ast.unparse(fixed_effect_level)}`. Arithmetic "
                "is not supported directly in varying-slope fixed-effect levels. "
                "Create a grouping column first, for example `group[z]`."
            )

        if isinstance(fixed_effect_slopes, ast.List):
            intercept = False
            slope_nodes = tuple(fixed_effect_slopes.elts)
        else:
            intercept = True
            slope_nodes = (
                tuple(fixed_effect_slopes.elts)
                if isinstance(fixed_effect_slopes, ast.Tuple)
                else (fixed_effect_slopes,)
            )

        if not slope_nodes:
            raise FormulaSyntaxError(
                f"Invalid fixed-effect term `{term}`: a varying-slope term must "
                "specify at least one slope, for example `f1[z]` or `f1[[z]]`. "
                "For an ordinary fixed effect without slopes, use `f1`."
            )
        for node in slope_nodes:
            if not _is_slope_expression(node):
                raise FormulaSyntaxError(
                    f"Invalid fixed-effect term `{term}`: unsupported slope "
                    f"expression `{ast.unparse(node)}`. Slopes must be column "
                    "names, arithmetic expressions, or transform calls, for "
                    "example `f1[z]`, `f1[z**2]`, or `f1[[np.log(z)]]`. "
                    "Separate multiple slopes with commas."
                )
        return cls(
            levels=Term(term.factors[:-1] + _term_from_ast(fixed_effect_level).factors),
            intercept=intercept,
            slopes=tuple(
                _term_from_ast(node, preserve_arithmetic=True) for node in slope_nodes
            ),
        )


def _normalize_fixed_effect_specifications(
    specifications: Iterable[FixedEffectSpecification],
) -> tuple[FixedEffectSpecification, ...]:
    """Normalize fixed effects by merging groups and deduplicating slopes.

    Normalization follows these rules:

    1. Combine specifications whose grouping levels compare equal as Formulaic
       terms, including equivalent interactions such as `f1:f2` and `f2:f1`.
    2. Keep each distinct slope once, removing duplicates both within a
       specification and across specifications on the same levels.
    3. Include group-specific intercepts if any component requests them: a
       plain `firm` or single-bracket `firm[...]` does; `firm[[...]]` does not.
       Combining only slope-only specifications keeps the result slope-only.
    4. Preserve the first occurrence's position for each grouping and each
       slope within it. This is the order received from the formula parser,
       which may already have normalized or reordered the original formula.
    5. Keep different grouping terms separate, even when they share a slope.
       Equality is symbolic: no data values are compared and no algebraic
       simplification of slope expressions is performed.

    For example:

    - `firm[z, z]` resolves to `firm[z]`.
    - `firm[[z, z]]` resolves to `firm[[z]]`.
    - `firm[[z]] + firm[w, w]` resolves to `firm[z, w]`.
    - `firm[[z]] + firm[w, z]` resolves to `firm[z, w]`.
    - `firm + firm[[z]]` resolves to `firm[z]`.
    - `firm[[z]] + firm[[w]]` resolves to `firm[[z, w]]`.
    - `firm[z] + firm[[w]]` resolves to `firm[z, w]`.
    - `firm[z] + firm[w]` resolves to `firm[z, w]`.
    - `firm[[z]] + region[[z]]` remains two separate slope-only effects.
    """
    merged: dict[Term, FixedEffectSpecification] = {}
    for specification in specifications:
        existing = merged.get(specification.levels)
        if existing is None:
            merged[specification.levels] = FixedEffectSpecification(
                levels=specification.levels,
                intercept=specification.intercept,
                slopes=tuple(dict.fromkeys(specification.slopes)),
            )
            continue
        slopes = tuple(dict.fromkeys(existing.slopes + specification.slopes))
        merged[specification.levels] = FixedEffectSpecification(
            levels=existing.levels,
            intercept=existing.intercept or specification.intercept,
            slopes=slopes,
        )
    return tuple(merged.values())


def _is_slope_expression(node: ast.AST) -> bool:
    """Accept names, dotted names, calls, and arithmetic as slope expressions.

    Scalar literals are allowed inside expressions (such as `z / 2` or
    transform keyword arguments), but an expression must contain a name.
    This is a syntactic check, not proof of dependence on a data column:
    names in expressions such as `np.pi` and `transform()` also qualify. Only
    `+`, `-`, `*`, `/`, `//`, `%`, `**`, and unary `+`/`-` are supported.
    Containers, indexing, slices, comparisons, comprehensions, and argument
    unpacking are excluded, including when nested inside a transform.

    Supported examples include `z`, `-z`, `z**2`, `(z1 + z2) / 2`,
    `np.log(z)`, and `transform(z, mode="scale", center=True)`. Unsupported
    examples include `0`, `"z"`, `z[0]`, `z > 0`, and `[x for x in z]`.
    """
    supported_nodes = (
        ast.Name,
        ast.Attribute,
        ast.Call,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.keyword,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    nodes = tuple(ast.walk(node))
    return (
        any(isinstance(child, ast.Name) for child in nodes)
        and all(isinstance(child, supported_nodes) for child in nodes)
        and all(
            child.arg is not None for child in nodes if isinstance(child, ast.keyword)
        )
    )


def _term_from_ast(node: ast.expr, *, preserve_arithmetic: bool = False) -> Term:
    """Parse one extracted Python expression as one Formulaic term."""
    expression = ast.unparse(node)
    formula_expression = (
        "{" + expression + "}"
        if preserve_arithmetic and not isinstance(node, ast.Name)
        else expression
    )
    formula = formulaic.Formula(formula_expression, _parser=_PARSER_NO_INTERCEPT)
    if not isinstance(formula, formulaic.formula.SimpleFormula) or len(formula) != 1:
        raise FormulaSyntaxError(
            f"Invalid fixed-effect expression `{expression}`: expected exactly "
            f"one formula term, but parsed `{formula}`. Specify separate fixed "
            "effects with `+`, for example `f1[z] + f2[z]`. To specify multiple "
            "slopes for one fixed effect, separate them with commas, "
            "for example `f1[z1, z2]`."
        )
    return formula[0]


def _factor_ast(factor: Factor) -> ast.expr | None:
    """Return the AST for a Python-evaluated Formulaic factor."""
    if not is_python_expression(factor):
        return None
    try:
        return ast.parse(factor.expr, mode="eval").body
    except SyntaxError as exception:
        raise FormulaSyntaxError(
            f"Could not parse fixed-effect expression `{factor.expr}`: "
            f"{exception.msg}. Expected a valid Python expression, "
            "for example `f1[z]` or `f1[[z1, z2]]` for varying slopes."
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
                "Formula must have a dependent variable on the left of '~'. "
                "Use a formula such as 'y ~ x | group'. "
                "See https://pyfixest.org/tutorials/formula-syntax.html#basic-syntax"
            )
        if (
            isinstance(self._formula.rhs, tuple)
            and len(self._formula.rhs) > self._max_parts
        ):
            raise FormulaSyntaxError(
                f"Formula can have at most {self._max_parts} parts separated by '|'; "
                f"got {len(self._formula.rhs)}. Put all fixed effects in one part, "
                "as in 'y ~ x | f1 + f2'."
            )
        if len(self.dependent) > 1:
            raise FormulaSyntaxError(
                f"Formula must have exactly one term on the left hand side. "
                f"Received {[str(term) for term in self.dependent]}. "
                "Use one dependent expression or use sw(y1, y2) for separate models."
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
                f"The instrument block contains {[str(term) for term in self.endogenous]}. "
                "Use a single endogenous regressor in [endogenous ~ instruments]. "
                "See https://github.com/py-econometrics/pyfixest/issues/791"
            )
        missing_endogenous = [
            term for term in self.endogenous if term not in self._right_hand_side.root
        ]
        if missing_endogenous:
            raise FormulaSyntaxError(
                "Formula operations removed or transformed the endogenous terms "
                f"{missing_endogenous}, but their instrument block remains. "
                "Put the intended endogenous term inside [endogenous ~ instruments] "
                "and keep that term in the main equation."
            )
        underdetermined = len(self.endogenous) > len(self.instruments)
        if underdetermined:
            raise UnderDeterminedIVError(
                "The IV system is underdetermined. "
                f"Found {len(self.endogenous)} endogenous regressors and "
                f"{len(self.instruments)} instruments. Supply at least one "
                "excluded instrument per endogenous regressor."
            )
        excluded_instruments = set(self.instruments).difference(
            terms_without_intercept(self.exogenous)
        )
        if not excluded_instruments:
            raise IVIdentificationError(
                "The instruments add no explanatory power for the endogenous variable: "
                f"all instrument terms {[str(term) for term in self.instruments]} "
                "are also specified as exogenous regressors. Supply an excluded "
                "instrument with independent first-stage variation. "
                "See https://pyfixest.org/tutorials/instrumental-variables.html"
            )

    def __repr__(self) -> str:
        return self.render()

    @property
    def _max_parts(self) -> int:
        return 2

    def render(self) -> str:
        """Render an expanded formula preserving lookup and expression semantics.

        This is the readable public model name. Distinct parsed formulas render
        to distinct strings, so the rendered text also identifies models and
        fixed-effect groups internally. Materialization uses the retained
        parsed stages.
        """
        exogenous = _render_formula_part(
            self.exogenous, explicit_no_intercept=not self.absorbs_intercept
        )
        result = f"{_render_formula_part(self.dependent)} ~ {exogenous}"
        if self.is_instrumental_variable:
            result += f" + [{_render_formula_part(self.endogenous)} ~ {_render_formula_part(self.instruments)}]"
        if self.is_fixed_effects:
            result += " | " + _render_formula_part(self.fixed_effects)
        return result

    @property
    def fixed_effects_key(self) -> str | None:
        """Rendered fixed effects, grouping models that share demeaning caches."""
        if not self.is_fixed_effects:
            return None
        return _render_formula_part(self.fixed_effects)

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
        return isinstance(self._formula.rhs, tuple) and any(
            terms_without_intercept(self._formula.rhs[-1])
        )

    @property
    def absorbs_intercept(self) -> bool:
        """Whether any fixed effect supplies group-specific intercepts.

        Ordinary fixed effects such as `firm` absorb the model constant
        through their group-specific intercepts. Effects with varying slopes
        and intercepts, such as `firm[z]`, do so as well.

        Returns False when there are no fixed effects or all effects are
        slope-only, such as `firm[[z]]`. This describes the declared fixed
        effects independently of whether the structural formula requests an
        intercept; it does not inspect numerical values or matrix rank.
        """
        return any(
            specification.intercept
            for specification in self.fixed_effect_specifications
        )

    @property
    def has_varying_slopes(self) -> bool:
        """Whether any fixed effect has varying slopes, as in `f1[z]`."""
        return any(
            specification.slopes for specification in self.fixed_effect_specifications
        )

    @property
    def dependent(self) -> formulaic.formula.SimpleFormula:
        """The dependent variable."""
        return self._left_hand_side

    @property
    def exogenous(self) -> formulaic.formula.SimpleFormula:
        """Exogenous aka covariates aka independent variables."""
        exogenous = self._right_hand_side
        if self.is_instrumental_variable:
            exogenous = formulaic.formula.SimpleFormula(
                term for term in exogenous.root if term not in self.endogenous
            )

        exogenous_terms = tuple(terms_without_intercept(exogenous))
        if self.absorbs_intercept and exogenous_terms:
            exogenous = formulaic.formula.SimpleFormula(
                exogenous_terms, _ordering=exogenous.ordering
            )

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
        """Instruments of an instrumental variable specification.

        The main formula sets the intercept of both stages, so this excludes
        the intercept Formulaic adds to every IV block.
        """
        if not self.is_instrumental_variable:
            raise AttributeError(
                "Instruments are available only in instrumental variables specifications."
            )
        instruments = get_first_multistage_rhs(self._right_hand_side)
        return formulaic.formula.SimpleFormula(
            terms_without_intercept(instruments), _ordering=instruments.ordering
        )

    @property
    def fixed_effects(self) -> formulaic.formula.SimpleFormula:
        """The fixed effects of a formula."""
        if not self.is_fixed_effects:
            raise AttributeError("Not a fixed effects specification")
        return formulaic.formula.SimpleFormula(
            terms_without_intercept(self._formula.rhs[1])
        )

    @property
    def fixed_effect_specifications(self) -> tuple[FixedEffectSpecification, ...]:
        """Fixed effects represented as symbolic `within::Effect` terms.

        Effects on the same levels are merged, so `f1 + f1[[z]]` is `f1[z]`.
        """
        if not self.is_fixed_effects:
            return ()
        return _normalize_fixed_effect_specifications(
            FixedEffectSpecification.from_term(term) for term in self.fixed_effects
        )

    @property
    def formulaic_fixed_effects(self) -> formulaic.formula.SimpleFormula:
        """Fixed-effect level terms in Formulaic's internal representation."""
        return formulaic.formula.SimpleFormula(
            as_formulaic_fixed_effect_term(term=specification.levels)
            for specification in self.fixed_effect_specifications
        )

    @property
    def fixed_effect_slopes(self) -> formulaic.formula.SimpleFormula:
        """Distinct varying-slope terms in normalized specification order.

        Visit normalized fixed-effect specifications in their stored order,
        then each specification's slopes in order, keeping only the first
        occurrence of each slope term across all specifications. This can
        differ from textual order: `firm[[z]] + region[[w]] + firm[[v]]`
        normalizes to firm slopes `(z, v)` followed by region slope `(w,)`,
        so the returned formula contains `(z, v, w)`.
        """
        slopes: list[Term] = []
        for specification in self.fixed_effect_specifications:
            slopes.extend(
                slope for slope in specification.slopes if slope not in slopes
            )
        return formulaic.formula.SimpleFormula(slopes)

    @property
    def second_stage(self) -> formulaic.formula.StructuredFormula:
        """Parsed response and regressors for second-stage materialization."""
        terms = list(self.exogenous)
        if self.is_instrumental_variable:
            terms.extend(self.endogenous)
        if self.absorbs_intercept and terms and not any(term == "1" for term in terms):
            terms.insert(0, Term([Factor("1", eval_method=Factor.EvalMethod.LITERAL)]))
        return formulaic.formula.StructuredFormula(
            lhs=self.dependent,
            rhs=formulaic.formula.SimpleFormula(
                terms, _ordering=self.exogenous.ordering
            ),
        )

    def as_first_stage(self) -> Formula:
        """Return the parsed OLS specification for an IV first stage."""
        if not self.is_instrumental_variable:
            raise TypeError("Not an instrumental variable specification.")
        rhs = formulaic.formula.SimpleFormula(
            dict.fromkeys(itertools.chain(self.exogenous, self.instruments)),
            _ordering="none",
        )
        return Formula(
            _formula=formulaic.formula.StructuredFormula(
                lhs=self.endogenous,
                rhs=(rhs, self.fixed_effects) if self.is_fixed_effects else rhs,
            )
        )

    @classmethod
    def parse(cls, formula: str) -> list[Formula]:
        """
        Parse fixest-style formula. In case of multiple estimation syntax,
        returns a list of multiple regression formulas.
        """
        formula = _preprocess(formula)
        formulas = [
            Formula(_formula=formulaic.Formula(formulaic_compliant, _parser=_PARSER))
            for formulaic_compliant in _expand_all_multiple_estimation(formula)
        ]
        is_iv = {f.is_instrumental_variable for f in formulas}
        if len(is_iv) > 1:
            iv = [f.render() for f in formulas if f.is_instrumental_variable]
            non_iv = [f.render() for f in formulas if not f.is_instrumental_variable]
            raise FormulaSyntaxError(
                "A multiple estimation call cannot mix IV and non-IV "
                f"specifications. IV: {iv}; non-IV: {non_iv}. Fit the IV and "
                "non-IV models in separate calls."
            )
        return formulas


def _expand_first_multiple_estimation(formula: str) -> list[str] | None:
    """Expand the first multiple estimation syntax in formula.

    The call may have a space before its parenthesis and arguments on several
    lines, as in `csw (X1,` followed by `X2)` on the next line.
    """
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
    if kind is _MultipleEstimationType.mvsw:
        arguments = [
            " + ".join(combination)
            for combination in itertools.chain.from_iterable(
                itertools.combinations(arguments, r=length)
                for length in range(1, len(arguments) + 1)
            )
        ]
    elif kind is _MultipleEstimationType.csw or kind is _MultipleEstimationType.csw0:
        arguments = [" + ".join(arguments[: i + 1]) for i, _ in enumerate(arguments)]
    if (
        kind is _MultipleEstimationType.sw0
        or kind is _MultipleEstimationType.csw0
        or kind is _MultipleEstimationType.mvsw
    ):
        arguments = ["1", *arguments]
    multiple_estimation_call = formula[match.start() : parenthesis_closed + 1]
    return [
        formula.replace(multiple_estimation_call, f"({argument})")
        for argument in arguments
    ]


def _expand_all_multiple_estimation(formula: str) -> Iterator[str]:
    """Recursively expand all multiple estimation calls."""
    expansion = _expand_first_multiple_estimation(formula)
    if expansion is None:
        yield formula
    else:
        for formula_expanded in expansion:
            yield from _expand_all_multiple_estimation(formula_expanded)
