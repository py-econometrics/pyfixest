"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Factor, Operator, OrderedSet, Term, Token
from formulaic.utils import variables as formulaic_variables
from formulaic.utils.code import format_expr
from formulaic.utils.stateful_transforms import _is_stateful_transform
from formulaic.utils.structured import Structured
from formulaic.utils.variables import Variable

from pyfixest.errors import FormulaSyntaxError
from pyfixest.estimation.formula.transforms.factor_interaction import (
    bin_mapping_state_key,
    is_i_contrast_state_key,
    variable_from_contrast_state_key,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING_STATE_KEY,
    FixedEffectEncoding,
)
from pyfixest.estimation.formula.utils import ColumnIdentifier, term_key

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


# Adapted from Formulaic 1.2.2.
# Copyright 2019 Matthew Wardrop and contributors.
# Licensed under MIT; see THIRD_PARTY_NOTICES.md.
def _get_ast_node_variables_formulaic_284(
    node: ast.AST, env: Mapping[str, Any], aliases: Mapping[str, str]
) -> list[Variable]:
    """Work around unavailable callback arguments in Formulaic's dependency walk.

    Adapted from Formulaic 1.2.2's ``_get_ast_node_variables``:
    https://github.com/matthewwardrop/formulaic/blob/8579df0d49638acbae87e7f4171f97bf66d7d1c8/formulaic/utils/variables.py#L120-L161

    Its stateful dependency callbacks evaluate arguments even when data is unavailable,
    aborting dependency extraction instead of walking the arguments:
    https://github.com/matthewwardrop/formulaic/issues/284
    """
    variables: list[Variable] = []
    todo = deque([node])
    while todo:
        node = todo.popleft()
        if _is_stateful_transform(node, env):
            call = cast(ast.Call, node)
            delegated = ast.Call(
                func=ast.Attribute(
                    value=call.func,
                    attr="get_required_variables",
                    ctx=ast.Load(),
                ),
                args=call.args,
                keywords=call.keywords,
            )
            try:
                delegated_variables = eval(
                    compile(format_expr(delegated), "", "eval"), {}, env
                )
            except (NameError, AttributeError):
                # Missing data names, or nested Q() without its data context,
                # must not prevent the argument walk below. Literal Q() calls
                # still contribute their quoted lookup names through the callback.
                delegated_variables = ()
            variables.extend(delegated_variables)
        if not isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
            todo.extend(ast.iter_child_nodes(node))
            continue
        name = formulaic_variables._get_ast_node_name(node)
        name = aliases.get(name, name)
        if isinstance(node, ast.Call):
            variables.append(Variable(name, roles=["callable"]))
            todo.extend(node.args)
            todo.extend(node.keywords)
        else:
            variables.append(Variable(name, roles=["value"]))
    return variables


if formulaic.__version__ == "1.2.2":
    formulaic_variables._get_ast_node_variables = _get_ast_node_variables_formulaic_284


class FormulaicCompatibilityError(RuntimeError):
    """Raised when formulaic internals no longer match pyfixest expectations."""


def _iv_block(
    lhs: OrderedSet[Term], rhs: OrderedSet[Term]
) -> Structured[OrderedSet[Term]]:
    """Preserve multistage formula terms without generating `_hat` placeholders."""
    if isinstance(lhs, Structured) or isinstance(rhs, Structured):
        raise FormulaSyntaxError(
            "Nested instrumental-variable specifications are not supported. "
            "Use a single `[endogenous ~ instruments]` block."
        )
    return Structured(OrderedSet(), deps=(Structured(lhs=lhs, rhs=rhs),))


class PyfixestOperatorResolver(DefaultOperatorResolver):
    """Let pyfixest construct IV stages from their original parsed terms."""

    @staticmethod
    def _is_multistage_formula_operator(operator: Operator) -> bool:
        """Identify the `~` operator used inside `[endogenous ~ instruments]`."""
        return (
            operator.symbol == "~"
            # A multistage `~` takes two arguments (endogenous ~ exogenous)
            and operator.arity == 2
            # A multistage `~` only accepts brackets as context
            and operator.accepts_context([Token("[", kind=Token.Kind.CONTEXT)])
            # A top-level `~` only accepts an empty context `[]`
            and not operator.accepts_context([])
        )

    @property
    def operators(self) -> list[Operator]:
        """Replace only the bracketed IV operator, preserving Formulaic's grammar."""
        operators = super().operators
        multistage_operator_positions = [
            position
            for position, operator in enumerate(operators)
            if self._is_multistage_formula_operator(operator)
        ]
        if len(multistage_operator_positions) != 1:
            raise FormulaicCompatibilityError(
                "Formulaic's operator definitions changed: expected exactly one "
                "multistage '~' operator definition."
            )
        position = multistage_operator_positions[0]
        operator = operators[position]
        operators[position] = Operator(
            operator.symbol,
            arity=operator.arity,
            precedence=operator.precedence,
            associativity=operator.associativity,
            fixity=operator.fixity,
            to_terms=_iv_block,
            accepts_context=operator.accepts_context,
            structural=operator.structural,
            disabled=operator.disabled,
        )
        return operators


def _model_spec_column_identifiers(
    *, model_spec: ModelSpec, columns: Iterable[str]
) -> tuple[ColumnIdentifier, ...]:
    """Identify retained columns by term semantics and fitted encoding position.

    Display names alone can coincide for different evaluated factors. The
    encoded name also distinguishes full-rank and treatment-coded columns.
    """
    identifiers_by_name = {}
    # IV specifications can repeat the intercept term. The fitted structure
    # retains its encoded columns; term_indices collapses repeated terms.
    for term, _, encoded_columns in model_spec.structure or ():
        identity = term_key(term=term)
        for encoded_position, name in enumerate(encoded_columns):
            identifiers_by_name[name] = ColumnIdentifier(
                term=identity, encoded_position=encoded_position, name=name
            )
    return tuple(identifiers_by_name[column] for column in columns)


def get_fixed_effect_encoding(
    *, transform_state: Mapping[str, Any], column: str
) -> FixedEffectEncoding:
    """Read pyfixest's typed encoding from Formulaic's transform-state layout."""
    try:
        return cast(
            FixedEffectEncoding,
            transform_state[column][FIXED_EFFECT_ENCODING_STATE_KEY],
        )
    except KeyError as exc:
        raise FormulaicCompatibilityError(
            f"Fixed-effect encoding for `{column}` is missing from the "
            "formulaic transform state."
        ) from exc


def terms_without_intercept(formula: formulaic.formula.Formula) -> Iterator[Any]:
    """Yield formula terms excluding Formulaic's intercept term."""
    return (term for term in formula if term != "1")


def formula_required_variables(formula: formulaic.formula.SimpleFormula) -> set[str]:
    """Collect dependencies without normalizing literal lookup names."""
    return {
        str(variable)
        for term in formula
        for factor in term.factors
        for variable in factor.required_variables
        if Variable.Role.CALLABLE not in variable.roles  # ignore transforms
    }


def is_structured_formula(rhs: formulaic.formula.Formula) -> bool:
    """Return whether formulaic parsed an IV RHS as a StructuredFormula."""
    return isinstance(rhs, formulaic.formula.StructuredFormula)


def count_multistage_blocks(rhs: formulaic.formula.Formula) -> int:
    """Count formulaic MULTISTAGE deps on a StructuredFormula RHS."""
    # formulaic internal: MULTISTAGE stores the parsed first-stage formulas in
    # `.deps`; formulaic does not currently expose a documented accessor.
    deps = getattr(rhs, "deps", ())
    return len(deps) if isinstance(deps, tuple) else int(bool(deps))


def get_first_multistage_lhs(
    rhs: formulaic.formula.Formula,
) -> formulaic.formula.Formula:
    """Return the endogenous formula from a formulaic MULTISTAGE RHS."""
    return _get_single_multistage_block(rhs).lhs


def get_first_multistage_rhs(
    rhs: formulaic.formula.Formula,
) -> formulaic.formula.Formula:
    """Return the instrument formula from a formulaic MULTISTAGE RHS."""
    return _get_single_multistage_block(rhs).rhs


def _get_single_multistage_block(rhs: formulaic.formula.Formula) -> Any:
    # formulaic internal: `.deps[0]` is the parsed `[endog ~ instr]` block.
    deps = getattr(rhs, "deps", None)
    if not isinstance(deps, tuple) or len(deps) != 1:
        raise FormulaicCompatibilityError(
            "formulaic MULTISTAGE structure changed: expected a one-element "
            "`.deps` tuple containing the IV sub-formula."
        )
    block = deps[0]
    if not hasattr(block, "lhs") or not hasattr(block, "rhs"):
        raise FormulaicCompatibilityError(
            "formulaic MULTISTAGE structure changed: expected `.deps[0]` to "
            "expose `.lhs` and `.rhs` formulas."
        )
    return block


def flatten_model_matrix(model_matrix: formulaic.ModelMatrix) -> list[pd.DataFrame]:
    """Return the leaf data frames from a possibly structured ModelMatrix."""
    # formulaic internal: `_flatten()` is private and its iteration order is
    # documented as unstable. Callers must not rely on the returned order.
    return list(model_matrix._flatten())


def materialize_model_spec_with_unseen_mask(
    rhs_spec: ModelSpec,
    newdata: pd.DataFrame,
    context: Mapping[str, Any],
) -> tuple[formulaic.ModelMatrix, np.ndarray]:
    """Materialize a prediction matrix and flag unseen categorical levels."""
    materializer = rhs_spec.get_materializer(newdata, context=context)
    model_matrix = materializer.get_model_matrix(rhs_spec)
    unseen = rows_with_unseen_contrast_levels(
        rhs_spec, newdata, materializer.factor_cache
    )
    for variable, levels, state in iter_i_categorical_levels(rhs_spec, newdata):
        column = newdata[variable]
        bin_key = bin_mapping_state_key(variable)
        if bin_key in state:
            column = column.replace(state[bin_key])
        unseen |= (~column.isin(levels) & column.notna()).to_numpy()
    return model_matrix, unseen


def rows_with_unseen_contrast_levels(
    rhs_spec: ModelSpec,
    newdata: pd.DataFrame,
    evaluated_factors: Mapping[str, Any],
) -> np.ndarray:
    """
    Flag `newdata` rows whose formulaic-encoded categorical level was not seen at fit time.

    `ModelSpec.factor_contrasts` records the levels of the *evaluated* factor:
    `C(np.floor(X2))` stores levels of `floor(X2)`, not of `X2`. The check can
    therefore not be done on the factor's source columns, which is what
    `ModelSpec.factor_variables` reports.

    Formulaic's materializer caches each evaluated factor before contrast
    encoding. Comparing those values with the fitted contrast levels avoids
    confusing source columns such as `X2` with evaluated factors such as
    `C(np.floor(X2))`.

    Returns
    -------
    np.ndarray
        Boolean mask of length `len(newdata)`; True where a row carries an
        unseen (or missing) categorical level.
    """
    mask = np.zeros(newdata.shape[0], dtype=bool)
    for factor, contrast_state in rhs_spec.factor_contrasts.items():
        evaluated_factor = evaluated_factors.get(factor.expr)
        if evaluated_factor is None or not hasattr(evaluated_factor, "values"):
            raise FormulaicCompatibilityError(
                "formulaic materializer factor cache changed: expected an "
                f"evaluated factor for '{factor.expr}'."
            )
        values = np.asarray(evaluated_factor.values)
        if values.ndim != 1 or values.shape[0] != newdata.shape[0]:
            raise FormulaicCompatibilityError(
                "formulaic materializer factor cache changed: expected evaluated "
                f"factor '{factor.expr}' to contain one value per data row."
            )
        mask |= ~pd.Series(values).isin(contrast_state.levels).to_numpy()
    return mask


def i_term_columns(rhs_spec: ModelSpec) -> list[str]:
    """Return the model-matrix columns produced by terms that call ``i()``."""
    i_factor_exprs = {
        factor_expr
        for factor_expr, value in rhs_spec.encoder_state.items()
        if any(is_i_contrast_state_key(key) for key in _unpack_encoder_state(value)[1])
    }
    columns: list[str] = []
    # formulaic internal: `ModelSpec.structure` lists one EncodedTermStructure
    # per term with the exact model-matrix columns that term produced.
    for term_structure in rhs_spec.structure or ():
        if any(factor.expr in i_factor_exprs for factor in term_structure.term.factors):
            columns.extend(term_structure.columns)
    return columns


def iter_i_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    """Yield the levels pyfixest's `i()` transform recorded for each variable."""
    for _factor_expr, value in rhs_spec.encoder_state.items():
        kind, state = _unpack_encoder_state(value)
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        for key, substate in state.items():
            if is_i_contrast_state_key(key):
                variable = variable_from_contrast_state_key(key)
                if variable in newdata.columns and "categories" in substate:
                    yield variable, set(substate["categories"]), state


def _unpack_encoder_state(value: Any) -> tuple[Factor.Kind, dict[str, Any]]:
    # formulaic internal: encoder_state values are
    # (Factor.Kind, state_dict) tuples produced by formulaic materializers.
    if not isinstance(value, tuple) or len(value) != 2:
        raise FormulaicCompatibilityError(
            "formulaic ModelSpec.encoder_state structure changed: expected each "
            "value to be a two-tuple of (Factor.Kind, state_dict)."
        )
    kind, state = value
    if not isinstance(kind, Factor.Kind) or not isinstance(state, dict):
        raise FormulaicCompatibilityError(
            "formulaic ModelSpec.encoder_state structure changed: expected each "
            "value to be a two-tuple of (Factor.Kind, state_dict)."
        )
    return kind, state


def is_python_expression(factor: Factor) -> bool:
    """Return whether Formulaic represents the factor as Python syntax."""
    return factor.eval_method is Factor.EvalMethod.PYTHON
