"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import formulaic
import formulaic.formula
import pandas as pd
from formulaic.parser.types import Factor
from formulaic.utils.variables import Variable

from pyfixest.estimation.formula.transforms.factor_interaction import (
    is_contrast_state_key,
    variable_from_contrast_state_key,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec

_FIXED_EFFECT_PREFIX = "__fixed_effect__("
_FIXED_EFFECT_SUFFIX = ")"


class FormulaicCompatibilityError(RuntimeError):
    """Raised when formulaic internals no longer match pyfixest expectations."""


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


def filter_multistage_endogenous_terms(
    exogenous: formulaic.formula.Formula,
    endogenous_variables: Iterable[Any],
) -> formulaic.formula.SimpleFormula:
    """Drop formulaic's generated ``<endogenous>_hat`` second-stage terms."""
    # formulaic internal: MULTISTAGE renames endogenous vars to "<name>_hat" in
    # the second-stage RHS. If formulaic changes the suffix, compat tests catch it.
    generated_endogenous = {f"{variable}_hat" for variable in endogenous_variables}
    terms = list(exogenous.root)
    missing = generated_endogenous - {str(term) for term in terms}
    if missing:
        raise FormulaicCompatibilityError(
            "formulaic MULTISTAGE endogenous suffix changed: expected generated "
            f"second-stage terms {sorted(missing)} to be present before filtering."
        )
    return formulaic.formula.SimpleFormula(
        [term for term in terms if str(term) not in generated_endogenous]
    )


def flatten_model_matrix(model_matrix: formulaic.ModelMatrix) -> list[pd.DataFrame]:
    """Return the leaf data frames from a possibly structured ModelMatrix."""
    # formulaic internal: `_flatten()` is private and its iteration order is
    # documented as unstable. Callers must not rely on the returned order.
    return list(model_matrix._flatten())


def iter_model_spec_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    """Yield categorical variable levels encoded in a formulaic ModelSpec."""
    yield from _iter_documented_categorical_levels(rhs_spec, newdata)
    yield from _iter_i_categorical_levels(rhs_spec, newdata)


def _iter_documented_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    for factor, contrast_state in rhs_spec.factor_contrasts.items():
        levels = contrast_state.levels
        for variable in rhs_spec.factor_variables.get(factor, set()):
            variable_name = str(variable)
            if variable_name in newdata.columns:
                yield variable_name, set(levels), {}


def get_fixed_effect_encoding(
    transform_state: Mapping[str, Any], column: str
) -> pd.DataFrame | None:
    """Return pyfixest's stored fixed-effect encoding DataFrame, if present."""
    fe_state = transform_state.get(column, {})
    return fe_state.get(FIXED_EFFECT_ENCODING)


def get_fixed_effect_code_values(
    transform_state: Mapping[str, Any], column: str
) -> set[str]:
    """Return the non-null fixed-effect ngroup codes seen during fitting."""
    encoding_df = get_fixed_effect_encoding(transform_state, column)
    if encoding_df is None:
        return set()
    code_col = encoding_df.columns[-1]
    return set(str(code) for code in encoding_df[code_col].dropna())


def decode_fixed_effect_dict(
    internal: Mapping[str, Mapping[str, float]],
    transform_state: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Decode ngroup-coded fixed-effect estimates to user-facing labels."""
    res: dict[str, dict[str, float]] = {}
    for col, levels in internal.items():
        code_to_value = _fixed_effect_code_to_value(transform_state, col)
        var_name = _fixed_effect_variable_name(col)
        res[var_name] = {}
        for level, coefficient in levels.items():
            decoded_level = code_to_value.get(level, level)
            res[var_name][decoded_level] = coefficient
    return res


def _fixed_effect_code_to_value(
    transform_state: Mapping[str, Any], column: str
) -> dict[str, str]:
    encoding_df = get_fixed_effect_encoding(transform_state, column)
    if encoding_df is None:
        return {}

    code_col = encoding_df.columns[-1]
    value_cols = list(encoding_df.columns[:-1])
    valid_rows = encoding_df.dropna(subset=[code_col])
    codes = valid_rows[code_col].astype(str)
    if len(value_cols) == 1:
        values = valid_rows[value_cols[0]].astype(str)
    else:
        values = valid_rows[value_cols].astype(str).agg(",".join, axis=1)
    return dict(zip(codes, values, strict=True))


def _fixed_effect_variable_name(column: str) -> str:
    if column.startswith(_FIXED_EFFECT_PREFIX) and column.endswith(
        _FIXED_EFFECT_SUFFIX
    ):
        inner = column[len(_FIXED_EFFECT_PREFIX) : -len(_FIXED_EFFECT_SUFFIX)]
        return inner.replace(", ", "^").replace(",", "^")
    return column


def get_fixed_effect_columns(fe_spec: ModelSpec, fixed_effect: str) -> list[str]:
    """Return input columns for a given encoded fixed effect."""
    variables = fe_spec.factor_variables[fixed_effect]
    return [str(v) for v in variables if Variable.Role.VALUE in v.roles]


def _iter_i_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    for _factor_expr, value in rhs_spec.encoder_state.items():
        kind, state = _unpack_encoder_state(value)
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        for key, substate in state.items():
            if is_contrast_state_key(key):
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
