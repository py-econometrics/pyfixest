"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import formulaic
import formulaic.formula
import pandas as pd
from formulaic.materializers.types import FactorValues
from formulaic.parser.types import Factor
from formulaic.transforms.contrasts import TreatmentContrasts, encode_contrasts
from formulaic.utils.sentinels import UNSET

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec

_CONTRASTS_PREFIX = "__contrasts_"
_CONTRASTS_SUFFIX = "__"
_BIN_MAPPING_PREFIX = "__bin_mapping_"
_BIN_MAPPING_SUFFIX = "__"
_FIXED_EFFECT_ENCODING = "__fixed_effect_encoding__"
_FIXED_EFFECT_PREFIX = "__fixed_effect__("
_FIXED_EFFECT_SUFFIX = ")"


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
    # formulaic internal: `.deps[0]` is the parsed `[endog ~ instr]` block.
    return rhs.deps[0].lhs


def get_first_multistage_rhs(
    rhs: formulaic.formula.Formula,
) -> formulaic.formula.Formula:
    """Return the instrument formula from a formulaic MULTISTAGE RHS."""
    # formulaic internal: `.deps[0]` is the parsed `[endog ~ instr]` block.
    return rhs.deps[0].rhs


def filter_multistage_endogenous_terms(
    exogenous: formulaic.formula.Formula,
    endogenous_variables: Iterable[Any],
) -> formulaic.formula.SimpleFormula:
    """Drop formulaic's generated ``<endogenous>_hat`` second-stage terms."""
    # formulaic internal: MULTISTAGE renames endogenous vars to "<name>_hat" in
    # the second-stage RHS. If formulaic changes the suffix, compat tests catch it.
    generated_endogenous = {f"{variable}_hat" for variable in endogenous_variables}
    return formulaic.formula.SimpleFormula(
        [term for term in exogenous.root if term not in generated_endogenous]
    )


def flatten_model_matrix(model_matrix: formulaic.ModelMatrix) -> list[pd.DataFrame]:
    """Return the leaf data frames from a possibly structured ModelMatrix."""
    # formulaic internal: `_flatten()` is private and its iteration order is
    # documented as unstable. Callers must not rely on the returned order.
    return list(model_matrix._flatten())


def unwrap_factor_values(value: Any) -> Any:
    """Return raw values from formulaic's FactorValues proxy."""
    # formulaic internal: FactorValues is a wrapt proxy; `.__wrapped__` exposes
    # the object consumed by pyfixest's custom encoders.
    return value.__wrapped__ if isinstance(value, FactorValues) else value


def contrast_state_key(variable: str) -> str:
    """Return pyfixest's formulaic encoder-state key for i() contrasts."""
    return f"{_CONTRASTS_PREFIX}{variable}{_CONTRASTS_SUFFIX}"


def is_contrast_state_key(key: str) -> bool:
    """Return whether a key stores pyfixest i() contrast state."""
    return key.startswith(_CONTRASTS_PREFIX) and key.endswith(_CONTRASTS_SUFFIX)


def variable_from_contrast_state_key(key: str) -> str:
    """Extract the variable name from a pyfixest i() contrast state key."""
    return key[len(_CONTRASTS_PREFIX) : -len(_CONTRASTS_SUFFIX)]


def get_or_create_contrast_state(
    encoder_state: dict[str, Any], variable: str
) -> dict[str, Any]:
    """Return the nested formulaic encoder state used for one i() contrast."""
    return encoder_state.setdefault(contrast_state_key(variable), {})


def get_contrast_levels(encoder_state: Mapping[str, Any], variable: str) -> list[Any]:
    """Return stored i() contrast levels for a variable."""
    return encoder_state[contrast_state_key(variable)]["levels"]


def set_contrast_levels(
    encoder_state: Mapping[str, Any], variable: str, levels: list[Any]
) -> None:
    """Store i() contrast levels in pyfixest's nested formulaic encoder state."""
    encoder_state[contrast_state_key(variable)].update({"levels": levels})


def bin_mapping_state_key(variable: str) -> str:
    """Return pyfixest's formulaic encoder-state key for binned i() levels."""
    return f"{_BIN_MAPPING_PREFIX}{variable}{_BIN_MAPPING_SUFFIX}"


def encode_contrasts_with_state(
    data: pd.Series,
    *,
    ref: Hashable | None,
    levels: Iterable[Any] | None,
    reduced_rank: bool,
    state: dict[str, Any],
    model_spec: ModelSpec,
) -> pd.DataFrame:
    """Call formulaic encode_contrasts with explicit state injection."""
    # formulaic internal: `_state`/`_spec` are normally supplied by formulaic's
    # stateful_transform machinery; pyfixest calls the transform directly.
    encoded = encode_contrasts(
        data,
        contrasts=TreatmentContrasts(base=ref if ref is not None else UNSET),
        levels=levels,
        reduced_rank=reduced_rank,
        output="pandas",
        _state=state,
        _spec=model_spec,
    )
    return unwrap_factor_values(encoded)


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


def _iter_i_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    for _factor_expr, value in rhs_spec.encoder_state.items():
        # formulaic internal: encoder_state values are
        # (Factor.Kind, state_dict) tuples produced by formulaic materializers.
        kind, state = value
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        for key, substate in state.items():
            if is_contrast_state_key(key):
                variable = variable_from_contrast_state_key(key)
                if variable in newdata.columns and "categories" in substate:
                    yield variable, set(substate["categories"]), state


def get_fixed_effect_encoding(
    transform_state: Mapping[str, Any], column: str
) -> pd.DataFrame | None:
    """Return pyfixest's stored fixed-effect encoding DataFrame, if present."""
    fe_state = transform_state.get(column, {})
    return fe_state.get(_FIXED_EFFECT_ENCODING)


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
