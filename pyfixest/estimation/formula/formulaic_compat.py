"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
from formulaic.parser.types import Factor, Term
from numpy.typing import NDArray

from pyfixest.estimation.formula.transforms.factor_interaction import (
    is_contrast_state_key,
    variable_from_contrast_state_key,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


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
) -> pd.DataFrame:
    """Return pyfixest's stored fixed-effect encoding DataFrame."""
    try:
        return transform_state[column][FIXED_EFFECT_ENCODING]
    except KeyError as exc:
        raise FormulaicCompatibilityError(
            f"Fixed-effect encoding for `{column}` is missing from the "
            "formulaic transform state."
        ) from exc


def get_fixed_effect_encoding_data(
    fixed_effect_name: str,
    transform_state: Mapping[str, Any],
) -> tuple[str, NDArray[np.int64], tuple[NDArray[Any], ...]]:
    """Return normalized encoding data for one fixed effect."""
    encoding = get_fixed_effect_encoding(transform_state, fixed_effect_name)
    value_columns = [
        column for column in encoding.columns if column != FIXED_EFFECT_ENCODING
    ]
    return (
        "^".join(value_columns),
        encoding[FIXED_EFFECT_ENCODING].to_numpy(dtype=np.int64),
        tuple(encoding[column].to_numpy(copy=True) for column in value_columns),
    )


def get_fixed_effect_coefficient_positions(
    term: Term,
    model_spec: ModelSpec,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Align one fixed-effect term's codes with positions in the coefficient vector.

    Formulaic stores the coefficients for all fixed-effect terms in one model
    matrix. The returned coefficient positions select the entries belonging to
    `term`. The returned codes identify which encoded fixed-effect levels those
    entries represent.

    For a full-rank term, every encoded level has a coefficient. For a
    reduced-rank term, formulaic omits the reference level, so its code is absent
    from the returned codes. The caller can therefore initialize coefficients for
    all codes to zero and fill only the returned code-position pairs.

    Returns
    -------
    coefficient_codes : NDArray[np.int64]
        Encoded fixed-effect levels represented in the coefficient vector.
    coefficient_indices : NDArray[np.int64]
        Positions of this term's coefficients in the complete fixed-effect
        coefficient vector.
    """
    (factor,) = term.factors
    contrasts_state = model_spec.factor_contrasts[factor]
    coefficient_indices = model_spec.term_indices[term]
    coefficient_codes = contrasts_state.contrasts.get_coding_column_names(
        contrasts_state.levels,
        reduced_rank=len(coefficient_indices) < len(contrasts_state.levels),
    )
    return (
        np.asarray(coefficient_codes, dtype=np.int64),
        np.asarray(coefficient_indices, dtype=np.int64),
    )


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
