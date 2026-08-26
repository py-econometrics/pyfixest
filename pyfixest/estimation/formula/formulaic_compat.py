"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
from formulaic.parser.types import Factor

from pyfixest.estimation.formula.transforms.factor_interaction import (
    bin_mapping_state_key,
    is_contrast_state_key,
    variable_from_contrast_state_key,
)

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


class FormulaicCompatibilityError(RuntimeError):
    """Raised when formulaic internals no longer match pyfixest expectations."""


def terms_without_intercept(formula: formulaic.formula.Formula) -> Iterator[Any]:
    """Yield formula terms excluding Formulaic's intercept term."""
    return (term for term in formula if term != "1")


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
    endogenous_terms: Iterable[Any],
) -> formulaic.formula.SimpleFormula:
    """Drop formulaic's generated ``<endogenous term>_hat`` second-stage terms."""
    # formulaic internal: MULTISTAGE renames each endogenous *term* to
    # "<term>_hat" in the second-stage RHS -- `log(X2)` becomes `log(X2)_hat`,
    # not `X2_hat`. If formulaic changes the suffix, compat tests catch it.
    generated_endogenous = {f"{term}_hat" for term in endogenous_terms}
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


def iter_i_categorical_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame
) -> Iterator[tuple[str, set[Any], dict[str, Any]]]:
    """Yield the levels pyfixest's `i()` transform recorded for each variable."""
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
