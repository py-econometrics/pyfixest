"""Compatibility helpers for formulaic internals used by pyfixest."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import formulaic
import formulaic.formula
import numpy as np
import pandas as pd
from formulaic import ModelSpec
from formulaic.errors import DataMismatchWarning
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.types import Factor

from pyfixest.estimation.formula.transforms.factor_interaction import (
    is_contrast_state_key,
    variable_from_contrast_state_key,
)


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


def rows_with_unseen_contrast_levels(
    rhs_spec: ModelSpec, newdata: pd.DataFrame, context: Mapping[str, Any]
) -> np.ndarray:
    """
    Flag `newdata` rows whose formulaic-encoded categorical level was not seen at fit time.

    `ModelSpec.factor_contrasts` records the levels of the *evaluated* factor:
    `C(np.floor(X2))` stores levels of `floor(X2)`, not of `X2`. The check can
    therefore not be done on the factor's source columns, which is what
    `ModelSpec.factor_variables` reports.

    Re-encoding each factor on its own side-steps evaluating the expression by
    hand. At full rank and with the stored encoder state, formulaic casts an
    unseen level to NaN, which dummy encoding turns into an all-zero row, while
    every seen level contributes exactly one indicator.

    Returns
    -------
    np.ndarray
        Boolean mask of length `len(newdata)`; True where a row carries an
        unseen (or missing) categorical level.
    """
    mask = np.zeros(newdata.shape[0], dtype=bool)
    for factor in rhs_spec.factor_contrasts:
        encoded = _encode_factor_at_full_rank(factor, rhs_spec, newdata, context)
        mask |= np.asarray(encoded.sum(axis=1)).ravel() == 0
    return mask


def _encode_factor_at_full_rank(
    factor: Factor,
    rhs_spec: ModelSpec,
    newdata: pd.DataFrame,
    context: Mapping[str, Any],
) -> Any:
    if factor.expr not in rhs_spec.encoder_state:
        raise FormulaicCompatibilityError(
            "formulaic ModelSpec.encoder_state structure changed: expected a "
            f"stored state for the encoded categorical factor '{factor.expr}'."
        )
    factor_spec = ModelSpec(
        formula=formulaic.Formula(
            f"{factor.expr} - 1",
            _parser=DefaultFormulaParser(include_intercept=False),
        ),
        ensure_full_rank=False,
        output="sparse",
        # deepcopy: materializing writes the observed categories back into the
        # state, which must not leak into the fitted model's spec.
        encoder_state=copy.deepcopy({factor.expr: rhs_spec.encoder_state[factor.expr]}),
    )
    with warnings.catch_warnings():
        # The caller materializes the same data against the fitted spec, which
        # already warns about levels outside the nominated ones.
        warnings.simplefilter("ignore", DataMismatchWarning)
        return factor_spec.get_model_matrix(
            newdata, context=context, na_action="ignore"
        )


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
