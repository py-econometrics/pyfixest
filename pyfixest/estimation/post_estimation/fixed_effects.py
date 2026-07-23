from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import formulaic
import numpy as np
import pandas as pd
import scipy.sparse
from formulaic import ModelSpec
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.types import Term
from numpy._typing import NDArray
from scipy.sparse import spmatrix

from pyfixest.estimation.formula.formulaic_compat import (
    FormulaicCompatibilityError,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)


@dataclass(kw_only=True, frozen=True, slots=True)
class FixedEffect:
    """
    Columnar coefficient data belonging to one fixed effect.

    Attributes
    ----------
    fixed_effect : str
        Internal encoded fixed-effect name used as the corresponding model-matrix
        column, for example `__fixed_effect__(firm)`.
    variable : str
        User-facing fixed-effect name. Interacted variables are joined with `^`,
        for example `firm^year`.
    codes : NDArray[np.int64]
        Integer codes identifying fixed-effect levels observed in the estimation
        sample after singleton removal.
    values : tuple[NDArray[Any], ...]
        Original level values, stored as one array per fixed-effect component.
        Every array is aligned with `codes`.
    coefficients : NDArray[np.float64]
        Estimated coefficient indexed by fixed-effect code. Omitted reference
        levels have coefficient zero. Levels removed before estimation have
        coefficient `NaN`.
    """

    fixed_effect: str
    variable: str
    codes: NDArray[np.int64]
    values: tuple[NDArray[Any], ...]
    coefficients: NDArray[np.float64]

    def levels(self) -> pd.Series:
        """Combine values of fixed effect levels into comma-separated string."""
        value_columns = pd.DataFrame(dict(enumerate(self.values))).astype("string")
        return value_columns.agg(",".join, axis="columns").astype("string")


@dataclass(kw_only=True, frozen=True, slots=True)
class FixedEffectCoefficientPositions:
    """
    Fixed-effect codes and their positions in the complete coefficient vector.

    Attributes
    ----------
    observed_codes : NDArray[np.int64]
        Encoded levels observed in the estimation sample, including an omitted
        reference level.
    coefficient_codes : NDArray[np.int64]
        Encoded levels represented in the complete coefficient vector.
    coefficient_indices : NDArray[np.int64]
        Positions of those levels in the complete coefficient vector.
    """

    observed_codes: NDArray[np.int64]
    coefficient_codes: NDArray[np.int64]
    coefficient_indices: NDArray[np.int64]


@dataclass(kw_only=True, frozen=True, slots=True)
class FixedEffectContrastCoding:
    """
    Sparse dummy matrix and coefficient alignment for fixed effects.

    Attributes
    ----------
    matrix : spmatrix
        Sparse one-hot encoded fixed-effect matrix used to estimate coefficients.
    coefficient_positions : Mapping[str, FixedEffectCoefficientPositions]
        Observed and retained codes with their positions in the complete
        coefficient vector, keyed by fixed effect.
    """

    matrix: spmatrix
    coefficient_positions: Mapping[str, FixedEffectCoefficientPositions]


def build_fixed_effects(
    fixed_effect_coefficients: np.ndarray,
    contrast_coding: FixedEffectContrastCoding,
    transform_state: Mapping[str, Any],
) -> dict[str, FixedEffect]:
    """Build fixed-effect coefficient records keyed by encoded name."""
    fixed_effects: dict[str, FixedEffect] = {}
    for fixed_effect_name, positions in contrast_coding.coefficient_positions.items():
        fixed_effect = _build_fixed_effect(
            name=fixed_effect_name,
            coefficients=fixed_effect_coefficients,
            positions=positions,
            transform_state=transform_state,
        )
        fixed_effects[fixed_effect.fixed_effect] = fixed_effect

    return fixed_effects


def _build_fixed_effect(
    name: str,
    coefficients: np.ndarray,
    positions: FixedEffectCoefficientPositions,
    transform_state: Mapping[str, Any],
) -> FixedEffect:
    """Build the coefficient record for one fixed effect."""
    variable, codes, values = get_fixed_effect_encoding_data(name, transform_state)
    coefficient_by_code = np.full(codes.max() + 1, np.nan, dtype=np.float64)
    coefficient_by_code[positions.observed_codes] = 0.0
    coefficient_by_code[positions.coefficient_codes] = coefficients[
        positions.coefficient_indices
    ]
    observed = np.isin(codes, positions.observed_codes)
    return FixedEffect(
        fixed_effect=name,
        variable=variable,
        codes=codes[observed],
        values=tuple(value[observed] for value in values),
        coefficients=coefficient_by_code,
    )


def fixed_effects_to_frame(
    fixed_effects: Mapping[str, FixedEffect],
) -> pd.DataFrame:
    """Convert fixed-effect coefficient records to a tidy DataFrame."""
    frames: list[pd.DataFrame] = []
    for fixed_effect in fixed_effects.values():
        frames.append(
            pd.DataFrame(
                {
                    "variable": fixed_effect.variable,
                    "code": fixed_effect.codes,
                    "level": fixed_effect.levels(),
                    "coefficient": fixed_effect.coefficients[fixed_effect.codes],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def check_fe_dtype_compatibility(
    model_spec: formulaic.ModelSpec,
    newdata: pd.DataFrame,
) -> None:
    """Raise if new fixed-effect dtypes cannot match the fitted encodings."""
    checked_columns: set[str] = set()
    for fixed_effect in model_spec.column_names:
        encoding = get_fixed_effect_encoding(model_spec.transform_state, fixed_effect)
        source_columns = [
            column for column in encoding.columns if column != FIXED_EFFECT_ENCODING
        ]
        for column in source_columns:
            if column in checked_columns or column not in newdata.columns:
                continue
            checked_columns.add(column)
            fit_is_numeric = pd.api.types.is_numeric_dtype(encoding[column])
            new_is_numeric = pd.api.types.is_numeric_dtype(newdata[column])
            if fit_is_numeric != new_is_numeric:
                raise ValueError(
                    f"Fixed effect column '{column}' has dtype "
                    f"{newdata[column].dtype} in newdata but "
                    f"{encoding[column].dtype} in the data used for fitting, "
                    "so its levels cannot be matched. Convert the column to a "
                    "matching type before calling predict()."
                )


def predict_fixed_effects(
    model_matrix: pd.DataFrame,
    coefficients: Mapping[str, FixedEffect],
) -> np.ndarray:
    """Return summed fixed-effect contributions for each row."""
    contributions = np.zeros(len(model_matrix), dtype=np.float64)
    for fixed_effect in model_matrix.columns:
        codes = model_matrix[fixed_effect].to_numpy(dtype=np.int64)
        contributions += coefficients[str(fixed_effect)].coefficients[codes]

    return contributions


def warn_on_unseen_fixed_effect_levels(
    model_matrix: pd.DataFrame,
    model_spec: formulaic.ModelSpec,
    newdata: pd.DataFrame,
) -> None:
    """Warn about fixed-effect levels not observed during fitting."""
    for fixed_effect in model_matrix.columns:
        fixed_effect = str(fixed_effect)
        encoding = get_fixed_effect_encoding(model_spec.transform_state, fixed_effect)
        source_columns = [
            column for column in encoding.columns if column != FIXED_EFFECT_ENCODING
        ]
        unseen = (
            model_matrix[fixed_effect].isna().to_numpy()
            & newdata[source_columns].notna().all(axis=1).to_numpy()
        )
        if unseen.any():
            missing = newdata.loc[unseen, source_columns].drop_duplicates()
            warnings.warn(
                f"{missing.shape[0]} unseen level(s) for fixed effect "
                f"`{'^'.join(source_columns)}`: {missing.iloc[:20]}\n"
                "Predictions for affected observations will be NaN",
                UserWarning,
                stacklevel=3,
            )


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
) -> FixedEffectCoefficientPositions:
    """
    Align one fixed-effect term's codes with positions in the coefficient vector.

    Formulaic stores the coefficients for all fixed-effect terms in one model
    matrix. The returned coefficient positions select the entries belonging to
    `term`. The returned codes identify which encoded fixed-effect levels those
    entries represent.

    For a full-rank term, every encoded level observed in the estimation sample
    has a coefficient. For a reduced-rank term, formulaic omits the reference
    level, so its code is absent from `coefficient_codes` but remains in
    `observed_codes`. Codes absent from `observed_codes`, such as singleton
    levels removed before estimation, must not be treated as reference levels.

    Returns
    -------
    FixedEffectCoefficientPositions
        Observed codes, codes represented in the coefficient vector, and their
        positions in that vector.
    """
    (factor,) = term.factors
    contrasts_state = model_spec.factor_contrasts[factor]
    coefficient_indices = model_spec.term_indices[term]
    coefficient_codes = contrasts_state.contrasts.get_coding_column_names(
        contrasts_state.levels,
        reduced_rank=len(coefficient_indices) < len(contrasts_state.levels),
    )
    return FixedEffectCoefficientPositions(
        observed_codes=np.asarray(contrasts_state.levels, dtype=np.int64),
        coefficient_codes=np.asarray(coefficient_codes, dtype=np.int64),
        coefficient_indices=np.asarray(coefficient_indices, dtype=np.int64),
    )


def contrast_code_fixed_effects(
    fixed_effects: Iterable[Term],
    fixed_effect_names: Iterable[str],
    data: pd.DataFrame,
    context: Mapping[str, Any],
    transform_state: Mapping[str, Any],
) -> FixedEffectContrastCoding:
    """Build the sparse FE dummy matrix and record its coefficient alignment."""
    contrast_coding = formulaic.Formula(
        [f"C({fixed_effect})" for fixed_effect in fixed_effects],
        _parser=DefaultFormulaParser(include_intercept=False),
    )
    matrix = contrast_coding.get_model_matrix(
        data,
        output="sparse",
        ensure_full_rank=True,
        context=context,
        transform_state=transform_state,
    )
    coefficient_positions: dict[str, FixedEffectCoefficientPositions] = {}
    for fixed_effect_name, term in zip(
        fixed_effect_names, matrix.model_spec.terms, strict=True
    ):
        coefficient_positions[fixed_effect_name] = (
            get_fixed_effect_coefficient_positions(term, matrix.model_spec)
        )

    return FixedEffectContrastCoding(
        matrix=cast(scipy.sparse.spmatrix, matrix),
        coefficient_positions=coefficient_positions,
    )
