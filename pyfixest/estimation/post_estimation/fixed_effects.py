from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import formulaic
import numpy as np
import pandas as pd
from numpy._typing import NDArray
from scipy.sparse import csc_matrix

from pyfixest.estimation.formula.formulaic_compat import (
    FormulaicCompatibilityError,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)
from pyfixest.utils.dev_utils import _find_stack_level


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
        User-facing fixed-effect name. Interacted variables are joined with `:`,
        for example `firm:year`.
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
class FixedEffectEstimates:
    """Fixed-effect estimates recovered by `fixef()`.

    `fixef()` solves the least-squares problem in `alpha` under treatment
    coding, which drops a reference level of the second and every further
    fixed effect. With more than one fixed effect the individual levels are
    therefore identified only up to that normalization, while `sumFE` and
    contrasts within one fixed effect are invariant to it.

    Parameters
    ----------
    coefficients : Mapping[str, FixedEffect]
        Coefficient records keyed by encoded fixed-effect name, for example
        `__fixed_effect__(f1)`. `fixef()` returns their tidy frame.
    alpha : NDArray[np.float64]
        Solution of the least-squares problem in the dummy-coded fixed
        effects, shape (n_fixed_effect_coefficients,), ordered as the columns
        of the contrast-coded fixed-effect matrix.
    sumFE : NDArray[np.float64]
        Fixed-effect contribution of each observation, shape (n_rows,), in
        the units of the dependent variable. For GLMs it is on the scale of
        the linear predictor and excludes the offset. Named as in `fixest`.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 | f1", pf.get_data())
    fit.fixef().head()
    ```

    ```{python}
    estimates = fit.fixef_estimates
    estimates.sumFE[:5]
    ```
    """

    coefficients: Mapping[str, FixedEffect]
    alpha: NDArray[np.float64]
    sumFE: NDArray[np.float64]


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
    matrix : csc_matrix
        Sparse one-hot encoded fixed-effect matrix used to estimate coefficients.
    coefficient_positions : Mapping[str, FixedEffectCoefficientPositions]
        Observed and retained codes with their positions in the complete
        coefficient vector, keyed by fixed effect.
    """

    matrix: csc_matrix
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
                f"`{':'.join(source_columns)}`: {missing.iloc[:20]}\n"
                "Predictions for affected observations will be NaN",
                UserWarning,
                stacklevel=_find_stack_level(),
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
        ":".join(value_columns),
        encoding[FIXED_EFFECT_ENCODING].to_numpy(dtype=np.int64),
        tuple(encoding[column].to_numpy(copy=True) for column in value_columns),
    )


def contrast_code_fixed_effects(
    *,
    fixed_effects: pd.DataFrame,
    column_names: Sequence[str],
) -> FixedEffectContrastCoding:
    """Build sparse dummies directly from retained estimation-sample codes.

    Keep every observed level of the first FE. Omit the smallest observed code
    of each subsequent FE.
    Gaps caused by missing-row or singleton removal stay absent.
    """

    def index_observed_codes(
        codes: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.intp]]:
        """Return sorted observed codes and each row's position among them.

        Preserve original codes while assigning compact positions to observed levels.
        """
        # Get number of occurrences of each fixed-effect level code in ascending order
        fixed_effect_level_counts = np.bincount(codes)
        observed_codes = np.flatnonzero(fixed_effect_level_counts)
        # Map each code to its position in the sorted observed-code array.
        code_to_position = np.full(
            fixed_effect_level_counts.shape[0], -1, dtype=np.intp
        )
        code_to_position[observed_codes] = np.arange(observed_codes.shape[0])
        return observed_codes, code_to_position[codes]

    def select_coefficient_levels(
        observed_codes: NDArray[np.int64],
        observed_positions: NDArray[np.intp],
        *,
        keep_all: bool,
    ) -> tuple[NDArray[np.int64], NDArray[np.intp], int]:
        """Select coefficient codes and rows contributing dummy entries.

        Unless keep_all is True, omit the smallest observed code as the reference.
        """
        first_coefficient_position = 0 if keep_all else 1
        coefficient_codes = observed_codes[first_coefficient_position:]
        dummy_rows = np.flatnonzero(observed_positions >= first_coefficient_position)
        return coefficient_codes, dummy_rows, first_coefficient_position

    coefficient_positions: dict[str, FixedEffectCoefficientPositions] = {}
    rows = []
    columns = []
    offset = 0
    for position, (name, column) in enumerate(
        zip(column_names, fixed_effects.columns, strict=True)
    ):
        codes = fixed_effects[column].to_numpy(dtype=np.int64)
        observed_codes, observed_positions = index_observed_codes(codes)
        # Apply normalization for fixed-effect levels
        coefficient_codes, dummy_rows, first_coefficient_position = (
            select_coefficient_levels(
                observed_codes,
                observed_positions,
                keep_all=position == 0,
            )
        )
        rows.append(dummy_rows)
        columns.append(
            observed_positions[dummy_rows] - first_coefficient_position + offset
        )
        coefficient_positions[name] = FixedEffectCoefficientPositions(
            observed_codes=observed_codes,
            coefficient_codes=coefficient_codes,
            coefficient_indices=np.arange(offset, offset + len(coefficient_codes)),
        )
        offset += len(coefficient_codes)

    row_indices = np.concatenate(rows)
    column_indices = np.concatenate(columns)
    matrix = csc_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(fixed_effects), offset),
    )
    return FixedEffectContrastCoding(
        matrix=matrix, coefficient_positions=coefficient_positions
    )
