from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from scipy.sparse import csc_matrix

from pyfixest.estimation.formula.formulaic_compat import (
    get_fixed_effect_encoding,
)


@dataclass(kw_only=True, frozen=True, slots=True)
class FixedEffect:
    """Columnar coefficient data belonging to one fixed effect.

    Access as ``fit.fixef_estimates.fixed_effects[formulaic_name]``. Created by ``fit.fixef()``; keys are Formulaic column names.

    Attributes
    ----------
    formulaic_column : str
        Internal column name in the Formulaic spec. Its factor metadata
        retains the original parsed term; `ModelMatrix.fixed_effects` exposes
        the original labels as column names.
    name : str
        User-facing fixed-effect name. Interacted variables are joined with `:`,
        for example `firm:year`.
    observed_codes : NDArray[np.int64]
        Integer codes identifying fixed-effect levels observed in the estimation
        sample after singleton removal.
    level_values : tuple[NDArray[Any], ...]
        Original level values, stored as one array per fixed-effect component.
        Every array is aligned with `observed_codes`.
    coefficients : NDArray[np.float64]
        Estimated coefficient indexed by fixed-effect code. Omitted reference
        levels have coefficient zero. Levels removed before estimation have
        coefficient `NaN`.

    See Also
    --------
    [fixef()](/reference/estimation.models.feols_.Feols.fixef.qmd),
    [FixedEffectEstimates](/reference/estimation.state.FixedEffectEstimates.qmd).
    """

    formulaic_column: str
    name: str
    observed_codes: NDArray[np.int64]
    level_values: tuple[NDArray[Any], ...]
    coefficients: NDArray[np.float64]

    def level_labels(self) -> pd.Series:
        """Combine values of fixed effect levels into comma-separated string."""
        value_columns = pd.DataFrame(dict(enumerate(self.level_values))).astype(
            "string"
        )
        return value_columns.agg(",".join, axis="columns").astype("string")


@dataclass(kw_only=True, frozen=True, slots=True)
class FixedEffectEstimates:
    """Fixed-effect estimates recovered by `fixef()`.

    Access as ``fit.fixef_estimates``. Created by ``fit.fixef()``.

    `fixef()` solves the least-squares problem in `alpha` under treatment
    coding, which drops a reference level of the second and every further
    fixed effect. With more than one fixed effect the individual levels are
    therefore identified only up to that normalization, while `sumFE` and
    contrasts within one fixed effect are invariant to it.

    Parameters
    ----------
    fixed_effects : Mapping[str, FixedEffect]
        Coefficient records keyed by Formulaic column name. `fixef()` returns
        their tidy frame with the original factor labels.
    alpha : NDArray[np.float64]
        Solution of the least-squares problem in the dummy-coded fixed
        effects, shape `(n_fixed_effect_coefficients,)`, ordered as the columns
        of the fixed-effect dummy matrix.
    sumFE : NDArray[np.float64]
        Fixed-effect contribution of each observation, shape `(n_rows,)`, in
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

    See Also
    --------
    [fixef()](/reference/estimation.models.feols_.Feols.fixef.qmd),
    [FixedEffect](/reference/estimation.state.FixedEffect.qmd).
    """

    fixed_effects: Mapping[str, FixedEffect]
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
class FixedEffectDummies:
    """
    Sparse dummy matrix and coefficient alignment for fixed effects.

    Attributes
    ----------
    matrix : csc_matrix
        Sparse one-hot encoded fixed-effect matrix used to estimate coefficients.
    coefficient_positions : Mapping[str, FixedEffectCoefficientPositions]
        Observed and retained codes with their positions in the complete
        coefficient vector, keyed by Formulaic column name.
    """

    matrix: csc_matrix
    coefficient_positions: Mapping[str, FixedEffectCoefficientPositions]


def build_fixed_effects(
    alpha: np.ndarray,
    dummies: FixedEffectDummies,
    transform_state: Mapping[str, Any],
) -> dict[str, FixedEffect]:
    """Build fixed-effect coefficient records keyed by Formulaic column name."""
    fixed_effects: dict[str, FixedEffect] = {}
    for formulaic_column, positions in dummies.coefficient_positions.items():
        encoding = get_fixed_effect_encoding(
            transform_state=transform_state, column=formulaic_column
        )
        coefficients = np.full(len(encoding.code_index), np.nan, dtype=np.float64)
        # Observed reference levels are zero; removed levels remain NaN.
        coefficients[positions.observed_codes] = 0.0
        coefficients[positions.coefficient_codes] = alpha[positions.coefficient_indices]
        fixed_effects[formulaic_column] = FixedEffect(
            formulaic_column=formulaic_column,
            name=encoding.name,
            observed_codes=positions.observed_codes,
            level_values=encoding.decode(codes=positions.observed_codes),
            coefficients=coefficients,
        )
    return fixed_effects


def fixed_effects_to_frame(
    fixed_effects: Mapping[str, FixedEffect],
) -> pd.DataFrame:
    """Convert fixed-effect coefficient records to a tidy DataFrame."""
    frames: list[pd.DataFrame] = []
    for fixed_effect in fixed_effects.values():
        frames.append(
            pd.DataFrame(
                {
                    "variable": fixed_effect.name,
                    "code": fixed_effect.observed_codes,
                    "level": fixed_effect.level_labels(),
                    "coefficient": fixed_effect.coefficients[
                        fixed_effect.observed_codes
                    ],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def sum_fixed_effects(
    codes: pd.DataFrame,
    fixed_effects: Mapping[str, FixedEffect],
) -> np.ndarray:
    """Return summed fixed-effect contributions for each row of FE codes."""
    contributions = np.zeros(len(codes), dtype=np.float64)
    for column in codes.columns:
        row_codes = codes[column].to_numpy(dtype=np.int64)
        contributions += fixed_effects[str(column)].coefficients[row_codes]

    return contributions


def build_fixed_effect_dummies(
    *,
    codes: pd.DataFrame,
    formulaic_columns: Sequence[str],
) -> FixedEffectDummies:
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
    for position, (formulaic_column, code_column) in enumerate(
        zip(formulaic_columns, codes.columns, strict=True)
    ):
        row_codes = codes[code_column].to_numpy(dtype=np.int64)
        observed_codes, observed_positions = index_observed_codes(row_codes)
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
        coefficient_positions[formulaic_column] = FixedEffectCoefficientPositions(
            observed_codes=observed_codes,
            coefficient_codes=coefficient_codes,
            coefficient_indices=np.arange(offset, offset + len(coefficient_codes)),
        )
        offset += len(coefficient_codes)

    row_indices = np.concatenate(rows)
    column_indices = np.concatenate(columns)
    matrix = csc_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(codes), offset),
    )
    return FixedEffectDummies(
        matrix=matrix, coefficient_positions=coefficient_positions
    )
