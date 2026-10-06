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
    """
    Columnar coefficient data belonging to one fixed effect.

    Attributes
    ----------
    fixed_effect : str
        Internal encoded fixed-effect name used as the corresponding model-matrix
        column, whose numeric wrapper ID identifies the parsed term.
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
        Coefficient records keyed by encoded fixed-effect name. `fixef()`
        returns their tidy frame with the original factor labels.
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
    for name, positions in contrast_coding.coefficient_positions.items():
        encoding = get_fixed_effect_encoding(
            transform_state=transform_state, column=name
        )
        coefficients = np.full(len(encoding.combinations), np.nan, dtype=np.float64)
        # Observed reference levels are zero; removed levels remain NaN.
        coefficients[positions.observed_codes] = 0.0
        coefficients[positions.coefficient_codes] = fixed_effect_coefficients[
            positions.coefficient_indices
        ]
        fixed_effects[name] = FixedEffect(
            fixed_effect=name,
            variable=encoding.variable,
            codes=positions.observed_codes,
            values=encoding.decoded_values(codes=positions.observed_codes),
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
                    "variable": fixed_effect.variable,
                    "code": fixed_effect.codes,
                    "level": fixed_effect.levels(),
                    "coefficient": fixed_effect.coefficients[fixed_effect.codes],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


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


def contrast_code_fixed_effects(
    *, fixed_effects: pd.DataFrame, column_names: Sequence[str]
) -> FixedEffectContrastCoding:
    """Build recovery dummies directly from retained estimation-sample codes.

    Keep every observed level of the first FE. For each subsequent FE, omit
    its smallest observed code as the reference, matching Formulaic treatment
    coding. Gaps caused by missing-row or singleton removal stay absent.
    """
    coefficient_positions: dict[str, FixedEffectCoefficientPositions] = {}
    rows = []
    columns = []
    offset = 0
    for position, (name, column) in enumerate(
        zip(column_names, fixed_effects.columns, strict=True)
    ):
        codes = fixed_effects[column].to_numpy(dtype=np.int64)
        observed_codes = np.unique(codes)
        coefficient_codes = observed_codes if position == 0 else observed_codes[1:]
        retained = (
            np.ones(len(codes), dtype=bool)
            if position == 0
            else codes != observed_codes[0]
        )
        rows.append(np.flatnonzero(retained))
        columns.append(np.searchsorted(coefficient_codes, codes[retained]) + offset)
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
