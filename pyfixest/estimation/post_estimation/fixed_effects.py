from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import formulaic
import numpy as np
import pandas as pd
from formulaic.parser.types import Term
from numpy.typing import NDArray

from pyfixest.estimation.formula.formulaic_compat import (
    get_fixed_effect_coefficient_positions,
    get_fixed_effect_encoding,
    get_fixed_effect_encoding_data,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


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
        Integer codes identifying the observed fixed-effect levels.
    values : tuple[NDArray[Any], ...]
        Original level values, stored as one array per fixed-effect component.
        A one-way fixed effect has one array; an interaction has one array for
        each interacted variable. Every array is aligned with `codes`.
    coefficients : NDArray[np.float64]
        Estimated coefficient for every encoded level, aligned with `codes`.
        Omitted reference levels have coefficient zero.
    """

    fixed_effect: str
    variable: str
    codes: NDArray[np.int64]
    values: tuple[NDArray[Any], ...]
    coefficients: NDArray[np.float64]


def build_fixed_effect_coefficients(
    fixed_effect_names: Iterable[str],
    fixed_effect_coefficients: np.ndarray,
    model_spec: ModelSpec,
    transform_state: Mapping[str, Any],
) -> dict[str, FixedEffect]:
    """Build fixed-effect coefficient records keyed by encoded name."""
    results: dict[str, FixedEffect] = {}
    for fixed_effect_name, term in zip(
        fixed_effect_names, model_spec.terms, strict=True
    ):
        result = _build_fixed_effect(
            fixed_effect_name=fixed_effect_name,
            term=term,
            fixed_effect_coefficients=fixed_effect_coefficients,
            model_spec=model_spec,
            transform_state=transform_state,
        )
        results[result.fixed_effect] = result

    return results


def _build_fixed_effect(
    fixed_effect_name: str,
    term: Term,
    fixed_effect_coefficients: np.ndarray,
    model_spec: ModelSpec,
    transform_state: Mapping[str, Any],
) -> FixedEffect:
    """Build the coefficient record for one fixed effect."""
    variable, codes, values = get_fixed_effect_encoding_data(
        fixed_effect_name, transform_state
    )
    coefficient_codes, coefficient_indices = get_fixed_effect_coefficient_positions(
        term, model_spec
    )
    coefficient_by_code = np.zeros(codes.max() + 1, dtype=np.float64)
    coefficient_by_code[coefficient_codes] = fixed_effect_coefficients[
        coefficient_indices
    ]

    return FixedEffect(
        fixed_effect=fixed_effect_name,
        variable=variable,
        codes=codes,
        values=values,
        coefficients=coefficient_by_code[codes],
    )


def fixed_effects_to_frame(
    fixed_effects: Mapping[str, FixedEffect],
) -> pd.DataFrame:
    """Convert fixed-effect coefficient records to a tidy DataFrame."""
    frames: list[pd.DataFrame] = []
    for fixed_effect in fixed_effects.values():
        level = pd.Series(fixed_effect.values[0]).astype(str)
        for values in fixed_effect.values[1:]:
            level = level.str.cat(pd.Series(values).astype(str), sep=",")
        frames.append(
            pd.DataFrame(
                {
                    "variable": fixed_effect.variable,
                    "code": fixed_effect.codes,
                    "level": level,
                    "coefficient": fixed_effect.coefficients,
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
    valid_idx: np.ndarray,
) -> np.ndarray:
    """Return summed fixed-effect contributions for valid rows."""
    contributions: list[np.ndarray] = []
    for fixed_effect in model_matrix.columns:
        coefficient_record = coefficients[str(fixed_effect)]
        coefficient_table = pd.DataFrame(
            {
                "code": coefficient_record.codes,
                "coefficient": coefficient_record.coefficients,
            }
        )
        contribution = (
            model_matrix.loc[valid_idx, [fixed_effect]]
            .merge(
                coefficient_table.rename(columns={"code": fixed_effect}),
                on=fixed_effect,
                how="left",
                sort=False,
            )["coefficient"]
            .to_numpy()
        )
        contributions.append(contribution)

    return np.column_stack(contributions).sum(axis=1)


def warn_on_unseen_fixed_effects(
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
