from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import formulaic
import numpy as np
import pandas as pd
from formulaic import ModelSpec
from formulaic.parser.types import Term
from numpy._typing import NDArray
from scipy.sparse import csc_matrix

from pyfixest.estimation.formula.formulaic_compat import (
    get_fixed_effect_encoding,
)
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    fixed_effect_context,
    wrap_fixed_effect,
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
    model_spec: ModelSpec,
    data: pd.DataFrame,
    context: Mapping[str, Any],
) -> FixedEffectContrastCoding:
    """Build the sparse FE dummy matrix and record its coefficient alignment."""
    contrast_coding = formulaic.formula.SimpleFormula(
        wrap_fixed_effect(term=term.factors[0].metadata["term"], dummies=True)
        for term in model_spec.formula
    )
    matrix = contrast_coding.get_model_matrix(
        data,
        output="sparse",
        ensure_full_rank=True,
        context=fixed_effect_context(terms=contrast_coding, data=data, context=context),
        transform_state=model_spec.transform_state,
    )
    coefficient_positions: dict[str, FixedEffectCoefficientPositions] = {}
    for fixed_effect_name, term in zip(
        model_spec.column_names, matrix.model_spec.terms, strict=True
    ):
        coefficient_positions[fixed_effect_name] = (
            get_fixed_effect_coefficient_positions(term, matrix.model_spec)
        )

    return FixedEffectContrastCoding(
        matrix=cast(csc_matrix, matrix),
        coefficient_positions=coefficient_positions,
    )
