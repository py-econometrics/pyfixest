from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from formulaic.materializers.types import FactorValues

from pyfixest.estimation.formula.formulaic_compat import (
    bin_mapping_state_key,
    encode_contrasts_with_state,
    get_contrast_levels,
    get_or_create_contrast_state,
    set_contrast_levels,
    unwrap_factor_values,
)

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


def factor_interaction(
    data: Any,
    var2: Any = None,
    *,
    ref: Hashable | None = None,
    ref2: Hashable | None = None,
    bin: dict | None = None,
    bin2: dict | None = None,
) -> FactorValues:
    """
    Fixest-style i() operator for categorical encoding with interactions.

    Parameters
    ----------
    data : array-like
        The categorical variable to encode.
    var2 : array-like, optional
        Optional second variable to interact with (continuous or categorical).
    ref : Hashable, optional
        Reference level to drop from `data`.
    ref2 : Hashable, optional
        Reference level to drop from `var2` (only if `var2` is categorical).
    bin : dict, optional
        Mapping of `new_level -> [old_levels]` for binning `data`.
    bin2 : dict, optional
        Mapping of `new_level -> [old_levels]` for binning `var2`.

    Returns
    -------
    FactorValues
        The encoded factor values, ready for use in a formulaic model matrix.

    Examples
    --------
    Implements the `i()` operator and is used by writing `i(...)` in a formula
    rather than by calling it directly. Expands a categorical variable into
    indicators, optionally dropping a reference level and interacting it with a
    second variable. Commonly used for event study specifications. See the
    [formula syntax tutorial](/tutorials/formula-syntax.qmd).

    ```{python}
    import pyfixest as pf

    data = pf.get_data()

    fit = pf.feols("Y ~ i(f1, ref=0)", data)
    fit.tidy().head()
    ```

    Interacting with a continuous variable gives group-specific slopes.

    ```{python}
    pf.feols("Y ~ i(f1, X2, ref=0)", data).tidy().head()
    ```

    Notes
    -----
    Naming convention (matches R fixest)::

        i(cyl)            -> cyl::4, cyl::6, cyl::8
        i(cyl, ref=4)     -> cyl::6, cyl::8
        i(cyl, wt)        -> cyl::4:wt, cyl::6:wt, cyl::8:wt
        i(cyl, wt, ref=4) -> cyl::6:wt, cyl::8:wt
    """
    # Try to get variable names from Series.name attribute
    factor_name = _get_series_name(data, default="factor")
    var2_name = _get_series_name(var2, default="var") if var2 is not None else None

    def encoder(
        values: Any,
        reduced_rank: bool,
        drop_rows: list[int],
        encoder_state: dict[str, Any],
        model_spec: "ModelSpec",
    ) -> FactorValues:
        """Run encoder callback during materialization."""
        return _encode_i(
            values=values,
            factor_name=factor_name,
            var2_name=var2_name,
            ref=ref,
            ref2=ref2,
            bin=bin,
            bin2=bin2,
            reduced_rank=reduced_rank,
            drop_rows=drop_rows,
            encoder_state=encoder_state,
            model_spec=model_spec,
        )

    # When var2 is provided, wrap both variables in a dict so that find_nulls()
    # will check both for null values. This ensures drop_rows is correctly populated.
    wrapped_data = {"__data__": data, "__var2__": var2} if var2 is not None else data

    return FactorValues(
        wrapped_data,
        kind="categorical",
        spans_intercept=var2 is None,
        encoder=encoder,
    )


def _get_series_name(data: Any, default: str = "var") -> str:
    """Extract name from Series/DataFrame column, or return default."""
    if data is None:
        return default
    if isinstance(data, FactorValues):
        data = unwrap_factor_values(data)
    if isinstance(data, pd.Series) and data.name is not None:
        return str(data.name)
    return default


def _encode_i(
    values: Any,
    factor_name: str,
    var2_name: str | None,
    ref: Hashable | None,
    ref2: Hashable | None,
    bin: dict | None,
    bin2: dict | None,
    reduced_rank: bool,
    drop_rows: list[int],
    encoder_state: dict[str, Any],
    model_spec: "ModelSpec",
) -> FactorValues:
    """
    Actual encoding logic, called during materialization.

    Uses formulaic's native encode_contrasts + TreatmentContrasts for the core
    dummy encoding, then applies fixest-style naming and handles interactions.
    """
    # Extract values - may be wrapped in dict for null detection
    unwrapped = unwrap_factor_values(values)
    data = unwrapped["__data__"] if var2_name is not None else unwrapped
    var2 = unwrapped.get("__var2__") if var2_name is not None else None
    # Convert to pandas Series and drop specified rows
    data = pd.Series(data)
    data.name = factor_name
    data.drop(index=data.index[drop_rows], inplace=True)
    if var2 is not None:
        var2 = pd.Series(var2)
        var2.name = var2_name
        var2.drop(index=var2.index[drop_rows], inplace=True)
    dummies = _encode_factor(
        pd.Series(data),
        ref=ref,
        bins=bin,
        reduced_rank=reduced_rank and var2 is None,
        encoder_state=encoder_state,
        model_spec=model_spec,
    )
    # Three options: (i) no interaction, (ii) interaction with continuous variable, (ii) factor-factor interaction
    if var2 is None:
        # (i) No interaction: return categorical encoding of single variable
        dummies.rename(
            columns={level: f"{factor_name}::{level}" for level in dummies.columns},
            inplace=True,
        )
        return FactorValues(
            dummies,
            kind="categorical",
            # spans_intercept is True only when no reference level was dropped
            # (i.e., ref is None and reduced_rank is False)
            spans_intercept=(ref is None and not reduced_rank),
            column_names=tuple(dummies.columns),
            format="{field}",  # Use column names directly
        )
    elif ref2 is None and bin2 is None and _is_numeric(var2):
        # (ii) interaction with continuous variable
        result = dummies.multiply(var2.to_numpy(), axis=0)
        result.rename(
            columns={
                level: f"{factor_name}::{level}:{var2_name}"
                for level in dummies.columns
            },
            inplace=True,
        )
        return FactorValues(
            result,
            kind="numerical",
            spans_intercept=False,
            column_names=tuple(result.columns),
            format="{field}",
        )
    else:
        # (iii) factor-factor interaction
        assert var2_name is not None
        dummies2 = _encode_factor(
            data=var2,
            ref=ref2,
            bins=bin2,
            reduced_rank=False,
            encoder_state=encoder_state,
            model_spec=model_spec,
        )
        interacted = pd.DataFrame(
            _interact_dummies(
                left=dummies.to_numpy(),
                right=dummies2.to_numpy(),
            ),
            columns=[
                f"{factor_name}::{l1}:{var2_name}::{l2}"
                for l1 in dummies.columns
                for l2 in dummies2.columns
            ],
            index=dummies.index,
        )
        # Drop reference level
        if ref is None:
            ref = get_contrast_levels(encoder_state, factor_name)[0]
        if ref2 is None:
            ref2 = get_contrast_levels(encoder_state, var2_name)[0]
        interacted.drop(
            f"{factor_name}::{ref}:{var2_name}::{ref2}",
            axis=1,
            inplace=True,
            errors="ignore",
        )
        return FactorValues(
            interacted,
            kind="categorical",
            spans_intercept=True,
            column_names=tuple(interacted.columns),
            format="{field}",  # Use column names directly
        )


def _encode_factor(
    data: pd.Series,
    ref: Hashable | None,
    bins: dict | None,
    reduced_rank: bool,
    encoder_state: dict[str, Any],
    model_spec: "ModelSpec",
) -> pd.DataFrame:
    # --- Binning (optional) ---
    if bins is not None:
        data = _apply_binning(data, bins, encoder_state)
    contrasts_state = get_or_create_contrast_state(encoder_state, str(data.name))
    # Drop a level if: (1) model has intercept (reduced_rank=True), OR (2) ref is explicitly specified
    # This replicates the old monkey-patched behavior: drop=reduced_rank or ref is not None
    dummies = encode_contrasts_with_state(
        data,
        ref=ref,
        levels=contrasts_state.get("levels"),
        reduced_rank=reduced_rank or ref is not None,
        state=contrasts_state,
        model_spec=model_spec,
    )
    if "levels" not in contrasts_state:
        # Store the full category list (including the reference level dropped by
        # reduced_rank) so that during prediction encode_contrasts can correctly
        # identify and drop the same reference level rather than dropping the
        # first level of the already-reduced set.
        levels = contrasts_state.get("categories", dummies.columns.tolist())
        set_contrast_levels(encoder_state, str(data.name), levels)
    return dummies


def _interact_dummies(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Compute all pairwise products using broadcasting
    # arr1[:, :, None] has shape (n_rows, n_levels1, 1)
    # arr2[:, None, :] has shape (n_rows, 1, n_levels2)
    return np.reshape(
        # Product has shape (n_rows, n_levels1, n_levels2)
        left[:, :, None] * right[:, None, :],
        shape=(len(left), -1),
    )


def _is_numeric(series: pd.Series) -> bool:
    """Check if series is numeric (not categorical/object)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _apply_binning(series: pd.Series, bins: dict, state: dict) -> pd.Series:
    """
    Apply binning: bin={'low': ['a','b'], 'high': ['c','d']}.

    Values not in the mapping are kept unchanged (matches R fixest behavior).
    The bin mapping is namespaced per variable to avoid collisions when
    both sides of an interaction are binned (``i(a, b, bin=..., bin2=...)``).
    """
    bin_key = bin_mapping_state_key(str(series.name))
    if bin_key not in state:
        mapping = {}
        for new_level, old_levels in bins.items():
            for old in old_levels:
                mapping[old] = new_level
        state[bin_key] = mapping
    return series.replace(state[bin_key])
