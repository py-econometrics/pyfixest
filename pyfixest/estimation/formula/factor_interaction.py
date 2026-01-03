from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from formulaic.materializers.types import FactorValues
from formulaic.transforms.contrasts import TreatmentContrasts, encode_contrasts
from formulaic.utils.sentinels import UNSET

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec


def factor_interaction(
    data: Any,
    var2: Any = None,
    *,
    ref: Optional[Hashable] = None,
    ref2: Optional[Hashable] = None,
    bin: Optional[dict] = None,
    bin2: Optional[dict] = None,
) -> FactorValues:
    """
    Fixest-style i() operator for categorical encoding with interactions.

    Args:
        data: The categorical variable
        var2: Optional second variable for interaction (continuous or categorical)
        ref: Reference level to drop from data
        ref2: Reference level to drop from var2 (if categorical)
        bin: Dict mapping new_level -> [old_levels] for binning

    Naming convention (matches R fixest):
        i(cyl)           -> cyl::4, cyl::6, cyl::8
        i(cyl, ref=4)    -> cyl::6, cyl::8
        i(cyl, wt)       -> cyl::4:wt, cyl::6:wt, cyl::8:wt
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
        data = data.__wrapped__
    if isinstance(data, pd.Series) and data.name is not None:
        return str(data.name)
    return default


def _encode_i(
    values: Any,
    factor_name: str,
    var2_name: Optional[str],
    ref: Optional[Hashable],
    ref2: Optional[Hashable],
    bin: Optional[dict],
    bin2: Optional[dict],
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
    unwrapped = values.__wrapped__ if isinstance(values, FactorValues) else values

    # Extract data and var2 from dict if present
    if isinstance(unwrapped, dict) and "__data__" in unwrapped:
        data = unwrapped["__data__"]
        var2 = unwrapped.get("__var2__")
    else:
        data = unwrapped
        var2 = None

    # Convert to pandas Series and drop specified rows
    factor_series = pd.Series(data)
    factor_series = factor_series.drop(index=factor_series.index[drop_rows])

    # --- Binning (optional) ---
    if bin is not None:
        factor_series = _apply_binning(factor_series, bin, encoder_state)

    # --- Get levels from state or data ---
    levels = encoder_state.get("levels")

    # --- Use formulaic's encode_contrasts for the dummy encoding ---
    # Create a dedicated sub-state for encode_contrasts to avoid key collisions
    contrasts_state = encoder_state.setdefault("_contrasts_state", {})

    # Build contrasts: TreatmentContrasts with base (ref or UNSET) and drop
    contrasts = TreatmentContrasts(
        base=ref if ref is not None else UNSET, drop=reduced_rank or ref is not None
    )

    encoded = encode_contrasts(
        factor_series,
        contrasts=contrasts,
        levels=levels,
        reduced_rank=ref is not None,
        output="pandas",
        _state=contrasts_state,
        _spec=model_spec,
    )

    # Extract the underlying DataFrame and levels from state
    dummies = encoded.__wrapped__
    levels_encoded = list(dummies.columns)  # These are the levels that were kept

    # Store levels in our state for consistency across train/predict
    if "levels" not in encoder_state:
        encoder_state["levels"] = contrasts_state.get("categories", levels_encoded)

    # --- No interaction: apply fixest naming and return ---
    if var2 is None or var2_name is None:
        col_names = [f"{factor_name}::{level}" for level in levels_encoded]
        dummies.columns = col_names
        return FactorValues(
            dummies,
            kind="categorical",
            spans_intercept=(ref is None and not reduced_rank),
            column_names=tuple(col_names),
            encoded=True,
            format="{field}",  # Use column names directly
        )

    # # --- Check if user specified to force var2 to categorical ---
    # force_categorical_prefix = re.match(r"^i\.(?P<variable>.+)$", var2)
    # if force_categorical := force_categorical_prefix is not None:
    #     var2 = force_categorical_prefix["variable"]

    # --- Handle interaction with var2 ---
    var2_series = pd.Series(
        var2.__wrapped__ if isinstance(var2, FactorValues) else var2
    )
    var2_series = var2_series.drop(index=var2_series.index[drop_rows])
    if bin2 is not None:
        var2_series = _apply_binning(var2_series, bin2, encoder_state)

    if ref2 is None and _is_numeric(var2_series):
        # Factor x Continuous interaction
        # Fixest naming: factor_name::level:var2_name (e.g., cyl::4:wt)
        result = dummies.multiply(var2_series, axis=0)
        col_names = [f"{factor_name}::{level}:{var2_name}" for level in levels_encoded]
        result.columns = col_names
        return FactorValues(
            result,
            kind="numerical",
            spans_intercept=False,
            column_names=tuple(col_names),
            encoded=True,
            format="{field}",
        )
    else:
        # Factor x Factor interaction
        return _factor_factor_interaction(
            dummies,
            levels_encoded,
            var2_series,
            ref,
            ref2,
            factor_name,
            var2_name,
            reduced_rank,
            encoder_state,
            model_spec,
        )


def _is_numeric(series: pd.Series) -> bool:
    """Check if series is numeric (not categorical/object)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _apply_binning(series: pd.Series, bin: dict, state: dict) -> pd.Series:
    """
    Apply binning: bin={'low': ['a','b'], 'high': ['c','d']}.

    Values not in the mapping are kept unchanged (matches R fixest behavior).
    """
    if "bin_mapping" not in state:
        mapping = {}
        for new_level, old_levels in bin.items():
            for old in old_levels:
                mapping[old] = new_level
        state["bin_mapping"] = mapping
    # Use replace() instead of map() to keep unmapped values unchanged
    return series.replace(state["bin_mapping"])


def _factor_factor_interaction(
    dummies1: pd.DataFrame,
    levels1: list,
    var2: pd.Series,
    ref: Optional[Hashable],
    ref2: Optional[Hashable],
    factor_name: str,
    var2_name: str,
    reduced_rank: bool,
    state: dict,
    model_spec: "ModelSpec",
) -> FactorValues:
    """Handle Factor x Factor interaction using encode_contrasts for var2."""
    # Create a dedicated sub-state for var2's encode_contrasts
    contrasts_state2 = state.setdefault("_contrasts_state2", {})

    # Get existing levels from state, or None to infer from data
    levels2 = state.get("levels2")

    # Use encode_contrasts for var2
    contrasts2 = TreatmentContrasts(
        base=ref2 if ref2 is not None else UNSET, drop=reduced_rank or ref2 is not None
    )

    encoded2 = encode_contrasts(
        var2,
        contrasts=contrasts2,
        levels=levels2,
        reduced_rank=False,
        output="pandas",
        _state=contrasts_state2,
        _spec=model_spec,
    )

    dummies2 = encoded2.__wrapped__
    levels2_encoded = list(dummies2.columns)

    # Store levels2 in state for consistency
    if "levels2" not in state:
        state["levels2"] = contrasts_state2.get("categories", levels2_encoded)

    # Create all pairwise interactions with fixest-style names
    # For factor x factor: factor1::level1:factor2::level2 (e.g., cyl_f::4:gear_f::4)
    result_cols = {}
    col_names = []
    for l1 in levels1:
        for l2 in levels2_encoded:
            col_name = f"{factor_name}::{l1}:{var2_name}::{l2}"
            result_cols[col_name] = dummies1[l1] * dummies2[l2]
            col_names.append(col_name)

    # To match R's fixest behavior: when no explicit references are provided,
    # drop the first combination (reference levels of both factors).
    # This handles collinearity with the intercept in typical models.
    # Note: reduced_rank is always False for factor-factor interactions,
    # so we use ref/ref2 to determine when to drop.
    if ref is None and ref2 is None and len(col_names) > 0:
        # Remove first combination from result
        first_col = col_names[0]
        del result_cols[first_col]
        col_names = col_names[1:]

    result = pd.DataFrame(result_cols, index=dummies1.index)

    return FactorValues(
        result,
        kind="categorical",
        spans_intercept=False,
        column_names=tuple(col_names),
        encoded=True,
        format="{field}",  # Use column names directly
    )
