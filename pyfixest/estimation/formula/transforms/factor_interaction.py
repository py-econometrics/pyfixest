from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from formulaic import FactorValues
from formulaic.transforms import encode_contrasts
from formulaic.transforms.contrasts import TreatmentContrasts
from formulaic.utils.sentinels import UNSET

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
        model_spec: ModelSpec,
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

    # Only a complete set of factor levels or interaction cells spans the
    # intercept. Explicit references remove groups, so later terms must remain
    # eligible for full-rank encoding.
    spans_intercept = (
        ref is None
        and ref2 is None
        and (
            var2 is None
            or _is_factor_interaction(
                var2=pd.Series(unwrap_factor_values(var2)), ref2=ref2, bin2=bin2
            )
        )
    )
    return FactorValues(
        wrapped_data,
        kind="categorical",
        spans_intercept=spans_intercept,
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
    model_spec: ModelSpec,
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
    data = pd.Series(data)
    data.name = factor_name
    data = _observed_levels(data, bins=bin, encoder_state=encoder_state)
    data.drop(index=data.index[drop_rows], inplace=True)
    var2_is_numeric = False
    if var2 is not None:
        var2 = pd.Series(var2)
        var2.name = var2_name
        var2_is_numeric = not _is_factor_interaction(var2=var2, ref2=ref2, bin2=bin2)
        if not var2_is_numeric:
            var2 = _observed_levels(var2, bins=bin2, encoder_state=encoder_state)
        var2.drop(index=var2.index[drop_rows], inplace=True)
    dummies = _encode_factor(
        data,
        ref=ref,
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
    elif var2_is_numeric:
        # (ii) interaction with continuous variable
        result = dummies.multiply(var2.to_numpy(dtype=np.float64), axis=0)
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
        # Without ref/ref2 the cells span the intercept. Drop the first cell
        # only when the model already spans it (intercept or an earlier
        # full-rank term); otherwise keep every cell.
        spans_intercept = ref is None and ref2 is None
        if spans_intercept and reduced_rank:
            first = get_contrast_levels(encoder_state, factor_name)[0]
            first2 = get_contrast_levels(encoder_state, var2_name)[0]
            interacted.drop(
                f"{factor_name}::{first}:{var2_name}::{first2}",
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return FactorValues(
            interacted,
            kind="categorical",
            spans_intercept=spans_intercept and not reduced_rank,
            column_names=tuple(interacted.columns),
            format="{field}",  # Use column names directly
        )


def _observed_levels(
    series: pd.Series, bins: dict | None, encoder_state: dict[str, Any]
) -> pd.Series:
    """
    Bin a factor and restrict its levels to the values in the supplied data.

    Like fixest, i() fixes the levels before the model drops rows with missing
    values: a level seen only on dropped rows remains a level (and can be the
    reference), while unused categories of a pandas Categorical are not levels.
    """
    if bins is not None:
        series = _apply_binning(series, bins, encoder_state)
    return series.astype("category").cat.remove_unused_categories()


def _encode_factor(
    data: pd.Series,
    ref: Hashable | None,
    reduced_rank: bool,
    encoder_state: dict[str, Any],
    model_spec: ModelSpec,
) -> pd.DataFrame:
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
        (len(left), -1),
    )


def _is_factor_interaction(
    var2: pd.Series, ref2: Hashable | None, bin2: dict | None
) -> bool:
    """Return whether `i(f, var2)` interacts two factors rather than a slope."""
    return ref2 is not None or bin2 is not None or not _is_numeric(var2)


def _is_numeric(series: pd.Series) -> bool:
    """Check whether a second variable encodes numeric slopes.

    Booleans count as numeric: fixest interacts a logical second variable as a
    0/1 slope, not as a factor, including object columns of booleans.
    """
    return pd.api.types.is_numeric_dtype(series) or (
        pd.api.types.is_object_dtype(series)
        and pd.api.types.infer_dtype(series, skipna=True) == "boolean"
    )


def _apply_binning(series: pd.Series, bins: dict, state: dict) -> pd.Series:
    """
    Apply binning: bin={'low': ['a','b'], 'high': ['c','d']}.

    Values not in the mapping are kept unchanged. The mapping is validated and
    cached when it is first built from the estimation data, so prediction reuses
    it; the cache key is per variable, so ``bin`` and ``bin2`` do not collide.
    """
    bin_key = bin_mapping_state_key(str(series.name))
    if bin_key not in state:
        if not bins:
            raise ValueError(
                f"The binning of i() for the variable `{series.name}` is empty. "
                "Map each new level to the values it bins, as in "
                "{'low': ['a', 'b']}."
            )
        mapping = {}
        duplicated = []
        for new_level, old_levels in bins.items():
            if isinstance(old_levels, str) or not isinstance(old_levels, Iterable):
                old_levels = [old_levels]
            old_levels = list(old_levels)
            if not old_levels:
                raise ValueError(
                    f"The binning of i() assigns no values of the variable "
                    f"`{series.name}` to the bin {new_level!r}."
                )
            for old in old_levels:
                if old in mapping and old not in duplicated:
                    duplicated.append(old)
                mapping[old] = new_level
        if duplicated:
            raise ValueError(
                f"The binning of i() assigns the values {duplicated!r} of the "
                f"variable `{series.name}` to more than one bin. Each value may "
                "belong to only one bin."
            )
        observed = set(series.dropna().unique())
        unmatched = [old for old in mapping if old not in observed]
        if unmatched:
            raise ValueError(
                f"The binning of i() names values that do not occur in the "
                f"variable `{series.name}`. No match was found for {unmatched!r}."
            )
        state[bin_key] = mapping
    return series.replace(state[bin_key])


def get_or_create_contrast_state(
    encoder_state: dict[str, Any], variable: str
) -> dict[str, Any]:
    """Return the nested formulaic encoder state used for one i() contrast."""
    return encoder_state.setdefault(contrast_state_key(variable), {})


def unwrap_factor_values(value: Any) -> Any:
    """Return raw values from formulaic's FactorValues proxy."""
    # formulaic internal: FactorValues is a wrapt proxy; `.__wrapped__` exposes
    # the object consumed by pyfixest's custom encoders.
    return value.__wrapped__ if isinstance(value, FactorValues) else value


def contrast_state_key(variable: str) -> str:
    """Return pyfixest's formulaic encoder-state key for i() contrasts."""
    return f"{_CONTRASTS_PREFIX}{variable}{_CONTRASTS_SUFFIX}"


def is_i_contrast_state_key(key: str) -> bool:
    """Return whether a key stores pyfixest i() contrast state."""
    return key.startswith(_CONTRASTS_PREFIX) and key.endswith(_CONTRASTS_SUFFIX)


def variable_from_contrast_state_key(key: str) -> str:
    """Extract the variable name from a pyfixest i() contrast state key."""
    return key[len(_CONTRASTS_PREFIX) : -len(_CONTRASTS_SUFFIX)]


def get_contrast_levels(encoder_state: Mapping[str, Any], variable: str) -> list[Any]:
    """Return stored i() contrast levels for a variable."""
    return encoder_state[contrast_state_key(variable)]["levels"]


def set_contrast_levels(
    encoder_state: Mapping[str, Any], variable: str, levels: list[Any]
) -> None:
    """Store i() contrast levels in pyfixest's nested formulaic encoder state."""
    encoder_state[contrast_state_key(variable)].update({"levels": levels})


def bin_mapping_state_key(variable: str) -> str:
    """Return pyfixest's formulaic encoder-state key for binned i() levels."""
    return f"{_BIN_MAPPING_PREFIX}{variable}{_BIN_MAPPING_SUFFIX}"


def encode_contrasts_with_state(
    data: pd.Series,
    *,
    ref: Hashable | None,
    levels: Iterable[Any] | None,
    reduced_rank: bool,
    state: dict[str, Any],
    model_spec: ModelSpec,
) -> pd.DataFrame:
    """Call formulaic encode_contrasts with explicit state injection."""
    # formulaic internal: `_state`/`_spec` are normally supplied by formulaic's
    # stateful_transform machinery; pyfixest calls the transform directly.
    encoded = encode_contrasts(
        data,
        contrasts=TreatmentContrasts(base=ref if ref is not None else UNSET),
        levels=levels,
        reduced_rank=reduced_rank,
        output="pandas",
        _state=state,
        _spec=model_spec,
    )
    return unwrap_factor_values(encoded)


_CONTRASTS_PREFIX: Final[str] = "__contrasts_"
_CONTRASTS_SUFFIX: Final[str] = "__"
_BIN_MAPPING_PREFIX: Final[str] = "__bin_mapping_"
_BIN_MAPPING_SUFFIX: Final[str] = "__"
