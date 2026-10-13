from __future__ import annotations

import keyword
import unicodedata
import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from formulaic.parser.types import Factor, Term
from formulaic.utils.stateful_transforms import stateful_transform

from pyfixest.errors import FixedEffectEvaluationError, FormulaSyntaxError
from pyfixest.utils.dev_utils import _find_stack_level

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"
_FIXED_EFFECT_TRANSFORM: Final[str] = "__fixed_effect__"


# Prediction matches labels only within the kind seen during fitting, so
# booleans, numbers, and strings never match each other.
_NUMERIC_TYPES: Final[frozenset[str]] = frozenset(
    {"integer", "floating", "mixed-integer-float"}
)
_MIXED_TYPES: Final[frozenset[str]] = frozenset(
    {"mixed", "mixed-integer", "unknown-array"}
)
_TYPE_ALIASES: Final[dict[str, str]] = {
    "datetime64": "datetime",
    "timedelta64": "timedelta",
}


@dataclass(frozen=True, slots=True)
class FixedEffectEncoding:
    """Observed factor levels and combinations in fitted FE-code order."""

    term: Term
    levels: tuple[pd.Index, ...]
    combinations: pd.RangeIndex | pd.MultiIndex
    kinds: tuple[str | None, ...]

    @property
    def variable(self) -> str:
        """Original factor labels used in fixed-effect output and diagnostics."""
        return ":".join(str(factor) for factor in self.term.factors)

    @classmethod
    def fit(
        cls, *, term: Term, values: tuple[pd.Series, ...]
    ) -> tuple[FixedEffectEncoding, pd.Series]:
        """Learn the level mappings and return the training codes."""
        kinds = tuple(cls._label_kind(column=column) for column in values)
        # Sorted factorization preserves declared categorical order, returning
        # codes and observed levels together.
        factorized = [pd.factorize(column, sort=True) for column in values]
        per_factor = [codes for codes, _ in factorized]
        levels = tuple(pd.Index(levels) for _, levels in factorized)
        if len(values) == 1:
            # The sorted level index already defines the complete code order.
            combinations = pd.RangeIndex(len(levels[0]))
            codes = per_factor[0].astype(np.float64)
        else:
            valid = np.all(np.column_stack(per_factor) >= 0, axis=1)
            observed = pd.MultiIndex.from_arrays(per_factor)
            combinations = observed[valid].unique().sort_values()
            codes = combinations.get_indexer(observed).astype(np.float64)
        codes[codes < 0] = np.nan
        return (
            cls(term=term, levels=levels, combinations=combinations, kinds=kinds),
            pd.Series(codes, index=values[0].index),
        )

    def encode(self, *, values: tuple[pd.Series, ...]) -> pd.Series:
        """Map values to fitted FE codes under exact label equality; unmatched values are NaN."""
        per_factor = []
        for level, kind, column in zip(self.levels, self.kinds, values, strict=True):
            query_kind = self._label_kind(column=column)
            if None not in (kind, query_kind) and kind != query_kind:
                raise FixedEffectEvaluationError(
                    f"Fixed-effect column `{column.name}` contains {query_kind} "
                    f"values (dtype {column.dtype}), but the model was fitted on "
                    f"{kind} values (dtype {level.dtype}). Convert the column to "
                    f"{kind} values before calling predict()."
                )
            per_factor.append(self._match_levels(levels=level, column=column))
        if len(values) == 1:
            codes = per_factor[0].astype(np.float64)
        else:
            codes = self.combinations.get_indexer(
                pd.MultiIndex.from_arrays(per_factor)
            ).astype(np.float64)
        codes[codes < 0] = np.nan
        return pd.Series(codes, index=values[0].index)

    @staticmethod
    def _label_kind(*, column: pd.Series) -> str | None:
        """Return the label kind, or None when the column has no labels."""
        if not column.notna().any():
            return None
        labels = (
            column.cat.categories
            if isinstance(column.dtype, pd.CategoricalDtype)
            else column
        )
        inferred = pd.api.types.infer_dtype(labels, skipna=True)
        if inferred in _MIXED_TYPES:
            raise FixedEffectEvaluationError(
                f"Fixed-effect column `{column.name}` mixes value types or "
                "contains unsupported values. Convert it to a single type, such "
                "as numbers, booleans, strings, or datetimes."
            )
        if inferred in _NUMERIC_TYPES:
            return "numeric"
        kind = _TYPE_ALIASES.get(inferred, inferred)
        if kind != "datetime":
            return kind
        if isinstance(labels.dtype, pd.DatetimeTZDtype):
            return "timezone-aware datetime"
        if labels.dtype != object:
            return "timezone-naive datetime"
        aware = {
            getattr(value, "tzinfo", None) is not None for value in labels.dropna()
        }
        if len(aware) > 1:
            raise FixedEffectEvaluationError(
                f"Fixed-effect column `{column.name}` mixes timezone-aware and "
                "timezone-naive datetimes. Convert it to a single type."
            )
        return f"timezone-{'aware' if aware.pop() else 'naive'} datetime"

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        """Unwrap NumPy scalars so labels compare without rounding or unit changes."""
        # `.item()` turns datetime64/timedelta64 into ints, dates, or datetimes
        # depending on the unit; pandas scalars keep the value at any unit.
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value)
        if isinstance(value, np.timedelta64):
            return pd.Timedelta(value)
        return value.item() if isinstance(value, np.generic) else value

    @classmethod
    def _match_levels(cls, *, levels: pd.Index, column: pd.Series) -> np.ndarray:
        """Return fitted positions under exact label equality.

        Labels are looked up as Python scalars without casting either side,
        so rounding, overflow, or wraparound cannot create a match.
        """
        # Identical NumPy numeric/bool dtypes need neither coercion nor custom
        # index semantics. Keep this common prediction path vectorized.
        if (
            isinstance(levels.dtype, np.dtype)
            and levels.dtype.kind in "biuf"
            and levels.dtype == column.dtype
        ):
            return levels.get_indexer(column)
        codes, uniques = pd.factorize(column)
        lookup = {cls._python_scalar(value): code for code, value in enumerate(levels)}
        matched = np.fromiter(
            (lookup.get(cls._python_scalar(value), -1) for value in uniques),
            dtype=np.intp,
            count=len(uniques),
        )
        # Missing query codes (-1) select the sentinel, even with no uniques.
        return np.append(matched, -1)[codes]

    def decoded_values(self, *, codes: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return original level values for the requested fitted FE codes."""
        if len(self.levels) == 1:
            return (self.levels[0].to_numpy()[codes],)
        combinations = self.combinations[codes]
        return tuple(
            level.to_numpy()[
                combinations.get_level_values(position).to_numpy(dtype=np.int64)
            ]
            for position, level in enumerate(self.levels)
        )


def wrap_fixed_effect(term: Term) -> Term:
    """Expose ordinary column lookups to Formulaic and retain the original term."""
    for factor in term.factors:
        if factor.eval_method is not Factor.EvalMethod.LOOKUP:
            raise FormulaSyntaxError(
                f"Unsupported fixed-effect term `{term}`. Fixed effects must be "
                f"an existing data column or an interaction of multiple data columns "
                "specified with `:`."
            )
        name = factor.expr
        if not name.isidentifier() or keyword.iskeyword(name):
            raise FormulaSyntaxError(
                f"Fixed-effect column name `{name}` must be a valid Python "
                "identifier and cannot be a Python keyword. Rename the column "
                "before estimation."
            )
        if unicodedata.normalize("NFKC", name) != name:
            raise FormulaSyntaxError(
                f"Fixed-effect column name `{name}` must use Python's normalized "
                "identifier spelling. Rename the column before estimation."
            )
    arguments = ", ".join(factor.expr for factor in term.factors)
    return Term(
        [
            Factor(
                f"{_FIXED_EFFECT_TRANSFORM}({arguments})",
                eval_method=Factor.EvalMethod.PYTHON,
                metadata={"term": term},
            )
        ]
    )


@stateful_transform
def encode_fixed_effects(*_columns, _state=None, _metadata=None, _context: Any = None):
    """Look up data columns and code their observed level combinations.

    The column arguments expose dependencies to Formulaic. Read the columns
    from its data layer to reject observation vectors captured in context.
    Only fitted level mappings are retained for prediction; observation values
    always come from data/newdata.
    """
    state = cast(MutableMapping[str, Any], _state)
    metadata = cast(Mapping[str, Any], _metadata)
    term: Term = metadata["term"]
    unseen_message: str | None = None
    try:
        columns = []
        for factor in term.factors:
            try:
                column = _context.data[factor.expr]
            except KeyError as exc:
                raise FixedEffectEvaluationError(
                    f"Column `{factor.expr}` required by fixed effect `{term}` "
                    "is missing."
                ) from exc
            columns.append(column.rename(str(factor)))
        values = tuple(columns)
        if FIXED_EFFECT_ENCODING not in state:
            encoding, codes = FixedEffectEncoding.fit(term=term, values=values)
            state[FIXED_EFFECT_ENCODING] = encoding
        else:
            encoding = cast(FixedEffectEncoding, state[FIXED_EFFECT_ENCODING])
            codes = encoding.encode(values=values)
            unseen = codes.isna().to_numpy() & np.all(
                np.column_stack([column.notna().to_numpy() for column in values]),
                axis=1,
            )
            if unseen.any():
                missing = (
                    # Discard Series labels to combine series by position.
                    pd.concat(
                        [column.reset_index(drop=True) for column in values],
                        axis=1,
                    )
                    .iloc[unseen]
                    .drop_duplicates()
                )
                unseen_message = (
                    f"{missing.shape[0]} unseen level(s) for fixed effect "
                    f"`{encoding.variable}`: {missing.iloc[:20]}\n"
                    "Predictions for affected observations will be NaN"
                )
    except FixedEffectEvaluationError:
        raise
    except Exception as exc:
        raise FixedEffectEvaluationError(
            f"Unable to encode fixed effect `{term}`. [{type(exc).__name__}: {exc}]"
        ) from exc
    if unseen_message is not None:
        warnings.warn(unseen_message, UserWarning, stacklevel=_find_stack_level())
    return codes
