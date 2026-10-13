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


@dataclass(frozen=True, slots=True)
class FixedEffectEncoding:
    """Observed factor levels and combinations in fitted FE-code order."""

    term: Term
    levels: tuple[pd.Index, ...]
    combinations: pd.RangeIndex | pd.MultiIndex

    @property
    def variable(self) -> str:
        """Original factor labels used in fixed-effect output and diagnostics."""
        return ":".join(str(factor) for factor in self.term.factors)

    @classmethod
    def fit(
        cls, *, term: Term, values: tuple[pd.Series, ...]
    ) -> tuple[FixedEffectEncoding, pd.Series]:
        """Learn the level mappings and return the training codes."""
        # Sorted factorization preserves declared categorical order and safely
        # orders mixed numbers/strings, returning codes and observed levels together.
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
            cls(term=term, levels=levels, combinations=combinations),
            pd.Series(codes, index=values[0].index),
        )

    def encode(self, *, values: tuple[pd.Series, ...]) -> pd.Series:
        """Use the fitted indexes for both training and prediction matching."""
        per_factor = [
            level.get_indexer(column)
            for level, column in zip(self.levels, values, strict=True)
        ]
        if len(values) == 1:
            codes = per_factor[0].astype(np.float64)
        else:
            codes = self.combinations.get_indexer(
                pd.MultiIndex.from_arrays(per_factor)
            ).astype(np.float64)
        codes[codes < 0] = np.nan
        return pd.Series(codes, index=values[0].index)

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
