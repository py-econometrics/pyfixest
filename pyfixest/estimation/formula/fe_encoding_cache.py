from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    FIXED_EFFECT_ENCODING,
)


@dataclass
class FixedEffectEncodingCache:
    """Full-input FE codes for one runner block with fixed data and row order.

    Entries precede Formulaic's missing-row selection and singleton removal.
    Each materialization receives its own prediction state. The runner must
    replace this cache when input data, split, context, or FE specification changes.
    """

    _entries: dict[tuple[str, ...], tuple[pd.Series, dict[str, Any]]] = field(
        default_factory=dict
    )

    @contextmanager
    def transform(
        self, *, data: pd.DataFrame, original: Callable
    ) -> Iterator[Callable]:
        """Temporarily wrap the standard FE transform without retaining fit inputs."""
        cache: FixedEffectEncodingCache | None = self
        source: pd.DataFrame | None = data

        @wraps(original)
        def encode(*args, **kwargs):
            state = kwargs.get("_state")
            # Prediction with an existing encoding must use the original merge.
            if (
                cache is None
                or source is None
                or state is None
                or FIXED_EFFECT_ENCODING in state
                or not args
                or not all(_is_source_column(arg, source) for arg in args)
            ):
                return original(*args, **kwargs)

            key = tuple(arg.name for arg in args)
            entry = cache._entries.get(key)
            if entry is None:
                codes = original(*args, **kwargs)
                # Retain immutable codes; never share mutable prediction state.
                codes.to_numpy(copy=False).setflags(write=False)
                cache._entries[key] = (codes, deepcopy(state))
                return codes.copy(deep=False)
            codes, encoding_state = entry
            state.update(deepcopy(encoding_state))
            return codes.copy(deep=False)

        try:
            yield encode
        finally:
            # Even if Formulaic retains the callable, it cannot retain this cache
            # or use training codes on a later prediction materialization.
            cache = None
            source = None


def _is_source_column(value: Any, data: pd.DataFrame) -> bool:
    """Only reuse direct data columns, never evaluated/context expressions.

    Memory sharing is an O(1) identity guard, not a scan or hash of N rows.
    Extension arrays (including categorical, string and nullable columns) are
    eligible only when both Series wrap the same array object.
    """
    if not isinstance(value, pd.Series) or value.name not in data.columns:
        return False
    source = data[value.name]
    if not isinstance(source, pd.Series) or not value.index.equals(source.index):
        return False
    if not isinstance(value.dtype, np.dtype) or not isinstance(source.dtype, np.dtype):
        return value.array is source.array
    return (
        _same_view(value.to_numpy(copy=False), source.to_numpy(copy=False))
        and value.dtype == source.dtype
    )


def _same_view(left: np.ndarray, right: np.ndarray) -> bool:
    # Sharing a buffer alone would also accept reversed or offset views.
    return (
        left.shape == right.shape
        and left.strides == right.strides
        and left.__array_interface__["data"][0] == right.__array_interface__["data"][0]
    )
