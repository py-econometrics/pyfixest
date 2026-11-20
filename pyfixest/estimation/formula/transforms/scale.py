from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def scale(
    data: Any,
    center: Any = True,
    scale: Any = True,
    ddof: float = 1,
    _state: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """Center and rescale `data` by statistics of its non-missing values.

    Missing values stay missing and are dropped later with the rest of the row.

    Parameters
    ----------
    data : array-like
        Values to transform.
    center : bool or array-like, default=True
        Subtract the column means (`True`), nothing (`False`), or the given values.
    scale : bool or array-like, default=True
        Divide by the column root mean squares after centering, which is the
        standard deviation when `center=True` (`True`), by nothing (`False`),
        or by the given values.
    ddof : float, default=1
        Delta degrees of freedom of the root mean square.
    """
    state = {} if _state is None else _state
    values = np.asarray(data, dtype=float)
    ddof = state.setdefault("ddof", ddof)

    if "center" not in state:
        if isinstance(center, bool):
            state["center"] = np.nanmean(values, axis=0) if center else None
        else:
            state["center"] = np.asarray(center)
    if state["center"] is not None:
        values = values - state["center"]

    if "scale" not in state:
        if isinstance(scale, bool) and scale:
            n_observed = np.sum(~np.isnan(values), axis=0)
            state["scale"] = np.sqrt(np.nansum(values**2, axis=0) / (n_observed - ddof))
        elif isinstance(scale, bool):
            state["scale"] = None
        else:
            state["scale"] = np.asarray(scale)
    if state["scale"] is not None:
        values = values / state["scale"]

    return values


@stateful_transform
def center(data: Any, _state: MutableMapping[str, Any] | None = None) -> np.ndarray:
    """Subtract the mean of the non-missing values from `data`."""
    return scale(data, scale=False, _state=_state)
