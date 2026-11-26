from __future__ import annotations

import warnings

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrDemeaner,
    MapDemeaner,
)
from pyfixest.utils.dev_utils import _find_stack_level


def _resolve_demeaner(demeaner: AnyDemeaner | None) -> AnyDemeaner:
    """Return the typed demeaner configuration, defaulting to within LSMR."""
    if demeaner is None:
        return LsmrDemeaner()
    if isinstance(demeaner, (MapDemeaner, LsmrDemeaner)):
        return demeaner
    raise TypeError(
        "demeaner must be created with pf.MapDemeaner() or pf.LsmrDemeaner(); "
        f"got {type(demeaner).__name__}. To select a demeaner backend, pass for example "
        "demeaner=pf.MapDemeaner(backend='rust')."
    )


def _warn_if_experimental_torch_demeaner(demeaner: object) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "torch":
        warnings.warn(
            (
                "The torch LSMR demeaner backend is experimental. "
                "Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=_find_stack_level(),
        )
