from __future__ import annotations

import warnings

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrDemeaner,
    MapDemeaner,
)


def _resolve_demeaner(demeaner: AnyDemeaner | None) -> AnyDemeaner:
    """Return the typed demeaner configuration, defaulting to MapDemeaner."""
    return demeaner if demeaner is not None else MapDemeaner()


def _warn_if_experimental_torch_demeaner(demeaner: object) -> None:
    if isinstance(demeaner, LsmrDemeaner) and demeaner.backend == "torch":
        warnings.warn(
            (
                "The torch LSMR demeaner backend is experimental. "
                "Behavior and performance may change in future releases."
            ),
            UserWarning,
            stacklevel=3,
        )
