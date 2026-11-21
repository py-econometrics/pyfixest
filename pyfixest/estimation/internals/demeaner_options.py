from __future__ import annotations

import warnings

from pyfixest.demeaners import (
    AnyDemeaner,
    LsmrDemeaner,
)
from pyfixest.utils.dev_utils import _find_stack_level


def _resolve_demeaner(demeaner: AnyDemeaner | None) -> AnyDemeaner:
    """Return the typed demeaner configuration, defaulting to within LSMR."""
    if demeaner is None:
        return LsmrDemeaner()
    if not isinstance(demeaner, AnyDemeaner):
        raise TypeError(
            f"demeaner must be a MapDemeaner or LsmrDemeaner; got {type(demeaner).__name__}. "
            "For example, pass MapDemeaner(backend='rust') or omit demeaner for the default. "
            "See https://pyfixest.org/how-to/demeaner-backends.html#usage"
        )
    return demeaner


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
