from __future__ import annotations

import warnings

from pyfixest.demeaners import LsmrDemeaner


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
