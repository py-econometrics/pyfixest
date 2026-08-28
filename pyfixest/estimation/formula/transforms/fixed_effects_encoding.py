from __future__ import annotations

from typing import Any, Final, cast

import pandas as pd
from formulaic.utils.stateful_transforms import stateful_transform

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"


@stateful_transform
def encode_fixed_effects(
    *args: pd.Series[Any],
    _state: dict[str, pd.DataFrame] | None = None,
    _metadata: Any = None,
    _spec: Any = None,
) -> pd.Series[Any]:
    """Encode fixed effect interactions for model matrix construction."""
    state = cast(dict[str, pd.DataFrame], _state)
    data = pd.concat(args, axis=1)
    if FIXED_EFFECT_ENCODING not in state:
        data[FIXED_EFFECT_ENCODING] = data.groupby(data.columns.tolist()).ngroup()
        encoded_state = data.dropna(subset=[FIXED_EFFECT_ENCODING]).drop_duplicates()
        state[FIXED_EFFECT_ENCODING] = encoded_state
        return data[FIXED_EFFECT_ENCODING]

    return data.merge(
        state[FIXED_EFFECT_ENCODING], on=data.columns.tolist(), how="left"
    )[FIXED_EFFECT_ENCODING]
