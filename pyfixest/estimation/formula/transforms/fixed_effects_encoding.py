from typing import Final

import pandas as pd
from formulaic.utils.stateful_transforms import stateful_transform

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"


@stateful_transform
def encode_fixed_effects(*args, _state=None, _metadata=None, _spec=None):
    """Encode fixed effect interactions for model matrix construction."""
    data = pd.concat(args, axis=1)
    if FIXED_EFFECT_ENCODING not in _state:
        data[FIXED_EFFECT_ENCODING] = data.groupby(data.columns.tolist()).ngroup()
        encoded_state = data.dropna(subset=[FIXED_EFFECT_ENCODING]).drop_duplicates()
        _state[FIXED_EFFECT_ENCODING] = encoded_state
        return data[FIXED_EFFECT_ENCODING]

    return data.merge(
        _state[FIXED_EFFECT_ENCODING], on=data.columns.tolist(), how="left"
    )[FIXED_EFFECT_ENCODING]
