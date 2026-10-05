from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Final, cast

import pandas as pd
from formulaic.utils.stateful_transforms import stateful_transform
from formulaic.utils.variables import Variable

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"


# Register literal column-name dependencies using the hook added in Formulaic 1.2.2:
# https://github.com/matthewwardrop/formulaic/pull/268
@stateful_transform(
    get_required_variables=lambda *args, **kwargs: (
        Variable(arg, roles={Variable.Role.VALUE})
        for arg in args
        if isinstance(arg, str)
    )
)
def encode_fixed_effects(
    *args, _state=None, _metadata=None, _spec=None, _context: Any = None
):
    """Encode FE interactions, resolving literal column names from the context.

    Quoted lookups are passed as strings to avoid nested stateful Q() calls
    during Formulaic's dependency collection. Evaluated expressions remain
    Series, and the stored encoding is reused when predicting on new data.
    """
    state = cast(MutableMapping[str, Any], _state)
    data = pd.concat(
        [_context.data[arg] if isinstance(arg, str) else arg for arg in args], axis=1
    )
    if FIXED_EFFECT_ENCODING not in state:
        data[FIXED_EFFECT_ENCODING] = data.groupby(data.columns.tolist()).ngroup()
        encoded_state = data.dropna(subset=[FIXED_EFFECT_ENCODING]).drop_duplicates()
        state[FIXED_EFFECT_ENCODING] = encoded_state
        return data[FIXED_EFFECT_ENCODING]

    return data.merge(
        state[FIXED_EFFECT_ENCODING], on=data.columns.tolist(), how="left"
    )[FIXED_EFFECT_ENCODING]
