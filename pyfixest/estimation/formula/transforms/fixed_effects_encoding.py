import functools
import itertools
from typing import Final

import pandas as pd
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Operator, OrderedSet
from formulaic.utils.stateful_transforms import stateful_transform

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"


@stateful_transform
def encode_fixed_effects(*args, _state=None, _metadata=None, _spec=None):
    """Encode fixed effect interactions for model matrix construction."""
    data = pd.concat(args, axis=1)
    if FIXED_EFFECT_ENCODING not in _state:
        data[FIXED_EFFECT_ENCODING] = data.groupby(data.columns.tolist()).ngroup()
        _state[FIXED_EFFECT_ENCODING] = data.dropna(
            subset=[FIXED_EFFECT_ENCODING]
        ).drop_duplicates()
        return data[FIXED_EFFECT_ENCODING]

    return data.merge(
        _state[FIXED_EFFECT_ENCODING], on=data.columns.tolist(), how="left"
    )[FIXED_EFFECT_ENCODING]


class _FixedEffectsOperatorResolver(DefaultOperatorResolver):
    def __init__(self):
        super().__init__()

    @property
    def operators(self) -> list[Operator]:
        operators = [
            operator for operator in super().operators if operator.symbol != "^"
        ]

        operators.append(
            Operator(
                symbol="^",
                arity=2,
                precedence=500,
                associativity="left",
                to_terms=lambda *term_sets: OrderedSet(
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*term_sets)
                ),
            )
        )
        return operators
