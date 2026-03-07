import functools
import itertools

import pandas as pd
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Operator, OrderedSet
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def fixed_effect_interactions(*args, _state=None, _metadata=None, _spec=None):
    """Encode fixed effect interactions for model matrix construction."""
    data = pd.concat(args, axis=1)
    if "__id__" not in _state:
        print(_state)
        data["__id__"] = data.groupby(data.columns.tolist()).ngroup()
        _state["__id__"] = data.drop_duplicates()
        return data["__id__"]

    return data.merge(_state, on=data.columns, how="left")["__id__"]


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
