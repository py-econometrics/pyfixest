from __future__ import annotations

import keyword
import unicodedata
from collections.abc import MutableMapping
from typing import Any, Final, cast

import pandas as pd
from formulaic.parser.types import Factor, Term
from formulaic.utils.stateful_transforms import stateful_transform

from pyfixest.errors import FormulaSyntaxError

FIXED_EFFECT_ENCODING: Final[str] = "__fixed_effect_encoding__"
_FIXED_EFFECT_TRANSFORM: Final[str] = "__fixed_effect__"


def wrap_fixed_effect(term: Term) -> Term:
    """Expose ordinary column lookups to Formulaic as one encoded factor."""
    for factor in term.factors:
        if factor.eval_method is not Factor.EvalMethod.LOOKUP:
            raise FormulaSyntaxError(
                f"Unsupported fixed-effect term `{term}`. Fixed effects must be "
                f"an existing data column or an interaction of multiple data columns "
                "specified with `:`."
            )
        name = factor.expr
        if not name.isidentifier() or keyword.iskeyword(name):
            raise FormulaSyntaxError(
                f"Fixed-effect column name `{name}` must be a valid Python "
                "identifier and cannot be a Python keyword. Rename the column "
                "before estimation."
            )
        if unicodedata.normalize("NFKC", name) != name:
            raise FormulaSyntaxError(
                f"Fixed-effect column name `{name}` must use Python's normalized "
                "identifier spelling. Rename the column before estimation."
            )
    arguments = ", ".join(factor.expr for factor in term.factors)
    return Term(
        [
            Factor(
                f"{_FIXED_EFFECT_TRANSFORM}({arguments})",
                eval_method=Factor.EvalMethod.PYTHON,
            )
        ]
    )


@stateful_transform
def encode_fixed_effects(*args, _state=None, _metadata=None, _spec=None):
    """Encode fixed effect interactions for model matrix construction."""
    state = cast(MutableMapping[str, Any], _state)
    data = pd.concat(args, axis=1)
    if FIXED_EFFECT_ENCODING not in state:
        data[FIXED_EFFECT_ENCODING] = data.groupby(data.columns.tolist()).ngroup()
        encoded_state = data.dropna(subset=[FIXED_EFFECT_ENCODING]).drop_duplicates()
        state[FIXED_EFFECT_ENCODING] = encoded_state
        return data[FIXED_EFFECT_ENCODING]

    return data.merge(
        state[FIXED_EFFECT_ENCODING], on=data.columns.tolist(), how="left"
    )[FIXED_EFFECT_ENCODING]
