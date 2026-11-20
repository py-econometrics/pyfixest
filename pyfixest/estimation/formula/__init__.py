from __future__ import annotations

from collections.abc import Callable
from typing import Final

from formulaic.parser import DefaultFormulaParser

from pyfixest.estimation.formula.transforms.factor_interaction import factor_interaction
from pyfixest.estimation.formula.transforms.fixed_effects_encoding import (
    encode_fixed_effect,
)
from pyfixest.estimation.formula.transforms.scale import center, scale

FORMULAIC_FEATURE_FLAG: Final[DefaultFormulaParser.FeatureFlags] = (
    DefaultFormulaParser.FeatureFlags.ALL
)

FORMULAIC_TRANSFORMS: Final[dict[str, Callable]] = {
    "i": factor_interaction,  # fixest::i()-style syntax
    "__fixed_effect__": encode_fixed_effect,
    "center": center,  # NaN-aware replacements for formulaic's transforms
    "scale": scale,
}
