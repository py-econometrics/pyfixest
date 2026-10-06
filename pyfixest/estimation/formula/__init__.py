from collections.abc import Callable
from typing import Final

from formulaic.parser import DefaultFormulaParser

from pyfixest.estimation.formula.transforms.factor_interaction import factor_interaction
from pyfixest.estimation.formula.transforms.misc import log

FORMULAIC_FEATURE_FLAG: Final[DefaultFormulaParser.FeatureFlags] = (
    DefaultFormulaParser.FeatureFlags.ALL
)

FORMULAIC_TRANSFORMS: Final[dict[str, Callable]] = {
    "i": factor_interaction,  # fixest::i()-style syntax
    "log": log,  # custom log settings infinite to nan
}
