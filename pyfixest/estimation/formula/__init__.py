"""Internal formula parsing and model-matrix construction."""

from typing import Final

from formulaic.parser import DefaultFormulaParser

FORMULAIC_FEATURE_FLAG: Final[DefaultFormulaParser.FeatureFlags] = (
    DefaultFormulaParser.FeatureFlags.DEFAULT
)
