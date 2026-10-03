from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from pyfixest.core.demean import Preconditioner
from pyfixest.estimation.formula.fe_encoding_cache import FixedEffectEncodingCache
from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.internals.demean_ import DemeanedData
from pyfixest.estimation.internals.families import PROBIT
from pyfixest.estimation.internals.model_state import (
    GlmEstimationOptions,
    ModelDescription,
    SampleSplit,
)
from pyfixest.estimation.models.feglm_ import Feglm


class Feprobit(Feglm):
    "Class for the estimation of a fixed-effects probit model."

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        *,
        options: GlmEstimationOptions,
        lookup_demeaned_data: dict[frozenset[int], DemeanedData],
        lookup_preconditioner: dict[frozenset[int], Preconditioner] | None = None,
        fixed_effect_encoding_cache: FixedEffectEncodingCache | None = None,
        sample_split: SampleSplit | None = None,
    ):
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            options=options,
            lookup_demeaned_data=lookup_demeaned_data,
            lookup_preconditioner=lookup_preconditioner,
            fixed_effect_encoding_cache=fixed_effect_encoding_cache,
            sample_split=sample_split,
            family=PROBIT,
        )

    def _describe_model(self, **kwargs: Any) -> ModelDescription:
        """Name the probit estimation function."""
        return replace(super()._describe_model(**kwargs), method="feglm-probit")
